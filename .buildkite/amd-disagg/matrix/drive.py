#!/usr/bin/env python3
"""Drive the disagg model matrix on Crusoe.

Keeps at most MAX_INFLIGHT jobs on the cluster, polls them every POLL_S seconds
over ssh, and prints one line per event (SUBMIT, RESULT, EP BLOCKED, MATRIX
DONE). State lives in state.json, so a restart picks up where it stopped.
On the Spur login node it runs as the mxmatrix systemd user unit
(run_driver.sh); the account lingers, so the unit outlives every ssh session.

EP runs are gated on the pilot: the pilot runs alone among EP runs, and the
other EP runs start only once it passes. Every failed run is retried once on
its own (out-of-memory with a smaller KV cap), so the matrix keeps going
unattended; only a pilot that fails twice blocks the EP runs. A preflight
rejection (dirty node) is resubmitted without counting as a retry. A job whose
gsm8k makes no progress for STALL_S is cancelled and fails as class "stall".

Commands go in control.txt, one per line, and are consumed on the next loop:
  requeue <id> [kv=<GiB>] [mtu=<n>]   run <id> again with optional overrides
  unblock-ep                          let EP runs start again
  pause | resume                      stop / restart submitting new runs
  stop                                exit (running jobs are left alone)
"""
import base64
import json
import os
import re
import socket
import subprocess
import sys
import time

HOST = "csrikris@crs-spur.crusoe.amd.com"
# MX_LOCAL=1 runs the cluster commands directly (driver on the Spur login node).
LOCAL = os.environ.get("MX_LOCAL") == "1"
HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "state.json")
CONTROL = os.path.join(HERE, "control.txt")
RESULTS = os.path.join(HERE, "results.tsv")
LOCK = os.path.join(HERE, "driver.lock")
ME = f"{socket.gethostname()}:{os.getpid()}"
REMOTE = "/shared_nfs/csrikris/matrix"
POLL_S = int(os.environ.get("POLL_S", "300"))
MAX_INFLIGHT = 2
OOM_RETRY_KV_GIB = 24
MAX_PREFLIGHT_RESUBMITS = 2
# gsm8k completes requests every few seconds even on the slowest layout, so this
# long without progress is a hang (e.g. an EP collective that never returns).
STALL_S = 1200

MODELS = {
    "dsv4": ("DeepSeek-V4-Pro", "/shared_nfs/csrikris/dsv4_env.sh"),
    "minimax": ("MiniMax-M3-MXFP8", "/shared_nfs/csrikris/minimax_m3_env.sh"),
    "kimi": ("Kimi-K2.6-MXFP4", "/shared_nfs/csrikris/kimi_k26_env.sh"),
    "glm": ("GLM-5.2-FP8", "/shared_nfs/csrikris/glm52_env.sh"),
}
PILOT = "minimax_2p1d-ep"
TP_ORDER = [
    "dsv4_1p1d-tp8", "minimax_1p1d-tp8", "kimi_1p1d-tp8", "glm_1p1d-tp8",
    "dsv4_2p1d-tp8", "minimax_2p1d-tp8", "kimi_2p1d-tp8", "glm_2p1d-tp8",
]
EP_ORDER = [
    PILOT, "minimax_1p1d-ep", "minimax_1p2d-ep",
    "kimi_1p1d-ep", "kimi_2p1d-ep", "kimi_1p2d-ep",
    "glm_1p1d-ep", "glm_2p1d-ep", "glm_1p2d-ep",
    "dsv4_1p1d-ep", "dsv4_2p1d-ep", "dsv4_1p2d-ep",
]
ACTIVE = ("RUNNING", "COMPLETING", "PENDING", "CONFIGURING", "REQUEUED")
TSV_COLS = ["id", "model", "layout", "attempt", "job", "verdict", "score", "nodes",
            "image", "vllm", "kv_gib", "mtu", "class", "note"]


def say(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def new_state():
    runs = {}
    for rid in TP_ORDER + EP_ORDER:
        model, layout = rid.split("_", 1)
        runs[rid] = {"model": model, "layout": layout, "kind": "ep" if layout.endswith("-ep") else "tp",
                     "status": "queued", "next": {}, "attempts": [], "preflight_resubmits": 0}
    return {"runs": runs, "ep_gate": "pilot", "paused": False}


def load():
    st = new_state()
    if os.path.exists(STATE):
        with open(STATE) as f:
            saved = json.load(f)
        # Runs added to TP_ORDER/EP_ORDER after the state was written join as queued.
        for rid, r in st["runs"].items():
            saved["runs"].setdefault(rid, r)
        st = saved
    return st


def save(st):
    tmp = STATE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, STATE)
    rows = ["\t".join(TSV_COLS)]
    for rid in TP_ORDER + EP_ORDER:
        r = st["runs"][rid]
        for i, a in enumerate(r["attempts"], 1):
            rows.append("\t".join(str(x) for x in [
                rid, MODELS[r["model"]][0], r["layout"], i, a.get("job") or "-", a.get("verdict") or "-",
                a.get("score") or "-", a.get("nodes") or "-", a.get("image") or "-", a.get("vllm") or "-",
                a.get("kv") or "default", a.get("mtu") or "default", a.get("class") or "-",
                a.get("note") or "-"]))
    with open(RESULTS, "w") as f:
        f.write("\n".join(rows) + "\n")


def ssh(body, timeout=300):
    if LOCAL:
        try:
            p = subprocess.run(["bash", "-s"], input=body.encode(), stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
        return p.stdout.decode(errors="replace").replace("\r", "")
    script = ("export TERM=dumb PAGER=cat GIT_PAGER=cat; PS2=''; stty -echo\n" + body + "\nexit\n")
    # Bytes, not text=True: universal newlines would turn a stray \r inside a
    # remote line into a line break.
    try:
        p = subprocess.run(["ssh", "-tt", "-o", "ConnectTimeout=20", HOST], input=script.encode(),
                           capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    if p.returncode == 255:
        return None
    return p.stdout.decode(errors="replace").replace("\r", "")


def push_results_cmd():
    with open(RESULTS, "rb") as f:
        b64 = base64.encodebytes(f.read()).decode()
    return f"base64 -d > {REMOTE}/results.tsv <<'B64END'\n{b64}B64END\n"


def inflight(st):
    return [rid for rid, r in st["runs"].items() if r["status"] in ("submitting", "inflight")]


def classify_finish(st, rid, a):
    """Decide what happens after attempt `a` of run `rid` ended."""
    r = st["runs"][rid]
    cls = a.get("class", "unknown")
    if "requeue_after" in r:
        r["status"], r["next"] = "queued", r.pop("requeue_after")
        if rid == PILOT:
            st["ep_gate"] = "pilot"
        say(f"REQUEUE {rid}: requested by control {r['next']}")
        return
    if a["verdict"] == "PASS":
        r["status"] = "done"
        if rid == PILOT and st["ep_gate"] == "pilot":
            st["ep_gate"] = "open"
            say(f"EP gate open: pilot {rid} passed")
        return
    if cls == "preflight" and r["preflight_resubmits"] < MAX_PREFLIGHT_RESUBMITS:
        r["preflight_resubmits"] += 1
        r["status"], r["next"] = "queued", {k: a[k] for k in ("kv", "mtu") if a.get(k)}
        say(f"REQUEUE {rid}: preflight rejected a node, resubmitting")
        return
    retries = sum(1 for x in r["attempts"] if x.get("class") != "preflight") - 1
    if retries < 1:
        if cls == "oom":
            r["status"], r["next"] = "queued", {"kv": OOM_RETRY_KV_GIB}
            say(f"REQUEUE {rid}: out of memory, retrying once with a {OOM_RETRY_KV_GIB} GiB KV cap")
        else:
            r["status"], r["next"] = "queued", {k: a[k] for k in ("kv", "mtu") if a.get(k)}
            say(f"REQUEUE {rid}: failed ({cls}), retrying once")
        return
    r["status"] = "done"
    if rid == PILOT and st["ep_gate"] != "blocked":
        st["ep_gate"] = "blocked"
        say(f"EP BLOCKED: pilot {rid} job {a.get('job')} failed twice ({cls}); no new EP runs until triage")


def poll(st):
    ids = inflight(st)
    if not ids:
        return True
    jobs = [st["runs"][i]["attempts"][-1].get("job") for i in ids]
    body = f"cd {REMOTE}\n" + push_results_cmd()
    body += "bash status.sh " + " ".join(j for j in jobs if j) + "\n"
    pending = [i for i in ids if st["runs"][i]["status"] == "submitting"]
    if pending:
        body += "grep -aE '^(" + "|".join(pending) + ") ' jobs.log | sed 's/^/MX_LOGGED /'\n"
    out = ssh(body)
    if out is None:
        say("DRIVER WARN: status poll over ssh failed; will retry next loop")
        return False
    for m in re.finditer(r"MX_LOGGED (\S+) (\d+) ", out):
        rid, job = m.group(1), m.group(2)
        r = st["runs"].get(rid)
        if r and r["status"] == "submitting" and not r["attempts"][-1].get("job"):
            r["attempts"][-1]["job"], r["status"] = job, "inflight"
            say(f"SUBMIT {rid} job={job} (recovered from jobs.log)")
    by_job = {}
    for m in re.finditer(r"MX_STATUS (\d+)\|(.*)", out):
        fields = dict(kv.split("=", 1) for kv in m.group(2).split("|") if "=" in kv)
        by_job[m.group(1)] = fields
    for rid in ids:
        r = st["runs"][rid]
        a = r["attempts"][-1]
        if r["status"] == "submitting":
            # Never confirmed: no MX_JOB and nothing in jobs.log. Safe to submit again.
            if time.time() - a["submitted"] > 2 * POLL_S:
                r["attempts"].pop()
                r["status"] = "queued"
                say(f"SUBMIT FAILED {rid}: no job id came back; requeued")
            continue
        f = by_job.get(a["job"])
        if not f:
            continue
        a.update({k: f.get(k, "-") for k in ("nodes", "image", "vllm")})
        state, rc, gate = f.get("state", "-"), f.get("rc", "-"), f.get("gate", "-")
        if state in ACTIVE and rc == "-":
            a["gone_polls"] = 0
            idle = f.get("idle", "-")
            if gate == "-" and idle.isdigit() and int(idle) > STALL_S and not a.get("stalled"):
                a["stalled"] = f"no gsm8k progress for {int(idle) // 60} min; cancelled by the driver"
                ssh(f"scancel {a['job']}", timeout=60)
                say(f"STALLED {rid} job={a['job']}: {a['stalled']}")
            if a.get("seen_state") != state:
                a["seen_state"] = state
                say(f"STATE {rid} job={a['job']} {state} healthy={f.get('healthy')} nodes={a['nodes']}")
            continue
        if rc == "-" and state == "-" and gate == "-":
            # scontrol lost the job (or timed out) and there is no verdict yet.
            a["gone_polls"] = a.get("gone_polls", 0) + 1
            if a["gone_polls"] < 2:
                continue
        if state in ACTIVE:
            # Verdict is in but the job is still tearing down; wait for the nodes.
            continue
        score = gate.split("=", 1)[1] if "=" in gate else "-"
        verdict = "PASS" if rc == "0" or (rc == "-" and gate.startswith("PASS")) else "FAIL"
        a.update({"verdict": verdict, "score": score, "class": f.get("class", "-"),
                  "note": f.get("note", "-"), "state": state, "ended": time.time()})
        if a.get("stalled") and verdict == "FAIL":
            a.update({"class": "stall", "note": a["stalled"]})
        r["status"] = "ended"
        say(f"RESULT {rid} {verdict} score={score} job={a['job']} nodes={a['nodes']} "
            f"kv={a.get('kv') or 'default'} class={a['class']} state={state} note={a['note']}")
        classify_finish(st, rid, a)
    return True


def pick(st):
    runs = st["runs"]
    fl = inflight(st)
    if st["paused"] or len(fl) >= MAX_INFLIGHT:
        return None
    ep_inflight = any(runs[i]["kind"] == "ep" for i in fl)
    tp_q = [i for i in TP_ORDER if runs[i]["status"] == "queued"]
    if st["ep_gate"] == "pilot":
        ep_q = [PILOT] if runs[PILOT]["status"] == "queued" and not ep_inflight else []
    elif st["ep_gate"] == "open":
        ep_q = [i for i in EP_ORDER if runs[i]["status"] == "queued"]
    else:
        ep_q = []
    if ep_q and not ep_inflight:
        return ep_q[0]
    if tp_q:
        return tp_q[0]
    if ep_q and st["ep_gate"] == "open":
        return ep_q[0]
    return None


def submit(st, rid):
    r = st["runs"][rid]
    nxt = r.get("next") or {}
    a = {"submitted": time.time(), "kv": nxt.get("kv"), "mtu": nxt.get("mtu")}
    r["attempts"].append(a)
    r["status"], r["next"] = "submitting", {}
    save(st)
    env = f"MX_ID={rid}"
    if a["kv"]:
        env += f" KV_GIB={a['kv']}"
    if a["mtu"]:
        env += f" MORI_IB_PATH_MTU={a['mtu']}"
    body = (f"cd {REMOTE} && {env} bash submit.sh {MODELS[r['model']][1]} {r['layout']} 2>&1"
            " | grep -aE 'MX_|ERROR|rror' | tail -n 5")
    out = ssh(body, timeout=180)
    m = re.search(r"MX_JOB (\d+)", out or "")
    if m:
        a["job"] = m.group(1)
        r["status"] = "inflight"
        say(f"SUBMIT {rid} job={a['job']} kv={a['kv'] or 'default'} mtu={a['mtu'] or 'default'}")
    else:
        # Leave it as "submitting": the next poll checks jobs.log before resubmitting.
        tail = " / ".join(l.strip() for l in (out or "ssh failed").splitlines()[-3:] if l.strip())
        say(f"DRIVER WARN: submit of {rid} returned no job id yet ({tail[:300]})")
    save(st)


def control(st):
    if not os.path.exists(CONTROL):
        return True
    with open(CONTROL) as f:
        lines = [l.split() for l in f if l.strip() and not l.startswith("#")]
    os.remove(CONTROL)
    keep_going = True
    for w in lines:
        if w[0] == "requeue" and len(w) >= 2 and w[1] in st["runs"]:
            r = st["runs"][w[1]]
            opts = dict(x.split("=", 1) for x in w[2:] if "=" in x)
            nxt = {k: opts[k] for k in ("kv", "mtu") if k in opts}
            if r["status"] in ("submitting", "inflight"):
                r["requeue_after"] = nxt
                say(f"CONTROL requeue {w[1]} {nxt} once its current job ends")
                continue
            r["status"] = "queued"
            r["next"] = nxt
            if w[1] == PILOT:
                st["ep_gate"] = "pilot"
            say(f"CONTROL requeue {w[1]} {r['next']}")
        elif w[0] == "unblock-ep":
            st["ep_gate"] = "open"
            say("CONTROL EP gate open")
        elif w[0] in ("pause", "resume"):
            st["paused"] = w[0] == "pause"
            say(f"CONTROL {w[0]}")
        elif w[0] == "stop":
            keep_going = False
        else:
            say(f"CONTROL ignored: {' '.join(w)}")
    save(st)
    return keep_going


def alive(owner):
    """False only for a driver on this host whose process is gone (a restart)."""
    host, _, pid = owner.partition(":")
    if host != socket.gethostname() or not pid.isdigit():
        return True
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def beat():
    """Refuse to run next to another live driver; otherwise refresh our heartbeat."""
    if os.path.exists(LOCK):
        owner, _, ts = open(LOCK).read().strip().rpartition(" ")
        if owner != ME and time.time() - float(ts or 0) < 2 * POLL_S + 60 and alive(owner):
            say(f"DRIVER ERROR: driver {owner} is still alive (see {LOCK}); exiting")
            sys.exit(1)
    with open(LOCK, "w") as f:
        f.write(f"{ME} {time.time()}\n")


def main():
    beat()
    st = load()
    save(st)
    say(f"driver up: {sum(r['status'] == 'queued' for r in st['runs'].values())} queued, "
        f"in flight {inflight(st)}, EP gate {st['ep_gate']}, poll {POLL_S}s")
    ssh_failures = 0
    while True:
        if not control(st):
            say("DRIVER STOPPED by control (running jobs left alone)")
            save(st)
            return
        ok = poll(st)
        ssh_failures = 0 if ok else ssh_failures + 1
        if ssh_failures == 6:
            say("DRIVER ERROR: ssh has failed 6 polls in a row")
        save(st)
        while ok:
            rid = pick(st)
            if not rid:
                break
            submit(st, rid)
            if st["runs"][rid]["status"] != "inflight":
                break
        runs = st["runs"].values()
        if all(r["status"] == "done" for r in runs):
            ssh(f"cd {REMOTE}\n" + push_results_cmd())
            say("MATRIX DONE: " + ", ".join(
                f"{rid}={r['attempts'][-1].get('verdict', '-')}" for rid, r in st["runs"].items()))
            return
        if not inflight(st) and not pick(st):
            blocked = [i for i, r in st["runs"].items() if r["status"] == "queued"]
            if blocked and not st.get("idle_reported"):
                st["idle_reported"] = True
                say(f"MATRIX IDLE: nothing can start (EP gate {st['ep_gate']}, paused={st['paused']}); "
                    f"waiting on control for {len(blocked)} queued runs")
        else:
            st["idle_reported"] = False
        save(st)
        beat()
        time.sleep(POLL_S)


if __name__ == "__main__":
    sys.exit(main())
