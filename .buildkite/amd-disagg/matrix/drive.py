#!/usr/bin/env python3
"""Drive the disagg gsm8k model matrix on Crusoe Spur.

    drive.py [--models dsv4,glm] [--layouts 1p1d-tp8,2p1d-ep] [--cells glm_2p1d-ep]
             [--image REF] [--router-image REF] [--ep-gate] [--list]

matrix/austin/nightly.sh runs it every night from the Austin node, after copying
.buildkite/amd-disagg to the run's folder on Crusoe (MX_REMOTE_DIR). State,
results.tsv, summary.md and the job logs go to MX_STATE_DIR on this host, so a
restart with the same MX_STATE_DIR picks up where it stopped.

Every POLL_S seconds, one session on the login node checks the jobs in flight
and counts the nodes of all the user's Spur jobs, running or pending. A cell
starts only if that count plus its own nodes stays within MX_MAX_NODES, and no
cell starts after MX_NO_NEW_AFTER. Every cell runs the same vLLM and vllm-router
images: the nightly tags, resolved to digests when the run starts.

A failed cell is retried once on its own (out-of-memory with a smaller KV cap),
except the expected failures in MX_XFAIL. A preflight rejection (dirty node) is
resubmitted without counting as a retry. A job whose gsm8k makes no progress
for STALL_S is cancelled and fails as class "stall". With --ep-gate the EP
pilot runs first among the EP cells and the others start only once it passes;
a pilot that fails twice blocks them. SIGTERM, SIGINT or SIGHUP cancels the
jobs in flight. The exit code is 0 only if every cell passed or failed as
expected.

Environment:
  MX_REMOTE_DIR    the run's folder on Crusoe, holding .buildkite/amd-disagg (required)
  MX_STATE_DIR     the run's folder on this host (default: this directory)
  MX_HOST          the login node (default: $USER@crs-spur.crusoe.amd.com)
  MX_SSH_MODE      tty (default) feeds each script to an interactive session, for
                   login nodes that refuse `ssh host cmd`; cmd runs it as the command
  MX_SSH_KEY       identity file for ssh and rsync (default: ssh's own choice)
  MX_LOG_ROOT      job logs on Crusoe (default: /shared_nfs/<user>/logs)
  MX_MAX_NODES     node budget across all the user's Spur jobs (default: 3)
  MX_NO_NEW_AFTER  HH:MM IST after which no cell starts (default: 09:00; empty: none)
  MX_XFAIL         expected-fail cells, comma-separated (default: glm_2p1d-ep)
  MX_COMMIT        the code version to report (default: this checkout's HEAD)
  POLL_S           seconds between polls (default: 120)

Commands go in MX_STATE_DIR/control.txt, one per line, consumed on the next loop:
  requeue <id> [kv=<GiB>] [mtu=<n>]   run <id> again with optional overrides
  unblock-ep                          let EP cells start again
  pause | resume                      stop / restart starting cells
  stop                                exit (running jobs are left alone)
"""
import argparse
import base64
import getpass
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
HOST = os.environ.get("MX_HOST") or f"{getpass.getuser()}@crs-spur.crusoe.amd.com"
SPUR_USER = HOST.rpartition("@")[0] or getpass.getuser()
SSH_MODE = os.environ.get("MX_SSH_MODE", "tty")
SSH_KEY = os.environ.get("MX_SSH_KEY", "")
REMOTE_ROOT = os.environ.get("MX_REMOTE_DIR", "").rstrip("/")
REMOTE = f"{REMOTE_ROOT}/.buildkite/amd-disagg/matrix"
STATE_DIR = os.environ.get("MX_STATE_DIR") or HERE
STATE = os.path.join(STATE_DIR, "state.json")
CONTROL = os.path.join(STATE_DIR, "control.txt")
RESULTS = os.path.join(STATE_DIR, "results.tsv")
SUMMARY = os.path.join(STATE_DIR, "summary.md")
LOCK = os.path.join(STATE_DIR, "driver.lock")
LOGS = os.path.join(STATE_DIR, "logs")
LOG_ROOT = os.environ.get("MX_LOG_ROOT") or f"/shared_nfs/{SPUR_USER}/logs"
COMMIT = os.environ.get("MX_COMMIT", "")
ME = f"{socket.gethostname()}:{os.getpid()}"
POLL_S = int(os.environ.get("POLL_S", "120"))
MAX_NODES = int(os.environ.get("MX_MAX_NODES", "3"))
NO_NEW_AFTER = os.environ.get("MX_NO_NEW_AFTER", "09:00")
XFAIL = {c for c in os.environ.get("MX_XFAIL", "glm_2p1d-ep").split(",") if c}
EP_GATE = False
OOM_RETRY_KV_GIB = 24
MAX_PREFLIGHT_RESUBMITS = 2
# gsm8k completes requests every few seconds even on the slowest layout, so this
# long without progress is a hang (e.g. an EP collective that never returns).
STALL_S = 1200
IST = timezone(timedelta(hours=5, minutes=30))
IMAGES = {"vllm": "vllm/vllm-openai-rocm:nightly", "router": "vllm/vllm-router:nightly"}

MODELS = {  # key: (MODEL_NAME, MODEL_DIR)
    "dsv4": ("DeepSeek-V4-Pro", "/shared_nfs/models"),
    "minimax": ("MiniMax-M3-MXFP8", "/shared_nfs/models/MiniMaxAI"),
    "kimi": ("Kimi-K2.6-MXFP4", "/shared_nfs/models/amd"),
    "glm": ("GLM-5.2-FP8", "/shared_nfs/models/zai-org"),
}
# Nodes per layout, as submit.sh lays them out.
LAYOUTS = {"1p1d-tp8": 2, "2p1d-tp8": 3, "1p1d-ep": 2, "2p1d-ep": 3, "1p2d-ep": 3}
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
STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
ssh_failures = 0
wait_note = None


class Stop(Exception):
    pass


def say(msg):
    print(f"{datetime.now(IST):%H:%M:%S} {msg}", flush=True)


def select_cells(models=None, layouts=None, cells=None):
    """The cells that match every given filter, in TP_ORDER + EP_ORDER order."""
    every = TP_ORDER + EP_ORDER
    for kind, given, known in (("model", models, MODELS), ("layout", layouts, LAYOUTS), ("cell", cells, every)):
        bad = [x for x in given or [] if x not in known]
        if bad:
            raise SystemExit(f"unknown {kind}: {', '.join(bad)} (known: {', '.join(known)})")
    return [c for c in every
            if (not models or c.split("_", 1)[0] in models)
            and (not layouts or c.split("_", 1)[1] in layouts)
            and (not cells or c in cells)]


def new_state(cells=None):
    runs = {}
    for rid in cells or TP_ORDER + EP_ORDER:
        model, layout = rid.split("_", 1)
        runs[rid] = {"model": model, "layout": layout, "kind": "ep" if layout.endswith("-ep") else "tp",
                     "status": "queued", "next": {}, "attempts": [], "preflight_resubmits": 0}
    return {"runs": runs, "ep_gate": "pilot" if EP_GATE and PILOT in runs else "open", "paused": False,
            "images": dict(IMAGES)}


def load(cells=None):
    st = new_state(cells)
    if os.path.exists(STATE):
        with open(STATE) as f:
            saved = json.load(f)
        # Cells added to TP_ORDER/EP_ORDER after the state was written join as queued.
        for rid, r in st["runs"].items():
            saved["runs"].setdefault(rid, r)
        for k, v in st.items():
            saved.setdefault(k, v)
        st = saved
    return st


def save(st):
    tmp = STATE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, STATE)
    rows = ["\t".join(TSV_COLS)]
    for rid in TP_ORDER + EP_ORDER:
        r = st["runs"].get(rid)
        for i, a in enumerate(r["attempts"] if r else [], 1):
            rows.append("\t".join(str(x) for x in [
                rid, MODELS[r["model"]][0], r["layout"], i, a.get("job") or "-", a.get("verdict") or "-",
                a.get("score") or "-", a.get("nodes") or "-", a.get("image") or "-", a.get("vllm") or "-",
                a.get("kv") or "default", a.get("mtu") or "default", a.get("class") or "-",
                a.get("note") or "-"]))
    with open(RESULTS, "w") as f:
        f.write("\n".join(rows) + "\n")


def ssh_cmd():
    """ssh and its options, shared by ssh() and the rsync log copy."""
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20"]
    if SSH_KEY:
        cmd += ["-i", SSH_KEY, "-o", "IdentitiesOnly=yes"]
    return cmd


def tty_script(body):
    """A session script that runs body and frames its output with @@BEGIN/@@END.

    The body travels as base64 in a heredoc and runs with stdin from /dev/null, so
    nothing in it is read as terminal input. The markers are split here so that
    only the output contains them whole."""
    b64 = base64.b64encode(body.encode()).decode()
    return ("stty -echo; export TERM=dumb PAGER=cat GIT_PAGER=cat PS1= PS2=; unset PROMPT_COMMAND; "
            "bind 'set enable-bracketed-paste off' 2>/dev/null\n"
            "echo; echo \"@@BEG\"\"IN\"; bash -c \"$(base64 -d <<'@@B64'\n"
            + "".join(b64[i:i + 76] + "\n" for i in range(0, len(b64), 76))
            + "@@B64\n)\" < /dev/null; echo; echo \"@@EN\"\"D\"; exit\n")


def tty_output(raw):
    out = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", raw.replace("\r", ""))
    m = re.search(r"@@BEGIN\n(.*)@@END", out, re.S)
    return m.group(1) if m else None


def ssh(body, timeout=300):
    """Run body with bash on the login node; return its output, or None if the session failed."""
    if SSH_MODE == "cmd":
        b64 = base64.b64encode(body.encode()).decode()
        argv, stdin = ssh_cmd() + [HOST, f"bash -c \"$(echo {b64} | base64 -d)\" < /dev/null"], b""
    else:
        argv, stdin = ssh_cmd() + ["-tt", HOST], tty_script(body).encode()
    # Bytes, not text=True: universal newlines would turn a stray \r inside a
    # remote line into a line break.
    try:
        p = subprocess.run(argv, input=stdin, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    raw = p.stdout.decode(errors="replace")
    if SSH_MODE == "cmd":
        return None if p.returncode == 255 else raw.replace("\r", "")
    return tty_output(raw)


def push_results_cmd():
    cmd = ""
    for path in (RESULTS, SUMMARY):
        if os.path.exists(path):
            with open(path, "rb") as f:
                b64 = base64.encodebytes(f.read()).decode()
            cmd += f"base64 -d > {REMOTE_ROOT}/{os.path.basename(path)} <<'B64END'\n{b64}B64END\n"
    return cmd


def pull_logs(job):
    """Copy a finished job's logs, minus the gsm8k samples, to LOGS."""
    os.makedirs(LOGS, exist_ok=True)
    cmd = ["rsync", "-a", "--exclude", "samples_*.jsonl", "-e", shlex.join(ssh_cmd()),
           f"{HOST}:{LOG_ROOT}/vllm-disagg-pd-{job}.log", f"{HOST}:{LOG_ROOT}/{job}", LOGS + "/"]
    try:
        ok = subprocess.run(cmd, capture_output=True, timeout=600).returncode == 0
    except subprocess.TimeoutExpired:
        ok = False
    if not ok:
        say(f"DRIVER WARN: copying the logs of job {job} failed; they are still in {LOG_ROOT} on Crusoe")


def resolve(ref):
    """repo:tag -> repo@sha256:..., asking Docker Hub directly (no Docker needed).

    Returns ref unchanged if it is already a digest or the registry can't be reached."""
    repo, _, tag = ref.rpartition(":")
    if "@" in ref or not repo or "/" in tag:
        return ref
    accept = ("application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, "
              "application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.manifest.v1+json")
    try:
        url = f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull"
        with urllib.request.urlopen(url, timeout=30) as r:
            token = json.load(r)["token"]
        req = urllib.request.Request(f"https://registry-1.docker.io/v2/{repo}/manifests/{tag}", method="HEAD",
                                     headers={"Authorization": f"Bearer {token}", "Accept": accept})
        with urllib.request.urlopen(req, timeout=30) as r:
            digest = r.headers.get("Docker-Content-Digest")
        if not digest:
            raise ValueError("no Docker-Content-Digest header")
        return f"{repo}@{digest}"
    except (OSError, ValueError, KeyError) as e:
        say(f"DRIVER WARN: could not resolve {ref} to a digest ({e}); cells will pull the tag")
        return ref


def git_head():
    p = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return p.stdout.strip() or "-"


def inflight(st):
    return [rid for rid, r in st["runs"].items() if r["status"] in ("submitting", "inflight")]


def classify_finish(st, rid, a):
    """Decide what happens after attempt `a` of run `rid` ended."""
    r = st["runs"][rid]
    cls = a.get("class", "unknown")
    if "requeue_after" in r:
        r["status"], r["next"] = "queued", r.pop("requeue_after")
        if rid == PILOT and EP_GATE:
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
    if rid in XFAIL:
        r["status"] = "done"
        say(f"XFAIL {rid}: failed as expected ({cls}); not retrying")
        return
    retries = sum(1 for x in r["attempts"] if x.get("class") not in ("preflight", "cancelled")) - 1
    if retries < 1:
        if cls == "oom":
            r["status"], r["next"] = "queued", {"kv": OOM_RETRY_KV_GIB}
            say(f"REQUEUE {rid}: out of memory, retrying once with a {OOM_RETRY_KV_GIB} GiB KV cap")
        else:
            r["status"], r["next"] = "queued", {k: a[k] for k in ("kv", "mtu") if a.get(k)}
            say(f"REQUEUE {rid}: failed ({cls}), retrying once")
        return
    r["status"] = "done"
    if rid == PILOT and st["ep_gate"] == "pilot":
        st["ep_gate"] = "blocked"
        say(f"EP BLOCKED: pilot {rid} job {a.get('job')} failed twice ({cls}); no new EP runs until triage")


def held_nodes(st, out):
    """Nodes held by all the user's Spur jobs per the poll's squeue section, plus
    cells in flight that squeue doesn't list (yet); None if squeue failed or
    printed something unexpected."""
    m = re.search(r"MX_SQ_BEGIN\n(.*?)MX_SQ_END rc=(\d+)", out, re.S)
    if not m or m.group(2) != "0":
        say("DRIVER WARN: squeue failed; starting no cells this round")
        return None
    listed, held = set(), 0
    for line in m.group(1).split("\n"):
        line = line.strip()
        if not line:
            continue
        f = re.fullmatch(r"(\d+)(?:_\S+)?:(\d+):[A-Z_]+", line)
        if not f:
            say(f"DRIVER WARN: unexpected squeue line {line!r}; starting no cells this round")
            return None
        listed.add(f.group(1))
        held += int(f.group(2))
    for rid in inflight(st):
        r = st["runs"][rid]
        if r["attempts"][-1].get("job") not in listed:
            held += LAYOUTS[r["layout"]]
    return held


def poll(st):
    """Check the jobs in flight and count the user's Spur nodes, in one session.

    Returns the nodes held (see held_nodes), or None if that is unknown."""
    global ssh_failures
    ids = inflight(st)
    jobs = [st["runs"][i]["attempts"][-1].get("job") for i in ids]
    body = f"cd {REMOTE}\nexport LOG_ROOT={LOG_ROOT}\n" + push_results_cmd()
    if any(jobs):
        body += "bash status.sh " + " ".join(j for j in jobs if j) + "\n"
    pending = [i for i in ids if st["runs"][i]["status"] == "submitting"]
    if pending:
        body += "grep -aE '^(" + "|".join(pending) + ") ' jobs.log | sed 's/^/MX_LOGGED /'\n"
    body += f"echo MX_SQ_BEGIN; timeout 30 squeue -u {SPUR_USER} -h -o '%i:%D:%T'; echo \"MX_SQ_END rc=$?\"\n"
    out = ssh(body)
    if out is None:
        ssh_failures += 1
        say(f"DRIVER WARN: status poll over ssh failed ({ssh_failures} in a row); will retry next loop")
        return None
    ssh_failures = 0
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
        pull_logs(a["job"])
    return held_nodes(st, out)


def pick(st, free):
    """The next queued cell that fits in `free` nodes, or None."""
    runs = st["runs"]
    if st["paused"]:
        return None
    fl = inflight(st)
    ep_inflight = any(runs[i]["kind"] == "ep" for i in fl)

    def fits(ids):
        return [i for i in ids if i in runs and runs[i]["status"] == "queued" and LAYOUTS[runs[i]["layout"]] <= free]
    tp_q = fits(TP_ORDER)
    if st["ep_gate"] == "pilot":
        ep_q = fits([PILOT]) if not ep_inflight else []
    elif st["ep_gate"] == "open":
        ep_q = fits(EP_ORDER)
    else:
        ep_q = []
    if ep_q and not ep_inflight:
        return ep_q[0]
    if tp_q:
        return tp_q[0]
    if ep_q and st["ep_gate"] == "open":
        return ep_q[0]
    return None


def start_cells(st, held):
    """Submit the next cells that fit in the node budget, given the nodes held."""
    global wait_note
    if held is None or past_cutoff(st):
        return
    free = MAX_NODES - held
    while True:
        rid = pick(st, free)
        if not rid:
            nxt = None if inflight(st) else pick(st, MAX_NODES)
            note = nxt and (f"WAIT {nxt}: your Spur jobs hold {held} node(s); it needs "
                            f"{LAYOUTS[st['runs'][nxt]['layout']]} of the {MAX_NODES}-node budget")
            if note and note != wait_note:
                say(note)
            wait_note = note
            return
        submit(st, rid)
        if st["runs"][rid]["status"] != "inflight":
            return
        free -= LAYOUTS[st["runs"][rid]["layout"]]


def submit(st, rid):
    r = st["runs"][rid]
    nxt = r.get("next") or {}
    a = {"submitted": time.time(), "kv": nxt.get("kv"), "mtu": nxt.get("mtu")}
    r["attempts"].append(a)
    r["status"], r["next"] = "submitting", {}
    save(st)
    name, model_dir = MODELS[r["model"]]
    env = {"MX_ID": rid, "MODEL_NAME": name, "MODEL_DIR": model_dir, "IMAGE": st["images"]["vllm"],
           "VLLM_ROUTER_IMAGE": st["images"]["router"], "DISAGG_SCRIPTS_STAGE": f"{REMOTE_ROOT}/stage",
           "LOG_ROOT": LOG_ROOT}
    if a["kv"]:
        env["KV_GIB"] = str(a["kv"])
    if a["mtu"]:
        env["MORI_IB_PATH_MTU"] = str(a["mtu"])
    body = (f"cd {REMOTE} && " + " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
            + f" bash submit.sh {r['layout']} 2>&1 | grep -aE 'MX_|ERROR|rror' | tail -n 5")
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
            if w[1] == PILOT and EP_GATE:
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


def cutoff_after(start, hhmm):
    """The first hhmm (IST) after start, in epoch seconds; None if hhmm is empty."""
    if not hhmm:
        return None
    h, m = (int(x) for x in hhmm.split(":"))
    t = datetime.fromtimestamp(start, IST).replace(hour=h, minute=m, second=0, microsecond=0)
    return (t if t.timestamp() > start else t + timedelta(days=1)).timestamp()


def past_cutoff(st):
    return bool(st.get("cutoff")) and time.time() >= st["cutoff"]


def finished(st):
    """Nothing in flight, and nothing more will start."""
    if inflight(st):
        return False
    return past_cutoff(st) or (not st["paused"] and pick(st, MAX_NODES) is None)


def tally(st):
    c = {"pass": 0, "fail": 0, "xfail": 0, "queued": 0, "active": 0}
    for rid, r in st["runs"].items():
        a = r["attempts"][-1] if r["attempts"] else {}
        if r["status"] == "done":
            c["pass" if a.get("verdict") == "PASS" else "xfail" if rid in XFAIL else "fail"] += 1
        else:
            c["queued" if r["status"] == "queued" else "active"] += 1
    return c


def ist(ts, fmt="%H:%M"):
    return datetime.fromtimestamp(ts, IST).strftime(fmt)


def cutoff_text(st):
    """The cutoff in IST, with its date unless that's the day the run started."""
    c = st["cutoff"]
    same_day = ist(c, "%Y-%m-%d") == ist(st.get("started", c), "%Y-%m-%d")
    return ist(c, "%H:%M" if same_day else "%Y-%m-%d %H:%M") + " IST"


def cell_text(st, rid, final):
    r = st["runs"].get(rid)
    if not r:
        return ""
    a = r["attempts"][-1] if r["attempts"] else {}
    tries = f" ({len(r['attempts'])} tries)" if len(r["attempts"]) > 1 else ""
    if r["status"] == "done":
        if a.get("verdict") == "PASS":
            try:
                return f"PASS {float(a.get('score')):.4f}{tries}"
            except (TypeError, ValueError):
                return f"PASS{tries}"
        return f"{'XFAIL' if rid in XFAIL else 'FAIL'} ({a.get('class') or '-'}){tries}"
    if r["status"] in ("inflight", "ended"):
        return f"running, job {a.get('job')}"
    if r["status"] == "submitting":
        return "submitting"
    if a.get("class") == "cancelled":
        return "cancelled"
    return "not run" if final else ("retry queued" if r["attempts"] else "queued")


def write_summary(st, final=False):
    """summary.md: what ran on which images, a model x layout table and every attempt."""
    c, now = tally(st), time.time()
    started = st.get("started", now)
    mins = int(now - started) // 60
    vllm = next((a["vllm"] for r in st["runs"].values() for a in r["attempts"]
                 if a.get("vllm") not in (None, "-")), "-")
    out = [f"# Disagg gsm8k matrix, run {os.path.basename(os.path.abspath(STATE_DIR))}", ""]
    if final:
        out += [f"- Ran from {ist(started, '%Y-%m-%d %H:%M')} to {ist(now)} IST ({mins // 60} h {mins % 60} min).",
                f"- Result: {c['pass']} of {len(st['runs'])} cells passed, {c['fail']} failed, "
                f"{c['xfail']} failed as expected, {c['queued']} not run."]
    else:
        status = (f"stopped at {ist(st['stopped']['at'])} IST ({st['stopped']['why']})" if st.get("stopped")
                  else f"in progress, updated {ist(now)} IST")
        out += [f"- Started {ist(started, '%Y-%m-%d %H:%M')} IST; {status}.",
                f"- So far: {c['pass']} passed, {c['fail']} failed, {c['xfail']} failed as expected, "
                f"{c['active']} running, {c['queued']} to go."]
    if st.get("ep_gate") == "blocked":
        out.append(f"- EP gate blocked: the pilot `{PILOT}` failed twice, so EP cells that hadn't started didn't run.")
    if c["queued"] and past_cutoff(st):
        out.append(f"- No cell starts after {cutoff_text(st)}, so the cells that hadn't started by then didn't run.")
    out += [
        f"- Code: `{st.get('commit', '-')}`, staged to `{st.get('remote_root', REMOTE_ROOT)}` on Crusoe.",
        f"- vLLM image: `{st['images']['vllm']}` (vLLM `{vllm}`).",
        f"- Router image: `{st['images']['router']}`.",
        f"- At most {MAX_NODES} nodes at a time across all of {SPUR_USER}'s Spur jobs"
        + (f"; no new cell after {cutoff_text(st)}." if st.get("cutoff") else "."),
        f"- Expected failures: {', '.join(sorted(XFAIL)) or 'none'}.",
        f"- Job logs: `logs/` next to this file (without the gsm8k samples), and everything in `{LOG_ROOT}` on Crusoe.",
        "",
    ]
    models = [k for k in MODELS if any(r["model"] == k for r in st["runs"].values())]
    layouts = [k for k in LAYOUTS if any(r["layout"] == k for r in st["runs"].values())]
    out += ["| Model | " + " | ".join(k.upper().replace("-", " ") for k in layouts) + " |",
            "|---|" + "---|" * len(layouts)]
    for m in models:
        out.append(f"| {MODELS[m][0]} | " + " | ".join(cell_text(st, f"{m}_{k}", final) for k in layouts) + " |")
    attempts = sorted(((a.get("submitted", 0), rid, i, a) for rid, r in st["runs"].items()
                       for i, a in enumerate(r["attempts"], 1)), key=lambda x: x[0])
    out += ["", "## Attempts", "",
            "| Cell | Try | Job | Result | exact_match | Class | Nodes | Minutes | Note |",
            "|---|---|---|---|---|---|---|---|---|"]
    for t0, rid, i, a in attempts:
        r = st["runs"][rid]
        took = int(((a.get("ended") or now) - t0) // 60) if t0 else "-"
        nodes = re.sub(r"crsuse2-m2m-v3-", "", a.get("nodes") or "-")
        note = (a.get("note") or "-").replace("|", "/").replace("`", "'")[:160]
        out.append(f"| {MODELS[r['model']][0]} {r['layout'].upper().replace('-', ' ')} | {i} | {a.get('job') or '-'} | "
                   f"{a.get('verdict') or 'running'} | {a.get('score') or '-'} | {a.get('class') or '-'} | "
                   f"{nodes} | {took} | {note} |")
    tmp = SUMMARY + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(out) + "\n")
    os.replace(tmp, SUMMARY)


def loop(st):
    while True:
        if not control(st):
            st["stopped"] = {"at": time.time(), "why": "stop in control.txt; its jobs in flight were left running"}
            say("DRIVER STOPPED by control (running jobs left alone)")
            save(st)
            write_summary(st)
            ssh(push_results_cmd(), timeout=120)
            return 1
        start_cells(st, poll(st))
        done = finished(st)
        save(st)
        write_summary(st, final=done)
        if done:
            ssh(push_results_cmd(), timeout=120)
            c = tally(st)
            say(f"MATRIX DONE: {c['pass']} passed, {c['fail']} failed, {c['xfail']} failed as expected, "
                f"{c['queued']} not run; {SUMMARY}")
            return 0 if c["fail"] == 0 and c["queued"] == 0 else 1
        beat()
        time.sleep(POLL_S)


def parse_args(argv):
    def names(s):
        return [x for x in s.split(",") if x]
    p = argparse.ArgumentParser(description="Drive the disagg gsm8k model matrix on Crusoe Spur.")
    p.add_argument("--models", type=names, help=f"model keys (default: all of {','.join(MODELS)})")
    p.add_argument("--layouts", type=names, help=f"layouts (default: all of {','.join(LAYOUTS)})")
    p.add_argument("--cells", type=names, help="cells such as glm_2p1d-ep (default: all)")
    p.add_argument("--image", help="vLLM image (default: the nightly tag, resolved to a digest)")
    p.add_argument("--router-image", help="vllm-router image (default: the nightly tag, resolved to a digest)")
    p.add_argument("--ep-gate", action="store_true", help="run the EP pilot first; other EP cells wait for it to pass")
    p.add_argument("--list", action="store_true", help="print the selected cells and exit")
    return p.parse_args(argv)


def main(argv=None):
    global EP_GATE
    args = parse_args(argv)
    cells = select_cells(args.models, args.layouts, args.cells)
    if args.list:
        print("\n".join(cells))
        return 0
    if not cells:
        say("DRIVER ERROR: no cell matches the selection")
        return 2
    if SSH_MODE not in ("tty", "cmd"):
        say(f"DRIVER ERROR: MX_SSH_MODE must be tty or cmd, not {SSH_MODE!r}")
        return 2
    if not REMOTE_ROOT:
        say("DRIVER ERROR: set MX_REMOTE_DIR to the run's folder on Crusoe")
        return 2
    EP_GATE = args.ep_gate
    os.makedirs(STATE_DIR, exist_ok=True)
    beat()
    st = load(cells)
    if "started" not in st:
        now = time.time()
        st.update(started=now, cutoff=cutoff_after(now, NO_NEW_AFTER), commit=COMMIT or git_head(),
                  remote_root=REMOTE_ROOT, images={"vllm": args.image or resolve(IMAGES["vllm"]),
                                                   "router": args.router_image or resolve(IMAGES["router"])})
    elif args.image or args.router_image:
        say("DRIVER WARN: resuming a run, which keeps its images; --image and --router-image are ignored")
    st.pop("stopped", None)
    save(st)
    write_summary(st)
    say(f"driver up: {len(st['runs'])} cells, {sum(r['status'] == 'queued' for r in st['runs'].values())} queued, "
        f"budget {MAX_NODES} nodes, no new cell after {cutoff_text(st) if st.get('cutoff') else 'never'}, "
        f"EP gate {st['ep_gate']}, expected failures {','.join(sorted(XFAIL)) or 'none'}, poll {POLL_S}s")
    say(f"images {st['images']['vllm']} and {st['images']['router']}; code {st['commit']}")
    if "MX_REMOTE_OK" not in (ssh(f"test -f {REMOTE}/submit.sh && echo MX_REMOTE_OK\n", timeout=120) or ""):
        say(f"DRIVER ERROR: no {REMOTE}/submit.sh on {HOST}, or the login failed")
        return 1

    def on_signal(signum, frame):
        raise Stop(signal.Signals(signum).name)
    for s in STOP_SIGNALS:
        signal.signal(s, on_signal)
    try:
        return loop(st)
    except (Stop, Exception) as e:
        for s in STOP_SIGNALS:
            signal.signal(s, signal.SIG_IGN)
        ids = inflight(st)
        jobs = [j for j in (st["runs"][i]["attempts"][-1].get("job") for i in ids) if j]
        unconfirmed = [i for i in ids if not st["runs"][i]["attempts"][-1].get("job")]
        now = time.time()
        for i in ids:
            st["runs"][i]["attempts"][-1].update({"verdict": "CANCELLED", "class": "cancelled", "ended": now,
                                                  "note": "cancelled when the driver stopped"})
            st["runs"][i]["status"] = "queued"
        st["stopped"] = {"at": now, "why": f"{type(e).__name__}: {e}"}
        save(st)
        write_summary(st)
        body = "scancel " + " ".join(jobs) + "\n" if jobs else ""
        if unconfirmed:
            # A submit cut short may still have queued a job that only jobs.log knows.
            body += (f"cd {REMOTE} && grep -aE '^(" + "|".join(unconfirmed)
                     + ") ' jobs.log | cut -d' ' -f2 | xargs -r scancel\n")
        if ssh(body + push_results_cmd(), timeout=60) is None and ids:
            st["stopped"]["why"] += "; the scancel over ssh failed, so its jobs may still be running"
            write_summary(st)
            what = f"could not cancel the jobs of {', '.join(ids)} over ssh; they may still be running"
        else:
            what = f"cancelled jobs: {' '.join(jobs) or 'none'}" + (
                f", plus any that jobs.log has for {', '.join(unconfirmed)}" if unconfirmed else "")
        say(f"DRIVER STOPPED ({type(e).__name__}: {e}); {what}")
        if not isinstance(e, Stop):
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())
