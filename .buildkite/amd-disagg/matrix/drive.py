#!/usr/bin/env python3
"""Drive the disagg gsm8k model matrix and its benchmarks on Crusoe Spur.

    drive.py [--models dsv4,glm] [--layouts 1p1d-tp8,2p1d-ep] [--cells glm_2p1d-ep]
             [--no-bench] [--image REF] [--router-image REF] [--ep-gate] [--list]

matrix/austin/nightly.sh runs it every night from the Austin node, after copying
.buildkite/amd-disagg to the run's folder on Crusoe (MX_REMOTE_DIR). State,
results.tsv, bench.tsv, summary.md and the job logs go to MX_STATE_DIR on this
host, so a restart with the same MX_STATE_DIR picks up where it stopped.

Each model x layout has a gsm8k accuracy cell (glm_2p1d-ep) and a benchmark
cell (glm_2p1d-ep_bench), which runs `vllm bench serve` at each MX_BENCH_CON
level. Benchmarks start once no accuracy cell is left to run, and only for
cells whose accuracy passed; a benchmark selected without its accuracy cell
runs regardless. The benchmarks that the other runs next to MX_STATE_DIR ran
least recently go first.

Every POLL_S seconds, one session on the login node checks the jobs in flight
and counts the nodes of all the user's Spur jobs, running or pending. A cell
starts only if that count plus its own nodes stays within MX_MAX_NODES; a cell
that needs more nodes than that starts only while the count is zero. No cell
starts after MX_NO_NEW_AFTER. Every cell runs the same vLLM and vllm-router
images: the nightly tags, resolved to digests when the run starts.

A failed cell is retried once on its own (out-of-memory with a smaller KV cap),
except the expected failures in MX_XFAIL and benchmarks that got through at
least one level. A preflight rejection (a busy node, or one whose docker cannot
start containers) is resubmitted without counting as a retry, and that node gets
no more of the run's jobs. A job whose gsm8k or benchmark makes no progress for
STALL_S is cancelled and fails as class "stall". With --ep-gate the EP pilot
runs first among the EP cells and the others start only once it passes; a
pilot that fails twice blocks them. SIGTERM, SIGINT or SIGHUP cancels the jobs
in flight. The exit code is 0 only if every cell passed or failed as expected;
a benchmark skipped because its accuracy failed doesn't count on its own.

Environment:
  MX_REMOTE_DIR    the run's folder on Crusoe, holding .buildkite/amd-disagg (required)
  MX_STATE_DIR     the run's folder on this host (default: this directory)
  MX_HOST          the login node (default: $USER@crs-spur.crusoe.amd.com)
  MX_SSH_MODE      tty (default) feeds each script to an interactive session, for
                   login nodes that refuse `ssh host cmd`; cmd runs it as the command
  MX_SSH_KEY       identity file for ssh and rsync (default: ssh's own choice)
  MX_LOG_ROOT      job logs on Crusoe (default: /shared_nfs/<user>/logs)
  MX_MAX_NODES     node budget across all the user's Spur jobs (default: 3)
  MX_NO_NEW_AFTER  HH:MM IST after which no cell starts (default: 23:00; empty: none)
  MX_XFAIL         expected-fail cells, comma-separated (default: glm_2p1d-ep)
  MX_EXCLUDE       nodes no job may get, comma-separated (default: none)
  MX_BENCH_ISL_OSL     benchmark input/output lengths, space-separated (default: 8192/1024)
  MX_BENCH_CON         benchmark concurrency levels (default: 8 16 32 64)
  MX_BENCH_PROMPTS     prompts per level, times its concurrency, at least 32 (default: 2)
  MX_BENCH_TIME_LIMIT  walltime of a benchmark job (default: 02:00:00)
  MX_BENCH_DEFER_TIMEOUT  seconds a benchmark's prefill waits for decode KV blocks
                       before MoRIIO fails the request (default: 600; empty: MoRIIO's 60)
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
import glob
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
BENCH_TSV = os.path.join(STATE_DIR, "bench.tsv")
SUMMARY = os.path.join(STATE_DIR, "summary.md")
LOCK = os.path.join(STATE_DIR, "driver.lock")
LOGS = os.path.join(STATE_DIR, "logs")
LOG_ROOT = os.environ.get("MX_LOG_ROOT") or f"/shared_nfs/{SPUR_USER}/logs"
COMMIT = os.environ.get("MX_COMMIT", "")
ME = f"{socket.gethostname()}:{os.getpid()}"
POLL_S = int(os.environ.get("POLL_S", "120"))
MAX_NODES = int(os.environ.get("MX_MAX_NODES", "3"))
NO_NEW_AFTER = os.environ.get("MX_NO_NEW_AFTER", "23:00")
XFAIL = {c for c in os.environ.get("MX_XFAIL", "glm_2p1d-ep").split(",") if c}
EXCLUDE = [h for h in os.environ.get("MX_EXCLUDE", "").split(",") if h]
BENCH = {"isl_osl": os.environ.get("MX_BENCH_ISL_OSL", "8192/1024"),
         "con": os.environ.get("MX_BENCH_CON", "8 16 32 64"),
         "prompts": os.environ.get("MX_BENCH_PROMPTS", "2"),
         "time_limit": os.environ.get("MX_BENCH_TIME_LIMIT", "02:00:00"),
         "defer_timeout": os.environ.get("MX_BENCH_DEFER_TIMEOUT", "600")}
BENCH_MIN_PROMPTS = 32
GPUS_PER_NODE = 8
EP_GATE = False
OOM_RETRY_KV_GIB = 24
MAX_PREFLIGHT_RESUBMITS = 2
# gsm8k and vllm bench serve complete requests every few seconds even on the
# slowest layout, so this long without progress is a hang (e.g. an EP
# collective that never returns).
STALL_S = 1200
IST = timezone(timedelta(hours=5, minutes=30))
IMAGES = {"vllm": "vllm/vllm-openai-rocm:nightly", "router": "vllm/vllm-router:nightly"}

MODELS = {  # key: (MODEL_NAME, MODEL_DIR)
    "dsv4": ("DeepSeek-V4-Pro", "/shared_nfs/models"),
    "minimax": ("MiniMax-M3-MXFP8", "/shared_nfs/models/MiniMaxAI"),
    "kimi": ("Kimi-K2.6-MXFP4", "/shared_nfs/models/amd"),
    "glm": ("GLM-5.2-FP8", "/shared_nfs/models/zai-org"),
    "kimik3": ("Kimi-K3", "/shared_nfs/models/moonshotai"),
    "minimax31": ("MiniMax-M3.1-Flash-Quark-MXFP4", "/shared_nfs/models/amd"),
}
# Extra submit.sh environment for a model's cells, then for one cell (which wins);
# an empty value drops the variable.
SUBMIT_ENV = {
    # 190 GiB of weights per GPU in TP8 leave room for a 24 GiB KV cache, and a node
    # without the checkpoint in its page cache reads 1.5 TB from NFS for ~90 min.
    "kimik3": {"KV_GIB": "24", "HEALTH_TIMEOUT_S": "9000", "SLURM_TIME_LIMIT": "04:00:00"},
    # A 16 GiB KV cache holds ~180 KDA states per TP1 rank, and vLLM's warmup batch
    # of max_num_seqs must fit in them.
    "kimik3_2p2d-ep": {"KV_GIB": "", "CLUSTER_SERVE_FLAGS": "--kv-cache-memory-bytes 17179869184 "
                       "--max-num-batched-tokens 1024 --max-num-seqs 128"},
    # Mono decode serves steps of at most 16 tokens.
    "minimax31": {"ACCURACY_NUM_CONCURRENT": "16"},
}
# Nodes per layout, as submit.sh lays them out.
LAYOUTS = {"1p1d-tp8": 2, "2p1d-tp8": 3, "1p1d-tp4": 2, "2p1d-tp4": 3,
           "1p1d-ep": 2, "2p1d-ep": 3, "1p2d-ep": 3, "2p2d-ep": 4}
PILOT = "minimax_2p1d-ep"
TP_ORDER = [
    "dsv4_1p1d-tp8", "minimax_1p1d-tp8", "kimi_1p1d-tp8", "glm_1p1d-tp8", "kimik3_1p1d-tp8", "minimax31_1p1d-tp4",
    "dsv4_2p1d-tp8", "minimax_2p1d-tp8", "kimi_2p1d-tp8", "glm_2p1d-tp8", "minimax31_2p1d-tp4",
]
EP_ORDER = [
    PILOT, "minimax_1p1d-ep", "minimax_1p2d-ep",
    "kimi_1p1d-ep", "kimi_2p1d-ep", "kimi_1p2d-ep",
    "glm_1p1d-ep", "glm_2p1d-ep", "glm_1p2d-ep",
    "dsv4_1p1d-ep", "dsv4_2p1d-ep", "dsv4_1p2d-ep",
    "minimax31_1p1d-ep", "minimax31_2p1d-ep", "minimax31_1p2d-ep",
    "kimik3_2p2d-ep",
]
ACC_ORDER = TP_ORDER + EP_ORDER
BENCH_ORDER = [f"{c}_bench" for c in ACC_ORDER]
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


def acc_id(rid):
    """The accuracy cell a benchmark cell belongs to; an accuracy cell's own id."""
    return rid.removesuffix("_bench")


def select_cells(models=None, layouts=None, cells=None, bench=True):
    """The cells that match every given filter: accuracy cells in ACC_ORDER, then benchmarks."""
    every = ACC_ORDER + BENCH_ORDER
    for kind, given, known in (("model", models, MODELS), ("layout", layouts, LAYOUTS), ("cell", cells, every)):
        bad = [x for x in given or [] if x not in known]
        if bad:
            raise SystemExit(f"unknown {kind}: {', '.join(bad)} (known: {', '.join(known)})")
    return [c for c in every
            if (not models or acc_id(c).split("_", 1)[0] in models)
            and (not layouts or acc_id(c).split("_", 1)[1] in layouts)
            and (not cells or c in cells)
            and (bench or c in ACC_ORDER)]


def new_state(cells=None):
    runs = {}
    for rid in cells or ACC_ORDER + BENCH_ORDER:
        model, layout = acc_id(rid).split("_", 1)
        runs[rid] = {"model": model, "layout": layout, "kind": "ep" if layout.endswith("-ep") else "tp",
                     "workload": "accuracy" if rid in ACC_ORDER else "bench",
                     "status": "queued", "next": {}, "attempts": [], "preflight_resubmits": 0}
    return {"runs": runs, "ep_gate": "pilot" if EP_GATE and PILOT in runs else "open", "paused": False,
            "images": dict(IMAGES), "bench": dict(BENCH), "exclude": {}}


def load(cells=None):
    st = new_state(cells)
    if os.path.exists(STATE):
        with open(STATE) as f:
            saved = json.load(f)
        # Cells added to ACC_ORDER/BENCH_ORDER after the state was written join as queued.
        for rid, r in st["runs"].items():
            saved["runs"].setdefault(rid, r)
        for rid, r in saved["runs"].items():
            r.setdefault("workload", "accuracy" if rid in ACC_ORDER else "bench")
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
    for rid in ACC_ORDER + BENCH_ORDER:
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
    for path in (RESULTS, BENCH_TSV, SUMMARY):
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


def last_benched():
    """{benchmark cell: when it last started} over the other runs next to STATE_DIR."""
    here, seen = os.path.realpath(STATE_DIR), {}
    for path in glob.glob(os.path.join(os.path.dirname(here), "*", "state.json")):
        if os.path.realpath(os.path.dirname(path)) == here:
            continue
        try:
            with open(path) as f:
                runs = json.load(f)["runs"]
            for rid in set(runs) & set(BENCH_ORDER):
                for a in runs[rid]["attempts"]:
                    if a.get("class") != "cancelled":
                        seen[rid] = max(seen.get(rid, 0), a.get("submitted") or 0)
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
    return seen


def inflight(st):
    return [rid for rid, r in st["runs"].items() if r["status"] in ("submitting", "inflight")]


def exclude_node(st, host, why):
    if host not in st["exclude"]:
        st["exclude"][host] = why
        say(f"EXCLUDE {host}: {why}; it gets no more of this run's jobs")


def classify_finish(st, rid, a):
    """Decide what happens after attempt `a` of run `rid` ended."""
    r = st["runs"][rid]
    cls = a.get("class", "unknown")
    m = re.search(r"PREFLIGHT-REJECTED: rank \d+ on ([^\s:]+): (.+)", a.get("note") or "")
    if cls == "preflight" and m:
        exclude_node(st, m.group(1), m.group(2).strip())
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
    if r["workload"] == "bench" and a.get("levels", "0") not in ("0", "-"):
        # Load-dependent: a second sweep would most likely fail the same way.
        r["status"] = "done"
        say(f"BENCH FAILED {rid}: {cls} after {a['levels']} level(s) wrote results; not retrying")
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
        if r["workload"] == "bench":
            a["levels"] = f.get("levels", "-")
        state, rc, gate = f.get("state", "-"), f.get("rc", "-"), f.get("gate", "-")
        if state in ACTIVE and rc == "-":
            a["gone_polls"] = 0
            idle = f.get("idle", "-")
            if gate == "-" and idle.isdigit() and int(idle) > STALL_S and not a.get("stalled"):
                what = "benchmark" if r["workload"] == "bench" else "gsm8k"
                a["stalled"] = f"no {what} progress for {int(idle) // 60} min; cancelled by the driver"
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


def acc_passed(st, rid):
    """Whether benchmark cell rid may run: its accuracy cell passed, or isn't in this run."""
    r = st["runs"].get(acc_id(rid))
    return r is None or (r["status"] == "done" and bool(r["attempts"])
                         and r["attempts"][-1].get("verdict") == "PASS")


def acc_failed(st, rid):
    """Whether benchmark cell rid never will: its accuracy cell is done without passing."""
    r = st["runs"].get(acc_id(rid))
    return r is not None and r["status"] == "done" and not acc_passed(st, rid)


def room_for(st, rid, free):
    """Whether cell rid can start with `free` nodes of the budget unused. A cell that
    needs more than MAX_NODES starts only while the user's Spur jobs hold none."""
    return LAYOUTS[st["runs"][rid]["layout"]] <= free or free == MAX_NODES


def pick(st, free):
    """The next queued cell that fits in `free` nodes, or None."""
    return None if st["paused"] else next_cell(st, free)


def next_cell(st, free):
    """pick() regardless of a pause. Benchmarks wait until no accuracy cell is
    in flight or could start, and go in st["bench_order"]."""
    if past_cutoff(st):
        return None
    rid = acc_pick(st, free)
    if rid:
        return rid
    runs = st["runs"]
    if any(r["workload"] == "accuracy" for r in (runs[i] for i in inflight(st))) or acc_pick(st, MAX_NODES):
        return None
    order = st.get("bench_order") or []
    for rid in order + [c for c in BENCH_ORDER if c not in order]:
        r = runs.get(rid)
        if r and r["status"] == "queued" and acc_passed(st, rid) and room_for(st, rid, free):
            return rid
    return None


def acc_pick(st, free):
    """The next queued accuracy cell that fits in `free` nodes, or None."""
    runs = st["runs"]
    fl = inflight(st)
    ep_inflight = any(runs[i]["kind"] == "ep" and runs[i]["workload"] == "accuracy" for i in fl)

    def fits(ids):
        return [i for i in ids if i in runs and runs[i]["status"] == "queued" and room_for(st, i, free)]
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
            need = LAYOUTS[st["runs"][nxt]["layout"]] if nxt else 0
            note = nxt and (f"WAIT {nxt}: your Spur jobs hold {held} node(s); it needs " + (
                f"{need} of the {MAX_NODES}-node budget" if need <= MAX_NODES
                else f"{need} nodes, more than the {MAX_NODES}-node budget, so it starts only once they hold none"))
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
    if r["workload"] == "bench" and not r["attempts"] and not nxt:
        # Serve the way the accuracy cell last ran, e.g. with its out-of-memory KV cap.
        acc = st["runs"].get(acc_id(rid))
        last = acc["attempts"][-1] if acc and acc["attempts"] else {}
        nxt = {k: last[k] for k in ("kv", "mtu") if last.get(k)}
    a = {"submitted": time.time(), "kv": nxt.get("kv"), "mtu": nxt.get("mtu")}
    r["attempts"].append(a)
    r["status"], r["next"] = "submitting", {}
    save(st)
    name, model_dir = MODELS[r["model"]]
    env = {"MX_ID": rid, "MODEL_NAME": name, "MODEL_DIR": model_dir, "IMAGE": st["images"]["vllm"],
           "VLLM_ROUTER_IMAGE": st["images"]["router"], "DISAGG_SCRIPTS_STAGE": f"{REMOTE_ROOT}/stage",
           "LOG_ROOT": LOG_ROOT}
    if r["workload"] == "bench":
        b = st["bench"]
        env.update({"RUN_AFTER_HEALTH": "bench", "BENCHMARK_COMBINATIONS": b["isl_osl"],
                    "BENCHMARK_CON": b["con"], "NUM_PROMPTS_FACTOR": b["prompts"],
                    "BENCHMARK_MIN_PROMPTS": str(BENCH_MIN_PROMPTS), "SLURM_TIME_LIMIT": b["time_limit"]})
        if b.get("defer_timeout"):
            env["MORIIO_DEFER_TIMEOUT"] = b["defer_timeout"]
    for key in (r["model"], acc_id(rid)):
        env.update(SUBMIT_ENV.get(key, {}))
    env = {k: v for k, v in env.items() if v}
    if st["exclude"]:
        env["EXCLUDE_NODES"] = ",".join(sorted(st["exclude"]))
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
    """Nothing in flight, and nothing more will start, even once a pause ends."""
    return not inflight(st) and next_cell(st, MAX_NODES) is None


def tally(st):
    """Counts per workload, "accuracy" and "bench"."""
    c = {w: {"cells": 0, "pass": 0, "fail": 0, "xfail": 0, "skipped": 0, "queued": 0, "active": 0}
         for w in ("accuracy", "bench")}
    for rid, r in st["runs"].items():
        t = c[r["workload"]]
        t["cells"] += 1
        a = r["attempts"][-1] if r["attempts"] else {}
        if r["status"] == "done":
            t["pass" if a.get("verdict") == "PASS" else "xfail" if rid in XFAIL else "fail"] += 1
        elif r["status"] == "queued" and acc_failed(st, rid):
            t["skipped"] += 1
        else:
            t["queued" if r["status"] == "queued" else "active"] += 1
    return c


def tally_text(t, final, workload):
    """One workload's counts, for summary.md and the log."""
    parts = [f"{t['pass']} of {t['cells']} cells passed" if final else f"{t['pass']} passed", f"{t['fail']} failed"]
    if workload == "accuracy" or t["xfail"]:
        parts.append(f"{t['xfail']} failed as expected")
    if t["skipped"]:
        parts.append(f"{t['skipped']} skipped because the accuracy cell didn't pass")
    if not final:
        parts.append(f"{t['active']} running")
    parts.append(f"{t['queued']} not run" if final else f"{t['queued']} to go")
    return ", ".join(parts)


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
    # A benchmark's score is its levels, e.g. 8/8.
    levels = f" {a['score']}" if r["workload"] == "bench" and a.get("score") not in (None, "-") else ""
    if r["status"] == "done":
        if a.get("verdict") == "PASS":
            if levels:
                return f"PASS{levels}{tries}"
            try:
                return f"PASS {float(a.get('score')):.4f}{tries}"
            except (TypeError, ValueError):
                return f"PASS{tries}"
        return f"{'XFAIL' if rid in XFAIL else 'FAIL'}{levels} ({a.get('class') or '-'}){tries}"
    if r["status"] in ("inflight", "ended"):
        return f"running, job {a.get('job')}"
    if r["status"] == "submitting":
        return "submitting"
    if a.get("class") == "cancelled":
        return "cancelled"
    if acc_failed(st, rid):
        return "skipped"
    return "not run" if final else ("retry queued" if r["attempts"] else "queued")


def cell_name(r):
    return f"{MODELS[r['model']][0]} {r['layout'].upper().replace('-', ' ')}"


def cell_gpus(r):
    """GPUs the cell's servers run on: a -tpN layout's take N of each node's GPUs."""
    tp = re.search(r"-tp(\d+)$", r["layout"])
    return LAYOUTS[r["layout"]] * (int(tp.group(1)) if tp else GPUS_PER_NODE)


def num(v, digits, div=1):
    """v / div with `digits` decimals; "-" if v isn't a number."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return "-"
    return f"{v / div:.{digits}f}"


def bench_levels(st):
    """The (isl, osl, concurrency) levels of the run's benchmarks, in the order they run."""
    b = st["bench"]
    return [(int(io.split("/")[0]), int(io.split("/")[1]), int(con))
            for io in b["isl_osl"].split() for con in b["con"].split()]


def level_ok(x):
    return x.get("completed") == x.get("num_prompts") and not x.get("failed")


def bench_results(job):
    """{(isl, osl, concurrency): vllm bench serve result} of a job, from its logs copied to LOGS."""
    res = {}
    for path in glob.glob(os.path.join(LOGS, str(job), "bench_*", "bench_*_con*.json")):
        m = re.search(r"bench_(\d+)_(\d+)_con(\d+)\.json$", path)
        try:
            with open(path) as f:
                x = json.load(f)
        except (OSError, ValueError):
            continue
        if m and isinstance(x, dict):
            res[tuple(int(g) for g in m.groups())] = x
    return res


def bench_done(st):
    """(id, run, attempt) of each benchmark cell whose newest job has ended, in table order."""
    out = []
    for m in MODELS:
        for k in LAYOUTS:
            r = st["runs"].get(f"{m}_{k}_bench")
            a = next((a for a in reversed(r["attempts"]) if a.get("job")), None) if r else None
            if a and a.get("ended"):
                out.append((f"{m}_{k}_bench", r, a))
    return out


def bench_tables(st, done, results):
    """Output tokens/s per GPU of every finished benchmark, then each one's levels."""
    levels = bench_levels(st)
    multi = len({lv[:2] for lv in levels}) > 1

    def label(lv):
        return f"{lv[0]}/{lv[1]} @ {lv[2]}" if multi else str(lv[2])
    def per_gpu(x, gpus):
        v = num(x.get("output_throughput"), 1, gpus)
        return v if level_ok(x) else f"{v} ({x.get('failed') or 0} failed)"
    out = ["Output tokens/s per GPU at each concurrency (N failed: that many requests failed; -: didn't run):", "",
           "| Cell | GPUs | " + " | ".join(label(lv) for lv in levels) + " |",
           "|---|---|" + "---|" * len(levels)]
    for rid, r, a in done:
        gpus = cell_gpus(r)
        row = [per_gpu(results[rid][lv], gpus) if lv in results[rid] else "-" for lv in levels]
        out.append(f"| {cell_name(r)} | {gpus} | " + " | ".join(row) + " |")
    out.append("")
    for rid, r, a in done:
        gpus = cell_gpus(r)
        out += [f"### {cell_name(r)}: job {a['job']} on {gpus} GPUs", "",
                f"| {'Lengths @ concurrency' if multi else 'Concurrency'} | Requests | Output tok/s | "
                "Output tok/s/GPU | Total tok/s | TTFT ms, median / P99 | TPOT ms, median / P99 | "
                "ITL ms, median / P99 | E2EL s, median / P99 | Duration s |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for lv in levels:
            x = results[rid].get(lv)
            if x is None:
                out.append(f"| {label(lv)} | not run |" + " - |" * 8)
                continue
            reqs = f"{x.get('completed')}/{x.get('num_prompts')}" + (f", {x['failed']} failed" if x.get("failed") else "")
            out.append(f"| {label(lv)} | {reqs} | {num(x.get('output_throughput'), 1)} | "
                       f"{num(x.get('output_throughput'), 1, gpus)} | {num(x.get('total_token_throughput'), 1)} | "
                       f"{num(x.get('median_ttft_ms'), 0)} / {num(x.get('p99_ttft_ms'), 0)} | "
                       f"{num(x.get('median_tpot_ms'), 1)} / {num(x.get('p99_tpot_ms'), 1)} | "
                       f"{num(x.get('median_itl_ms'), 1)} / {num(x.get('p99_itl_ms'), 1)} | "
                       f"{num(x.get('median_e2el_ms'), 1, 1000)} / {num(x.get('p99_e2el_ms'), 1, 1000)} | "
                       f"{num(x.get('duration'), 0)} |")
        out.append("")
    return out


BENCH_TSV_COLS = (["run", "id", "model", "layout", "gpus", "job", "vllm", "isl", "osl", "concurrency",
                   "num_prompts", "completed", "failed", "duration", "request_throughput", "output_throughput",
                   "output_throughput_per_gpu", "total_token_throughput"]
                  + [f"{s}_{m}_ms" for m in ("ttft", "tpot", "itl", "e2el") for s in ("mean", "median", "p99")])


def write_bench_tsv(done, results):
    """bench.tsv: one row per level of every finished benchmark."""
    run = os.path.basename(os.path.abspath(STATE_DIR))
    rows = ["\t".join(BENCH_TSV_COLS)]
    for rid, r, a in done:
        gpus = cell_gpus(r)
        for (isl, osl, con), x in sorted(results[rid].items()):
            tput = x.get("output_throughput")
            own = {"run": run, "id": rid, "model": MODELS[r["model"]][0], "layout": r["layout"], "gpus": gpus,
                   "job": a["job"], "vllm": a.get("vllm") or "-", "isl": isl, "osl": osl, "concurrency": con,
                   "output_throughput_per_gpu": tput / gpus if isinstance(tput, (int, float)) else "-"}
            vals = (own[k] if k in own else x.get(k, "-") for k in BENCH_TSV_COLS)
            rows.append("\t".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in vals))
    with open(BENCH_TSV, "w") as f:
        f.write("\n".join(rows) + "\n")


def write_summary(st, final=False):
    """summary.md: what ran on which images, the accuracy and benchmark tables and
    every attempt; and bench.tsv if the run has benchmarks."""
    c, now = tally(st), time.time()
    started = st.get("started", now)
    mins = int(now - started) // 60
    vllm = next((a["vllm"] for r in st["runs"].values() for a in r["attempts"]
                 if a.get("vllm") not in (None, "-")), "-")
    out = [f"# Disagg matrix, run {os.path.basename(os.path.abspath(STATE_DIR))}", ""]
    if final:
        out.append(f"- Ran from {ist(started, '%Y-%m-%d %H:%M')} to {ist(now)} IST ({mins // 60} h {mins % 60} min).")
    else:
        status = (f"stopped at {ist(st['stopped']['at'])} IST ({st['stopped']['why']})" if st.get("stopped")
                  else f"in progress, updated {ist(now)} IST")
        out.append(f"- Started {ist(started, '%Y-%m-%d %H:%M')} IST; {status}.")
    for w, title in (("accuracy", "Accuracy"), ("bench", "Benchmarks")):
        if c[w]["cells"]:
            out.append(f"- {title}{'' if final else ' so far'}: {tally_text(c[w], final, w)}.")
    if st.get("ep_gate") == "blocked":
        out.append(f"- EP gate blocked: the pilot `{PILOT}` failed twice, so EP cells that hadn't started didn't run.")
    if (c["accuracy"]["queued"] or c["bench"]["queued"]) and past_cutoff(st):
        out.append(f"- No cell starts after {cutoff_text(st)}, so the cells that hadn't started by then didn't run.")
    big = {}
    for rid, r in st["runs"].items():
        if LAYOUTS[r["layout"]] > MAX_NODES:
            big.setdefault(acc_id(rid), r)
    out += [
        f"- Code: `{st.get('commit', '-')}`, staged to `{st.get('remote_root', REMOTE_ROOT)}` on Crusoe.",
        f"- vLLM image: `{st['images']['vllm']}` (vLLM `{vllm}`).",
        f"- Router image: `{st['images']['router']}`.",
        f"- At most {MAX_NODES} nodes at a time across all of {SPUR_USER}'s Spur jobs"
        + "".join(f"; {cell_name(r)} needs {LAYOUTS[r['layout']]}, so its cells start only while those jobs hold none"
                  for r in big.values())
        + (f"; no new cell after {cutoff_text(st)}." if st.get("cutoff") else "."),
        f"- Expected failures: {', '.join(sorted(XFAIL)) or 'none'}.",
    ]
    if st["exclude"]:
        out.append("- Nodes kept out of this run: "
                   + "; ".join(f"{h} ({why})" for h, why in sorted(st["exclude"].items())) + ".")
    if c["bench"]["cells"]:
        b = st["bench"]
        defer = (f"; MoRIIO lets a prefill wait {b['defer_timeout']} s for decode KV blocks" if b.get("defer_timeout")
                 else "")
        out.append(f"- Benchmark settings: `vllm bench serve` on random prompts of {b['isl_osl']} input/output tokens at "
                   f"concurrency {b['con']}, with {b['prompts']} prompts per unit of concurrency (at least "
                   f"{BENCH_MIN_PROMPTS}){defer}. They start after the accuracy cells, and only for cells whose "
                   "accuracy passed.")
    out += [f"- Job logs: `logs/` next to this file (without the gsm8k samples), and everything in `{LOG_ROOT}` on Crusoe.",
            ""]
    for w, title, suffix in (("accuracy", "## Accuracy: gsm8k exact_match", ""), ("bench", "## Benchmarks", "_bench")):
        runs = [r for r in st["runs"].values() if r["workload"] == w]
        if not runs:
            continue
        models = [k for k in MODELS if any(r["model"] == k for r in runs)]
        layouts = [k for k in LAYOUTS if any(r["layout"] == k for r in runs)]
        out += [title, "", "| Model | " + " | ".join(k.upper().replace("-", " ") for k in layouts) + " |",
                "|---|" + "---|" * len(layouts)]
        for m in models:
            out.append(f"| {MODELS[m][0]} | "
                       + " | ".join(cell_text(st, f"{m}_{k}{suffix}", final) for k in layouts) + " |")
        out.append("")
    done = bench_done(st)
    results = {rid: bench_results(a["job"]) for rid, r, a in done}
    if done:
        out += bench_tables(st, done, results)
    attempts = sorted(((a.get("submitted", 0), rid, i, a) for rid, r in st["runs"].items()
                       for i, a in enumerate(r["attempts"], 1)), key=lambda x: x[0])
    out += ["## Attempts", "",
            "| Cell | Try | Job | Result | Score | Class | Nodes | Minutes | Note |",
            "|---|---|---|---|---|---|---|---|---|"]
    for t0, rid, i, a in attempts:
        r = st["runs"][rid]
        took = int(((a.get("ended") or now) - t0) // 60) if t0 else "-"
        nodes = re.sub(r"crsuse2-m2m-v3-", "", a.get("nodes") or "-")
        note = (a.get("note") or "-").replace("|", "/").replace("`", "'")[:160]
        out.append(f"| {cell_name(r)}{' bench' if r['workload'] == 'bench' else ''} | {i} | {a.get('job') or '-'} | "
                   f"{a.get('verdict') or 'running'} | {a.get('score') or '-'} | {a.get('class') or '-'} | "
                   f"{nodes} | {took} | {note} |")
    tmp = SUMMARY + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(out) + "\n")
    os.replace(tmp, SUMMARY)
    if c["bench"]["cells"]:
        write_bench_tsv(done, results)


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
            say("MATRIX DONE: " + "; ".join(f"{name} {tally_text(c[w], True, w)}" for w, name in
                                            (("accuracy", "accuracy"), ("bench", "benchmarks")) if c[w]["cells"])
                + f"; {SUMMARY}")
            return 0 if all(t["fail"] == 0 and t["queued"] == 0 for t in c.values()) else 1
        beat()
        time.sleep(POLL_S)


def parse_args(argv):
    def names(s):
        return [x for x in s.split(",") if x]
    p = argparse.ArgumentParser(description="Drive the disagg gsm8k model matrix and its benchmarks on Crusoe Spur.")
    p.add_argument("--models", type=names, help=f"model keys (default: all of {','.join(MODELS)})")
    p.add_argument("--layouts", type=names, help=f"layouts (default: all of {','.join(LAYOUTS)})")
    p.add_argument("--cells", type=names, help="cells such as glm_2p1d-ep or glm_2p1d-ep_bench (default: all)")
    p.add_argument("--no-bench", action="store_true", help="leave out the benchmark cells")
    p.add_argument("--image", help="vLLM image (default: the nightly tag, resolved to a digest)")
    p.add_argument("--router-image", help="vllm-router image (default: the nightly tag, resolved to a digest)")
    p.add_argument("--ep-gate", action="store_true", help="run the EP pilot first; other EP cells wait for it to pass")
    p.add_argument("--list", action="store_true", help="print the selected cells and exit")
    return p.parse_args(argv)


def main(argv=None):
    global EP_GATE
    args = parse_args(argv)
    cells = select_cells(args.models, args.layouts, args.cells, bench=not args.no_bench)
    if args.list:
        print("\n".join(cells))
        return 0
    if not cells:
        say("DRIVER ERROR: no cell matches the selection")
        return 2
    try:
        if not bench_levels({"bench": BENCH}) or int(BENCH["prompts"]) < 1:
            raise ValueError("no levels")
        if BENCH["defer_timeout"] and not re.fullmatch(r"[0-9]+(\.[0-9]+)?", BENCH["defer_timeout"]):
            raise ValueError("defer timeout")
    except (ValueError, IndexError):
        say(f"DRIVER ERROR: bad benchmark settings MX_BENCH_ISL_OSL={BENCH['isl_osl']!r} "
            f"MX_BENCH_CON={BENCH['con']!r} MX_BENCH_PROMPTS={BENCH['prompts']!r} "
            f"MX_BENCH_DEFER_TIMEOUT={BENCH['defer_timeout']!r}")
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
        seen = last_benched()
        st.update(started=now, cutoff=cutoff_after(now, NO_NEW_AFTER), commit=COMMIT or git_head(),
                  remote_root=REMOTE_ROOT, bench_order=sorted(BENCH_ORDER, key=lambda c: seen.get(c, 0)),
                  images={"vllm": args.image or resolve(IMAGES["vllm"]),
                          "router": args.router_image or resolve(IMAGES["router"])})
    elif args.image or args.router_image:
        say("DRIVER WARN: resuming a run, which keeps its images; --image and --router-image are ignored")
    for h in EXCLUDE:
        exclude_node(st, h, "set in MX_EXCLUDE")
    st.pop("stopped", None)
    save(st)
    write_summary(st)
    n_bench = sum(r["workload"] == "bench" for r in st["runs"].values())
    say(f"driver up: {len(st['runs'])} cells ({n_bench} benchmarks), "
        f"{sum(r['status'] == 'queued' for r in st['runs'].values())} queued, "
        f"budget {MAX_NODES} nodes, no new cell after {cutoff_text(st) if st.get('cutoff') else 'never'}, "
        f"EP gate {st['ep_gate']}, expected failures {','.join(sorted(XFAIL)) or 'none'}, poll {POLL_S}s")
    if n_bench:
        b = st["bench"]
        say(f"benchmarks: {b['isl_osl']} at concurrency {b['con']}, {b['prompts']}x prompts, "
            f"walltime {b['time_limit']}, MoRIIO defer timeout {b.get('defer_timeout') or 'default'}; "
            f"order {','.join(c for c in st.get('bench_order') or BENCH_ORDER if c in st['runs'])}")
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
