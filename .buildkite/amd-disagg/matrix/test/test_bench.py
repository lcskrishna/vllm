"""Benchmark cells: they wait for the accuracy cells and run only where accuracy
passed; their submit settings, retries, order and summary tables; status.sh on
benchmark jobs; and the launcher's benchmark gate. Also the launcher's MoRIIO KV
config, and how it ends a job whose server dies during bring-up."""
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

from bench_fixtures import bench_json

sys.argv = ["x"]
T = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(T, "test_drive.py")).read().split("step()\nprint(\"start:\"")[0])
d.MAX_NODES = 3
d.EP_GATE = False
MX = os.path.dirname(T)
AD = os.path.dirname(MX)


def fresh(cells):
    global st
    jobs.clear()
    submits.clear()
    cancels.clear()
    foreign.clear()
    squeue.update(rc=0, hide_ours=False)
    st = d.new_state(cells)


def job(rid):
    return st["runs"][rid]["attempts"][-1]["job"]


# Every accuracy cell goes first, retries included. Then a benchmark runs where
# accuracy passed, served as the accuracy cell last ran (here with its OOM KV cap).
fresh(["minimax_1p1d-tp8", "glm_1p1d-tp8", "minimax_1p1d-tp8_bench", "glm_1p1d-tp8_bench"])
step()
assert inflight() == ["minimax_1p1d-tp8"], inflight()
finish("minimax_1p1d-tp8", "1", "-", "oom", "HSA_STATUS_ERROR_OUT_OF_RESOURCES")
step()
finish("minimax_1p1d-tp8", "0", "PASS: exact_match=0.93", "pass")
step()
assert inflight() == ["glm_1p1d-tp8"], inflight()
finish("glm_1p1d-tp8", "1", "FAIL: exact_match=0.50", "accuracy", "FAIL: exact_match=0.50")
step()
assert inflight() == ["glm_1p1d-tp8"] and len(st["runs"]["glm_1p1d-tp8"]["attempts"]) == 2
finish("glm_1p1d-tp8", "1", "FAIL: exact_match=0.51", "accuracy", "FAIL: exact_match=0.51")
step()
assert inflight() == ["minimax_1p1d-tp8_bench"], inflight()
assert "RUN_AFTER_HEALTH=bench" in submits[-1][2] and "KV_GIB=24" in submits[-1][2], submits[-1][2]
assert d.cell_text(st, "glm_1p1d-tp8_bench", True) == "skipped"
finish("minimax_1p1d-tp8_bench", "0", "PASS: bench=4/4", "pass")
step()
assert d.finished(st) and inflight() == []
c = d.tally(st)["bench"]
assert (c["pass"], c["skipped"], c["queued"]) == (1, 1, 0), c
assert d.cell_text(st, "minimax_1p1d-tp8_bench", True) == "PASS 4/4"

# A benchmark selected without its accuracy cell starts at once. A failure before
# any level wrote results is retried once; one after some levels isn't.
fresh(["kimi_1p1d-tp8_bench"])
step()
assert inflight() == ["kimi_1p1d-tp8_bench"]
finish("kimi_1p1d-tp8_bench", "1", "-", "bringup", "TIMEOUT waiting for x")
jobs[job("kimi_1p1d-tp8_bench")]["levels"] = "0"
step()
assert inflight() == ["kimi_1p1d-tp8_bench"] and len(st["runs"]["kimi_1p1d-tp8_bench"]["attempts"]) == 2
finish("kimi_1p1d-tp8_bench", "1", "FAIL: bench=2/4", "bench", "FAIL: bench=2/4 levels; 8192/1024 con=32 failed")
jobs[job("kimi_1p1d-tp8_bench")]["levels"] = "3"
step()
r = st["runs"]["kimi_1p1d-tp8_bench"]
assert r["status"] == "done" and len(r["attempts"]) == 2 and r["attempts"][-1]["score"] == "2/4", r
assert d.cell_text(st, "kimi_1p1d-tp8_bench", True) == "FAIL 2/4 (bench) (2 tries)"
fresh(["kimi_2p1d-tp8_bench"])
step()
finish("kimi_2p1d-tp8_bench", "1", "-", "stall", "-")
jobs[job("kimi_2p1d-tp8_bench")]["levels"] = "3"
step()
assert st["runs"]["kimi_2p1d-tp8_bench"]["status"] == "done" and len(submits) == 1

# While an accuracy cell waits for nodes, no benchmark takes them.
fresh(["glm_2p1d-tp8", "minimax_1p1d-tp8_bench"])
foreign.append(("999", 1, "RUNNING"))
step()
assert submits == []
foreign.clear()
step()
assert inflight() == ["glm_2p1d-tp8"]
finish("glm_2p1d-tp8", "0", "PASS: exact_match=0.95", "pass")
step()
assert inflight() == ["minimax_1p1d-tp8_bench"]

# The cutoff holds for benchmarks too.
fresh(["minimax_1p1d-tp8_bench"])
st["cutoff"] = time.time() - 1
step()
assert submits == [] and d.finished(st) and d.tally(st)["bench"]["queued"] == 1

# Benchmarks that the other runs started least recently go first. A cancelled
# attempt, this run's own state (also reached through a link) and unreadable
# files don't count.
root = tempfile.mkdtemp()
for name, runs in (("a", {"dsv4_1p1d-tp8_bench": [{"submitted": 100}], "kimi_1p1d-tp8_bench": [{"submitted": 300}]}),
                   ("b", {"dsv4_1p1d-tp8_bench": [{"submitted": 500, "class": "cancelled"}],
                          "minimax_1p1d-tp8_bench": [{"submitted": 200}], "glm_1p1d-tp8": [{"submitted": 50}]}),
                   ("cur", {"glm_1p1d-tp8_bench": [{"submitted": 900}]})):
    os.makedirs(os.path.join(root, name))
    with open(os.path.join(root, name, "state.json"), "w") as f:
        json.dump({"runs": {rid: {"attempts": a} for rid, a in runs.items()}}, f)
os.symlink("cur", os.path.join(root, "latest"))
os.makedirs(os.path.join(root, "broken"))
with open(os.path.join(root, "broken", "state.json"), "w") as f:
    f.write("{")
d.STATE_DIR = os.path.join(root, "cur")
seen = d.last_benched()
d.STATE_DIR = T
shutil.rmtree(root)
assert seen == {"dsv4_1p1d-tp8_bench": 100, "kimi_1p1d-tp8_bench": 300, "minimax_1p1d-tp8_bench": 200}, seen
order = sorted(d.BENCH_ORDER, key=lambda c: seen.get(c, 0))
assert order[0] == "glm_1p1d-tp8_bench"
assert order[-3:] == ["dsv4_1p1d-tp8_bench", "minimax_1p1d-tp8_bench", "kimi_1p1d-tp8_bench"], order
fresh(["dsv4_1p1d-tp8_bench", "glm_1p1d-tp8_bench"])
st["bench_order"] = ["glm_1p1d-tp8_bench", "dsv4_1p1d-tp8_bench"]
step()
assert inflight() == ["glm_1p1d-tp8_bench"]

# A failed level shows in summary.md; the levels after it didn't run.
fresh(["minimax_1p1d-tp8_bench"])
st["started"] = time.time()
step()
finish("minimax_1p1d-tp8_bench", "1", "FAIL: bench=2/4", "bench", "FAIL: bench=2/4 levels; 8192/1024 con=32 failed")
jobs[job("minimax_1p1d-tp8_bench")]["levels"] = "3"
bench_json(d.LOGS, job("minimax_1p1d-tp8_bench"), levels=(8, 16, 32), failed_at=32)
step()
d.write_summary(st, final=True)
summary = open(d.SUMMARY).read()
print(summary)
assert "| MiniMax-M3-MXFP8 1P1D TP8 | 16 | 50.0 | 100.0 | 200.0 (3 failed) | - |" in summary
assert "| 32 | 61/64, 3 failed |" in summary and "| 64 | not run | - | - |" in summary
assert "| MiniMax-M3-MXFP8 | FAIL 2/4 (bench) |" in summary
assert "- Benchmarks: 0 of 1 cells passed, 1 failed, 0 not run." in summary and "- Accuracy" not in summary
assert len(open(d.BENCH_TSV).read().splitlines()) == 1 + 3

# A defer timeout that isn't a number of seconds stops the driver before it starts.
d.BENCH["defer_timeout"] = "ten"
assert d.main(["--cells", "minimax_1p1d-tp8_bench"]) == 2
d.BENCH["defer_timeout"] = "600"

# status.sh on benchmark jobs: the gate, progress from the benchmark logs and the
# levels that wrote results. An accuracy job reads as before.
logs = tempfile.mkdtemp()
fake = os.path.join(logs, "bin")
os.makedirs(fake)
with open(os.path.join(fake, "scontrol"), "w") as f:
    f.write("#!/bin/bash\necho \"JobId=$3 JobState=COMPLETED\"\n")
os.chmod(os.path.join(fake, "scontrol"), 0o755)
for j, rc, line, levels in (
        ("501", 0, "PASS: bench=8/8 levels. Results in /x", 8),
        ("502", 1, "FAIL: bench=3/8 levels; 8192/1024 con=64 failed, so later levels didn't run. Results in /x", 4),
        ("503", 0, "PASS: exact_match=0.951234 >= threshold=0.90 (tasks=gsm8k). Log: /x", 0)):
    bench_dir = os.path.join(logs, j, "bench_20261106_013700")
    os.makedirs(bench_dir)
    for con in (8, 16, 32, 64, 128, 256, 512, 1024)[:levels]:
        for ext in ("log", "json"):
            open(os.path.join(bench_dir, f"bench_8192_1024_con{con}.{ext}"), "w").close()
    if not levels:
        open(os.path.join(logs, j, "accuracy_GLM_tp_20261106.log"), "w").close()
    with open(os.path.join(logs, j, ".disagg_done"), "w") as f:
        f.write(f"{rc}\n")
    with open(os.path.join(logs, f"vllm-disagg-pd-{j}.log"), "w") as f:
        f.write(f"SLURM_JOB_NODELIST: crsuse2-m2m-v3-[084,093]\n[vllm_disagg] {line}\n")
p = subprocess.run(["bash", os.path.join(MX, "status.sh"), "501", "502", "503"], capture_output=True, text=True,
                   env=dict(os.environ, LOG_ROOT=logs, PATH=f"{fake}:{os.environ['PATH']}"))
shutil.rmtree(logs)
got = {}
for line in p.stdout.splitlines():
    j, _, rest = line.removeprefix("MX_STATUS ").partition("|")
    got[j] = dict(kv.split("=", 1) for kv in rest.split("|"))
assert got["501"]["gate"] == "PASS: bench=8/8" and got["501"]["levels"] == "8" and got["501"]["class"] == "pass", got
assert got["501"]["idle"].isdigit()
assert got["502"]["gate"] == "FAIL: bench=3/8" and got["502"]["levels"] == "4" and got["502"]["class"] == "bench"
assert got["502"]["note"].startswith("FAIL: bench=3/8 levels; 8192/1024 con=64 failed"), got["502"]
assert got["503"]["gate"] == "PASS: exact_match=0.951234" and got["503"]["levels"] == "0", got["503"]

# The launcher's benchmark role, with a stand-in for vllm: an unrecorded warmup,
# a seed per level, a result JSON per level, and a gate line; it stops at the
# first failed level. Like Kimi-K3's tokenizer, the stand-in won't load without
# --trust-remote-code. As a server it stays up, or crashes with FAKE_SERVE=crash.
work = tempfile.mkdtemp()
fake = os.path.join(work, "bin")
os.makedirs(fake)
with open(os.path.join(fake, "vllm"), "w") as f:
    f.write("""#!/usr/bin/env python3
import json, os, sys, time
a = sys.argv[1:]
if a[0] == "serve":
    if os.environ.get("FAKE_SERVE") == "crash":
        sys.exit("RuntimeError: fake server crashed while loading")
    time.sleep(120)
    sys.exit(0)
if "--trust-remote-code" not in a:
    sys.exit("ValueError: the tokenizer has custom code; pass trust_remote_code=True")
def arg(k):
    return a[a.index(k) + 1]
n, con = int(arg("--num-prompts")), arg("--max-concurrency")
print(f"fake bench con={con} seed={arg('--seed')} prompts={n}")
if con == os.environ.get("FAKE_EXIT_AT"):
    sys.exit(3)
bad = 2 if con == os.environ.get("FAKE_FAIL_AT") else 0
if "--save-result" in a:
    with open(os.path.join(arg("--result-dir"), arg("--result-filename")), "w") as f:
        json.dump({"num_prompts": n, "completed": n - bad, "failed": bad, "output_throughput": 10.0 * int(con),
                   "total_token_throughput": 90.0 * int(con), "median_ttft_ms": 5.0, "median_tpot_ms": 1.5}, f)
""")
os.chmod(os.path.join(fake, "vllm"), 0o755)


def launcher(**env):
    logs = os.path.join(work, f"logs{len(os.listdir(work))}")
    e = dict(os.environ, PATH=f"{fake}:{os.environ['PATH']}", CLUSTER_ENV=os.path.join(AD, "cluster_crs.sh"),
             LOG_PATH=logs, MODEL_DIR=work, MODEL_NAME="Fake-Model", BENCHMARK_COMBINATIONS="8192/1024",
             BENCHMARK_CON="8 16 32", NUM_PROMPTS_FACTOR="2", BENCHMARK_MIN_PROMPTS="32", **env)
    p = subprocess.run(["bash", os.path.join(AD, "vllm_disagg.sh"), "bench"], env=e, capture_output=True,
                       text=True, timeout=120)
    results = sorted(os.path.basename(x) for x in
                     subprocess.run(["bash", "-c", f"ls {logs}/bench_*/*.json"], capture_output=True,
                                    text=True).stdout.split())
    warmup = subprocess.run(["bash", "-c", f"cat {logs}/bench_*/bench_warmup.log"], capture_output=True,
                            text=True).stdout
    return p, results, warmup


p, results, warmup = launcher()
assert p.returncode == 0 and "PASS: bench=3/3 levels" in p.stdout, (p.stdout, p.stderr)
assert "warmup: ISL=8192 OSL=1024 CON=8 num_prompts=32, not recorded" in p.stdout
assert warmup.strip() == "fake bench con=8 seed=0 prompts=32", warmup
assert "fake bench con=8 seed=1 prompts=32" in p.stdout and "fake bench con=16 seed=2 prompts=32" in p.stdout
assert "fake bench con=32 seed=3 prompts=64" in p.stdout
assert results == ["bench_8192_1024_con16.json", "bench_8192_1024_con32.json", "bench_8192_1024_con8.json"], results
assert "bench 8192/1024 con=32: 64/64 requests ok, 0 failed; output 320.0 tok/s" in p.stdout
p, results, _ = launcher(FAKE_FAIL_AT="16")
assert p.returncode == 1 and "FAIL: bench=1/3 levels; 8192/1024 con=16 failed" in p.stdout, p.stdout
assert "con=32" not in p.stdout and len(results) == 2
p, results, _ = launcher(FAKE_EXIT_AT="8")
assert p.returncode == 1 and "warmup failed (rc=3); going on with the levels" in p.stdout
assert "FAIL: bench=0/3 levels; 8192/1024 con=8 failed" in p.stdout and results == []


# A server's MoRIIO KV config, as DRY_RUN prints it: defer_timeout only when
# MORIIO_DEFER_TIMEOUT is set, in TP and EP mode alike; a non-number stops the launcher.
def kv_config(**env):
    e = {k: v for k, v in os.environ.items() if k != "MORIIO_DEFER_TIMEOUT"}
    e.update(CLUSTER_ENV=os.path.join(AD, "cluster_crs.sh"), LOG_PATH=os.path.join(work, "dry"),
             MODEL_DIR="/shared_nfs/models/MiniMaxAI", MODEL_NAME="MiniMax-M3-MXFP8", IPADDRS="10.0.0.1,10.0.0.2",
             xP="1", yD="1", NODE_RANK="0", GPUS_PER_NODE="8", DRY_RUN="1", **env)
    p = subprocess.run(["bash", os.path.join(AD, "vllm_disagg.sh"), "node"], env=e, capture_output=True,
                       text=True, timeout=120)
    if p.returncode:
        return p, None
    start = p.stdout.index("--kv-transfer-config {") + len("--kv-transfer-config ")
    return p, json.JSONDecoder().raw_decode(p.stdout[start:])[0]["kv_connector_extra_config"]


for ep in ("0", "1"):
    p, extra = kv_config(WIDE_EP_MODE=ep, MORIIO_DEFER_TIMEOUT="600")
    assert extra and extra["defer_timeout"] == 600, (ep, p.stdout[-2000:], p.stderr[-2000:])
    p, extra = kv_config(WIDE_EP_MODE=ep)
    assert extra is not None and "defer_timeout" not in extra, (ep, extra)
p, _ = kv_config(MORIIO_DEFER_TIMEOUT="abc")
assert p.returncode == 1 and "MORIIO_DEFER_TIMEOUT='abc' is not a number of seconds" in p.stderr, p.stderr


# A rank whose server dies during bring-up says so, and rank 0's health gate
# gives up at once instead of waiting out HEALTH_TIMEOUT_S. The verdict stays
# rank 0's, and status.sh reads it as a bring-up failure with the server's error.
def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return str(s.getsockname()[1])


root = os.path.join(work, "logs-up")
up = os.path.join(root, "601")
os.makedirs(up)
node_env = dict(os.environ, PATH=f"{fake}:{os.environ['PATH']}", CLUSTER_ENV=os.path.join(AD, "cluster_crs.sh"),
                LOG_PATH=up, MODEL_DIR="/shared_nfs/models/MiniMaxAI", MODEL_NAME="MiniMax-M3-MXFP8",
                IPADDRS="127.0.0.1,127.0.0.1", PREFILL_PORT=free_port(), DECODE_PORT=free_port(),
                xP="1", yD="1", GPUS_PER_NODE="8", HEALTH_TIMEOUT_S="60")


def node(rank, **env):
    return subprocess.run(["bash", os.path.join(AD, "vllm_disagg.sh"), "node"],
                          env=dict(node_env, NODE_RANK=str(rank), **env), capture_output=True, text=True, timeout=120)


p1 = node(1, FAKE_SERVE="crash")
assert p1.returncode == 1 and "exited before completion" in p1.stdout, (p1.stdout, p1.stderr)
assert open(os.path.join(up, ".disagg_server_exited")).read().strip() == "1"
assert not os.path.exists(os.path.join(up, ".disagg_done"))
t0 = time.time()
p0 = node(0)
assert p0.returncode == 1 and "rank 1 server exited while waiting" in p0.stdout, (p0.stdout, p0.stderr)
assert time.time() - t0 < 30, time.time() - t0
assert open(os.path.join(up, ".disagg_done")).read().strip() == "1"
with open(os.path.join(root, "vllm-disagg-pd-601.log"), "w") as f:
    f.write("SLURM_JOB_NODELIST: crsuse2-m2m-v3-[084,093]\n" + p0.stdout + p1.stdout)
with open(os.path.join(fake, "scontrol"), "w") as f:
    f.write("#!/bin/bash\necho \"JobId=$3 JobState=COMPLETED\"\n")
os.chmod(os.path.join(fake, "scontrol"), 0o755)
p = subprocess.run(["bash", os.path.join(MX, "status.sh"), "601"], capture_output=True, text=True,
                   env=dict(os.environ, LOG_ROOT=root, PATH=f"{fake}:{os.environ['PATH']}"))
got = dict(kv.split("=", 1) for kv in p.stdout.strip().partition("|")[2].split("|"))
assert got["rc"] == "1" and got["class"] == "bringup", p.stdout
assert "rank 1 server exited while waiting" in got["note"], got["note"]
assert "RuntimeError: fake server crashed while loading" in got["note"], got["note"]
shutil.rmtree(work)
print("BENCH OK")
