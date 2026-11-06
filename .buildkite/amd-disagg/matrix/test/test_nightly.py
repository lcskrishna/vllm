"""The nightly driver: node budget, expected failures, cutoff, cell selection,
summary.md, stop signals and the two ssh transports."""
import os
import subprocess
import sys
import time
from datetime import datetime

from bench_fixtures import bench_json

sys.argv = ["x"]
T = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(T, "test_drive.py")).read().split("step()\nprint(\"start:\"")[0])
d.MAX_NODES = 3
d.EP_GATE = False


def fresh(cells=None):
    global st
    jobs.clear()
    submits.clear()
    cancels.clear()
    foreign.clear()
    squeue.update(rc=0, hide_ours=False)
    st = d.new_state(cells)


# Defaults: no EP gate and a 3-node budget, so one cell at a time.
fresh()
assert st["ep_gate"] == "open"
step()
assert inflight() == ["minimax_2p1d-ep"], inflight()
finish("minimax_2p1d-ep", "0", "PASS: exact_match=0.91", "pass")
step()
assert inflight() == ["minimax_1p1d-ep"], inflight()

# Another job of the user's holds a node: a 3-node cell waits, a 2-node cell can start.
fresh(["glm_2p1d-tp8"])
foreign.append(("999", 1, "RUNNING"))
step()
assert submits == []
foreign.clear()
step()
assert inflight() == ["glm_2p1d-tp8"]
fresh(["glm_1p1d-tp8", "glm_2p1d-tp8"])
foreign.append(("999", 1, "RUNNING"))
step()
assert inflight() == ["glm_1p1d-tp8"]
finish("glm_1p1d-tp8", "0", "PASS: exact_match=0.94", "pass")
step()
assert inflight() == [] and st["runs"]["glm_2p1d-tp8"]["status"] == "queued"

# A cell larger than the budget starts only while the user's jobs hold no nodes,
# and then nothing starts beside it.
fresh(["glm_1p1d-tp8", "kimik3_2p2d-ep"])
foreign.append(("999", 1, "RUNNING"))
step()
assert inflight() == ["glm_1p1d-tp8"], inflight()
finish("glm_1p1d-tp8", "0", "PASS: exact_match=0.94", "pass")
step()
assert inflight() == [] and not d.finished(st)
foreign.clear()
step()
assert inflight() == ["kimik3_2p2d-ep"], inflight()
assert d.room_for(st, "kimik3_2p2d-ep", 3) and not d.room_for(st, "kimik3_2p2d-ep", 2)

# A cell the driver has just submitted counts even before squeue lists it.
fresh(["glm_1p1d-tp8", "kimi_1p1d-tp8"])
squeue["hide_ours"] = True
step()
step()
assert len(inflight()) == 1 and len(submits) == 1, inflight()

# squeue failing, or printing a line that doesn't parse, starts nothing.
fresh(["glm_1p1d-tp8"])
squeue["rc"] = 1
step()
assert submits == []
squeue["rc"] = 0
foreign.append(("not a job", 1, "RUNNING"))
step()
assert submits == []
foreign.clear()
step()
assert inflight() == ["glm_1p1d-tp8"]

# An expected failure is not retried, though a preflight rejection is still resubmitted.
fresh(["glm_2p1d-ep"])
step()
finish("glm_2p1d-ep", "1", "-", "preflight", "PREFLIGHT-REJECTED: rank 0")
step()
assert inflight() == ["glm_2p1d-ep"] and len(st["runs"]["glm_2p1d-ep"]["attempts"]) == 2
finish("glm_2p1d-ep", "1", "FAIL: exact_match=0.10", "accuracy", "FAIL: exact_match=0.10")
step()
assert st["runs"]["glm_2p1d-ep"]["status"] == "done" and len(st["runs"]["glm_2p1d-ep"]["attempts"]) == 2
assert d.finished(st) and d.tally(st)["accuracy"]["xfail"] == 1

# A whole pass through loop(): only the expected failure fails, so the exit code is 0.
# Its benchmark is skipped; the other 26 run after the last accuracy cell.
fresh()
st.update(started=time.time() - 3600, commit="abc1234 on csrikris-crusoe", remote_root="/remote/run")


def finish_all(_):
    for rid in d.inflight(st):
        if rid == "glm_2p1d-ep":
            finish(rid, "1", "-", "stall", "no gsm8k progress")
        elif rid.endswith("_bench"):
            finish(rid, "0", "PASS: bench=4/4", "pass")
            bench_json(d.LOGS, st["runs"][rid]["attempts"][-1]["job"])
        else:
            finish(rid, "0", "PASS: exact_match=0.951234", "pass")


d.time.sleep = finish_all
assert d.loop(st) == 0
summary = open(d.SUMMARY).read()
print(summary)
assert "- Accuracy: 26 of 27 cells passed, 0 failed, 1 failed as expected, 0 not run." in summary
assert ("- Benchmarks: 26 of 27 cells passed, 0 failed, 1 skipped because the accuracy cell didn't pass, "
        "0 not run.") in summary
assert "Spur jobs; Kimi-K3 2P2D EP needs 4, so its cells start only while those jobs hold none." in summary
assert "| GLM-5.2-FP8 | PASS 0.9512 | PASS 0.9512 |  |  | PASS 0.9512 | XFAIL (stall) | PASS 0.9512 |  |" in summary
assert "| GLM-5.2-FP8 | PASS 4/4 | PASS 4/4 |  |  | PASS 4/4 | skipped | PASS 4/4 |  |" in summary
assert "| Kimi-K3 | PASS 4/4 |  |  |  |  |  |  | PASS 4/4 |" in summary
# 100 * concurrency output tok/s over the GPUs the servers run on.
assert "| MiniMax-M3-MXFP8 1P1D TP8 | 16 | 50.0 | 100.0 | 200.0 | 400.0 |" in summary
assert "| GLM-5.2-FP8 2P1D TP8 | 24 | 33.3 | 66.7 | 133.3 | 266.7 |" in summary
assert "| MiniMax-M3.1-Flash-Quark-MXFP4 2P1D TP4 | 12 | 66.7 | 133.3 | 266.7 | 533.3 |" in summary
assert "| Kimi-K3 2P2D EP | 32 | 25.0 | 50.0 | 100.0 | 200.0 |" in summary
assert "| 64 | 128/128 | 6400.0 | 400.0 | 57600.0 | 500 / 900 | 10.0 / 12.0 | 9.0 / 20.0 | 11.0 / 13.0 | 124 |" in summary
assert "MoRIIO lets a prefill wait 600 s for decode KV blocks" in summary
assert len(summary.split("## Attempts")[1].strip().splitlines()) == 2 + 27 + 26
order = [s[0] for s in submits]
assert max(order.index(c) for c in d.ACC_ORDER) < min(order.index(c) for c in order if c.endswith("_bench"))
assert [c for c in order if c.endswith("_bench")] == [c for c in d.BENCH_ORDER if c != "glm_2p1d-ep_bench"]
bench_env = next(s[2] for s in submits if s[0] == "kimi_1p2d-ep_bench")
for kv in ("RUN_AFTER_HEALTH=bench", "BENCHMARK_COMBINATIONS=8192/1024", "BENCHMARK_CON='8 16 32 64'",
           "NUM_PROMPTS_FACTOR=2", "BENCHMARK_MIN_PROMPTS=32", "SLURM_TIME_LIMIT=02:00:00", "MORIIO_DEFER_TIMEOUT=600",
           "MX_ID=kimi_1p2d-ep_bench"):
    assert kv in bench_env, (kv, bench_env)
acc_env = next(s[2] for s in submits if s[0] == "kimi_1p2d-ep")
assert "RUN_AFTER_HEALTH" not in acc_env and "MORIIO_DEFER_TIMEOUT" not in acc_env
# Per-model settings reach every cell of the model; a cell's own win, and an empty one drops it.
env = {s[0]: s[2] for s in submits}
for rid in ("kimik3_1p1d-tp8", "kimik3_1p1d-tp8_bench"):
    assert all(kv in env[rid] for kv in ("KV_GIB=24", "HEALTH_TIMEOUT_S=9000", "SLURM_TIME_LIMIT=04:00:00")), env[rid]
assert "KV_GIB" not in env["kimik3_2p2d-ep"] and ("CLUSTER_SERVE_FLAGS='--kv-cache-memory-bytes 17179869184 "
                                                  "--max-num-batched-tokens 1024 --max-num-seqs 128'") in env["kimik3_2p2d-ep"]
assert "ACCURACY_NUM_CONCURRENT=16" in env["minimax31_1p1d-tp4"] and "bash submit.sh 1p1d-tp4" in env["minimax31_1p1d-tp4"]
assert "ACCURACY_NUM_CONCURRENT" not in env["minimax_1p1d-tp8"]
tsv = open(d.BENCH_TSV).read().splitlines()
assert len(tsv) == 1 + 26 * 4 and tsv[0].split("\t") == d.BENCH_TSV_COLS
row = dict(zip(d.BENCH_TSV_COLS, tsv[1].split("\t")))
assert row["id"] == "dsv4_1p1d-tp8_bench" and row["concurrency"] == "8" and row["output_throughput_per_gpu"] == "50.000"

# Past the cutoff nothing starts; what's left is "not run" and the exit code is 1.
fresh(["kimi_1p1d-tp8", "kimi_2p1d-tp8"])
st.update(started=time.time() - 60, cutoff=time.time() - 1)
d.time.sleep = lambda s: None
assert d.loop(st) == 1 and submits == []
summary = open(d.SUMMARY).read()
assert "0 of 2 cells passed, 0 failed, 0 failed as expected, 2 not run" in summary
assert "No cell starts after" in summary and "| Kimi-K2.6-MXFP4 | not run | not run |" in summary
night = datetime(2026, 11, 6, 1, 37, tzinfo=d.IST).timestamp()
assert d.cutoff_after(night, "09:00") == datetime(2026, 11, 6, 9, 0, tzinfo=d.IST).timestamp()
afternoon = datetime(2026, 11, 6, 14, 15, tzinfo=d.IST).timestamp()
assert d.cutoff_after(afternoon, "09:00") == datetime(2026, 11, 7, 9, 0, tzinfo=d.IST).timestamp()
assert d.cutoff_after(night, "") is None
assert d.cutoff_text({"started": night, "cutoff": d.cutoff_after(night, "09:00")}) == "09:00 IST"
assert d.cutoff_text({"started": afternoon, "cutoff": d.cutoff_after(afternoon, "09:00")}) == "2026-11-07 09:00 IST"

# Cell selection; with the gate on, a selection without the pilot doesn't wait for it.
assert d.select_cells(models=["glm"], layouts=["2p1d-ep", "1p1d-tp8"]) == [
    "glm_1p1d-tp8", "glm_2p1d-ep", "glm_1p1d-tp8_bench", "glm_2p1d-ep_bench"]
assert d.select_cells(models=["glm"], layouts=["2p1d-ep"], bench=False) == ["glm_2p1d-ep"]
assert d.select_cells(cells=["kimi_1p2d-ep"]) == ["kimi_1p2d-ep"] and len(d.select_cells()) == 54
assert d.select_cells(cells=["kimi_1p2d-ep_bench"]) == ["kimi_1p2d-ep_bench"]
assert len(d.select_cells(bench=False)) == 27
assert d.select_cells(models=["kimik3"], bench=False) == ["kimik3_1p1d-tp8", "kimik3_2p2d-ep"]
try:
    d.select_cells(models=["llama"])
    raise AssertionError("an unknown model was accepted")
except SystemExit as e:
    assert "unknown model: llama" in str(e)
d.EP_GATE = True
assert d.new_state(d.select_cells(models=["kimi"]))["ep_gate"] == "open" and d.new_state()["ep_gate"] == "pilot"
d.EP_GATE = False

# A stop signal cancels the jobs in flight; --image and --router-image reach every cell.
for f in (d.STATE, d.LOCK):
    if os.path.exists(f):
        os.remove(f)
fresh()


def stop(_):
    raise d.Stop("SIGTERM")


d.time.sleep = stop
rc = d.main(["--cells", "kimi_1p1d-tp8", "--image", "img@sha256:a", "--router-image", "rtr@sha256:b"])
assert rc == 1 and cancels == [submits[0][1]], (rc, cancels, submits)
assert "IMAGE=img@sha256:a" in submits[0][2] and "VLLM_ROUTER_IMAGE=rtr@sha256:b" in submits[0][2]
assert "DISAGG_SCRIPTS_STAGE=/remote/run/stage" in submits[0][2]
summary = open(d.SUMMARY).read()
assert "IST; stopped at " in summary and "(Stop: SIGTERM)." in summary, summary
assert "| Kimi-K2.6-MXFP4 | cancelled |" in summary and "| CANCELLED |" in summary

# A resume runs the cancelled cell again, and the cancelled attempt isn't its one retry.


def fail_then_pass(_):
    for f in jobs.values():
        if f["state"] == "PENDING":
            ok = fail_then_pass.n > 0
            f.update({"state": "COMPLETED" if ok else "FAILED", "rc": "0" if ok else "1", "class": "pass" if ok else "accuracy",
                      "gate": "PASS: exact_match=0.95" if ok else "FAIL: exact_match=0.10"})
            fail_then_pass.n += 1


fail_then_pass.n = 0
d.time.sleep = fail_then_pass
assert d.main(["--cells", "kimi_1p1d-tp8"]) == 0
saved = json.load(open(d.STATE))
assert [a["verdict"] for a in saved["runs"]["kimi_1p1d-tp8"]["attempts"]] == ["CANCELLED", "FAIL", "PASS"]
assert "stopped" not in saved and "1 of 1 cells passed" in open(d.SUMMARY).read()

# A stop while a submit is unconfirmed also cancels the job jobs.log has for that cell;
# if that ssh fails too, summary.md says the job may still be running.
for f in (d.STATE, d.LOCK):
    os.remove(f)
fresh()
bodies = []


def lost_submit(body, timeout=300):
    bodies.append(body)
    return None if "bash submit.sh" in body or "jobs.log | cut" in body else fake_ssh(body, timeout)


d.ssh = lost_submit
d.time.sleep = stop
assert d.main(["--cells", "kimi_1p1d-tp8", "--image", "i", "--router-image", "r"]) == 1
d.ssh = fake_ssh
assert "the scancel over ssh failed, so its jobs may still be running" in open(d.SUMMARY).read()
tmp = os.path.join(T, "jobs_t")
os.makedirs(tmp, exist_ok=True)
with open(os.path.join(tmp, "jobs.log"), "w") as f:
    f.write("kimi_1p1d-tp8 4242 2026-11-06T01:40:00Z mx-a\nkimi_2p1d-tp8 4343 2026-11-06T01:41:00Z mx-b\n")
cleanup = next(l for l in bodies[-1].splitlines() if "jobs.log | cut" in l)
cleanup = cleanup.replace(f"cd {d.REMOTE}", f"cd {tmp}").replace("xargs -r scancel", "xargs -r echo cancel")
assert subprocess.run(["bash", "-c", cleanup], capture_output=True, text=True).stdout == "cancel 4242\n", cleanup
shutil.rmtree(tmp)

# tty transport: the output is framed despite prompt noise, and the script
# can't read the session's input.
script = d.tty_script("echo hi; read x; echo got=$x")
assert "@@BEGIN" not in script and "@@END" not in script
raw = subprocess.run(["bash"], input=script.encode(), capture_output=True).stdout.decode()
assert d.tty_output("motd\r\n\x1b[?2004l$ " + raw) == "hi\ngot=\n\n", repr(raw)

# cmd transport: the same script runs as the ssh command.
seen = {}


class Done:
    returncode, stdout = 0, b"out\r\n"


def fake_run(argv, **kw):
    seen["argv"] = argv
    return Done


real_run, d.SSH_MODE, d.SSH_KEY = subprocess.run, "cmd", "/keys/id"
subprocess.run = fake_run
try:
    assert real_ssh("echo hi; read x; echo got=$x") == "out\n"
finally:
    subprocess.run, d.SSH_MODE, d.SSH_KEY = real_run, "tty", ""
assert seen["argv"][-2] == d.HOST and ["-i", "/keys/id"] == seen["argv"][5:7]
assert subprocess.run(["bash", "-c", seen["argv"][-1]], capture_output=True, text=True).stdout == "hi\ngot=\n"
print("NIGHTLY OK")
