import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys

T = os.path.dirname(os.path.abspath(__file__))
shutil.copy(os.path.join(T, "..", "drive.py"), os.path.join(T, "drive_t.py"))
for f in ("state.json", "results.tsv", "summary.md", "control.txt", "driver.lock"):
    if os.path.exists(os.path.join(T, f)):
        os.remove(os.path.join(T, f))
for k in [k for k in os.environ if k.startswith("MX_")]:
    del os.environ[k]
os.environ["MX_REMOTE_DIR"] = "/remote/run"
spec = importlib.util.spec_from_file_location("drive_t", os.path.join(T, "drive_t.py"))
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
# These scenarios run a TP and an EP cell side by side, which takes 5 nodes, and
# go through the EP pilot gate.
d.MAX_NODES = 5
d.EP_GATE = True
d.pull_logs = lambda job: None

jobs = {}        # job -> fields dict, plus "n": its node count
next_job = [1000]
submits = []
cancels = []
foreign = []     # (job, nodes, state) of the user's other Spur jobs
squeue = {"rc": 0, "hide_ours": False}


def fake_ssh(body, timeout=300):
    out = []
    if body.startswith("scancel "):
        j = body.split()[1]
        cancels.append(j)
        jobs[j]["state"] = "CANCELLED"
    if "bash submit.sh" in body:
        next_job[0] += 1
        j = str(next_job[0])
        rid = body.split("MX_ID=")[1].split()[0]
        layout = body.split("bash submit.sh ")[1].split()[0]
        submits.append((rid, j, body))
        jobs[j] = {"state": "PENDING", "rc": "-", "gate": "-", "class": "-", "note": "-", "n": d.LAYOUTS[layout]}
        out.append(f"MX_JOB {j}")
    if "bash status.sh" in body:
        for j in body.split("bash status.sh", 1)[1].split("\n")[0].split():
            f = {k: v for k, v in jobs[j].items() if k != "n"}
            out.append(f"MX_STATUS {j}|" + "|".join(f"{k}={v}" for k, v in f.items())
                       + "|healthy=0/2|nodes=n1,n2|image=abc|vllm=0.33")
    if "squeue" in body:
        out.append("MX_SQ_BEGIN")
        if not squeue["hide_ours"]:
            out += [f"{j}:{f['n']}:{f['state']}" for j, f in jobs.items() if f["state"] in ("PENDING", "RUNNING")]
        out += [f"{j}:{n}:{s}" for j, n, s in foreign]
        out.append(f"MX_SQ_END rc={squeue['rc']}")
    if "test -f" in body and "submit.sh" in body:
        out.append("MX_REMOTE_OK")
    return "\n".join(out) + "\n"


real_ssh, d.ssh = d.ssh, fake_ssh
st = d.new_state()


def finish(rid, rc, gate="-", cls="-", note="-"):
    j = st["runs"][rid]["attempts"][-1]["job"]
    jobs[j].update({"state": "COMPLETED" if rc == "0" else "FAILED", "rc": rc, "gate": gate,
                    "class": cls, "note": note})


def step():
    d.start_cells(st, d.poll(st))


def inflight():
    return sorted(d.inflight(st))


step()
print("start:", inflight())
assert inflight() == ["dsv4_1p1d-tp8", "minimax_2p1d-ep"], inflight()

# TP finishes first: next TP starts; EP stays the pilot only.
finish("dsv4_1p1d-tp8", "0", "PASS: exact_match=0.95", "pass")
step()
print("after dsv4 tp pass:", inflight())
assert inflight() == ["minimax_1p1d-tp8", "minimax_2p1d-ep"]

# Pilot OOMs -> auto retry with 24 GiB, still alone among EP.
finish("minimax_2p1d-ep", "1", "-", "oom", "HSA_STATUS_ERROR_OUT_OF_RESOURCES")
step()
print("after pilot oom:", inflight(), st["ep_gate"], submits[-1][2].split("&&")[1][:60])
assert "KV_GIB=24" in submits[-1][2] and st["runs"]["minimax_2p1d-ep"]["status"] == "inflight"

# Pilot passes on retry -> gate opens; the next free slot goes to EP.
finish("minimax_2p1d-ep", "0", "PASS: exact_match=0.93", "pass")
step()
print("after pilot pass:", inflight(), st["ep_gate"])
assert st["ep_gate"] == "open" and inflight() == ["minimax_1p1d-ep", "minimax_1p1d-tp8"]

# Accuracy fail on EP: retried once by itself; a second failure is final. Gate stays open.
finish("minimax_1p1d-ep", "1", "FAIL: exact_match=0.80", "accuracy", "FAIL: exact_match=0.80")
step()
print("after ep accuracy fail:", inflight(), st["ep_gate"])
assert st["ep_gate"] == "open" and "minimax_1p1d-ep" in inflight()
assert len(st["runs"]["minimax_1p1d-ep"]["attempts"]) == 2
finish("minimax_1p1d-ep", "1", "FAIL: exact_match=0.81", "accuracy", "FAIL: exact_match=0.81")
step()
assert st["runs"]["minimax_1p1d-ep"]["status"] == "done" and "minimax_1p2d-ep" in inflight()

# Bring-up failure on a non-pilot EP run: retried once, the gate stays open.
finish("minimax_1p2d-ep", "1", "-", "bringup", "TIMEOUT waiting for x")
step()
print("after ep bringup fail:", inflight(), st["ep_gate"])
assert st["ep_gate"] == "open" and "minimax_1p2d-ep" in inflight()
assert len(st["runs"]["minimax_1p2d-ep"]["attempts"]) == 2

# Preflight rejection on TP: resubmitted, not counted as a retry.
tp = [i for i in inflight()][0]
finish(tp, "1", "-", "preflight", "PREFLIGHT-REJECTED: rank 0")
step()
print("after preflight:", inflight(), st["runs"][tp]["preflight_resubmits"])
assert tp in inflight() and st["runs"][tp]["preflight_resubmits"] == 1

# Control: requeue the failed EP run and unblock.
with open(d.CONTROL, "w") as f:
    f.write("requeue minimax_1p1d-ep kv=16\nunblock-ep\n")
d.control(st)
assert st["ep_gate"] == "open" and st["runs"]["minimax_1p1d-ep"]["next"] == {"kv": "16"}

# Drain everything with passes.
for _ in range(200):
    for rid in inflight():
        finish(rid, "0", "PASS: exact_match=0.94", "pass")
    step()
    if all(r["status"] == "done" for r in st["runs"].values()):
        break
order = [s[0] for s in submits]
print("submit order:", order)
assert all(r["status"] == "done" for r in st["runs"].values())
# A cell that needs fewer nodes may start ahead of one that doesn't fit yet, so
# EP order holds among cells of the same size.
ep_first = [o for i, o in enumerate(order) if o.endswith("-ep") and o not in order[:i]]
for n in (2, 3):
    same = [o for o in ep_first if d.LAYOUTS[o.split("_", 1)[1]] == n]
    assert same == [o for o in d.EP_ORDER if o in same], (n, ep_first)
assert any("KV_GIB=16" in s[2] for s in submits if s[0] == "minimax_1p1d-ep")
d.save(st)
print(open(d.RESULTS).read())

# A pilot that fails twice blocks every other EP run.
st = d.new_state()
for rid in d.TP_ORDER:
    st["runs"][rid]["status"] = "done"
step()
assert inflight() == [d.PILOT]
finish(d.PILOT, "1", "-", "bringup", "TIMEOUT waiting for x")
step()
assert inflight() == [d.PILOT] and st["ep_gate"] == "pilot"
finish(d.PILOT, "1", "-", "bringup", "TIMEOUT waiting for x")
step()
assert inflight() == [] and st["ep_gate"] == "blocked" and d.pick(st, d.MAX_NODES) is None

# gsm8k stops making progress: the driver cancels the job once, records a stall, retries once.
st = d.new_state()
for rid in d.TP_ORDER:
    st["runs"][rid]["status"] = "done"
st["ep_gate"] = "open"
step()
rid = inflight()[0]
j = st["runs"][rid]["attempts"][-1]["job"]
jobs[j].update({"state": "RUNNING", "idle": "600"})
step()
assert cancels == []
jobs[j]["idle"] = "1500"
step()
assert cancels == [j]
step()
first = st["runs"][rid]["attempts"][0]
assert first["class"] == "stall" and first["verdict"] == "FAIL" and "25 min" in first["note"], first
assert len(st["runs"][rid]["attempts"]) == 2 and st["runs"][rid]["status"] == "inflight"
assert cancels == [j]

# A state written before the 1p1d-ep runs existed gains them as queued, untouched otherwise.
old = d.load()
for rid in [r for r in old["runs"] if r.endswith("_1p1d-ep")]:
    del old["runs"][rid]
old["runs"]["glm_2p1d-ep"]["status"] = "inflight"
with open(d.STATE, "w") as f:
    json.dump(old, f)
merged = d.load()
assert all(merged["runs"][f"{m}_1p1d-ep"]["status"] == "queued" for m in d.MODELS)
assert merged["runs"]["glm_2p1d-ep"]["status"] == "inflight" and merged["ep_gate"] == old["ep_gate"]

# A dead driver on this host does not hold the lock; a live or remote one does.
dead = subprocess.Popen(["true"])
dead.wait()
assert not d.alive(f"{socket.gethostname()}:{dead.pid}")
assert d.alive(f"{socket.gethostname()}:{os.getpid()}")
assert d.alive(f"other-host:{dead.pid}")
print("ALL OK")
