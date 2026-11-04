import os
import sys

sys.argv = ["x"]
T = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(T, "test_drive.py")).read().split("step()\nprint(\"start:\"")[0])

step()
assert inflight() == ["dsv4_1p1d-tp8", "minimax_2p1d-ep"], inflight()
with open(d.CONTROL, "w") as f:
    f.write("requeue minimax_2p1d-ep kv=20\n")
d.control(st)
assert st["runs"]["minimax_2p1d-ep"]["status"] == "inflight"
finish("minimax_2p1d-ep", "-", "-", "kvlen", "KV cache is needed, which is larger than the available")
jobs[st["runs"]["minimax_2p1d-ep"]["attempts"][-1]["job"]]["state"] = "CANCELLED"
step()
r = st["runs"]["minimax_2p1d-ep"]
print("pilot:", r["status"], st["ep_gate"], submits[-1][2].split("&&")[1][:50])
assert r["status"] == "inflight" and st["ep_gate"] == "pilot" and len(r["attempts"]) == 2
assert "KV_GIB=20" in submits[-1][2]
assert r["attempts"][0]["verdict"] == "FAIL" and r["attempts"][0]["class"] == "kvlen"
print("REQUEUE OK")
