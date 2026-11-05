"""Fake `vllm bench serve` result files for the driver tests."""
import json
import os

LEVELS = (8, 16, 32, 64)


def bench_json(logs, job, levels=LEVELS, failed_at=None):
    """Write each level's result JSON for benchmark job `job` under logs/<job>/, as
    pull_logs would leave them: 100 output and 900 total tok/s per unit of
    concurrency; the level failed_at has 3 failed requests."""
    out = os.path.join(logs, str(job), "bench_20261106_013700")
    os.makedirs(out, exist_ok=True)
    for con in levels:
        n = max(2 * con, 32)
        bad = 3 if con == failed_at else 0
        x = {"num_prompts": n, "completed": n - bad, "failed": bad, "max_concurrency": con,
             "duration": 60.0 + con, "request_throughput": con / 10, "output_throughput": 100.0 * con,
             "total_token_throughput": 900.0 * con}
        for m, (median, p99) in {"ttft": (500.0, 900.0), "tpot": (10.0, 12.0), "itl": (9.0, 20.0),
                                 "e2el": (11000.0, 13000.0)}.items():
            x.update({f"mean_{m}_ms": median, f"median_{m}_ms": median, f"p99_{m}_ms": p99})
        with open(os.path.join(out, f"bench_8192_1024_con{con}.json"), "w") as f:
            json.dump(x, f)
