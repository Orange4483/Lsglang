#!/usr/bin/env python3
"""Does concurrency amortise the per-step decode latency?

Uses the same /generate path as client.py (no chat template, so no reasoning
tokens distort the count).  One shared ~32K prefix, N concurrent streams, 400
generated tokens each, and reports aggregate throughput over the wall window.
"""
from __future__ import annotations

import json
import sys
import threading
import time

REPO = "/home/hello/project/Lsglang-egram-nvme-kv-nvme/lsglang165"
sys.path.insert(0, f"{REPO}/launch/lan_14_15")
from client import build_ids, stream_post  # noqa: E402

MODEL = sys.argv[1]
STREAMS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
GEN = int(sys.argv[3]) if len(sys.argv) > 3 else 400
CTX_TOK = int(sys.argv[4]) if len(sys.argv) > 4 else 32000
PORT = int(sys.argv[5]) if len(sys.argv) > 5 else 8070
lock = threading.Lock()
rows: list[dict] = []


def ask(sid: int) -> None:
    ids = build_ids(MODEL, CTX_TOK + 7 * max(sid, 0))
    payload = {
        "input_ids": ids,
        "sampling_params": {
            "max_new_tokens": GEN,
            "temperature": 0.0,
            "ignore_eos": True,
        },
        "stream": True,
    }
    meta = stream_post("127.0.0.1", PORT, "/generate", payload, 7200)
    with lock:
        rows.append({
            "stream": sid,
            "tokens": meta.get("completion_tokens", 0),
            "ttft": meta.get("_ttft") or 0.0,
        })


# Warm the shared prefix so each stream starts from a cached context.
ask(-1)
rows.clear()

t_start = time.time()
ts = [threading.Thread(target=ask, args=(i,)) for i in range(STREAMS)]
[t.start() for t in ts]
[t.join() for t in ts]
t_end = time.time()

tot = sum(r["tokens"] for r in rows)
window = t_end - t_start
print(json.dumps({
    "streams": STREAMS, "gen": GEN, "ctx_tokens": CTX_TOK,
    "window_s": round(window, 2), "tokens": tot,
    "aggregate_tps": round(tot / window, 2),
    "per_stream": [{"s": r["stream"], "gen_tps": round(r["tokens"] / max(
        window - r["ttft"], 1e-9), 2)} for r in rows],
}), flush=True)
