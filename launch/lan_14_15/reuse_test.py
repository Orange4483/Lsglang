#!/usr/bin/env python3
"""Prefix-cache / KV-reuse probe: send the same long prompt N times in a row.

The first send is cold (full prefill, KV written to L2 host / L3 NVMe); later
sends should mostly hit the reuse tier, which shows up as a large
``cached_tokens`` and a short TTFT.  Reports one JSON line per send.

  python reuse_test.py --tokens 131072 --max-new 8 --repeats 3
  python reuse_test.py --tokens 131072 --max-new 8 --repeats 3 --nonce 7
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from client import build_ids, stream_post  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8070)
    ap.add_argument("--model", default=os.environ.get(
        "MODEL", "/home/hello/models/DeepSeek-V4.1-Flash-1017"))
    ap.add_argument("--tokens", type=int, default=131072)
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--nonce", type=int, default=0)
    ap.add_argument("--gap", type=float, default=0.0,
                    help="seconds to wait between sends (lets async L3 writes land)")
    ap.add_argument("--timeout", type=float, default=7200)
    args = ap.parse_args()

    ids = build_ids(args.model, args.tokens)
    if args.nonce:
        h = abs(hash((args.nonce, args.tokens))) % 20000
        for i in range(min(16, len(ids))):
            ids[i] = 100 + (h * 7 + i * 131) % 20000
    for k in range(args.repeats):
        if k and args.gap:
            time.sleep(args.gap)
        payload = {
            "input_ids": ids,
            "sampling_params": {"max_new_tokens": args.max_new,
                                "temperature": 0.0, "ignore_eos": True},
            "stream": True,
        }
        meta = stream_post(args.host, args.port, "/generate", payload, args.timeout)
        ttft = meta.get("_ttft") or 0.0
        total = meta.get("_total") or 0.0
        print(json.dumps({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "send": k + 1,
            "len": args.tokens,
            "nonce": args.nonce,
            "prompt_tokens": meta.get("prompt_tokens"),
            "cached_tokens": meta.get("cached_tokens"),
            "ttft_s": round(ttft, 4),
            "total_s": round(total, 4),
            "prefill_tps": round((meta.get("prompt_tokens") or 0) / ttft, 1) if ttft else None,
        }), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
