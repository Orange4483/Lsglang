#!/usr/bin/env python3
"""Simulated "same repo, many turns" agent workload.

A fixed code context and system prompt are followed by N short instructions;
each turn appends the previous answer, so every turn shares an ever-growing
prefix with the one before it — the shape in which a KV reuse tier (and the
MoE expert cache) is supposed to pay off.

Prints one JSON line per turn:
  prompt_tokens, cached_tokens, ttft-ish wall, decode t/s

Run it twice with the same --nonce to compare cold vs warm, or restart the
server in between to see what the NVMe tier restores.

  python scenario_agent.py --tokens 24000 --turns 6 --gen 48 --nonce 101
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

sys.path.insert(0, "/home/hello/project/Lsglang-egram-nvme-kv-nvme/lsglang165/launch/lan_14_15")
from client import build_ids  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

SYS = (
    "You are a senior engineer working inside one Python repository. "
    "Answer briefly and concretely; do not restate the code."
)

TASKS = [
    "List the parsing functions in the code above and what each returns.",
    "Which of them can raise, and on what input?",
    "Sketch a unit test for the strict path.",
    "What would you change to stream records instead of listing them?",
    "Name one bug the current error handling hides.",
    "Summarise the module in three bullet points.",
]


def post(port: int, payload: dict, timeout: float):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return out, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8070)
    ap.add_argument("--model", default="/home/hello/models/DeepSeek-V4.1-Flash-1017")
    ap.add_argument("--tokens", type=int, default=24000, help="size of the fixed code context")
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--gen", type=int, default=48)
    ap.add_argument("--nonce", type=int, default=0, help="vary the context so runs do not share a prefix")
    ap.add_argument("--timeout", type=float, default=7200)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ids = build_ids(args.model, args.tokens)
    if args.nonce:
        h = abs(hash((args.nonce, args.tokens))) % 20000
        for i in range(min(16, len(ids))):
            ids[i] = 100 + (h * 7 + i * 131) % 20000
    context = tok.decode(ids)

    messages = [
        {"role": "system", "content": SYS},
        {"role": "user", "content": context + "\n\n---\nFirst: outline the module."},
    ]
    total_turns = min(args.turns, len(TASKS))
    for turn in range(total_turns):
        messages.append({"role": "user", "content": TASKS[turn]})
        payload = {
            "model": "DeepSeek-V4.1-Flash-1017",
            "messages": messages,
            "max_tokens": args.gen,
            "temperature": 0.0,
            "stream": False,
            "chat_template_kwargs": {"thinking": False},
        }
        out, dt = post(args.port, payload, args.timeout)
        usage = out.get("usage", {})
        details = usage.get("prompt_tokens_details") or {}
        text = out["choices"][0]["message"]["content"]
        completion = usage.get("completion_tokens") or 0
        print(json.dumps({
            "turn": turn + 1,
            "nonce": args.nonce,
            "context_tokens": args.tokens,
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": details.get("cached_tokens"),
            "completion_tokens": completion,
            "wall_s": round(dt, 3),
            "decode_tps": round(max(completion - 1, 1) / max(dt, 1e-6), 1),
        }), flush=True)
        messages.append({"role": "assistant", "content": text})
    return 0


if __name__ == "__main__":
    sys.exit(main())
