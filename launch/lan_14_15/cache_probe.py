#!/usr/bin/env python3
"""Prefix-cache probe: does a *new session* with the same system prompt still hit?

Sends three chat requests:
  A  long system context + question A            (cold)
  B  same system context + question B            (should hit A's prefix)
  C  same system context + question C + cache_salt (should NOT hit: namespaced)

Prints prompt_tokens / cached_tokens / wall for each.
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


def post(port: int, payload: dict, timeout: float = 7200.0):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read().decode())
    return out, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8070)
    ap.add_argument("--model", default="/home/hello/models/DeepSeek-V4.1-Flash-1017")
    ap.add_argument("--tokens", type=int, default=24000)
    ap.add_argument("--gen", type=int, default=32)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ids = build_ids(args.model, args.tokens)
    context = tok.decode(ids)

    cases = [
        ("A_cold", "List the parsing helpers in the code above.", None),
        ("B_same_sys_new_question", "Which helper can raise, and on what input?", None),
        ("C_same_sys_with_salt", "Sketch a unit test for the strict path.", "salt-9f3c"),
    ]
    for name, question, salt in cases:
        payload = {
            "model": "DeepSeek-V4.1-Flash-1017",
            "messages": [
                {"role": "system", "content": SYS + "\n\n" + context},
                {"role": "user", "content": question},
            ],
            "max_tokens": args.gen,
            "temperature": 0.0,
            "stream": False,
            "chat_template_kwargs": {"thinking": False},
        }
        if salt:
            payload["cache_salt"] = salt
        out, dt = post(args.port, payload)
        usage = out.get("usage", {})
        details = usage.get("prompt_tokens_details") or {}
        print(
            json.dumps(
                {
                    "case": name,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "cached_tokens": details.get("cached_tokens"),
                    "wall_s": round(dt, 2),
                }
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
