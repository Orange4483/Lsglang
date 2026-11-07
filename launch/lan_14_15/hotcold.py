#!/usr/bin/env python3
"""Hot-expert probe: same-domain vs alternating-domain, and dual-stream decode.

Every request shares one ~24k-token context, so its prefill is a prefix-cache
hit and the reported figure is pure decode.  `hot` repeats one domain (the
pool's resident experts should stay hot), `cold` alternates domains (the pool
keeps missing), `dual` runs two hot streams at once.
"""
from __future__ import annotations

import json
import sys
import threading
import time
import urllib.request

REPO = "/home/hello/project/Lsglang-egram-nvme-kv-nvme/lsglang165"
sys.path.insert(0, f"{REPO}/launch/lan_14_15")
from client import build_ids  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

MODEL = sys.argv[1]
MODE = sys.argv[2] if len(sys.argv) > 2 else "hot"
TURNS = int(sys.argv[3]) if len(sys.argv) > 3 else 6
TOK = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
CTX_CODE = TOK.decode(build_ids(MODEL, 24000))
POEM = "床前明月光，疑是地上霜。举头望明月，低头思故乡。"
_other = POEM
while len(TOK.encode(_other)) < 24000:
    _other += POEM
CTX_CODE = TOK.decode(TOK.encode(CTX_CODE)[:24000])
CTX_OTHER = TOK.decode(TOK.encode(_other)[:24000])
CTX = CTX_CODE

HOT = ["Refactor the functions in this module and list them."]
COLD = [
    "Refactor the functions in this module and list them.",
    "把这段代码的意图用一首中文七言绝句表达。",
    "用法律条文的口气写一段关于该模块风险责任的说明。",
    "Write a short poem in English about this code.",
]
lock = threading.Lock()
rows: list[dict] = []


def ask(stream_id: str, task: str, turn: int, ctx: str = "") -> None:
    body = {
        "model": "DeepSeek-V4.1-Flash-1017",
        "messages": [
            {"role": "system", "content": ctx or CTX},
            {"role": "user", "content": f"{task}\n(tag {stream_id}-{turn})"},
        ],
        "max_tokens": 48,
        "temperature": 0.0,
        "stream": True,
    }
    req = urllib.request.Request(
        "http://127.0.0.1:8070/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    ttft = None
    n = 0
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                delta = json.loads(payload)["choices"][0].get("delta", {})
            except Exception:  # noqa: BLE001
                continue
            if delta.get("content"):
                if ttft is None:
                    ttft = time.time() - t0
                n += 1
    total = time.time() - t0
    dec = (total - ttft) if ttft is not None else total
    row = {
        "mode": MODE, "stream": stream_id, "turn": turn,
        "ttft_s": round(ttft or 0, 3), "decode_s": round(dec, 3),
        "tokens": n, "decode_tps": round(n / dec, 2) if dec > 0 else 0,
        "wall_s": round(total, 2),
    }
    with lock:
        rows.append(row)
        print(json.dumps(row), flush=True)


if MODE == "dual":
    def worker(sid: str) -> None:
        for t in range(TURNS):
            ask(sid, HOT[0], t)
    ts = [threading.Thread(target=worker, args=(sid,)) for sid in ("A", "B")]
    t_start = time.time()
    [t.start() for t in ts]
    [t.join() for t in ts]
    window = time.time() - t_start
    total = sum(r["tokens"] for r in rows)
    print(json.dumps({"mode": "dual", "window_s": round(window, 2),
                      "tokens": total,
                      "aggregate_tps": round(total / window, 2),
                      "per_request_tps": round(total / window / max(1, len(rows)), 2),
                      "n": len(rows)}), flush=True)
else:
    tasks = HOT if MODE == "hot" else COLD
    for t in range(TURNS):
        ask("A", tasks[t % len(tasks)], t)
    vals = [r["decode_tps"] for r in rows]
    print(json.dumps({"mode": MODE, "mean_decode_tps": round(sum(vals) / len(vals), 2),
                      "n": len(vals)}), flush=True)
