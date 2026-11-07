#!/usr/bin/env python3
"""Benchmark / stability client for the DSV4.1 server on .14 / .15.

Three modes:
  health                      GET /health, /v1/models, /model_info
  gen    --tokens N           native /generate with exactly N input ids (streaming)
  chat   --tokens N           OpenAI /v1/chat/completions (text prompt, streaming)

Every run prints exactly one JSON line on stdout, so callers can append it to a
results file and diff runs.

Examples:
  python client.py health
  python client.py gen --tokens 32768 --max-new 32
  python client.py gen --tokens 524288 --max-new 16 --timeout 3600
  python client.py chat --tokens 4096 --max-new 64
"""
from __future__ import annotations

import argparse
import http.client
import json
import sys
import time
import urllib.request

SEED_TEXT = (
    "def parse_records(path, *, strict=True):\n"
    "    out = []\n"
    "    with open(path, 'r', encoding='utf-8') as fh:\n"
    "        for lineno, line in enumerate(fh, 1):\n"
    "            line = line.strip()\n"
    "            if not line or line.startswith('#'):\n"
    "                continue\n"
    "            try:\n"
    "                out.append(json.loads(line))\n"
    "            except ValueError as exc:\n"
    "                if strict:\n"
    "                    raise ValueError(f'{path}:{lineno}: {exc}') from exc\n"
    "    return out\n"
)


def build_ids(model: str, n_tokens: int) -> list[int]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    seed = tok.encode(SEED_TEXT, add_special_tokens=False)
    if not seed:
        raise SystemExit("tokenizer produced no ids")
    reps = n_tokens // len(seed) + 1
    return (seed * reps)[:n_tokens]


def http_json(host: str, port: int, path: str, payload=None, timeout: float = 30):
    url = f"http://{host}:{port}{path}"
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    return body, time.time() - t0


def stream_post(host: str, port: int, path: str, payload: dict, timeout: float):
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    conn.request("POST", path, json.dumps(payload), {"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        detail = resp.read()[:2000].decode(errors="replace")
        raise RuntimeError(f"HTTP {resp.status}: {detail}")
    t0 = time.time()
    ttft = None
    chunks = 0
    text_parts: list[str] = []
    meta: dict = {}
    while True:
        line = resp.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            break
        try:
            obj = json.loads(data)
        except ValueError:
            continue
        if isinstance(obj, dict) and obj.get("error"):
            raise RuntimeError("server error: " + str(obj["error"]))
        if ttft is None:
            ttft = time.time() - t0
        chunks += 1
        if "meta_info" in obj and obj["meta_info"]:
            meta.update(obj["meta_info"])
        if isinstance(obj.get("text"), str):
            text_parts.append(obj["text"])
        choices = obj.get("choices")
        if choices:
            delta = choices[0].get("delta") or {}
            if isinstance(delta.get("content"), str):
                text_parts.append(delta["content"])
            if "usage" in obj and obj["usage"]:
                meta.setdefault("usage", {}).update(obj["usage"])
    total = time.time() - t0
    meta["_text"] = "".join(text_parts)
    meta["_chunks"] = chunks
    meta["_ttft"] = ttft
    meta["_t0"] = t0
    meta["_total"] = total
    return meta


def summarize(mode: str, prompt_tokens: int, completion_tokens: int, meta: dict) -> dict:
    ttft = meta.get("_ttft") or 0.0
    total = meta.get("_total") or 0.0
    decode_time = max(total - ttft, 1e-9)
    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mode": mode,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_s": round(ttft, 4),
        "t0": round(meta.get("_t0") or 0.0, 3),
        "total_s": round(total, 4),
        "prefill_tps": round(prompt_tokens / ttft, 2) if ttft > 0 else None,
        "decode_tps": round(max(completion_tokens - 1, 0) / decode_time, 2),
        "chunks": meta.get("_chunks"),
        "text_head": (meta.get("_text") or "")[:120],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["health", "gen", "chat"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8070)
    ap.add_argument("--model", default="/home/hello/models/DeepSeek-V4.1-Flash-1017")
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--prefix-nonce", type=int, default=0,
                    help="constant-across-lengths seed prefix for growing-prefix runs")
    ap.add_argument("--nonce", type=int, default=0,
                    help="rewrite the first 16 ids so repeated runs cannot share the radix prefix")
    ap.add_argument("--timeout", type=float, default=1800.0)
    args = ap.parse_args()

    if args.mode == "health":
        out = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "mode": "health"}
        for path in ("/health", "/v1/models", "/model_info"):
            try:
                body, dt = http_json(args.host, args.port, path, timeout=20)
                if path == "/v1/models":
                    out["models"] = [m.get("id") for m in body.get("data", [])]
                    out["models_s"] = round(dt, 3)
                elif path == "/health":
                    out["health"] = body
                else:
                    out["model_info"] = {
                        k: body.get(k)
                        for k in ("model_path", "served_model_name", "context_length",
                                  "max_total_num_tokens", "tp_size")
                    }
            except Exception as exc:  # noqa: BLE001
                out[path] = f"ERR {exc}"
        print(json.dumps(out, ensure_ascii=False))
        return 0

    ids = build_ids(args.model, args.tokens)
    if args.nonce:
        h = abs(hash((args.nonce, args.tokens))) % 20000
        for i in range(min(16, len(ids))):
            ids[i] = 100 + (h * 7 + i * 131) % 20000
    if args.prefix_nonce:
        # Constant across lengths (unlike --nonce, which mixes in the token
        # count), so a growing-length run keeps one shared prefix while every
        # run still starts from a sequence the server has never seen.
        h = abs(hash(("prefix", args.prefix_nonce))) % 20000
        for i in range(min(16, len(ids))):
            ids[i] = 100 + (h * 7 + i * 131) % 20000
    if args.mode == "gen":
        payload = {
            "input_ids": ids,
            "sampling_params": {
                "max_new_tokens": args.max_new,
                "temperature": 0.0,
                "top_p": 1.0,
                "ignore_eos": True,
            },
            "stream": True,
        }
        meta = stream_post(args.host, args.port, "/generate", payload, args.timeout)
        prompt_tokens = meta.get("prompt_tokens", len(ids))
        completion_tokens = meta.get("completion_tokens", args.max_new)
        print(json.dumps(summarize("gen", prompt_tokens, completion_tokens, meta),
                         ensure_ascii=False))
        return 0

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    text = tok.decode(ids)
    payload = {
        "model": "DeepSeek-V4.1-Flash-1017",
        "messages": [{"role": "user", "content": text}],
        "max_tokens": args.max_new,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"thinking": False},
    }
    meta = stream_post(args.host, args.port, "/v1/chat/completions", payload, args.timeout)
    usage = meta.get("usage") or {}
    print(json.dumps(summarize("chat", usage.get("prompt_tokens", len(ids)),
                               usage.get("completion_tokens", 0), meta),
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
