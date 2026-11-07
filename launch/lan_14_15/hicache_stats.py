#!/usr/bin/env python3
"""Snapshot the HiCache counters plus the L3 directory footprint."""
import json
import os
import subprocess
import sys
import urllib.request

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8070
l3 = sys.argv[2] if len(sys.argv) > 2 else "/home/hello/hicache_l3_165"
out = {}
try:
    body = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5).read().decode()
    for line in body.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if "hicache" in line or "cache_backup" in line or "cache_load" in line:
            name, _, value = line.rpartition(" ")
            out[name] = value
except Exception as exc:  # noqa: BLE001
    out["metrics_error"] = str(exc)
files = 0
size = 0
for root, _dirs, names in os.walk(l3):
    for n in names:
        try:
            size += os.path.getsize(os.path.join(root, n))
            files += 1
        except OSError:
            pass
out["l3_files"] = files
out["l3_mb"] = round(size / 2**20, 1)
print(json.dumps(out, indent=1, ensure_ascii=False))
