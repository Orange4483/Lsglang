#!/usr/bin/env bash
# Two concurrent streams at a long context with a large output budget - the
# agentic shape.  Checks that both fit the KV pool, run concurrently, and OOM
# never happens; reports per-stream rates and the VRAM peak.
#   bash dualbig.sh 14 <ctx> <max_new>
set -uo pipefail
HOST_NUM=${1:?usage: dualbig.sh 14|15 <ctx> <max_new>}
CTX=${2:?ctx}
GEN=${3:?max_new}
cd "$HOME/project/Lsglang-egram-nvme-kv-nvme/lsglang165"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate Lsglang165
export PYTHONPATH=$PWD/python
M=$HOME/models/DeepSeek-V4.1-Flash-1017
[ "$HOST_NUM" = 14 ] && M=/media/hello/Windows/models/DeepSeek-V4.1-Flash-1017
PEAK=/tmp/gpu_peak_big_$$.txt
: >"$PEAK"

python launch/lan_14_15/client.py gen --model "$M" --max-new "$GEN" \
  --tokens "$CTX" --nonce 11 --timeout 14400 2>/dev/null >/tmp/big_a.jsonl &
A=$!
python launch/lan_14_15/client.py gen --model "$M" --max-new "$GEN" \
  --tokens $((CTX + 3)) --nonce 22 --timeout 14400 2>/dev/null >/tmp/big_b.jsonl &
B=$!
while kill -0 $A 2>/dev/null || kill -0 $B 2>/dev/null; do
  nvidia-smi --query-gpu=memory.used --format=csv,noheader >>"$PEAK" 2>/dev/null
  sleep 5
done
wait $A || echo "stream A failed" >&2
wait $B || echo "stream B failed" >&2
cat /tmp/big_a.jsonl /tmp/big_b.jsonl
python3 - "$PEAK" <<'PY'
import re, sys
vals = []
for line in open(sys.argv[1]):
    nums = [int(x) for x in re.findall(r"(\d+) MiB", line)]
    vals.extend(nums)
print("gpu_peak_mib:", max(vals) if vals else "n/a")
PY
