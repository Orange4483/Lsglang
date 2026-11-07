#!/usr/bin/env bash
# Two concurrent streams at growing lengths: does the pair OOM, and what does it
# cost?  Server must allow 2 running requests (our default).
set -uo pipefail
HOST_NUM=${1:?usage: dualgrow.sh 14|15}
cd "$HOME/project/Lsglang-egram-nvme-kv-nvme/lsglang165"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate Lsglang165
export PYTHONPATH=$PWD/python
M=$HOME/models/DeepSeek-V4.1-Flash-1017
[ "$HOST_NUM" = 14 ] && M=/media/hello/Windows/models/DeepSeek-V4.1-Flash-1017
LENGTHS=${LENGTHS:-32768 131072 524288}

for n in $LENGTHS; do
  echo "== dual $n" >&2
  python launch/lan_14_15/client.py gen --model "$M" --max-new 400 \
    --tokens "$n" --nonce 0 --timeout 7200 2>/dev/null >/tmp/dual_a.jsonl &
  A=$!
  python launch/lan_14_15/client.py gen --model "$M" --max-new 400 \
    --tokens $((n + 7)) --nonce 0 --timeout 7200 2>/dev/null >/tmp/dual_b.jsonl &
  B=$!
  wait $A $B || echo "one stream failed" >&2
  cat /tmp/dual_a.jsonl /tmp/dual_b.jsonl
  printf 'mem_avail=%s gpu=%s\n' \
    "$(free -g | awk '/Mem|内存/ {print $7}')" \
    "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')" >&2
done
