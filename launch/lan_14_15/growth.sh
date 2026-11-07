#!/usr/bin/env bash
# Growing-context OOM sweep: one shared prefix (constant nonce 0, same seed
# text), lengths sent in ascending order so each prompt extends the previous
# one, 400-token outputs, server default --max-running-requests 2.
# Reports per-length result plus host/GPU headroom; stops at the first failure.
set -uo pipefail
HOST_NUM=${1:?usage: growth.sh 14|15}
cd "$HOME/project/Lsglang-egram-nvme-kv-nvme/lsglang165"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate Lsglang165
export PYTHONPATH=$PWD/python
M=$HOME/models/DeepSeek-V4.1-Flash-1017
[ "$HOST_NUM" = 14 ] && M=/media/hello/Windows/models/DeepSeek-V4.1-Flash-1017
LENGTHS=${LENGTHS:-65536 131072 262144 524288 600000 660000}

for n in $LENGTHS; do
  echo "== $n" >&2
  out=$(python launch/lan_14_15/client.py gen --model "$M" --max-new 400 \
        --tokens "$n" --nonce 0 --timeout 7200 2>&1 | tail -1)
  echo "$out"
  printf 'mem_avail=%s gpu=%s\n' \
    "$(free -g | awk '/Mem|内存/ {print $7}')" \
    "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')" >&2
  printf '%s' "$out" | grep -q prompt_tokens || { echo "FAIL at $n" >&2; break; }
done
