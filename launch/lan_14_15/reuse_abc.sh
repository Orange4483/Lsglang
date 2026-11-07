#!/usr/bin/env bash
# A/B/A prefix-reuse probe.  Send A (cold, fills L1/L2), send B of the same
# size (should evict A from the device pool), then A again: if the host (L2) or
# NVMe (L3) tier works, the third send reports cached_tokens ~= len and a much
# shorter TTFT.
#   bash launch/lan_14_15/reuse_abc.sh 15 [tokens]
set -uo pipefail
HOST_NUM=${1:?usage: reuse_abc.sh 14|15 [tokens]}; shift || true
LEN=${1:-350000}
REPO=${REPO:-$HOME/project/Lsglang-egram-nvme-kv-nvme/lsglang165}
cd "$REPO"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate Lsglang165
export PYTHONPATH=$PWD/python
model=$HOME/models/DeepSeek-V4.1-Flash-1017
[ "$HOST_NUM" = 14 ] && model=/media/hello/Windows/models/DeepSeek-V4.1-Flash-1017

run() {
  echo "[$(date '+%T')] send nonce=$1 len=$LEN" >&2
  python launch/lan_14_15/reuse_test.py --tokens "$LEN" --max-new 8 --repeats 1 \
    --nonce "$1" --timeout 7200 --model "$model"
}
run 101          # A: cold
run 102          # B: same size, different prefix -> evicts A from the device pool
run 101          # A again: expect a host/NVMe hit
