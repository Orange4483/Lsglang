#!/usr/bin/env bash
# Phase 1 probe: warm the prefix cache with the repeated-work scenario, evict
# it from the device pool with a large distractor, then re-run the same
# scenario.  Turn 1 of run B is the measurement: if a host (L2) tier restored
# the prefix it should show cached_tokens ~= 24064 and a short wall time.
#   bash phase1_test.sh 14 [nonce]
set -uo pipefail
HOST_NUM=${1:-14}
NONCE=${2:-301}
REPO=${REPO:-$HOME/project/Lsglang-egram-nvme-kv-nvme/lsglang165}
cd "$REPO"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate Lsglang165
export PYTHONPATH=$PWD/python
model=$HOME/models/DeepSeek-V4.1-Flash-1017
[ "$HOST_NUM" = 14 ] && model=/media/hello/Windows/models/DeepSeek-V4.1-Flash-1017

echo "== A: repeated-work scenario (cold -> warm)"
bash launch/lan_14_15/phase_run.sh "$HOST_NUM" "p1_A_${NONCE}" 24000 6 "$NONCE" 48

echo "== distractor: 400k-token unrelated request (evicts the device pool)"
python launch/lan_14_15/client.py gen --tokens 400000 --max-new 16 \
  --nonce 999 --timeout 3600 --model "$model" || true

echo "== B: same scenario again (turn 1 tells us whether L2 restored it)"
bash launch/lan_14_15/phase_run.sh "$HOST_NUM" "p1_B_${NONCE}" 24000 3 "$NONCE" 48
