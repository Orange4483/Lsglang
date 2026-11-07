#!/usr/bin/env bash
# Run the repeated-work scenario against a local server and append the per-turn
# lines to logs/<tag>.jsonl.
#   bash phase_run.sh <14|15> <tag> [context_tokens] [turns] [nonce] [gen]
set -uo pipefail
HOST_NUM=${1:?usage: phase_run.sh 14|15 tag [tokens] [turns] [nonce] [gen]}
TAG=${2:?tag}
CTX=${3:-24000}
TURNS=${4:-6}
NONCE=${5:-101}
GEN=${6:-48}
REPO=${REPO:-$HOME/project/Lsglang-egram-nvme-kv-nvme/lsglang165}
OUT=$REPO/logs/${TAG}.jsonl
cd "$REPO"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate Lsglang165
export PYTHONPATH=$PWD/python
model=$HOME/models/DeepSeek-V4.1-Flash-1017
[ "$HOST_NUM" = 14 ] && model=/media/hello/Windows/models/DeepSeek-V4.1-Flash-1017
python launch/lan_14_15/scenario_agent.py --tokens "$CTX" --turns "$TURNS" \
  --nonce "$NONCE" --gen "$GEN" --model "$model" | tee -a "$OUT"
echo "results: $OUT" >&2
