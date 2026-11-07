#!/usr/bin/env bash
# DeepSeek-V4.1-Flash-1017 on 192.168.1.15 (2x48G: RTX 4090 SM89 + RTX PRO 5000 SM120),
# TP2, CPU-hybrid MoE (lk_moe), Engram host table, 665535-token context, 0.0.0.0:8070.
#
# Usage:  bash launch/lan_14_15/serve_15.sh [extra sglang args...]
# Env knobs (all optional):
#   DSV41_MEM_FRACTION   default 0.85
#   DSV41_POOL_SLOTS     default 24
#   DSV41_SUB_M          default 12288
#   DSV41_CHUNK          default 12288
#   DSV41_MAX_TOTAL      default 666112  (page-rounded pool must hold a full
#                        665535-token prompt plus a few output tokens)
#   DSV41_MAX_REQ        default 1
#   DSV41_ENGRAM         host (default) | nvme
#   DSV41_EXTRA          extra env assignments, e.g. "LK_POOL_LOG=100"
set -euo pipefail

REPO=${REPO:-$HOME/project/Lsglang-egram-nvme-kv-nvme/lsglang165}
MODEL=${MODEL:-$HOME/models/DeepSeek-V4.1-Flash-1017}
PORT=${PORT:-8070}
LOGDIR=${LOGDIR:-$REPO/logs}
LOG=${LOG:-$LOGDIR/serve_15.log}
mkdir -p "$LOGDIR"

exec >>"$LOG" 2>&1
echo "===== $(date '+%F %T') serve_15 pid=$$ ====="

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate Lsglang165
# The conda env ships nvcc 13.4 next to cudart 13.0 headers, so every cccl /
# cutlass JIT (tilelang) aborts with "CUDA compiler and CUDA toolkit headers
# are incompatible".  The system toolkit is self-consistent (13.2 + 13.2), so
# point CUDA_HOME/PATH at it for all JIT builders.
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export CUDA_PATH=$CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"

# The HiCache storage hash extension JIT-compiles against openssl.  The conda
# env ships both the headers and libcrypto, but neither is on the compiler
# default search path: the build then fails and every L3 lookup raises, so the
# storage tier silently never hits.  Point cc/ld at the env.
export CPATH="${CONDA_PREFIX:-$HOME/anaconda3/envs/Lsglang165}/include:$HOME/anaconda3/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="${CONDA_PREFIX:-$HOME/anaconda3/envs/Lsglang165}/lib:$HOME/anaconda3/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
cd "$REPO"
export PYTHONPATH=$PWD/python

MEM=${DSV41_MEM_FRACTION:-0.88}
SLOTS=${DSV41_POOL_SLOTS:-40}
SUBM=${DSV41_SUB_M:-12288}
CHUNK=${DSV41_CHUNK:-8192}
MAXTOT=${DSV41_MAX_TOTAL:-666112}
MAXREQ=${DSV41_MAX_REQ:-2}
ENGRAM=${DSV41_ENGRAM:-host}
EXTRA=${DSV41_EXTRA:-}

if [ "${DSV41_SPEC:-1}" = 1 ]; then
  SPEC_ENV=(LK_GPU_RESIDENT_MOE_LAYERS_DSPARK=0-2)
  SPEC_ARGS=(--speculative-algorithm DSPARK --speculative-dspark-block-size 5
             --speculative-num-draft-tokens 6 --speculative-attention-mode decode)
else
  SPEC_ENV=()
  SPEC_ARGS=()
fi

if [ "${DSV41_SWA_REPLAY:-1}" = 0 ]; then
  SWA_ARGS=()
else
  SWA_ARGS=(--enable-decoder-swa-bounded-replay)
fi

# How much of the token budget the paged SWA pool may hold.  Reuse only starts
# once that pool has been filled and reclaimed, so a smaller ratio makes prefix
# hits appear earlier in a session (default is the engine's 0.8).
if [ -n "${DSV41_SWA_RATIO:-}" ]; then
  SWA_RATIO_ARGS=(--swa-full-tokens-ratio "$DSV41_SWA_RATIO")
else
  SWA_RATIO_ARGS=()
fi

# Per-request SWA replay window.  Without it the SWA component gates prefix
# matching, so a finished request's full-KV prefix is never reusable (the tree
# reports a hit but the scheduler charges 0 cached tokens).  This is the switch
# that makes any KV reuse tier effective on DeepSeek-V4.1.
if [ "${DSV41_SWA_WINDOW:-0}" = 1 ]; then
  WINDOW_ARGS=(--enable-encoder-swa-bounded-replay)
else
  WINDOW_ARGS=()
fi

# Text-only serving (drops the vision tower).  sglang disables the radix cache
# for multimodal checkpoints on some model impls, which also kills every KV
# reuse tier; DSV41_TEXT_ONLY=1 is the switch for text workloads.
if [ "${DSV41_TEXT_ONLY:-0}" = 1 ]; then
  TEXT_ARGS=(--language-model-only)
else
  TEXT_ARGS=()
fi

if [ "${DSV41_METRICS:-0}" = 1 ]; then
  METRICS_ARGS=(--enable-metrics)
else
  METRICS_ARGS=()
fi

# Secondary goal: host-side KV reuse (ratio 4 by default); exploratory goal:
# an NVMe L3 tier behind it (DSV41_KV_NVME=<dir>).
HICACHE_ARGS=()
if [ "${DSV41_HICACHE:-1}" = 1 ]; then
  HICACHE_ARGS+=(--enable-hierarchical-cache
                 --hicache-host-memory-mode "${DSV41_HICACHE_MODE:-cache}"
                 --hicache-ratio "${DSV41_HICACHE_RATIO:-2}"
                 --hicache-mem-layout "${DSV41_HICACHE_LAYOUT:-page_first_direct}"
                 --hicache-io-backend "${DSV41_HICACHE_IO:-direct}"
                 --hicache-write-policy write_through
                 --hicache-host-memory-fraction "${DSV41_HICACHE_HOST_FRAC:-0.5}")
  KV_NVME_DIR=${DSV41_KV_NVME-/home/hello/hicache_l3_165}
if [ -n "$KV_NVME_DIR" ]; then
    mkdir -p "$KV_NVME_DIR"
    HICACHE_ARGS+=(--hicache-storage-backend file
                   --hicache-storage-prefetch-policy wait_complete
                   --file-storage-path "$KV_NVME_DIR"
                   --hicache-storage-backend-extra-config "{\"max_size\": \"${DSV41_KV_NVME_SIZE:-20G}\"}")
  fi
fi

if [ "$ENGRAM" = nvme ]; then
  ENGRAM_ENV=(LK_ENGRAM_NVME=1 SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=0)
else
  ENGRAM_ENV=(SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1)
fi

echo "params: mem=$MEM slots=$SLOTS sub_m=$SUBM chunk=$CHUNK maxtot=$MAXTOT maxreq=$MAXREQ engram=$ENGRAM extra=$EXTRA"

# shellcheck disable=SC2086
env "${ENGRAM_ENV[@]}" \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES=${DSV41_CUDA_VISIBLE:-0,1} \
  GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo \
  GLIBC_TUNABLES=glibc.malloc.thp_enabled=0 \
  OMP_NUM_THREADS=1 \
  LK_MOE_HYBRID_ENABLED=1 \
  LK_THREADS=${DSV41_THREADS:-48} \
  LK_THREAD_BINDING=NUMA_NODE \
  LK_GPU_PREFETCH_WINDOW=1 \
  LK_GPU_PREFILL_MIN_TOKENS=1024 \
  LK_GPU_PREFILL_ENGINE=cutlass \
  LK_GPU_PREFILL_SUB_M="$SUBM" \
  LK_GPU_PREFILL_D_SLOTS=16384 \
  LK_NUMA_INTERLEAVE=1 \
  LK_POWER_SAVING=1 \
  LK_GPU_RESIDENT_MOE_LAYERS="" \
  LK_POOL_SLOTS="$SLOTS" \
  LK_POOL_LOG=${LK_POOL_LOG:-0} \
  SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 \
  SGLANG_SKIP_P2P_CHECK=1 \
  SGLANG_SM8X_MARLIN_GROUP32=1 \
  SGLANG_SM8X_MARLIN_GROUP32_ONLY=1 \
  FLASHINFER_USE_CUDA_NORM=1 \
  FLASHINFER_DISABLE_VERSION_CHECK=1 \
  "${SPEC_ENV[@]}" \
  $EXTRA \
  sglang serve \
  --model "$MODEL" \
  --served-model-name DeepSeek-V4.1-Flash-1017 \
  --host 0.0.0.0 \
  --port "$PORT" \
  --trust-remote-code \
  "${SWA_ARGS[@]}" \
  "${WINDOW_ARGS[@]}" \
  "${SWA_RATIO_ARGS[@]}" \
  "${METRICS_ARGS[@]}" \
  "${TEXT_ARGS[@]}" \
  --enable-cache-report \
  --tensor-parallel-size 2 \
  --max-running-requests "$MAXREQ" \
  --reasoning-parser auto \
  --tool-call-parser auto \
  --disable-shared-experts-fusion \
  --disable-custom-all-reduce \
  --chunked-prefill-size "$CHUNK" \
  --context-length 665535 \
  --max-total-tokens "$MAXTOT" \
  --mem-fraction-static "$MEM" \
  --cuda-graph-backend-prefill disabled \
  --watchdog-timeout 1800 \
  "${SPEC_ARGS[@]}" \
  "${HICACHE_ARGS[@]}" \
  "$@"
