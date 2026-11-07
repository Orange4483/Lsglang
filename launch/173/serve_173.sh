#!/usr/bin/env bash
# Run the packaged 1.7.3 environment on a target host, mirroring the 1.6.5
# production knobs, so the L1/L2/L3 behaviour can be compared like for like.
#   bash launch/173/serve_173.sh 14|15
set -uo pipefail
HOST_NUM=${1:?usage: serve_173.sh 14|15}; shift || true
E=${ENV173:-$HOME/envs/Lsglang173}
PORT=${PORT:-8070}
LOG=${LOG:-$HOME/project/Lsglang-egram-nvme-kv-nvme/lsglang173/serve_${HOST_NUM}.log}
mkdir -p "$(dirname "$LOG")"
if [ "$HOST_NUM" = 14 ]; then
  MODEL=/media/hello/Windows/models/DeepSeek-V4.1-Flash-1017
  TP=4
  HICACHE_RATIO=${DSV41_HICACHE_RATIO:-4}
  TEXT_ARGS=(--language-model-only)
  ENGRAM_ENV=(LK_ENGRAM_NVME=1 SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=0)
  CUDA_VISIBLE=${DSV41_CUDA_VISIBLE:-0,1,2,3}
  THREADS=${DSV41_THREADS:-16}
else
  MODEL=$HOME/models/DeepSeek-V4.1-Flash-1017
  TP=2
  HICACHE_RATIO=${DSV41_HICACHE_RATIO:-2}
  TEXT_ARGS=()
  ENGRAM_ENV=(SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1)
  CUDA_VISIBLE=${DSV41_CUDA_VISIBLE:-0,1}
  THREADS=${DSV41_THREADS:-48}
fi
SUBM=${DSV41_SUB_M:-12288}
CHUNK=${DSV41_CHUNK:-8192}
SLOTS=${DSV41_POOL_SLOTS:-40}
MAXTOT=${DSV41_MAX_TOTAL:-666112}
MAXREQ=${DSV41_MAX_REQ:-2}
MEM=${DSV41_MEM_FRACTION:-0.88}
KV_NVME_DIR=${DSV41_KV_NVME-/home/hello/hicache_l3_173}
mkdir -p "$KV_NVME_DIR"
HICACHE_MODE=${DSV41_HICACHE_MODE:-$([ "${DSV41_HICACHE:-1}" = 1 ] && echo l3 || echo l1)}
case "$HICACHE_MODE" in
  l1)
    HICACHE_ARGS=() ;;
  l2)
    # host pool only: no L3 file backend, so no restore-from-disk path
    HICACHE_ARGS=(--enable-hierarchical-cache --hicache-ratio "$HICACHE_RATIO"
      --hicache-mem-layout page_first_direct --hicache-io-backend direct
      --hicache-write-policy write_through) ;;
  *)
    HICACHE_ARGS=(--enable-hierarchical-cache --hicache-ratio "$HICACHE_RATIO"
      --hicache-mem-layout page_first_direct --hicache-io-backend direct
      --hicache-write-policy write_through --hicache-storage-backend file
      --hicache-storage-prefetch-policy wait_complete
      --file-storage-path "$KV_NVME_DIR"
      --hicache-storage-backend-extra-config '{"max_size": "20G"}') ;;
esac
if [ "${DSV41_SWA_REPLAY:-1}" = 1 ]; then
  SWA_ARGS=(--enable-decoder-swa-bounded-replay)
else
  SWA_ARGS=(--no-enable-decoder-swa-bounded-replay)
fi

LOGREQ_ARGS=()
if [ "${DSV41_LOG_REQ:-0}" = 1 ]; then
  LOGREQ_ARGS=(--log-requests --log-requests-level "${DSV41_LOG_REQ_LEVEL:-3}"
               --log-requests-format json)
fi

if [ "$HOST_NUM" = 15 ] && [ "${DSV41_SPEC:-0}" = 1 ]; then  SPEC_ENV=(LK_GPU_RESIDENT_MOE_LAYERS_DSPARK=0-2)
  SPEC_ARGS=(--speculative-algorithm DSPARK --speculative-dspark-block-size 5
             --speculative-num-draft-tokens 6 --speculative-attention-mode decode)
else
  SPEC_ENV=()
  SPEC_ARGS=()
fi

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate "$E"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="${CONDA_PREFIX}/include:$HOME/anaconda3/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="${CONDA_PREFIX}/lib:$HOME/anaconda3/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"

echo "params: env=$E tp=$TP slots=$SLOTS sub_m=$SUBM chunk=$CHUNK maxtot=$MAXTOT maxreq=$MAXREQ hicache=$HICACHE_RATIO l3=$KV_NVME_DIR" | tee -a "$LOG"

exec env "${ENGRAM_ENV[@]}" \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE" \
  GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo \
  GLIBC_TUNABLES=glibc.malloc.thp_enabled=0 \
  OMP_NUM_THREADS=1 \
  LK_MOE_HYBRID_ENABLED=1 \
  LK_THREADS="$THREADS" \
  LK_THREAD_BINDING=NUMA_NODE \
  LK_GPU_PREFETCH_WINDOW=1 \
  LK_GPU_PREFILL_MIN_TOKENS=1024 \
  LK_GPU_PREFILL_ENGINE=cutlass \
  LK_GPU_PREFILL_SUB_M="$SUBM" \
  LK_GPU_PREFILL_D_SLOTS=16384 \
  LK_GPU_PREFILL_WAVE_EXPERTS="${DSV41_WAVE_EXPERTS:-64}" \
  LK_NUMA_INTERLEAVE=1 \
  LK_POWER_SAVING=1 \
  LK_GPU_RESIDENT_MOE_LAYERS="" \
  LK_POOL_SLOTS="$SLOTS" \
  LK_POOL_NO_STAGE=0 \
  "${SPEC_ENV[@]}" \
  LK_POOL_LOG=${LK_POOL_LOG:-0} \
  SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 \
  SGLANG_SKIP_P2P_CHECK=1 \
  FLASHINFER_USE_CUDA_NORM=1 \
  FLASHINFER_DISABLE_VERSION_CHECK=1 \
  ${DSV41_EXTRA:-} \
  "$E/bin/sglang" serve \
  --model "$MODEL" \
  --served-model-name DeepSeek-V4.1-Flash-1017 \
  --host 0.0.0.0 \
  --port "$PORT" \
  --trust-remote-code \
  "${SWA_ARGS[@]}" \
  "${SPEC_ARGS[@]}" \
  "${TEXT_ARGS[@]}" \
  --enable-cache-report \
  --enable-metrics \
  --tensor-parallel-size "$TP" \
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
  "${HICACHE_ARGS[@]}" \
  "${LOGREQ_ARGS[@]}" \
  "$@" >>"$LOG" 2>&1
