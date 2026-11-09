#!/usr/bin/env bash
# Run the packaged 1.7.3 environment on a target host, mirroring the 1.6.5
# production knobs, so the L1/L2/L3 behaviour can be compared like for like.
#   bash launch/173/serve_173.sh 14|15
set -uo pipefail
HOST_NUM=${1:?usage: serve_173.sh 14|15}; shift || true
# Default to the newest packaged env on the host, so a fresh checkout does not
# silently keep running the previous release.
E=${ENV173:-}
if [ -z "$E" ]; then
  for cand in "$HOME/envs/Lsglang175" "$HOME/envs/Lsglang174" "$HOME/envs/Lsglang173"; do
    if [ -d "$cand" ]; then E=$cand; break; fi
  done
fi
: "${E:?no lsglang env found under $HOME/envs}"
PORT=${PORT:-8070}
LOG=${LOG:-$HOME/project/Lsglang-egram-nvme-kv-nvme/lsglang173/serve_${HOST_NUM}.log}
mkdir -p "$(dirname "$LOG")"
if [ "$HOST_NUM" = 14 ]; then
  MODEL=/media/hello/Windows/models/DeepSeek-V4.1-Flash-1017
  TP=4
  # ratio 4 (the 1.7.2 recipe) needs ~60 GB of host KV on top of the host MoE
  # table and OOM-killed the scheduler on a 503 GB host; 1 is the safe default.
  HICACHE_RATIO=${DSV41_HICACHE_RATIO:-1}
  TEXT_ARGS=(--language-model-only)
  ENGRAM_ENV=(LK_ENGRAM_NVME=1 SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=0)
  CUDA_VISIBLE=${DSV41_CUDA_VISIBLE:-0,1,2,3}
  THREADS=${DSV41_THREADS:-16}
  # Measured on the 2026-11-09 sweep: chunk 8192 + SUB_M 20480 (CAP=20480,
  # inner=8192) lifts a cold 32K prefill from 2,318 to 3,022 t/s and 80K from
  # 2,225 to 3,149 t/s at 666,112 tokens. 24576 OOMs a single 32K prompt on the
  # 24 GB cards; 20480 survives two parallel 58K prefills.
  DEF_CHUNK=8192
  DEF_SUBM=20480
  DEF_POWER=1
else
  MODEL=$HOME/models/DeepSeek-V4.1-Flash-1017
  TP=2
  HICACHE_RATIO=${DSV41_HICACHE_RATIO:-2}
  TEXT_ARGS=()
  ENGRAM_ENV=(SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1)
  CUDA_VISIBLE=${DSV41_CUDA_VISIBLE:-0,1}
  THREADS=${DSV41_THREADS:-48}
  # Measured on the 2026-11-09 sweep: chunk 1024 + SUB_M 32768 (CAP=32768,
  # inner=1024) lifts a cold 32K prefill from 2,129 to 3,424 t/s and 80K to
  # 3,190 t/s while keeping max_total_num_tokens at 666,112. SUB_M 65536 OOMs
  # an 80K prefill on this box, so 32768 is the ceiling.
  DEF_CHUNK=1024
  DEF_SUBM=32768
  DEF_POWER=0
fi
SUBM=${DSV41_SUB_M:-$DEF_SUBM}
CHUNK=${DSV41_CHUNK:-$DEF_CHUNK}
SLOTS=${DSV41_POOL_SLOTS:-40}
MAXTOT=${DSV41_MAX_TOTAL:-666112}
MAXREQ=${DSV41_MAX_REQ:-2}
MEM=${DSV41_MEM_FRACTION:-0.88}
# Keep caches out of the home root: one stable directory per host, wiped when
# the build or the profile changes (see clean_l3.sh).
KV_NVME_DIR=${DSV41_KV_NVME-$HOME/kvcache/l3}
mkdir -p "$KV_NVME_DIR"
HICACHE_MODE=${DSV41_HICACHE_MODE:-$([ "${DSV41_HICACHE:-1}" = 1 ] && echo l3 || echo l1)}
# Host pool sizing. DeepSeek-V4 HiCache rejects --hicache-size, so the only
# knobs are the device-relative ratio and the host-memory fraction. The ratio
# alone ignores how much RAM the host actually has (ratio 4 drove a 503 GB host
# into the kernel OOM killer); the fraction makes sglang shrink its own ratio
# until the host pools fit.
if [ -n "${DSV41_HICACHE_HOST_FRACTION:-}" ]; then
  HICACHE_CAPACITY=(--hicache-host-memory-fraction "$DSV41_HICACHE_HOST_FRACTION")
else
  HICACHE_CAPACITY=(--hicache-ratio "$HICACHE_RATIO")
fi
case "$HICACHE_MODE" in
  l1)
    HICACHE_ARGS=() ;;
  l2)
    # host pool only: no L3 file backend, so no restore-from-disk path
    HICACHE_ARGS=(--enable-hierarchical-cache "${HICACHE_CAPACITY[@]}"
      --hicache-mem-layout page_first_direct --hicache-io-backend direct
      --hicache-write-policy write_through) ;;
  *)
    HICACHE_ARGS=(--enable-hierarchical-cache "${HICACHE_CAPACITY[@]}"
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
  LK_POWER_SAVING="${LK_POWER_SAVING:-$DEF_POWER}" \
  LK_GPU_RESIDENT_MOE_LAYERS="" \
  LK_POOL_SLOTS="$SLOTS" \
  LK_POOL_NO_STAGE=0 \
  LK_SWA_BOUNDED_FULL_PREFIX="${LK_SWA_BOUNDED_FULL_PREFIX:-1}" \
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


