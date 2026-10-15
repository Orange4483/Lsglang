#!/usr/bin/env bash
export LOG=${LOG:-/tmp/opencode/serve_logs/$(basename "$0" .sh)_$(date +%m%d_%H%M%S).log}
mkdir -p /tmp/opencode/serve_logs
exec > >(tee -a "$LOG") 2>&1
# DeepSeek-V4.1-Flash · 2× RTX 3090 (SM86), TP2, plain decode.

CUDA_DEVICE_ORDER=PCI_BUS_ID \
GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo \
GLIBC_TUNABLES=glibc.malloc.thp_enabled=0 \
CUDA_VISIBLE_DEVICES=0,3 \
LK_MOE_HYBRID_ENABLED=1 \
LK_THREADS=48 \
OMP_NUM_THREADS=1 \
LK_THREAD_BINDING=CPU_CORE \
LK_GPU_PREFETCH_WINDOW=1 \
LK_GPU_PREFILL_MIN_TOKENS=1024 \
LK_GPU_PREFILL_SUB_M=8192 \
LK_NUMA_INTERLEAVE=1 \
LK_POWER_SAVING=1 \
LK_GPU_RESIDENT_MOE_LAYERS="" \
LK_POOL_SLOTS=12 \
SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 \
SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 \
SGLANG_SKIP_P2P_CHECK=1 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
sglang serve \
  --model ~/Models/DeepSeek-V4.1-Flash \
  --served-model-name DeepSeek-V4.1-Flash \
  --host 0.0.0.0 \
  --port 8070 \
  --trust-remote-code \
  --enable-decoder-swa-bounded-replay \
  --tensor-parallel-size 2 \
  --max-running-requests 2 \
  --disable-shared-experts-fusion \
  --disable-custom-all-reduce \
  --chunked-prefill-size 1024 \
  --max-total-tokens 131072 \
  --mem-fraction-static 0.80 \
  --cuda-graph-backend-prefill disabled \
