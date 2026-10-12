#!/usr/bin/env bash
# Usage: run `conda activate Lsglang` first, then execute this script.
# Qwen3.8-Flash-Next-NVFP4 · 2× RTX 3090 + 2× RTX 5060 Ti (SM86+SM120 mixed), TP4, plain decode.
# Budget = 5060Ti rank (16.31G): fixed 5.3 (w3.2 TP-sliced / mamba 0.32 / ctx 0.5 / graph 0.5 / JIT 0.9)
#   + KV@256K 3.0 (kv heads=2<4 -> 1 head/rank REPLICATED, 11.4KB/tok same as TP2) + pool0 + hc 0.32 + act 0.6
#   => margin ~5.2GB >= 1.5 floor. 256K = model max_context. pool128 needs 196K (E line).
# Marlin flags MANDATORY on mixed TP: sm86=marlin vs sm120=auto(cutlass) fork => decode halved (10-09 case).

CUDA_DEVICE_ORDER=PCI_BUS_ID \
FLASHINFER_CUDA_ARCH_LIST="8.6 12.0f" \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 \
LK_MOE_HYBRID_ENABLED=1 \
LK_THREADS=48 \
OMP_NUM_THREADS=1 \
LK_THREAD_BINDING=CPU_CORE \
LK_GPU_PREFETCH_WINDOW=2 \
LK_GPU_PREFILL_MIN_TOKENS=256 \
LK_GPU_PREFILL_SUB_M=16384 \
LK_NUMA_INTERLEAVE=1 \
LK_POWER_SAVING=1 \
LK_GPU_RESIDENT_MOE_LAYERS="" \
LK_GPU_RESIDENT_MOE_LAYERS_SPEC="" \
LK_POOL_SLOTS=0 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
sglang serve \
  --model ~/Models/Qwen3.8-Flash-Next-NVFP4-W4A16-4o6-FP8 \
  --served-model-name Qwen3.8-Flash-Next-NVFP4 \
  --host 0.0.0.0 \
  --port 8072 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --max-running-requests 2 \
  --disable-shared-experts-fusion \
  --disable-custom-all-reduce \
  --ple-offload-embedding \
  --ple-offload-backend pinned \
  --chunked-prefill-size 8192 \
  --max-total-tokens 262144 \
  --mem-fraction-static 0.90 \
  --warmups prefill_shapes \
  --max-mamba-cache-size 11 \
  --reasoning-parser auto \
  --tool-call-parser auto \
  --default-chat-template-kwargs '{"thinking":true,"reasoning_effort":"medium"}'
