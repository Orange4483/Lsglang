#!/usr/bin/env bash
# Qwen3.8-Flash-Next-NVFP4 · 2× RTX 5060 Ti (SM120), TP2, plain decode (no speculation)
# pool0 by A/B: pool12 costs 2.06GB VRAM (170MB/slot), no prefill gain,
# decode 45->40. 192K ctx = 5060Ti ceiling; 256K OOM (p89 tested).
# 5060Ti ceiling (256K KV needs 8.7GB -> exceeds frac cap). Derived from the mtp
# script with --speculative-* removed.

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1,2 \
SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 \
LK_MOE_HYBRID_ENABLED=1 \
LK_THREADS=48 \
OMP_NUM_THREADS=1 \
LK_THREAD_BINDING=NUMA_NODE \
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
  --port 8071 \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --max-running-requests 2 \
  --disable-shared-experts-fusion \
  --disable-custom-all-reduce \
  --ple-offload-embedding \
  --ple-offload-backend pinned \
  --chunked-prefill-size 8192 \
  --max-total-tokens 196608 \
  --mem-fraction-static 0.92 \
  --warmups prefill_shapes \
  --max-mamba-cache-size 11 \
  --reasoning-parser auto \
  --tool-call-parser auto \
  --default-chat-template-kwargs '{"thinking":true,"reasoning_effort":"medium"}'
