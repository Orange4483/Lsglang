#!/usr/bin/env bash
# Qwen3.8-Flash-Next-NVFP4 · 2× RTX 5060 Ti (SM120), TP2, MTP speculative decode (NEXTN).

CUDA_DEVICE_ORDER=PCI_BUS_ID \
GLIBC_TUNABLES=glibc.malloc.thp_enabled=0 \
GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo \
CUDA_VISIBLE_DEVICES=1,2 \
SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 \
LK_MOE_HYBRID_ENABLED=1 \
LK_THREADS=48 \
OMP_NUM_THREADS=1 \
LK_THREAD_BINDING=NUMA_NODE \
LK_GPU_PREFETCH_WINDOW=1 \
LK_GPU_PREFILL_MIN_TOKENS=1024 \
LK_GPU_PREFILL_SUB_M=16384 \
LK_NUMA_INTERLEAVE=1 \
LK_POWER_SAVING=1 \
LK_GPU_RESIDENT_MOE_LAYERS="" \
LK_GPU_RESIDENT_MOE_LAYERS_SPEC="" \
LK_POOL_SLOTS=12 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
LK_NVME_DIRECT=1 \
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
  --ple-offload-backend nvme \
  --chunked-prefill-size 8192 \
  --max-total-tokens 46000 \
  --mem-fraction-static 0.88 \
  --warmups prefill_shapes \
  --mm-feature-transport=cpu \
  --mm-attention-backend flashinfer_cudnn \
  --image-processor-backend pil \
  --mm-process-config '{"image":{"max_pixels":3686400,"min_pixels":200704}}' \
  --mamba-radix-cache-strategy extra_buffer_lazy \
  --mamba-max-states-per-path 2 \
  --mamba-track-interval 256 \
  --max-mamba-cache-size 11 \
  --reasoning-parser auto \
  --tool-call-parser auto \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --default-chat-template-kwargs '{"thinking":true,"reasoning_effort":"medium"}'
