#!/usr/bin/env bash
# Qwen3.6-35B-A3B-NVFP4 · 2× RTX 5060 Ti (SM120), TP2, NEXTN d4 (MTP), resident 0-1,33-34 + pool.

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1,2 \
SGLANG_SKIP_P2P_CHECK=1 \
SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 \
LK_MOE_HYBRID_ENABLED=1 \
LK_THREAD_BINDING=NUMA_NODE \
LK_THREADS=48 \
OMP_NUM_THREADS=1 \
LK_POWER_SAVING=1 \
LK_NUMA_INTERLEAVE=1 \
LK_GPU_PREFILL_MIN_TOKENS=1024 \
LK_GPU_PREFILL_SUB_M=65536 \
LK_GPU_RESIDENT_MOE_LAYERS=0-1,33-34 \
LK_POOL_SLOTS=128 \
sglang serve \
    --model ~/Models/Qwen3.6-35B-A3B-NVFP4 \
    --served-model-name Qwen3.6-35B-A3B-NVFP4 \
    --host 0.0.0.0 \
    --port 8070 \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --max-running-requests 2 \
    --chunked-prefill-size 32768 \
    --max-total-tokens 26000 \
    --mem-fraction-static 0.90 \
    --tool-call-parser auto \
    --reasoning-parser auto \
    --disable-shared-experts-fusion \
    --cuda-graph-backend-prefill disabled \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4
