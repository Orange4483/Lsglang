#!/usr/bin/env bash
# GLM-5.3-Flash-NVFP4 · 2× RTX 3090 (SM86), TP2, MTP speculative decode (NEXTN).

CUDA_DEVICE_ORDER=PCI_BUS_ID \
GLIBC_TUNABLES=glibc.malloc.thp_enabled=0 \
GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo \
CUDA_VISIBLE_DEVICES=0,3 \
SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 \
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
LK_GPU_RESIDENT_MOE_LAYERS_SPEC="" \
LK_POOL_SLOTS=16 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
/home/guqiong/.conda/envs/Lsglang/bin/python /home/guqiong/Downloads/sglang/python/sglang/launch_server.py \
  --model ~/Models/GLM-5.3-Flash-NVFP4 \
  --served-model-name GLM-5.3-Flash-NVFP4 \
  --host 0.0.0.0 \
  --port 8070 \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --max-running-requests 2 \
  --max-mamba-cache-size 11 \
  --disable-shared-experts-fusion \
  --disable-custom-all-reduce \
  --chunked-prefill-size 4096 \
  --max-total-tokens 140000 \
  --mem-fraction-static 0.90 \
  --cuda-graph-backend-prefill disabled \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 
