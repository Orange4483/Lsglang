#!/usr/bin/env bash
LOG="${LOG:-/tmp/opencode/serve_logs/sglang_serve_qwen36_mtp_tp2_3090.log}"
mkdir -p /tmp/opencode/serve_logs
echo "===== $(date "+%F %T") launch pid=$$ =====" >> "$LOG"
if [ -t 1 ]; then
    # interactive terminal: console and log both live (closing the
    # terminal is a deliberate stop)
    exec > >(tee -a "$LOG") 2>&1
else
    # detached/tool launch: plain file append, no coprocess that a
    # session teardown could kill (EPIPE would take the engine down)
    exec >> "$LOG" 2>&1
fi
# Qwen3.6-35B-A3B-NVFP4 · 2× RTX 3090 (SM86), TP2, NEXTN d4 (MTP), resident 0-1,33-34 + pool.

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0,3 \
SGLANG_SKIP_P2P_CHECK=1 \
SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 \
LK_MOE_HYBRID_ENABLED=1 \
LK_THREAD_BINDING=NUMA_NODE \
LK_THREADS=48 \
OMP_NUM_THREADS=1 \
LK_POWER_SAVING=1 \
LK_NUMA_INTERLEAVE=1 \
LK_GPU_PREFILL_MIN_TOKENS=1024 \
LK_GPU_PREFILL_SUB_M=32768 \
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
    --chunked-prefill-size 8192 \
    --max-total-tokens 262144 \
    --mem-fraction-static 0.95 \
    --tool-call-parser auto \
    --reasoning-parser auto \
    --disable-shared-experts-fusion \
    --cuda-graph-backend-prefill disabled \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4
