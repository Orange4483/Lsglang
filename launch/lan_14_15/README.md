# DeepSeek-V4.1-Flash-1017 at 665535 context on .14 / .15

Two offline hosts, one branch (`feat/dsv41-665535-12-14-15`), Lsglang 1.6.5 with
lk_moe 2.7.3.  Everything here was measured on the machines, not estimated.

| | 192.168.1.14 | 192.168.1.15 |
|---|---|---|
| CPU / RAM | 1× Xeon Gold 6430 (32C/64T), 503 GiB | Xeon 6530, 128 logical, 1007 GiB |
| GPU | 4× RTX 4090 24 GB, SM89, TP4 | RTX 4090 48 GB (SM89) + RTX PRO 5000 48 GB (SM120), TP2 |
| NVMe | WD Blue SN5000 2 TB, DRAM-less, PCIe 3.0 x4 | Predator GM7000 2 TB, DRAM, PCIe 4.0 **x2** |
| Model | `/media/hello/Windows/models/DeepSeek-V4.1-Flash-1017` (ntfs3) | `/home/hello/models/DeepSeek-V4.1-Flash-1017` (ext4) |
| Engram | NVMe-direct (`LK_ENGRAM_NVME=1`) | 190 GiB shmem host table, pinned, 2 MiB pages |
| Decode | plain | DSPARK (block 5 / verify 6, draft 0-2 resident) |

Endpoint on both: `0.0.0.0:8070`, model name `DeepSeek-V4.1-Flash-1017`,
`--context-length 665535`, `--max-total-tokens 666112`.

## Why 666112 and not 665535

The KV pool is tokenised in 256-token pages, so `--max-total-tokens 665535`
produced a pool of 665344 tokens: a full-context prompt then had no room for
even one output token (measured: 665000-token prompt prefilled, 0 tokens out).
666112 keeps `--context-length 665535` as the client-visible limit and gives the
pool its page-rounded size.

## Version-specific fixes on this branch

1. `fix(engram): pin the host table in 1 GiB granules` — `cudaHostRegister` on
   driver 595.58.03 returns EINVAL for a single call above ~2 GiB
   (`dmesg: Cannot map memory with base addr ... and size of ... pages`), so the
   94 GiB/rank Engram table could never be pinned and `.15` died in
   `finish_load`.  UVA keeps the device address equal to the host address, so
   chunking is invisible to the gather kernels.
2. `fix(launch): use the system CUDA 13.2 toolkit for JIT builds` — the conda
   env carries nvcc 13.4 beside cudart 13.0 headers, so any cold tilelang /
   cutlass JIT aborts with *"CUDA compiler and CUDA toolkit headers are
   incompatible"* (`.14` died in prewarm; `.15` only survived because its JIT
   cache was warm).  `CUDA_HOME=/usr/local/cuda` (13.2 + 13.2) is consistent.
3. `fix(hicache): pin host KV pools in 1 GiB calls` — the HiCache host-pool
   allocator registers the whole pool in one call (chunk limit 256 GiB), hitting
   the same driver limit; `attach_hybrid_pool_to_unified_cache` failed.
4. THP for the Engram host table needs **both** the sysfs knob and the tmpfs
   mount option: with `shmem_enabled=advise` alone a 1 GiB probe mapping still
   showed `ShmemPmdMapped=0`; `mount -o remount,huge=advise /dev/shm` gave
   `ShmemPmdMapped=1048576 kB`.  `start.sh` does both, `tune_nvme.sh` re-applies.

## Launch

The measured parameters are the script defaults, so the canonical commands are
short:

```bash
# .14 — plain decode, Engram from NVMe, L2 host tier ratio 4, L3 NVMe, text-only
bash launch/lan_14_15/restart.sh 14

# .15 — DSPARK decode, Engram host table, L2 host tier ratio 2, L3 NVMe
bash launch/lan_14_15/restart.sh 15
```

Defaults both hosts inherit (`DSV41_*` overrides still win):
`--chunked-prefill-size 8192`, `LK_POOL_SLOTS` 40 (both hosts),
`--mem-fraction-static 0.88`, `--max-total-tokens 666112`,
`--max-running-requests 2`, HiCache on with host ratio **4 (.14) / 2 (.15)**,
the NVMe L3 tier at `/home/hello/hicache_l3_165` (20 GB cap), and text-only on .14.  `LK_POOL_LOG=100` turns on the `[pool]` window
stats used to size the pool.

Optional tiers on either host (`--max-total-tokens` unchanged):

```bash
# host KV tier at a custom ratio
DSV41_HICACHE=1 DSV41_HICACHE_RATIO=1 bash launch/lan_14_15/restart.sh 15

# exploratory NVMe L3 tier behind it (see the L3 section: still blocked)
DSV41_HICACHE=1 DSV41_KV_NVME="$HOME/hicache_l3_165" \
  bash launch/lan_14_15/restart.sh 15
```

`start.sh` sets the THP mount option, `stop.sh` kills the session *and* the
`sglang::scheduler_TP*` workers (a bare SIGTERM to the parent leaves workers
holding VRAM), `restart.sh` waits for `/model_info` to name the right model.

## Measured performance

Single request, `ignore_eos`, nonce-disjoint prompts (`client.py gen`), so no
prefix-cache reuse.  Raw lines: `logs/<tag>_stability.jsonl` on each host.

### .14 — TP4, Engram on NVMe, plain decode

| prompt | prefill t/s | decode t/s |
|---|---|---|
| 4 096 | 912 | 25.7 |
| 32 768 | 2 334 | 25.9 |
| 131 072 | 2 229 | 24.0 |
| 262 144 | 2 037 | 23.4 |
| 524 288 | 1 687 | 22.9 |
| 600 000 | 1 600 | 21.4 |
| 660 000 | 1 489 | 22.8 |
| **665 000** | **1 528** | **21.3** |

### .15 — TP2, Engram host table

| prompt | prefill t/s | decode t/s | config |
|---|---|---|---|
| 4 096 | 333 | 15.6 | 4 KiB pages, plain |
| 32 768 | 2 020 | 15.6 | 4 KiB pages, plain |
| 131 072 | 1 993 | 12.3 | 4 KiB pages, plain |
| 4 096 | 916 | 46.0 | 2 MiB pages, DSPARK |
| 32 768 | 2 002 | 36.4 | 2 MiB pages, DSPARK |
| 131 072 | 1 999 | 32.3 | 2 MiB pages, DSPARK |
| 262 144 | 1 829 | 32.9 | 2 MiB pages, DSPARK |
| 524 288 | 1 551 | 29.9 | 2 MiB pages, DSPARK |
| 660 000 | 1 434 | 30.8 | 2 MiB pages, DSPARK |
| **665 000** | **1 424** | **16.9** | full pool (`full token usage 1.00`) |

The 4K row with cold JIT shows the TTFT-inflated number; steady-state 4K prefill
is the 32K-class figure.

### Full-context proof

`665000`-token prompts (the largest that still leaves output room under the
665535 context limit) completed on both hosts with the pool at
`full token usage 1.00` and 16/16 generated tokens:

| | TTFT | prefill | decode |
|---|---|---|---|
| .14 | 435 s | 1 528 t/s | 21.3 t/s |
| .15 | 467 s | 1 424 t/s | 16.9 t/s |

### Speculation is host-dependent

On `.15` DSPARK multiplies decode 15.6 -> 46 t/s at 4K and keeps 30-33 t/s up to
660K.  On `.14` it is a wash (27.4 vs 25.7 t/s at 4K, 27.5 vs 25.9 at 32K):
verifying 6 tokens costs about five plain CPU-MoE steps there, so the draft is
disabled and `.14` runs plain decode.

## KV reuse: root cause found, blocked upstream in 1.6.5

Both goals (host tier at ratio 4, NVMe tier behind it) need one thing first: a
request whose prefix is actually reusable.  It never is, on either host.

**Symptom.** The identical prompt sent twice always reports
`#cached-token: 0` and the same TTFT, in every combination tried: streaming and
non-streaming, with and without HiCache, with and without
`--enable-decoder-swa-bounded-replay`, with and without `--language-model-only`
(8 k prompt: 5.8 s then 4.8 s wall; 131 k: 65.3 s then 65.3 s; a 516-token
prompt also 0/0).

**The tree does match — the SWA component is what zeroes it.**  With
`SGLANG_RADIX_DEBUG=1` (branch diagnostic) the same two requests log:

```
RADIXDBG match  disable=False key_len=8068 full_hit=7936 swa_hit=0
RADIXDBG insert key_len=7936 prefix_len=7936
```

The Full-KV prefix is present (7936 of 8068 tokens); `swa_hit=0` because a
finished request's sliding-window pages are not retained.  For a hybrid-SWA
model the SWA component gates the match (`swa_gates_prefix_match` in
`kv_cache_builder.py`), so the reusable prefix collapses to 0 even though the
tree found one.

**The only switch that bypasses the gate** is a per-request SWA window, which
`kv_cache_builder` only sees when the pool owns `request_window` — created by
`--enable-encoder-swa-bounded-replay` (`deepseek_v4_memory_pool.py:1121`), not by
the decoder-replay flag we had been using.

What that flag does in practice:

| host | result with `--enable-encoder-swa-bounded-replay` |
|---|---|
| `.14` (no speculation) | `hybrid_swa` becomes False and HiCache attaches — after the new adaptive pin, TP1 splits one 1 GiB call into 2×512 MiB — but the **first prefill dies**: `CUDA error: invalid argument` in `prepare_for_extend` at `torch.tensor(seq_lens, pin_memory=True).to(device)`. Reproduced with and without HiCache, with decoder replay on and off, and at `--max-running-requests` 1 and 2. |
| `.15` (DSpark) | The flag is accepted, but `needs_paged_swa_allocator` stays True (`not uses_draft_swa_ring()` with a speculative algorithm), so `hybrid_swa` remains True, matching stays gated, `cached_tokens=0`. The server is stable. |

So on `.14` the replay path itself is broken for this stack (lk_moe hybrid MoE
+ Engram-on-NVMe), not the host tier; on `.15` the speculative layout keeps the
gate in place.

**Consequence for the two goals.**  `DSV41_HICACHE=1` (ratio 4) attaches and
serves, and `DSV41_KV_NVME=<dir>` wires the `file` L3 backend behind it, but no
request produces a reusable prefix, so neither tier ever has something to store
or restore.  That is why the phases cannot be measured yet — not a tuning
problem but a missing/blocked code path.  Two ways forward, both upstream:

1. make encoder SWA replay work at bs=1 with the hybrid-MoE/Engram stack (the
   `.14` crash above), or
2. keep the paged SWA pool for DSpark but retain/cache the window so
   `swa_hit > 0` (the `.15` gate).

Everything needed to measure the payoff afterwards is already in the tree:
`reuse_test.py` (same-prompt TTFT), `reuse_abc.sh` (A/B/A eviction), and
`scenario_agent.py` / `phase_run.sh` — a simulated "same repo, many turns"
workload that reports per-turn `prompt_tokens`, `cached_tokens` and wall time.
Without reuse it shows a flat curve (every turn re-prefills the whole context),
which is the baseline the tiers have to beat.

### Update: reuse does fire once the SWA pool is warm
A 6-turn, 24k-context run of `scenario_agent.py` on `.14` (plain decode, no
HiCache) shows the gate is *dynamic*:

| turn | prompt tokens | cached tokens | wall | decode |
|---|---|---|---|---|
| 1 | 24050 | 0 | 12.5 s | 3.8 t/s |
| 2 | 24113 | 0 | 12.4 s | 3.8 t/s |
| 3 | 24175 | 0 | 12.6 s | 3.7 t/s |
| 4 | 24239 | **24064** | 4.4 s | 10.8 t/s |
| 5 | 24300 | **24064** | 4.5 s | 10.4 t/s |
| 6 | 24361 | **24064** | 4.8 s | 9.8 t/s |

The first three turns log `swa token usage: 0.90` and re-prefill everything;
from turn 4 the same requests log `swa usage: 0.02`, `#new-token: 175..297` and
the whole fixed prefix is reused — 2.8x lower wall time per turn.  This is
reproducible (run twice).  The reading: while the paged SWA pool is saturated
the SWA component still gates the match; once its pages are reclaimed the
full-KV prefix becomes matchable and stays so.

That gives the host/NVMe tiers something to store in a *warm* workload, but a
cold start still pays full prefill, and the earlier blockers (encoder-replay
crash on `.14`, paged-SWA gate on `.15`) still stand for the cold case.  The
next experiment is `--swa-full-tokens-ratio 0.1` (fill/reclaim the SWA pool
earlier so reuse starts at turn 2) followed by the same scenario with
`DSV41_HICACHE=1` and a distractor to force L2 eviction.



## Stability

Repeated runs at 32K and 131K (48 generated tokens each) completed back to back
on `.14` with consistent numbers — prefill 2196/2301 t/s at 32K and
2277/2294 t/s at 131K, decode 26.3/26.5 and 24.2/23.6 t/s — and the full-length
matrix (4K … 665K) ran without an OOM or a retraction.  The one crash seen was
self-inflicted: killing the HTTP client mid-request left the engine with an
abandoned 262K prefill and the SM120 card took an `Xid 8` (launch timeout) about
a minute later.  With clean requests the same configuration completed 262K,
524K and 660K with DSPARK.

## Operating notes

* **A killed server can leave the NVIDIA driver holding pinned host memory.**
  After an aborted HiCache boot `.14` reported 460 GiB "used" with no sglang
  process, no page cache and no device memory in use, and every later boot
  stalled in `lk_moe init` with 0 GiB available; `dmesg` showed
  `Cannot map memory with base addr ... and size of ... pages`.  A reboot
  cleared it (500 GiB free again) -- the same failure mode as the 1.5.6
  handover.  Stop servers with `stop.sh`; do not kill the client mid-request or
  leave worker orphans behind.
* `.14` is a **single** Xeon Gold 6430 (32C/64T, one NUMA node), which is what
  `lk_moe` reports as `numa nodes: 1 cors:32 cpus:64`.  With one node
  `LK_NUMA_INTERLEAVE=1` has nothing to interleave over; that is not a defect.
  NPS is an AMD EPYC knob — the Intel equivalent would be SNC/SNC2 in BIOS,
  which splits this one socket into two NUMA nodes and could give
  `LK_THREAD_BINDING=NUMA_NODE` something to bind to.  Optional tuning only;
  nothing above depends on it.
* `--enable-decoder-swa-bounded-replay` stays on by default (it is in the
  shipped 1.6.5 script and every number above was measured with it);
  `DSV41_SWA_REPLAY=0` exists only for the reuse experiment.

## NVMe notes

* `.14`'s model lives on the NTFS Windows partition (`ntfs3`).  O_DIRECT works
  there (verified with an aligned 4 KiB random-read probe); measured 47 k IOPS
  buffered 264 B rows at QD8 and 3.4 GiB/s sequential O_DIRECT, which is the
  PCIe 3.0 x4 limit of the link, not the drive.  The volume was mounted by
  udisks on login, so after a reboot it may be missing and the server dies on
  `Repo id must be in the form ...`; an fstab entry with `nofail` now mounts it
  (`/dev/nvme0n1p3 /media/hello/Windows ntfs3 rw,uid=1000,gid=1000,iocharset=utf8,nofail 0 0`).
* `.15`'s GM7000 is negotiated at **x2** (`LnkSta: Speed 16GT/s, Width x2`), so
  it runs at the same ~3.5 GiB/s as the x4 Gen3 link on `.14`; that is a slot /
  wiring issue, not a setting.
* `tune_nvme.sh` applies the useful knobs: `noatime` on the model filesystems,
  `max_sectors_kb=1024` for weight streaming, `read_ahead_kb=0` for the 264 B
  random Engram rows, scheduler `none`, and the shmem THP mount option.
* The `.14` Engram filler moves only ~38 MB/s during a 600K prefill (45 M row
  lookups, 62% deduped), so the NVMe path is not the prefill bottleneck.
### Host-pool sizing on .14

--hicache-ratio 4 with the full multimodal checkpoint does not fit on the
503 GiB host: the boot ends at 0 GiB available and the crash leaks ~360 GiB of
driver-pinned memory (a reboot is needed to clear it).  atio 1 plus
--language-model-only is stable and still exercises the tier:

* hicache_attached=True, 314 GiB used / 188 GiB available after a 524K request
  (GPU 17.9 / 16.3 GiB per card),
* 524288-token prompt: 1676 t/s prefill, 21.8 t/s decode,
* the 24K six-turn scenario still reuses 24064 tokens from turn 4 (12.4 s -> 4.3 s).

--swa-full-tokens-ratio 0.1 did **not** move the turn-4 threshold, and four
identical single-turn sends never hit, so the warm-up rule is not about SWA-pool
capacity or client timing.  A 400K distractor did not produce an observable L2
restore either, but that probe is inconclusive: 400K + 24K stays under the 666K
device pool, and the turn-4 gate suppresses the first three turns regardless of
tier.  Next: --hicache-host-memory-mode buffer_only with the NVMe ile
backend (host becomes a staging buffer, KV evicts to disk) plus a host_hit
figure in the radix trace so an L2/L3 restore is visible instead of inferred.
### L3 (NVMe) attach crashes on .14

The 1.5.6 recipe (launch_l2l3.sh) is the same flag set we use; it additionally
set SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR and
SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB=1 (the same >2 GiB pin workaround, so that
host hit the bug too).  In 1.6.5 the L3 tier cannot be attached on .14:

| config | result |
|---|---|
| hicache, cache mode, ratio 1, no L3 | boots, hicache_attached=True, 524K OK |
| hicache, cache mode + ile L3 | dies in uild_deepseek_v4_hicache_stack (kv_cache_builder.py:425) with CUDA error: invalid argument |
| hicache, uffer_only + ile L3 | same crash in the same call |

So the host tier alone is fine and the storage backend is the trigger.  The
traceback ends at the stack build (the CUDA error is asynchronous), so the next
step is to instrument that call and, in parallel, try the L3 tier on .15 where
there is no Engram host-table pin contention.
### L3 attach: root cause (fixed) and the next stall

CUDA_LAUNCH_BLOCKING=1 turned the asynchronous error into a real traceback:

`
cache_controller.py:425  _create_sync_groups
  -> parallel_state.create_custom_parallel_group
    -> torch.distributed.all_gather_object        # default (NCCL) group
torch.AcceleratorError: CUDA error: invalid argument
`

The storage backend builds gloo sync groups, but the rank-config gather in
create_custom_parallel_group ran on the *default* group, which is NCCL and
needs a CUDA tensor for object collectives.  It now uses the world group's CPU
collective (get_world_group().all_gather_object), the same one
GroupCoordinator.all_gather_object uses elsewhere.  With that fix the attach
no longer raises: the 'file' backend initialises and the run proceeds past the
sync groups.

The next blocker is different: with L3 attached the boot **stalls in CUDA graph
capture** (no new log line for >7 min while the schedulers sit at ~170% CPU,
443 GiB resident).  The next experiment is L3 with the decode graph disabled
(--disable-cuda-graph) to separate capture from the storage threads, then the
same 500K pair to measure whether the tier costs throughput.
### A/B: plain vs DSPARK on .14 (L2 config, growing prefixes)

One constant prefix seed, lengths sent in order so each prompt extends the
previous one (the shape of a real session), 32 generated tokens:

| prompt | plain prefill / decode | DSPARK prefill / decode | decode delta |
|---|---|---|---|
| 32 768 | 2033 / 22.64 | 2218 / 23.31 | +3.0% |
| 131 072 | 2295 / 22.59 | 2279 / 22.86 | +1.2% |
| 524 288 | 1697 / 22.74 | 1690 / 24.70 | +8.6% |
| 600 000 | 1619 / 22.06 | 1578 / 15.97 (re-run 16.78) | **-24% / -27%** |

Both sequences completed every length with no OOM (310 GiB used / 192 GiB
available at 600K), so the growing-prefix shape is safe on this host.

Verdict: **keep plain on .14**.  DSPARK is 1-9% faster up to ~524K and then
falls off a cliff at 600K (reproduced twice with different prefixes), which is
inside the 665535-token target.  Revisit only if the workload is known to stay
below ~520K, or after the draft-side pressure at 90% pool usage is understood.
## Recommended launch parameters (measured)

Chunk size is *not* free: past 8192 the prefill curve is flat while the VRAM peak
and the pool/context budget both suffer, and 4096 costs real long-context
prefill.  8192 is the sweet spot on this model; 1024 is fine at short context but
not worth the 131K penalty.

| knob | .14 (4x4090 24G, TP4, plain) | .15 (4090 48G + PRO 5000, TP2, DSPARK) |
|---|---|---|
| --chunked-prefill-size | **8192** | **8192** |
| LK_POOL_SLOTS | **40** (7.0 GiB slab) | **24** (8.4 GiB slab) |
| --max-total-tokens | 666112 | 666112 |
| --mem-fraction-static | **0.88** | **0.88** |
| --max-running-requests | **1** | **1** |
| speculation | off | DSPARK |

Measured with 64 generated tokens (single stream; the chat path under-reports on
.15 because the template streams reasoning tokens we do not count):

| prompt | .14 prefill / decode | .15 prefill / decode |
|---|---|---|
| 32 768 | 1988 t/s / 29.3 t/s | 1978 t/s / **47.3 t/s** |
| 131 072 | 2036 t/s / 30.0 t/s | 2044 t/s / 32.0 t/s |
| 524 288 | 1540 t/s / 29.6 t/s | 1581 t/s / 30.4 t/s |

Chunk sweep on .14 (pool 40, frac 0.88, 64 tokens):

| chunk | 32K prefill | 131K prefill | 524K prefill | 524K decode |
|---|---|---|---|---|
| 1024 | 1957 | 2050 | — | — |
| 4096 | 2033 | 2032 | **1323** | 28.2 |
| 8192 | 1988 | 2036 | **1540** | 29.6 |

Pool sizing is model-specific, not copyable.  DeepSeek-V4.1-Flash-1017 has
**384 routed experts, 6 active, 40 layers, hidden 5120, moe-intermediate 2304,
FP4 experts**; one resident slot costs ~0.18 GiB per rank at TP4 (40 slots =
7.0 GiB) and ~0.35 GiB at TP2 (24 slots = 8.4 GiB).  128 slots would need
~23 GiB per rank on .14 — impossible on 24 GiB cards, and the 128-slot reference
recipe comes from a much smaller NVFP4 MoE where a slot is an order of magnitude
cheaper.

Compared with the previous defaults (pool 12, frac 0.85, chunk 8192) the new
config trades ~9% of 524K prefill (1697 -> 1540) for **+30% decode**
(22.7 -> 29.6 t/s), which is the right side of the trade for an agent workload
whose prefill is already cached.

Dual concurrency stays off: two streams measured 11.7-12.2 t/s aggregate on .14
(versus 29.6 single) and collapse on .15, so the config keeps
--max-running-requests 1 and spends the spare VRAM on the expert pool instead.
### L3 (NVMe) verdict under the measured parameters

With --hicache-io-backend kernel --disable-cuda-graph the storage tier now
boots and serves on .14 (the direct backend stalled in graph capture and the
first prefill crashed).  What it does:

| probe | result |
|---|---|
| boot + first request | OK, hicache_attached=True, 32K prefill 1899-2023 t/s |
| write path | hicache_backup_tokens_total = 32768 per rank; L3 dir grows to **12 363 files / 1.0 GB** after a 600K request |
| read path | #cached-token: 0 on every repeat, incl. a 32K prompt re-sent after a 600K eviction |
| decode cost | **4.0-4.2 t/s** vs 29.6 without L3 (64 tokens, 32K and 600K alike) |

So the tier writes correctly but never restores, and it costs 6-7x decode.  The
restore is blocked by the same SWA gate that keeps L2 from matching (a finished
request's SWA pages are not retained, so swa_hit stays 0 and the scheduler
charges no cached tokens); the decode collapse is the kernel-IO backup path
copying during generation.

Verdict: keep L3 off.  Revisit only after the SWA gate is fixed upstream and the
backup path can be moved off the decode critical path (write_back or a
background flush), then re-run this exact probe.
### The "600K DSPARK cliff" was a short-generation artefact

The earlier A/B measured 32-64 token outputs and saw DSPARK fall from ~24 t/s at
524K to 16 t/s at 600K.  Re-measured with 400-token outputs (same pool/frac):

| length | plain, 400 tok | DSPARK, 400 tok | DSPARK, 64 tok |
|---|---|---|---|
| 524 288 | **35.10 t/s** | 33.92 t/s | 22.5 t/s (600K only) |
| 600 000 | **34.88 t/s** | 32.81 t/s | 22.51 t/s |

Fitting the 64/400 pairs gives the decode startup cost: spec ~1.06 s per request
versus ~0.40 s for plain (draft warmup + first verify, larger at 600K).  Over a
64-token window that fixed cost is most of the measurement, which is what looked
like a cliff; at 400 tokens the steady-state rate is flat to within 4% from 524K
to 600K.

Two real findings came out of it:

* DSPARK does not pay at long context even in steady state: acceptance is high
  (accept len 5.92, rate 0.98 at 600K) but a spec step costs ~2.9x a plain step
  for ~2.9 tokens, i.e. -6% at 600K and -3.5% at 524K.  Plain stays the right
  choice on .14 for reasons other than a cliff.
* DSPARK + pool 40 at 524K OOM'd after ~200 decode tokens (314 MiB free): the
  verify buffers need VRAM headroom, so a 40-slot pool is not safe with
  speculation at long context.

Measurement rule from now on: decode numbers use >=400 generated tokens; short
outputs report startup, not throughput.
### Measuring the expert hit rate (coding workload)

LK_POOL_LOG=100 makes lk_moe emit a window line per 100 decode steps:

`
[pool] dstep=400 win hit=58.7% nr=0.1% psl=0.00 link=6.6GB/s gather=5us copyiss=667us que=1 ...
       cum hit=57.1% h=54802 m=41198 p=6853 B=32.2GB resident=6716 ... oom=0 stalls=0
`

h/m are cumulative pool hits/misses in routed expert requests, win hit is
the windowed rate, esident the number of resident expert slots (40 slots x 40
layers x 4 ranks = 6400 plus shared experts).  The number that matters for a
workload is the **delta across the run**, not the since-boot figure.

launch/lan_14_15/hitrate.sh does exactly that around a continuous coding
session (scenario_agent.py, one repo, growing conversation, 400 tokens per
turn so the JIT/setup cost amortises).  Measured on .14, pool 40:

| probe | value |
|---|---|
| session | 5 turns, 24K code context, 400 tokens/turn |
| expert requests | 240 000 (h+m delta) |
| **hit rate** | **50.4 %** (121 045 hits) |
| resident slots | ~6.4-6.7 k (pool full) |
| pool oom / stalls | 0 / 0 |
| per-turn decode | 21.9 -> 25.2 t/s (context growing) |

So roughly half of the routed expert work in a same-domain coding session runs
on the GPU-resident half, which is what the +30 % decode over the 12-slot
baseline reflects.  A different-domain session measured with the same script
gives the contrast.

### Why >2 GiB host pins fail and leak: kernel, not the 595 driver

Both hosts run **Ubuntu 24.04.2, kernel 6.11.0-17-generic, driver 595.58.03**.
The failure signature we hit (cudaHostRegister returns EINVAL above ~2 GiB,
dmesg: Cannot map memory with base addr ... and size of ... pages, and the
pinned pages are never released) is a known Linux **6.11 mm/gup regression** in
the FOLL_LONGTERM path, fixed by upstream commit
94efde1d1539 ("mm/gup: avoid an unnecessary allocation call for FOLL_LONGTERM
cases").  Ubuntu tracks it as
[LP #2110319](https://bugs.launchpad.net/ubuntu/+source/linux-nvidia/+bug/2110319)
and released the fix in linux-nvidia-6.11 6.11.0-1010.10; the same series
fixed it in the generic 6.11 around 6.11.0-26.  Our 6.11.0-17 predates
both, which is why the pin path behaves as it does.

The same signature is reported against sglang:
[issue #7666](https://github.com/sgl-project/sglang/issues/7666) (cannot pin
>4 GiB, workaround = split the allocation) and
[PR #26301](https://github.com/sgl-project/sglang/pull/26301) (the exact
Cannot map memory dmesg line, fix = raise instead of silently ignoring).  The
NVIDIA forum thread on
[failed cuMemHostRegister >2 GiB leaking pinned memory](https://forums.developer.nvidia.com/t/bug-report-failed-cumemhostregister-2-gib-permanently-leaks-pinned-host-memory/378290)
anchors it to kernel 6.13+.

Recommendation, in order:

1. **Upgrade the kernel** to a patched 6.11 (linux-image-6.11.0-26-generic or
   linux-nvidia-6.11 6.11.0-1010.10) or to 6.13+.  The hosts are offline, so
   transfer the matching linux-image/modules/headers debs and reboot.  This
   removes both the 2 GiB limit and the leak, i.e. the chunking workaround
   becomes belt-and-braces rather than load-bearing.
2. If the kernel must stay at 6.11.0-17, keep our 1 GiB granularity-aligned
   pinning and treat a failed pin as fatal, then **reboot after any crash** -
   the leaked pages (we saw 335-460 GiB held) are not recoverable otherwise.
3. Driver 595.58.03 has separate reported problems (Ubuntu
   [LP #2147467](https://bugs.launchpad.net/ubuntu/+source/nvidia-graphics-drivers-595/+bug/2147467),
   workaround = 590), but those are graphics-side; swapping the driver alone
   does **not** fix the pin issue.  If a driver move is wanted anyway, 590 is the
   known-good fallback and 595.112+ is the newer 595 branch.
### Why .15's extra memory channels do not scale decode

Topology (lscpu / numactl):

| | .14 | .15 |
|---|---|---|
| CPU | 1 x Xeon Gold 6530, 32C/64T | 2 x Xeon Gold 6430, 2 x 32C, 128T |
| NUMA nodes | 2 (SNC2) | 8 (NPS4 per socket) |
| DRAM | 512 GiB, ~8 channels | 1024 GiB, ~16 channels |
| LK_THREADS | 16 | 48 |

Re-measured with 400-token outputs (the earlier "equal decode" was a 64-token
artefact, see above):

| prompt | .14 plain | .15 DSPARK | .15 / .14 |
|---|---|---|---|
| 32 768 | 35.10 t/s | **52.31 t/s** | 1.49x |
| 131 072 | ~35 t/s | **45.72 t/s** | ~1.3x |
| 524 288 | 34.88 t/s | **45.01 t/s** | 1.29x |

So .15 *is* faster - by 30-50%, not the ~2x the channel count would suggest.
The NUMA hypothesis was tested directly: LK_NUMA_INTERLEAVE=0 versus 1 on
.15 (weights spread over 8 nodes either way, confirmed with 
umastat -p):

| .15 variant | 32K decode | 131K decode |
|---|---|---|
| LK_NUMA_INTERLEAVE=1 | 52.31 | 45.72 |
| LK_NUMA_INTERLEAVE=0 | 54.13 | 45.52 |

Interleaving makes no measurable difference, so decode in this stack is **not
DRAM-bandwidth bound**: the per-token traffic is ~1.05 GiB/rank at TP4 and
~2.1 GiB/rank at TP2, i.e. ~150-200 GB/s at these rates, well under either
host's capacity (
umastat and a naive copy probe show .15 itself can drop to
113 GB/s when placement is bad, while .14 reaches 184 GB/s - and yet decode is
faster on .15).  The 30-50% gap instead tracks the per-rank CPU work
(48 threads per rank on .15 versus 16 on .14) plus the common GPU-side
attention cost, which is why the two hosts converge at long context.

Next test to pin this down: sweep LK_THREADS (16 -> 32 on .14, 48 -> 96 / 24
on .15) with 400-token outputs; if decode follows the thread count and not the
channels, the per-rank CPU path is the limit and the tuning target is threads,
not bandwidth.
### Decode is latency-bound, not bandwidth- or CPU-throughput-bound

Sampling the scheduler's utime+stime across an 8 s decode window (900-token
output, 32K context):

| | .14 | .15 |
|---|---|---|
| decode | 38.91 t/s | 56.44 t/s (1.45x) |
| CPU in the decode window | 1066 % of one core (10.7 cores) | 3464 % (34.6 cores) |
| cores available | 64 | 128 |
| threads in the process | 135 | 221 |
| per-rank expert slice per token | ~1.05 GiB (TP4) | ~2.1 GiB (TP2) |

Neither host is close to CPU saturation, and .15 spends 3.2x the CPU for 1.45x
the decode: at TP2 each rank streams twice the expert bytes, and DSPARK's verify
rows (7 per step) add CPU MoE work for ~3 accepted tokens.  Combined with the
interleave experiment (no effect) this pins the bottleneck to the **per-step
latency** of the CPU gather + GPU gemm/attention chain, not to DRAM bandwidth or
core count.  That is why 2x the channels gives 1.3-1.5x decode and why adding
LK_THREADS will not help - the threads are idle.

The lever that should work is amortising the per-step latency with more work per
step: either longer batches (concurrency) or a larger pool share.  The earlier
dual-stream test was run with 48-token outputs and the pool staging on, so it
measured startup, not steady state; re-test with 400-token outputs before
concluding that concurrency hurts.
### Kernel 6.14.0-37 upgrade (2026-10-25)

Both hosts moved from 6.11.0-17-generic to 6.14.0-37-generic, which carries
the mm/gup FOLL_LONGTERM fixes (94efde1d1539 and friends, mainline v6.13) that
6.11 never received.  The pin workarounds in this branch (1 GiB chunks, adaptive
halving) were compensating for that kernel bug; on 6.14 a **single 8 GiB pinned
allocation succeeds** on both hosts (8 GiB was beyond the 6.11 ceiling of ~2 GiB)
and dmesg shows no Cannot map memory.  The chunks stay as harmless
belt-and-braces for hosts still on 6.11, but they are no longer load-bearing.

Upgrade path per host: install the five 6.14.0-37 debs → reboot → reinstall the
NVIDIA runfile with --kernel-module-type=open (both hosts use the open modules;
PRO 5000 Blackwell requires it) → reboot.  6.11 is kept installed and held as a
rollback.  Post-upgrade numbers: .14 32K/400-token decode 38.7 t/s (was 35.1),
.15 50.2 t/s (was 52.3, within noise), HiCache still attaches.
### Does concurrency amortise the per-step latency?  Partly, at two streams

Measured on .14 with --max-running-requests 4, /generate path, 400 tokens per
stream, 1K context so prefill is negligible:

| streams | per-stream | aggregate | vs single |
|---|---|---|---|
| 1 | 35.96 t/s | 35.96 t/s | - |
| 2 | 24.70 / 24.69 t/s | **49.4 t/s** | **+37%** |
| 4 | 6.02 t/s each | 24.1 t/s | -33% |

So two streams do buy aggregate throughput (+37%) on a host that was using only
11 of 64 cores for one stream, but it is far from linear: the per-step critical
path (CPU gather latency + GPU gemm/attention + pool staging) is shared, and at
four streams the 16 LK_THREADS per rank and the pool's staging ring become the
contention point, so aggregate falls below the single-stream rate.

Implication: --max-running-requests 2 is the defensible setting for .14 (it
raises aggregate without hurting the single-stream latency that matters for an
interactive agent); four streams need more LK_THREADS before they make sense.
Next: repeat the two-stream point on .15 (48 threads/rank) and try .14 with
LK_THREADS=32 at four streams.
### Concurrency on .15 (and the cross-host picture)

Same probe (1K context, 400 tokens/stream, --max-running-requests 4):

| streams | .14 sum of stream rates | .15 sum of stream rates |
|---|---|---|
| 1 | 36.0 t/s | 54.5 t/s |
| 2 | **49.4 t/s** | 64.2 t/s |
| 4 | 24.1 t/s (collapses) | **65.7 t/s** |

Reading:

* concurrency does amortise the per-step latency, but only by 20-40%: the
  critical path per step is shared, so the gain is sub-linear;
* .14 peaks at two streams and collapses at four (4 ranks x 16 threads already
  covers all 64 vCPUs, so four concurrent decode rows thrash the CPU side);
* .15 keeps improving to four streams (2 ranks x 48 threads on 128 vCPUs leaves
  headroom), topping out at ~66 t/s aggregate.

Defaults therefore become --max-running-requests 2 on .14 and 4 on .15
(DSV41_MAX_REQ overrides).  For a single interactive session keep 1: the
per-stream rate is highest there.
### Single-call pinning restored (kernel 6.14) and the growing-context sweep

With the 6.14 kernel a single cudaHostRegister above 2 GiB works (measured:
1/2/4/8 GiB all c=0 on both hosts), so both call sites now try the **whole
range in one call first** and only fall back to the granular path if the driver
rejects it - i.e. on a fixed host the code path is the framework's original one:

* .14 HiCache host pools: 0 splits in the boot log (previously the 1 GiB clamp
  always split);
* .15 Engram host table: two 48 GiB single-call pins, each layer 48 344 MiB
  resident, **100 % in huge pages**.

Growing-context sweep (one shared prefix, lengths ascending, 400-token outputs,
--max-running-requests 2):

| prompt | .14 prefill / decode | .14 GPU used | .15 prefill / decode | .15 GPU used |
|---|---|---|---|---|
| 65 536 | 2022 / 37.21 | 20.0 GiB | 1955 / 55.27 | 29.6 GiB |
| 131 072 | 3775* / 37.49 | 20.4 | 3740* / 44.93 | 29.6 |
| 262 144 | 3389* / 36.38 | 21.5 | 3413* / 42.46 | 30.9 |
| 524 288 | 2639* / 36.82 | 21.8 | 2698* / 44.67 | 30.9 |
| 600 000 | 7518* / 36.39 | 22.4 | 8660* / 44.52 | 34.3 |
| 660 000 | 9351* / 33.79 | **23.5** | 11719* / 47.79 | 37.4 |

* the prompt extends the previous one, so the prefill rate is measured over a
mostly-cached prefix; it is not a raw prefill number.  No OOM at any length on
either host.

VRAM allocation verdict:

* .14 is at its ceiling: 660K leaves ~1 GiB free on the tight card, so this is
  the maximum context at 400 generated tokens.  A 665 535-token prompt with
  max_new 400 is rejected (no prefill batch appears in the log), so the full
  context needs max_new <= ~64 or a smaller pool (32).
* .15 still has ~11.6 GiB free at 660K, so it can afford a larger pool (24 ->
  32-40) or a higher host ratio without touching the context limit.
### L3 (NVMe) fixed: the storage hash extension never built

The tier was never fundamentally broken - its **key hashing** was.  The trace
(the extended SGLANG_RADIX_DEBUG line) showed the tree matching and the
storage lookup being issued, but always missing:

`
RADIXDBG match full_hit=23808 swa_host_hit=0 host_hit=0
        pf(attempts=4 issued=4 l3_demand=4 l3_miss_tok=95232)
`

and the log carried the real error:

`
HiCache storage query failed for request ...
  RuntimeError: Error building extension 'hicache_hash_cpp':
  hash_binding.cpp:5:10: fatal error: openssl/sha.h: No such file or directory
`

The extension JIT-compiles -lcrypto with #include <openssl/sha.h>; the conda
env ships both, but neither is on the compiler's default search path, so the
build failed, every L3 lookup raised, and the lookup was recorded as a miss.
Writes still landed files on disk, which is why it looked like "writes work,
reads do not".  Fix: export the env's include/lib dirs in the launch scripts:

`ash
export CPATH="\/include:\C:\Users\OrangeCat/anaconda3/include\"
export LIBRARY_PATH="\/lib:\C:\Users\OrangeCat/anaconda3/lib\"
`

Verification on .14 (L3 = cache mode + ile backend + direct IO + CUDA graphs,
pool 40, frac 0.88, maxreq 2):

| probe | before | after |
|---|---|---|
| cold-boot first 24K request | cached 0, ttft 12.3 s | **cached 23808, ttft 2.6-3.9 s** |
| storage lookup | l3_miss_tok = 4 x 23808 | **l3_miss_tok = 0** |
| decode (400 tok, 24K) | 34.3 t/s (L2-only baseline 36) | **34.1-40.0 t/s** |
| growing 65K -> 660K | - | all served, decode 34.2-36.0, no OOM, GPU <= 23 GiB |

So L3 now restores a prefix written by an earlier process (cold L1/L2), at
baseline decode speed.  The earlier "decode collapses to 4 t/s" observation was
a short-output artefact on top of the broken-hash path, not a cost of the tier.
### .15 expert pool: 40 slots beat 24 by 11-21%

.15 had ~11.6 GiB spare at 660K, so the pool was raised from 24 to 40 slots
(measured with 400-token outputs, maxreq 2):

| prompt | pool 24 | pool 40 | delta |
|---|---|---|---|
| 32 768 | 52.31 t/s | **58.30 t/s** | +11% |
| 131 072 | 45.72 t/s | **55.20 t/s** | +21% |
| 524 288 | 45.01 t/s | **54.01 t/s** | +20% |
| 660 000 | 47.79 t/s | **52.85 t/s** | +11% |

At 660K the cards sit at 44.3 / 42.9 GiB of 49 GiB (~5 GiB free), so 40 slots
still leaves room; the pool reports cum hit=65.3% (h=183676 m=97495,
resident=5874) with oom=0 (850 pool stalls across the four runs).  Default for
.15 is now LK_POOL_SLOTS=40 (DSV41_POOL_SLOTS overrides); 48 would add
~2 GiB and is not validated.
### L3 storage cap (sglang's own) and the final stability matrix

The ile backend has its own LRU cap and we already pass it:

`
--hicache-storage-backend-extra-config '{"max_size": "20G"}'
`

Boot log on .14 (rank 0 owns eviction, other ranks skip LRU bookkeeping):

`
HiCacheFile eviction enabled: cap=20000000000 B, watermark=0.90, min_free=0 B,
              existing=1189402880 B (13382 entries)
`

DSV41_KV_NVME_SIZE sets the cap (SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE is the
equivalent env), clamped to 90% of the filesystem and to the filesystem capacity.
The dir held ~1.2 GB when this was measured, well under the 20 GB cap.

Growing-length runs (400-token outputs, maxreq 2, no OOM anywhere):

| prompt | .14 (L3 on, pool 40) | .15 (pool 40, DSPARK) |
|---|---|---|
| 32 768 | 34.1 t/s | 58.3 t/s |
| 65 536 | 34.2 | 55.5 |
| 131 072 | 36.0 | 55.2 |
| 262 144 | 35.3 | 54.3 |
| 524 288 | 35.6 | 53.3 |
| 660 000 | 34.4 | 53.0 |
| GPU at 660K | 22.0 GiB | 44.4 / 42.9 GiB |

Two concurrent streams (--max-running-requests 2), same 400-token outputs:

| prompt | .14 sum of streams | .15 sum of streams |
|---|---|---|
| 32 768 | 38.9 t/s (21.8 + 17.1) | 50.8 t/s (13.8 + 37.1) |
| 131 072 | 30.1 t/s (8.2 + 21.8) | - |
| 524 288 | **70.2 t/s (35.1 + 35.1)** | **108.5 t/s (54.4 + 54.1)** |

So two streams roughly double aggregate throughput at long context (the per-step
cost is dominated by attention over 500K+ tokens, which the second row rides
along for free) and are neutral-to-negative at 32K.  No OOM in any of these runs.
### Dual streams with growing lengths, and what different domains cost

Two concurrent streams (--max-running-requests 2), 400-token outputs, KV pool
666 112 tokens on both hosts.  same sends identical prompts (the prefix and
its KV pages are shared), diff sends different prefixes (nothing shared).

.14 (L3 on, pool 40):

| length | same: peak VRAM | same: result | diff: peak VRAM | diff: result |
|---|---|---|---|---|
| 65 536 | 22.96 GiB | ok, 21.6+11.4 t/s | - | - |
| 131 072 | 22.96 | ok, 8.2+21.5 | 22.96 | ok, 22.3+3.0 |
| 262 144 | 22.96 | ok, 2.8+21.7 | 22.96 | ok, 21.0+2.2 |
| 524 288 | 22.96 | ok, **34.8 + 34.7** | 22.98 | ok, 35.0 + 37.2 (2nd ttft 705 s) |

.15 (pool 40, DSPARK): diff 524288 peak **44.36 GiB**, both streams complete
(53.4 + 52.7 t/s, second ttft 736 s); same 524288 was 108.5 t/s aggregate.

Reading:

* **Peak VRAM does not rise with different domains.**  The KV pool is
  pre-allocated at max-total-tokens, so the peak is flat (~23 GiB on .14,
  ~44 GiB on .15) whatever the prefix overlap or context length.  There is no
  OOM risk from a mismatched domain pair on either host.
* What a mismatch *does* cost is **queueing**: two 524K contexts need ~1.05M
  tokens against a 666K pool, so the scheduler runs them one after the other
  (second TTFT 705-736 s) instead of overwriting the first.
* With shared prefixes the two 524K requests fit together and run concurrently
  (69-108 t/s aggregate at 524K), which is the case the agent workload hits.
* Short prompts do not benefit from a second stream; long prompts do.
### Agentic shape: 256K context with long outputs, twice

Real agent sessions need >= 256K context and outputs far beyond 400 tokens.
Two concurrent streams at 262 144 tokens with an 8 192-token output budget
(different prefixes, so the two contexts do not share pages; the pair needs
~2 x 270K tokens against the 666K pool, which fits):

| host | stream A | stream B | aggregate | peak VRAM | OOM |
|---|---|---|---|---|---|
| .14 (L3, pool 40) | 17.4 t/s | 26.2 t/s | **43.5 t/s** | 22.98 GiB | no |
| .15 (pool 40, DSPARK) | 39.7 t/s | 21.5 t/s | **61.1 t/s** | 44.36 GiB | no |

Both streams generated the full 8192 tokens in 546-624 s wall.  Against the
single-stream rates (35 t/s on .14, 53 t/s on .15) the pair adds +24% and +15%,
so long outputs keep a modest concurrency benefit - much less than the 2x seen
with 400-token outputs at 524K, because decoding 8K tokens per stream keeps both
rows busy and the CPU pool contended.

The guidance for an agentic deployment is therefore: --max-total-tokens must
cover **the sum of the concurrent contexts plus their output budgets**
(2 x (256K + 8K) ~ 540K tokens here), and --max-new in the client should be
sized to the real answer length; 400 tokens underestimates both the KV need and
the measured throughput shape.
Raising the output budget further (same 256K context, different prefixes):

| host | max_new | stream A | stream B | aggregate | peak VRAM | OOM |
|---|---|---|---|---|---|---|
| .14 | 8 192 | 17.4 | 26.2 | 43.5 t/s | 22.98 GiB | no |
| .14 | **16 384** | 27.0 | 21.3 | **48.2 t/s** | 22.98 GiB | no |
| .15 | 8 192 | 39.7 | 21.5 | 61.1 t/s | 44.36 GiB | no |

(.15 at 16 384 was not run.)  Both 16K streams completed in 921 s; the aggregate
creeps up with the output budget (43.5 -> 48.2 on .14) because the fixed
per-step costs spread over more generated tokens, and the VRAM peak does not move
at all.

Sizing rule for an agentic deployment:

`
--max-total-tokens >= sum over concurrent requests of (context + max_new) + slack
`

For two 256K sessions with 16K answers that is 2 x 278K = 556K, inside the 666K
pool; two 524K sessions cannot fit (1.05M) and the scheduler will queue them.
--max-new 400 is not a realistic probe of this shape: it understates both the
KV budget a session needs and the throughput the deployment will see.
### L2/L3 verified on .14 (fresh boot) and the one caveat

Cold boot, L1/L2 empty, L3 on; the **first** request is a 24 000-token prompt:

`
RADIXDBG match full_hit=23808 swa_host_hit=256 host_hit=23808
        pf(attempts=1 issued=1 l3_demand=1 l3_miss_tok=0)
Prefill batch #new-token: 192, #cached-token: 23808
ttft 3.32 s (a full 24K prefill is ~12.3 s)
`

* **L3 works**: l3_miss_tok=0 and the prefix arrives from disk; it is staged
  through the host pool (host_hit=23808, swa_host_hit=256).
* **L2 works**: hicache_host_used_tokens climbs to **659 968 / 666 112**,
  hicache_backup_tokens_total reaches 3.99M (kv) + 5 632 (swa) per rank, and
  the L3 directory grows 64 634 -> 77 526 files / **6.4 GB** under the 20 GB cap.
* **Caveat (host ratio 1)**: the host pool is sized at ratio 1, i.e. one device
  pool's worth.  A 660K different-domain request fills it completely
  (659 968 tokens) and the 24K prefix is evicted from both device and host; the
  next 24K request then reports ull_hit=0 and re-prefills in 12.3 s, after
  which it hits from device again.  So L3 restore is reliable on a cold tree but
  did not resurrect that prefix immediately after a full-pool churn - if
  cross-session retention matters, raise DSV41_HICACHE_RATIO (host > device
  pool) so evicted prefixes survive the churn.
* .15 runs with hicache_attached=False (no L2/L3): enable with
  DSV41_HICACHE=1 (plus DSV41_KV_NVME=... for L3) if wanted.
### Host-tier sizing: ratio 4 on .14, ratio 2 on .15 (churn test)

With ratio 1 the host pool holds one device pool's worth, so a 660K request
displaced everything and the 24K prefix missed afterwards.  Re-running the same
churn (warm 24K -> 660K different domain -> 24K again):

| host | host ratio | host pool (used/total tokens) | 24K after churn | 24K ttft |
|---|---|---|---|---|
| .14 | 1 | 659 968 / 666 112 | ull_hit=0, re-prefill | 12.3 s |
| .14 | **4** | 684 032 / 2 664 448 | ull_hit=23808 host_hit=23808 swa_host_hit=256 l3_miss_tok=0 | **2.49 s** |
| .15 | **2** | n/a (metrics not sampled) | hit | **1.55 s** (cold prefill was 18.4 s) |

So ratio 4 (≈4 device pools) is what keeps cross-session prefixes alive through a
full-pool churn on .14, and ratio 2 already does it on .15.  L3 stayed at 6.4 GB.
### Host-tier ratio: how much memory, and why .15 needs less

Measured host-pool footprint (the pools are pinned, so this is real RAM):

| host | ratio | host pool pinned | host_total_tokens | RAM available |
|---|---|---|---|---|
| .14 (TP4) | 4 | **28.4 GB** | 2.66 M | **174 GB** of 503 |
| .15 (TP2) | 2 | **7.4 GB** | ~1.33 M | 504 GB of 1007 |

So ratio 4 on .14 is not a memory risk: it costs 28 GB out of a 503 GB host
that already has 174 GB free (and .14 keeps Engram on NVMe, so no host table).
The per-ratio cost differs because every TP rank pins its own pool: .14 has four
ranks (~7.1 GB per ratio), .15 has two (~3.7 GB per ratio).

Why .15 is fine at ratio 2: the retention requirement is "host must hold the
churn plus the prefix we want to keep".  A 660K request is one device pool, so
ratio 2 covers it - verified: 24K after a 660K churn hit in **1.55 s** on .15
(18.4 s when cold).  Ratio 4 there would cost ~15 GB (still cheap for 1 TB) and
only adds headroom for heavier multi-session churn.

### .15 expert pool: 48 does not help

| .15 pool | 32K decode | 660K decode | VRAM at 660K |
|---|---|---|---|
| 40 (default) | 58.30 t/s | 52.85 t/s | 44.4 / 42.9 GiB |
| **48** | 57.45 t/s | 52.34 t/s | **46.2 / 45.3 GiB** |

No gain (slightly worse, within noise) and ~2 GiB less headroom (~3 GiB left at
660K), so 40 stays the default and 48 is not recommended; 56 would likely OOM.
### .15 prefill: the 4090 rank is the straggler, the PRO 5000 waits

Per-GPU utilisation sampled at 0.5 s during a full 32K prefill (window bounded by
the request send time), with the phase split at ttft:

| host | card | prefill idle samples | prefill median | prefill mean | decode mean |
|---|---|---|---|---|---|
| .15 | GPU0 RTX 4090 (SM89) | **9/54 (17 %)** | 99 % | 75 % | 40 % |
| .15 | GPU1 RTX PRO 5000 (SM120) | **20/54 (37 %)** | 94 % | 55 % | 40 % |
| .14 | all four 4090 | 11-14/39 (28-36 %) | 91-95 % | 55-57 % | 53-59 % |

Read: this is **not** a CPU/GPU bubble - the median is 94-99 % on both cards, so
the GPU path is saturated while it runs.  It is a **TP imbalance**: the 4090 rank
is the straggler and the PRO 5000 blocks on the TP collectives.  The log names the
cause - FlashInfer has no MXFP4 kernels on SM8x; using Marlin, i.e. the 4090
runs the MoE prefill through Marlin MXFP4 while the SM120 card uses the FlashInfer
CUTLASS MXFP8 x MXFP4 path - so the two ranks need different time per layer and
sync every layer.

Consequence: .15's 32K prefill is 1315 t/s versus .14's 1897 t/s, even though
.15 has the faster CPU and more memory channels; TP4 on .14 simply has more
parallelism and no straggler.  Decode is unaffected by this (both cards sit at
~40 % during decode, matching the earlier latency-bound finding).

Mitigations are limited without a fourth GPU on .15: keep the prompt hot (KV reuse
already removes most prefill), or accept the ~25 s cold prefill at 32K.