# Eviction scoring and selection in hipinfer

Status: **run on a GPU** (RTX 2080 Ti and RTX A6000, 2026-10-18, merged tree). The first GPU run
found a bug in `hi_evict`, fixed in `32c801d` (below); every check passes on the fixed build. This is
the HIP counterpart of the Triton kernels in SubCache (`subcache/kernels.py`; literature note in
SubCache `docs/eviction-kernels.md`).

| check | where | result |
|---|---|---|
| the default decode path is unchanged | laptop, SASS (sm_86) of `main` vs this branch | the plain decode-attention kernels and the fused QKV/RoPE GEMV are **instruction-identical** to `main`. Changed: the 4 prefill attention kernels (one null-checked `lse` store after the main loop) and `rope_cache_kernel` (null-checked shift) |
| selection logic: radix select, the top-2 thread decomposition, the incremental rescan | laptop, `tests/test_eviction_cpu.py` (emulates the kernels' algorithms in PyTorch) | all pass, including inputs full of exact ties; three deliberate bugs (tie direction, merge rule, rescan rule) each make it fail |
| the WFL reference used by the GPU tests vs SubCache's `policies.wfl_prune` | laptop, same file, `SUBCACHE_DIR` set | identical kept sets in 5 of 5 cases |
| kernels vs PyTorch (outputs, LSE, logits, scores, decisions) | 2080 Ti and A6000, `tests/test_eviction.py`, default and portable kernels | **475 checks pass** on each. Tracked attention outputs are **bitwise** equal to the plain kernels'. Scores against a float64 reference: decode within 9.9e-7 relative, prefill within 1.7e-6. WFL kept sets equal the reference in all 10 cases; the incremental rescan equals a full rescan |
| engine: tracking changes no output; eviction decisions; decode after eviction | 2080 Ti (TinyLlama, Llama-3.2-1B / 3B, Qwen2.5-1.5B) and A6000 (Llama-3.2-1B, Qwen2.5-1.5B), `tests/test_eviction_engine.py` | **all 38 checks pass** per model after the fix: tokens and logits with tracking on are bitwise equal to off, the kept slots equal PyTorch's from the engine's own state, and decode after an eviction adds exactly G of score mass per step |
| overheads | 2080 Ti, `scripts/bench_eviction.py`, no throttling (`bench_gpu_log.csv`) | below |
| AMD | — | compiles (gfx90a, gfx942, gfx1100) as part of the merged tree; never run |

**The bug the GPU run found.** `hi_evict` moved the kept slots' scores to `[0, capacity)` but left
the evicted slots' scores in `[capacity, n)`. Decode after an eviction writes its new key at cache
index `position - shift`, inside that range, and `decode_score_kernel` adds the new key's mass to
whatever the slot already held. So each step after an eviction appeared to add 5.4–10 G of mass
instead of G. The CPU emulation could not see it: it never ran decode after a compaction. The fix
clears both score arrays' tails after the move.

## Overheads (RTX 2080 Ti)

From `results/2080ti/eviction/bench_eviction.json`, on the fixed build, with no other process on
the GPU and no thermal throttling (median SM clock 1,770 MHz, peak 83 °C):

- **Decode attention with tracking:** +5–32 % on the attention kernel itself, across the shapes of
  four models, 1k–32k keys and batch 1 and 8 (Llama-3.2-1B, G = 8, D = 64: +20–31 %; Llama-3.2-3B
  and Llama-3.1-8B, D = 128: +14–26 %; Qwen2.5-1.5B: +5–32 %). Attention is a small part of a decode step: **a whole Llama-3.2-1B decode
  step at a 4.1k context goes from 4.96 to 5.05 ms (+1.7 %)**.
- **Prefill score pass:** the second, key-parallel pass costs 1.0–1.8× the FlashAttention kernel's
  own time at 2–4k tokens and 1.4–4.5× at 512 (few kv heads launch few blocks, as noted below).
  Storing the LSE in pass 1 adds at most 4 %.
- **Selection over every (layer, kv head):** H2O top-k takes 0.03–0.10 ms at 1k–4k slots. WFL prune
  is the expensive policy: 9–40 ms to evict 64 slots from 1,024 (8 rounds) and 0.1–0.6 s at 4,096;
  a full prune from 4,096 to 512 (448 rounds) takes 0.98–4.2 s. Where the PyTorch matrix version
  of the same algorithm fit in memory, the kernels are 6.4–25× faster (Llama-3.2-1B and
  Qwen2.5-1.5B at 1,024 slots).
- **In the engine** (Llama-3.2-1B, 4,096 → 1,024 slots): H2O selection 0.10 ms plus 0.88 ms to
  compact keys, values and scores; WFL 885 ms; WFL limited to a pool of 2 × m and 8 rounds 63 ms.
  The decode step after the eviction is 4.85 ms.

## What it does

SubCache's eviction policies need two things the FlashAttention-style kernels don't give: H2O
scores (the attention weight every cached key has received, summed over queries and over the query
heads that share its kv head), and, for WFL, every slot's best and second-best kept neighbour under
a key-and-value similarity. In SubCache, getting H2O scores meant running eager matmul attention, and
WFL selection took 71 % of decode time. This adds:

1. **Scores inside the attention kernels.** Turned on per engine with `hi_set_tracking` (off by
   default); the plain kernels run otherwise.
2. **Selection kernels.** Top-k (H2O, window score) and WFL prune with the similarity computed on
   the fly (no n x n matrix) and an incremental rescan.
3. **An eviction path in the engine** (`hi_evict`). hipinfer had none. It compacts the kept keys,
   values and scores to the front of the slot's cache; decode continues at the true positions.

### Decode: logits stored, weights after the reduction

The split-K decode kernels (`decode_attn_kernel`, `decode_attn_mma_kernel`) gain a template flag
`TRACK`. With it, each split block also writes the log2-domain logit of every key for each query
head of its group to a scratch buffer `lg [B][Hq][max_seq]` (fp32), and the block that finishes the
split reduction writes each head's log-sum-exp `lse [B][Hq]`. A second, small kernel
(`decode_score_kernel`, grid over keys) then computes `p = exp2(lg - lse)`, sums it over the G query
heads, and adds it to `score [B][Hkv][max_seq]`. Optional window: a decode query marked as window
row r also writes its softmax rows (fp16) and adds its mass to a window score. With `TRACK` off the
template instantiates the original code: the SASS is identical to `main`.

Per key and kv head, tracking adds `4G` bytes written (logits), `4G` read back and 8 bytes of score
read-modify-write, against `4D` bytes of K and V the attention reads anyway. That is +8 % traffic for
G=4, D=128 (Llama-3.2-3B), but +28 % for G=8, D=64 (Llama-3.2-1B), plus one extra launch per layer.
These are estimates from byte counts, not measurements.

### Prefill: two passes

As in MiniKV (arXiv 2411.18077, §3.2):

1. The FlashAttention pass (`flash_prefill_mma_kernel` / `flash_prefill_kernel`) also writes each
   row's log2-domain log-sum-exp (`lse [Hq][T]`) when given a buffer.
2. `prefill_score_kernel` runs over keys instead of queries: one block per (64 keys, kv head) keeps
   its K tile in shared memory and walks every query row that can see one of its keys (for every
   query head of the group), recomputes QKᵀ on WMMA, normalizes with the saved LSE, and sums down
   each column in registers. Each key belongs to exactly one block, so there are no atomics. Rows
   at or after `win_r0` are window rows: their mass also goes to the window score, and their
   softmax rows can be stored.

Pass 2 recomputes QKᵀ but not PV, so it costs roughly half the attention FLOPs plus one `exp2` per
visible (query, key) pair. MiniKV's Triton version of this design went from 0.118 ms
(FlashAttention alone) to 0.622 ms (their §4.6). Ours is not measured.

Chunked prefill works: each chunk adds its rows' mass to every key it sees, including earlier
chunks' keys. In the engine the window is the last `window` tokens of the prompt (SnapKV's
observation window; the question-window score).

### Selection

All selection kernels work per problem, one (layer, kv head) of one sequence, over a ground set of
n slots. All break ties toward the lower index.

- **`evict_select_kernel`**: the k largest (or smallest) values among eligible slots, by a 4-bit
  radix select (8 passes) over order-preserving uint32 keys of the fp32 values, then one ordered
  pass to take the right number of slots equal to the k-th key. One block per problem. H2O keeps
  the top m scores with it; WFL uses it to remove the k cheapest slots per round.
- **`wfl_prep_kernel`**: gathers the ground set, centres keys and values over it, and L2-normalizes
  each row (fp32; torch's 1e-12 floor), as SubCache's `slot_similarity` does. Also normalizes the
  weights `w = score / sum`.
- **`wfl_top2_kernel`**: for each listed row i, the best and second-best kept column under
  `sim(i, j) = max(cos_k, 0) * max(cos_v, 0)` (squared for `power` 2), with `sim(i, i) = 1`.
  Columns that are not kept are skipped, which matches `wfl_prune`'s -1 because kept columns are
  >= 0. 64 x 64 tiles; each thread owns 4 rows x 4 columns, computes both dot products in fp32
  from shared memory, keeps a running top-2 per row over its columns (visited in increasing order,
  strict `>`), and the 16 threads of a row merge with an xor butterfly. No n x n matrix is stored.
- **`wfl_round_kernel`**: one removal round of `policies.wfl_prune`:
  `loss[j] = mix * sum_{i: arg_i = j} w_i * max(best_i - max(sec_i, 0), 0) + (1 - mix) * w_j` over
  kept candidates, removes the k smallest, and lists the rows whose best or second-best column was
  removed. **Only those rows are rescanned** next round. This is exact: if neither of a row's two
  columns was removed, the maximum over the smaller kept set is the same pair, and with ties broken
  toward the lower index that includes the indices. On random data with near-duplicate keys, the
  CPU emulation rescanned 2–58 % of the rows a full rescan would (2 % at one removal per round, n=200).

The first round still computes all n² similarities per problem: `4 n² D` fp32 FLOPs, ≈2.1 GFLOP per
problem at n = 2048, D = 128. That is why `hi_evict` has SubCache's `pool` option, which limits the
ground set to the protected slots plus the `pool * m` candidates with the highest H2O scores.

The loss sums use float atomics, so their last bits depend on order. Decisions can then differ from
a sequential reference, but only where two candidates' losses are within rounding of each other.

### Engine path

`hi_set_tracking(e, on, window)` allocates `score` (and, with a window, `wscore`)
`[layers][max_batch][kv_heads][max_seq]` fp32, the decode logits scratch `[max_batch][heads][max_seq]`,
and a per-sequence `shift` (slots evicted so far), and drops captured decode graphs. While tracking:

- prefill with `p0 = 0` zeroes the sequence's scores and its shift; each chunk runs pass 1 with the
  LSE and then pass 2. Continuing a sequence with a prefill (`p0 > 0`) after it has evicted slots
  is refused.
- decode runs the `TRACK` kernels and the finish kernel per layer. The QKV epilogue (a separate
  instantiation, EPI 4) and `rope_cache_kernel` write the new key and value at cache index
  `position - shift`. RoPE still uses the true position, and attention covers `position - shift + 1`
  keys. Graphs stay valid across evictions, because they read `shift` from device memory.

`hi_evict(e, slot, capacity, policy, sink, recent, mix, step, rounds, pool, power, kept_out, ms_out)`
keeps the first `sink` and last `recent` slots and chooses the rest by policy 0 (H2O), 1 (WFL prune)
or 2 (window score). Then, per layer, it lists the kept slots in order, checks that every head kept
exactly `capacity`, and moves keys, values and scores to the front through a scratch buffer.
`Engine.evict` in `scripts/hipinfer.py` wraps it, with `set_tracking`, `scores`, `cache_read` and
`cache_len`.

Limits: one GPU (tensor-parallel engines refuse tracking); positions still run up to `max_seq` (the
RoPE table and token history are sized by it), so eviction shortens attention, not the context the
engine can address; the engine uses window rows only in prefill.

## Numerics

Logits and LSEs are in the log2 domain (scaled by `log2(e) / sqrt(D)`), as the attention kernels'
online softmax already is. QKᵀ is exact fp16 products with fp32 accumulation in every kernel, so
the scores should match a float64 reference to about 1e-6 relative; the GPU test asserts 1e-4
relative + 1e-6 absolute and prints the max abs and relative error. Pass 1 uses `mma.sync` and pass
2 uses WMMA. Their logits can differ in the last bits, so a row's weights sum to 1 only within
rounding; the tests check `sum(score) = G * queries`. The attention outputs with tracking on must
be **bitwise** equal to the plain kernels' outputs; the tests check this, at kernel level and as
engine tokens and logits.

## Running it

On the laptop (CPU only):

```bash
SUBCACHE_DIR=../SubCache python tests/test_eviction_cpu.py
```

On edmonds, from a checkout of this branch. Run `check` first, then pick a GPU with no processes. GPU
1 (the A6000) is refused, and other sessions planned GPU 0 (int8) and GPU 2 (speculative decoding):

```bash
scripts/eviction_gpu.sh check
```

```bash
GPU=3 scripts/eviction_gpu.sh start
```

```bash
scripts/eviction_gpu.sh status
```

```bash
scripts/eviction_gpu.sh fetch
```

It syncs to `/data/axd200066/hipinfer_evict` and runs under `nohup`, not tmux. It sets
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, runs `nvidia-smi` before every step, and skips a step if any other
process is on the GPU. It shows the newest thermal log from `/data/axd200066/gpu_thermal/` and the
clock-throttle reasons before and after the benchmarks. Steps: build, the CPU test, the existing
`tests/test_kernels.py` (regression), `tests/test_eviction.py` with the default and the portable
kernels, `tests/test_eviction_engine.py` on Llama-3.2-1B (D 64) and Qwen2.5-1.5B (D 128), and
`scripts/bench_eviction.py`. Logs and JSON go to `results/2080ti/eviction/`.

## Not done / open

- AMD: the code uses only portable paths there (scalar decode, WMMA prefill, ballot/popcount helpers
  for wave64). It compiles with hipcc 7.2.4 for gfx90a, gfx942 and gfx1100 but has never run.
- WFL keeps no state between evictions. A streaming caller pays the full first round each time;
  carrying best/second over and scanning only the new columns would remove that.
- `prefill_score_kernel` gives each block 64 keys of one kv head, so a short prompt with few kv heads
  launches few blocks (T = 512, 2 kv heads: 16 blocks). This is the 4.5× at 512 tokens above.
- No accuracy result: these checks show the kernels compute H2O and WFL exactly as specified, not
  how well either policy preserves a model's outputs at a given cache budget. That evaluation lives
  in SubCache.
