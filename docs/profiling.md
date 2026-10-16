# Where the time goes: profiling hipinfer

This document explains, from the committed result files only, where one decode step's time goes,
how close each kernel is to the hardware limit, and what limits prefill, batched decode and
tensor-parallel decode. Unless noted, the GPU is the RTX 2080 Ti (datasheet DRAM bandwidth
616 GB/s), the weights are fp16, and the data comes from `results/2080ti/<model>/bench.json`,
measured on the current build (commit 7370172).

## How the numbers were produced, and what wasn't used

| what | tool | detail |
|---|---|---|
| kernel microbenchmarks (`kernels` section) | CUDA events (`torch.cuda.Event`) | 10 calls captured in one CUDA graph, replayed 20 times; median per call ([scripts/bench.py:25](../scripts/bench.py:25)). Host launch overhead is excluded for both hipinfer and PyTorch. |
| batch-1 decode (`decode_b1`, `decode_ctx`) | HIP/CUDA events inside the engine (`time_ms`, [src/api.cu:858](../src/api.cu:858)) | 256 (or 128) graph-replayed decode steps after a prompt; best of 3 (or 2) |
| prefill (`prefill`) | CUDA events | median of 10 calls to `eng.prefill`, which includes the token upload and the final sync |
| batched decode (`decode_batched`) | engine events + **NVML** | 128 steps; board power from `nvmlDeviceGetPowerUsage` polled every 20 ms in a side thread, averaged ([bench.py:63](../scripts/bench.py:63)) |
| HF reference decode | `time.perf_counter` around synchronized steps | wall clock, not events |
| A6000 throttling check | `nvidia-smi` temperature/clock log, one sample per 5 s | [results/a6000/THERMAL.md](../results/a6000/THERMAL.md) |
| per-kernel prefill profile, GPU idle time, SM clock | `torch.profiler` (CUPTI) + `nvidia-smi` in [scripts/prof_prefill.py](../scripts/prof_prefill.py) | **output not committed**. The "~2.5 ms GPU idle per 3B prefill" in [docs/turing-prefill-gemm.md](turing-prefill-gemm.md) is quoted from it. |

**Not used:** Nsight Systems, Nsight Compute and rocprof. There are no hardware-counter captures in
the repo. The GPU server had no `nsys`/`ncu` ([plan, line 30](superpowers/plans/2026-10-15-prefill-3b-gap.md)).
So every GB/s and TFLOP/s figure below is *derived*: algorithmic bytes or FLOPs divided by measured
time. Nothing measures DRAM traffic, L2 hit rate, occupancy or stall reasons directly, and the
explanations of *why* a kernel is below the limit are inferences from timing shapes. Where that
matters it is called out.

**Peak bandwidth** is the datasheet figure (616 GB/s for the 2080 Ti, 768 GB/s for the A6000;
`KNOWN_BW` in [bench.py:238](../scripts/bench.py:238)), not a measured copy bandwidth. The fastest
committed kernel reaches 579 GB/s (94 %), so the practically achievable ceiling is at least that.

## The bandwidth floor for decode

At batch 1, each generated token must read every weight once, plus the KV cache up to the current
position. Nothing can be faster than

```
floor = (weight bytes per step + KV bytes per step) / peak DRAM bandwidth
```

- `weight bytes per step` is `hi_weight_bytes` ([api.cu:1086](../src/api.cu:1086)): every layer's
  matrices and norms, and the lm_head once. The embedding is excluded, since only one row of it is
  read.
- `KV bytes per step` is `2 · layers · kv_heads · head_dim · 2 B` per cached token, times the mean
  context during the run (`ctx + n/2`).

The `% of floor` in the tables is `floor / measured`.

| model | decode ms/token, ctx 128 | floor ms | % of floor | at ctx 1024 |
|---|---|---|---|---|
| Qwen2.5-0.5B | 2.344 | 1.609 | 68.6 % | 67.3 % |
| TinyLlama-1.1B | 4.135 | 3.368 | 81.4 % | 80.5 % |
| Llama-3.2-1B | 4.745 | 4.026 | 84.8 % | 84.2 % |
| Qwen2.5-1.5B | 6.214 | 5.024 | 80.8 % | 80.2 % |
| Llama-3.2-3B | 12.112 | 10.479 | 86.5 % | 84.9 % |

Across the 128–8,192-token context sweep (`decode_ctx`):

- Llama-3.2-1B holds 84–85 % and Llama-3.2-3B 85–86 %.
- Qwen2.5-1.5B falls from 81 % to 75–76 % at 4–8k tokens.
- Qwen2.5-0.5B stays at 64–69 %.

The floor includes the KV reads, so this measures how well attention keeps up as the cache grows,
not only the GEMVs.

## One decode step, batch 1

The kernel microbenchmarks time each decode kernel on its own, at the model's shapes. A step runs
5 kernels per layer (fused QKV+RoPE+cache-write GEMV, decode attention, O+residual GEMV, fused
norm+gate/up+SwiGLU GEMV, down+residual GEMV), then the lm_head GEMV and argmax: `5·layers + 3`
launches, replayed as one graph ([decode_step, api.cu:1319](../src/api.cu:1319)). Summing the
microbenchmarks over the layers gives a time budget that can be checked against the measured step.

**TinyLlama-1.1B (22 layers), context 1,024:**

| kernel | time per step | share | effective GB/s | % of 616 | time above its own floor |
|---|---|---|---|---|---|
| QKV GEMV (+RMSNorm) | 484 µs | 11.7 % | 476 | 77 % | 110 µs |
| O GEMV (+residual) | 396 µs | 9.5 % | 466 | 76 % | 96 µs |
| gate/up GEMV (+RMSNorm, SwiGLU) | 1,825 µs | 44.0 % | 556 | 90 % | 177 µs |
| down GEMV (+residual) | 979 µs | 23.6 % | 519 | 84 % | 155 µs |
| lm_head GEMV (+RMSNorm) | 229 µs | 5.5 % | 572 | 93 % | 16 µs |
| decode attention, L = 1,024 | 238 µs | 5.7 % | 97 (KV bytes) | 16 % | 200 µs |
| **sum** | **4,151 µs** | | | | **755 µs** |
| measured step (`decode_b1`, ctx 1,024) | 4,223 µs | | | | |

**Llama-3.2-3B (28 layers), context 1,024:**

| kernel | time per step | share | effective GB/s | % of 616 | time above its own floor |
|---|---|---|---|---|---|
| QKV GEMV (+RMSNorm) | 2,014 µs | 16.0 % | 437 | 71 % | 584 µs |
| O GEMV (+residual) | 1,059 µs | 8.4 % | 499 | 81 % | 201 µs |
| gate/up GEMV (+RMSNorm, SwiGLU) | 4,964 µs | 39.5 % | 568 | 92 % | 389 µs |
| down GEMV (+residual) | 2,724 µs | 21.6 % | 517 | 84 % | 436 µs |
| lm_head GEMV (+RMSNorm) | 1,361 µs | 10.8 % | 579 | 94 % | 82 µs |
| decode attention, L = 1,024 | 459 µs | 3.7 % | 256 (KV bytes) | 42 % | 269 µs |
| **sum** | **12,582 µs** | | | | **1,961 µs** |
| measured step | 12,544 µs | | | | |

Across all five models, the microbenchmark sum is 94–100 % of the measured step (Qwen2.5-0.5B
2,272 vs 2,417 µs; the others within 3 %). So **the step time is the sum of its kernels**, and
inter-kernel gaps inside the graph are small. The CUDA graph saves 0.01–0.04 ms per step on
these models (`ours_nograph_ms`). Both numbers are GPU-event times with the launches queued
asynchronously, so this is not a measure of host launch cost.

Caveats on the reconstruction:

- The microbenchmark's QKV row is the plain RMSNorm+GEMV (epilogue 0). The real step runs the
  fused QKV+bias+RoPE+cache-write epilogue (epilogue 3), which `bench.py` doesn't time on its own.
- Each microbenchmark replays the same weights, so a matrix that fits in L2 reads faster than DRAM
  would allow. Qwen2.5-1.5B's 4.7 MB O-projection measures 738 GB/s, *above* the 616 GB/s peak. In a
  real step every layer streams different weights, so those rows are optimistic.
- The decode-attention microbenchmark exists only at L = 1,024 and 4,096 (B = 1), so the budget is
  computed at context 1,024.

### Per-kernel efficiency across models (GEMV at M = 1, effective GB/s)

| model | QKV+norm | O+res | gate/up+norm+SwiGLU | down+res | lm_head+norm | decode attn L=1024 |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 355 | 426 | 510 | 424 | 569 | 10.4 µs |
| TinyLlama-1.1B | 476 | 466 | 556 | 519 | 572 | 10.8 µs |
| Llama-3.2-1B | 412 | 466 | 564 | 522 | 576 | 9.0 µs |
| Qwen2.5-1.5B | 388 | (738, L2-resident) | 558 | 483 | 577 | 13.1 µs |
| Llama-3.2-3B | 437 | 499 | 568 | 517 | 579 | 16.4 µs |

Against PyTorch (`F.linear`, i.e. cuBLAS GEMV) at the same shapes:

- The RMSNorm-fused GEMVs are faster because PyTorch has to run the norm as separate kernels:
  gate/up+norm+SwiGLU is 1.20–2.38× faster, and QKV+norm 1.30–3.64×. Llama-3.2-3B's QKV+norm is the
  exception, at parity (0.99×).
- down+residual is at parity (0.95–1.01×). O+residual is too (0.97–1.0×), except on the two Qwen
  models (1.54–1.59×), whose O matrices are small enough that PyTorch's own overhead shows.
- lm_head is level with or slightly ahead of PyTorch (1.00–1.10×).
- The large speedups are partly a statement about PyTorch's unfused baseline. The kernels' absolute
  GB/s is the more meaningful number.

### A simple model of GEMV time

For every model, the GEMV microbenchmarks that don't fit in L2 (> 5.5 MB; 3–5 per model) fit a
straight line closely:

```
t ≈ a + bytes / B      a = 4.0–9.9 µs per call,   B = 575–583 GB/s (93–95 % of 616)
```

For the four smaller models the residuals are within ±3.4 µs. Llama-3.2-3B's QKV GEMV sits 8 µs above
its line, which pulls that model's other points up to 5 µs below it.
Pooled over all 22 points: a = 6.2 µs, B = 581 GB/s. So **once a GEMV is streaming, it streams at
about 94 % of datasheet bandwidth**. The rest of the loss is a fixed per-call cost.

That fixed cost covers whatever doesn't scale with bytes: ramp-up and tail of the grid, the RMSNorm
prologue, the cross-warp reduction and epilogue, and the inter-kernel gap within a graph replay.
Splitting it among those needs Nsight Compute. It is also why the small matrices (QKV and O, 2–31 MB) sit at 58–81 %, while gate/up and lm_head
(17–788 MB) reach 83–94 %.

### Where the gap to the floor goes

Applying that fit to each model's measured step at context 1,024:

| model | gap to floor | streaming at 575–583 GB/s instead of 616 | fixed cost × (4·layers + 1) GEMVs | attention above its KV floor | unexplained |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | 790 µs | 114 (14 %) | 449 (57 %) | 230 (29 %) | −3 (0 %) |
| TinyLlama-1.1B | 822 µs | 195 (24 %) | 360 (44 %) | 200 (24 %) | 67 (8 %) |
| Llama-3.2-1B | 765 µs | 251 (33 %) | 360 (47 %) | 90 (12 %) | 65 (9 %) |
| Qwen2.5-1.5B | 1,252 µs | 296 (24 %) | 722 (58 %) | 318 (25 %) | −84 (−7 %) |
| Llama-3.2-3B | 1,898 µs | 585 (31 %) | 1,118 (59 %) | 269 (14 %) | −73 (−4 %) |

The three terms account for the gap to within ±9 %. Read as an estimate (it rests on a 3–5-point fit
per model), it says:

1. **The largest share is fixed per-kernel cost, not bandwidth.** It explains why Qwen2.5-0.5B is
   the outlier at 67–69 %: its matrices are so small that a 4.6 µs fixed cost is large next to their
   streaming time. More bandwidth efficiency inside the GEMV would recover at most the 14–33 %
   "streaming" column. Fewer, larger kernels (more fusion, or a persistent decode kernel) would
   target the bigger one.
2. **Decode attention at batch 1 is latency-bound, not bandwidth-bound.** At L = 1,024 it reads 1–4 MB
   of KV per layer and takes 9–16 µs, an effective 50–256 GB/s. At 4,096 tokens on TinyLlama's shapes it
   reaches 273 GB/s, and at B = 32, L = 2,048 it reaches 523 GB/s (85 %). So the kernel streams
   well once it has enough work, and at batch 1 it simply has too little.
3. **The difference between models is mostly matrix size**, not a model-specific inefficiency.

## Prefill

Prefill runs the same layers over T rows: tensor-core GEMMs instead of GEMVs, FlashAttention-style
causal attention, and a final GEMV for the last position's logits only. A budget from the M = 2,048
GEMM and T = 2,048 attention microbenchmarks, against the measured 2,000-token prefill:

| model (2080 Ti) | prefill 2,000 tok | 4 GEMMs × layers | of which lost to cuBLAS | flash attention | other |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | 43.1 ms | 34.5 ms (80 %) | +1.5 ms | 5.5 ms (13 %) | 3.0 ms |
| TinyLlama-1.1B | 100.5 ms | 85.1 ms (85 %) | +6.9 ms | 11.3 ms (11 %) | 4.1 ms |
| Llama-3.2-1B | 101.1 ms | 86.4 ms (85 %) | +5.4 ms | 8.5 ms (8 %) | 6.2 ms |
| Qwen2.5-1.5B | 140.9 ms | 122.9 ms (87 %) | +6.9 ms | 12.8 ms (9 %) | 5.3 ms |
| Llama-3.2-3B | 289.7 ms | 253.4 ms (87 %) | +17.1 ms | 24.6 ms (8 %) | 11.7 ms |

(The microbenchmarks use 2,048 rows against a 2,000-token prompt, which overstates GEMM time by about
2 % and attention by about 5 %. "Of which lost to cuBLAS" is our GEMM time minus cuBLAS's at the same
shapes.)

- **Prefill is GEMM-bound: 80–87 % of it is the four per-layer GEMMs.** On the 2080 Ti they run at
  37–48 TFLOP/s, against cuBLAS's 40–51 TFLOP/s on the same shapes (`bench.json`, M = 2,048). That
  is 0.84–0.95× cuBLAS, with two exceptions: Qwen2.5-0.5B's QKV is 1.40× (cuBLAS manages only
  27 TFLOP/s at N = 1,152), and Qwen2.5-1.5B's down projection is 1.03×. The repo doesn't record a compute peak
  for either GPU, so cuBLAS is used as the practical ceiling.
- **On Llama-3.2-3B, matching cuBLAS would save about 17 ms of the 289.7 ms in-suite prefill.**
  llama.cpp takes 261.8 ms, so even cuBLAS-parity GEMMs (≈ 272 ms in the suite's power-capped
  conditions) wouldn't quite close the gap measured there. The 273.6 ms isolated run is on a cooler,
  faster-clocked GPU and isn't directly comparable.
- **Why the 3B GEMMs aren't at parity:** the write-up estimates wave quantization. At M ≈ 2,048 the
  autotuned 256×128 tiles run one block per SM, and QKV launches 320 tiles (4.71 waves), O and down
  192 (2.82 waves), gate/up 1,024 (15.06 waves) on 68 SMs. That is about 6 % of GEMM time, or about
  15 ms per prefill ([plan, line 233](superpowers/plans/2026-10-15-prefill-3b-gap.md)). This is a
  tile-count estimate, not a measurement. A stream-K attempt to fix it was 1.3–2.4× slower per GEMM
  and was dropped ([docs/turing-prefill-gemm.md](turing-prefill-gemm.md)).
- **Attention is 8–13 % of prefill.** The 2080 Ti causal flash kernel runs at 23–33 TFLOP/s
  (causal FLOPs, T = 512–2,048, TinyLlama shapes). That is 20–29× PyTorch SDPA, but SDPA has no flash backend on Turing
  and measured only 1.1–1.2 TFLOP/s here, so the speedup is no evidence of a fast kernel. On the A6000, where SDPA has a FlashAttention backend, the same kernel is at
  0.90–0.92× SDPA (TinyLlama T = 1,024 / 2,048).
- **A6000 prefill GEMMs are further behind: 0.61–0.85× cuBLAS** (`results/a6000/*/bench.json`, M = 2,048).
  On TinyLlama (an unthrottled run) the GEMMs take 51.1 ms of a 59.6 ms prefill, where cuBLAS
  would take 37.4 ms. That 13.7 ms is 23 % of the prefill. The Ampere `cp.async` path hasn't had the
  tuning the Turing path got. Llama-3.2-1B, Qwen2.5-1.5B and Llama-3.2-3B were partly throttled
  on that card (THERMAL.md). Their GEMM rows are still usable as ratios, because hipinfer and cuBLAS
  were timed back to back at each shape, but their absolute TFLOP/s are not.

**A stale data file to be aware of:** [results/2080ti/gemm_sweep_tinyllama.json](../results/2080ti/gemm_sweep_tinyllama.json)
and `gemm_sweep_llama32-3b.json` cover configs 0–25 only. Config 26 (the 256×128 SwiGLU tile) was added
in the Turing GEMM change, so these sweeps predate the current build. At M = 2,048 they read
0.79–0.91× cuBLAS, consistent with the write-up's "before" numbers. Use them for the shape of
TFLOP/s vs M, not for current ratios.

## Batched decode

With B sequences, a step still reads each weight once but does B times the math. Ideally step time stays
flat until compute becomes the limit. Measured (ctx 512, 128 new tokens; bytes per step =
weights + B × KV at the mean context of 576 tokens):

| TinyLlama-1.1B | B=1 | B=4 | B=8 | B=16 | B=32 | B=64 |
|---|---|---|---|---|---|---|
| ms / step | 4.16 | 4.65 | 4.88 | 5.19 | 6.69 | 8.26 |
| step time vs B=1 | 1.00× | 1.12× | 1.17× | 1.25× | 1.61× | 1.98× |
| effective GB/s (% of 616) | 500 (81 %) | 456 (74 %) | 445 (72 %) | 439 (71 %) | 372 (60 %) | 351 (57 %) |
| ≈ TFLOP/s (2 × params × B / step) | 0.5 | 1.8 | 3.4 | 6.4 | 9.9 | 16.0 |
| board power (NVML) | 239 W | 247 W | 246 W | 246 W | 245 W | 248 W |

| Llama-3.2-3B | B=1 | B=4 | B=8 | B=16 (max that fits) |
|---|---|---|---|---|
| ms / step | 12.21 | 13.56 | 16.70 | 18.01 |
| effective GB/s (% of 616) | 532 (86 %) | 493 (80 %) | 416 (68 %) | 415 (67 %) |

- **Neither roofline is reached at B = 64.** TinyLlama at B = 64 moves 2.90 GB per step (29 % of it KV)
  at 351 GB/s (57 % of peak), and does about 16 TFLOP/s, about a third of what cuBLAS reaches on
  prefill-sized GEMMs on this card. At B = 64 the weights carry 64 FLOP per byte, below the ≈ 83 FLOP/byte
  ratio of cuBLAS's 51 TFLOP/s to 616 GB/s. So batched decode should still be bandwidth-limited, and the
  57 % points at kernel efficiency at these shapes rather than a hardware wall. The four models
  that fit B = 64 on the 11 GB card reach 44–68 % of peak bandwidth there (Llama-3.2-3B tops out
  at B = 16, at 67 %).
- **The switch from GEMV to GEMM shows.** Up to B = 4, decode uses the GEMV kernels. From B = 5
  (`HIPINFER_GEMV_MAX`, default 4, [api.cu:1312](../src/api.cu:1312)) it uses the weight-streaming
  tensor-core `gemm_skinny_kernel`. From B = 4 to B = 8, step time grows 32 % on Qwen2.5-1.5B,
  23 % on Llama-3.2-3B and 20 % on Qwen2.5-0.5B, but only 2–5 % on TinyLlama and Llama-3.2-1B. The
  committed `bench.json` has no GEMM timings at M = 5–64 (only 128 / 512 / 2,048), so this data can't
  say which projection is responsible. The README's figure of ~1.15–1.2× slower than cuBLAS at 32–64
  sequences is not backed by a committed current-build file (see [claims.md](claims.md)).
- **Power:** every 2080 Ti batched run draws 218–249 W, at or near the card's 250 W cap. This is
  also why long 2080 Ti runs settle at 1.4–1.55 GHz (README, thermal section).

## Tensor-parallel decode (2× RTX 2080 Ti, PCIe 3.0, no P2P)

Each GPU reads half the weights, so the per-GPU floor halves. Sharded decode reaches 46 % (Qwen2.5-0.5B),
62–76 % (1–3B) and 83 % (Llama-3.1-8B, Mistral-7B) of the per-GPU floor (`results/2080ti/*/tp2.json`).
Each layer adds two all-reduces of one hidden row, and the all-reduce microbenchmark gives their size:

| model | sharded step, ctx 128 | one-shot all-reduce, 1 row | × 2 per layer | share of step |
|---|---|---|---|---|
| Qwen2.5-0.5B | 1.76 ms | 7.9 µs | 0.38 ms | 22 % |
| TinyLlama-1.1B | 2.71 ms | 8.5 µs | 0.37 ms | 14 % |
| Llama-3.2-1B | 2.81 ms | 8.1 µs | 0.26 ms | 9 % |
| Qwen2.5-1.5B | 3.87 ms | 8.2 µs | 0.46 ms | 12 % |
| Llama-3.2-3B | 6.89 ms | 10.4 µs | 0.58 ms | 8 % |
| Mistral-7B-v0.3 | 13.99 ms | 11.7 µs | 0.75 ms | 5 % |
| Llama-3.1-8B | 14.68 ms | 11.1 µs | 0.71 ms | 5 % |

The rest of the shortfall on small models is the same fixed per-kernel cost as on one GPU, now paid on
matrices half the size. The kernel count per layer is unchanged, and the norms are replicated. The
all-reduce share is an estimate: the microbenchmark is an isolated, graph-replayed loop. It agrees
roughly with the measured gap between the one-shot and NCCL-only backends (e.g. Llama-3.2-3B:
0.38 ms measured vs 0.34 ms predicted).

## What would make this profile stronger

1. An Nsight Systems trace of one graph-replayed decode step and one prefill. It would replace the
   microbenchmark reconstruction with the real timeline (including the fused QKV epilogue) and
   settle how big the inter-kernel gaps are.
2. Nsight Compute on the QKV and O GEMVs (the 58–81 % kernels) and on decode attention at batch 1. It
   would measure DRAM bytes, achieved occupancy and stall reasons, and split the fixed per-call cost
   into ramp-up, tail and prologue.
3. GEMM microbenchmarks at M = 5–64 against cuBLAS on the current build, to explain the batched-decode
   step at B = 8–64.
4. Committing the `prof_prefill.py` output for Llama-3.2-3B alongside the write-up that quotes it.
5. rocprof on any AMD GPU: nothing has run there yet ([claims.md](claims.md)).
