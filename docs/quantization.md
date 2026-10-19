# int8 weights (W8A16)

hipinfer can store the projection weights as int8 with one fp16 scale per output row and keep
everything else in fp16: activations, the KV cache, norms, biases and the embedding. Batch-1
decode streams every weight once per token, so halving the weight bytes is the largest lever
left once the fp16 kernels run near the bandwidth floor. This page covers the format, the kernels,
how the int8 path is checked, and what it measured. int4 comes later and is not covered here.

## Format: weight-file version 4

`scripts/export.py --int8` writes version 4: the version-2 layout and header, plus an int32
quantization type at byte 56 (1 = int8 with one scale per row). These tensors are stored as int8
rows `q [N, K]`, each followed by its fp16 scales `s [N]` on its own 256-byte boundary:

- `wqkv` (q, k and v rows stacked)
- `wo`
- `wgu` (gate and up rows interleaved in groups of 16)
- `wdown`
- `lm_head`

The lm_head is always written, even for tied models such as Llama-3.2, because the embedding
stays fp16 and only B rows of it are gathered per step.

Quantization is symmetric round-to-nearest per output row (`quantize_rows`):

```
s[n]    = fp16(max_k |w[n, k]| / 127)
q[n, k] = clamp(round(w[n, k] / s[n]), -127, 127)      w[n, k] ≈ q[n, k] * s[n]
```

The rounding divides by the stored fp16 scale, so `q * s` is exactly what the file describes. A
row of zeros gets `s = 0, q = 0`. Granite's folded multipliers are applied before quantization.
MoE models are rejected (`--int8` is dense-only; the expert kernels are fp16), and fp16 files
(versions 2 and 3) load exactly as before.

A scale per output row, rather than per group of k, makes tensor parallelism simple, and the
loader's `proj()` handles both kinds of split:

- **Column-parallel layers** (qkv, gate/up, lm_head) are split by rows, and each rank takes the
  scales of its rows.
- **Row-parallel layers** (o, down) are split by columns. The scale belongs to the whole output
  row, so every rank keeps all N scales and scales its partial dot product. The all-reduce then
  sums `s[n] * partial_r[n]` over ranks, which equals `s[n] * full[n]`.

## Decode: the GEMV

`gemv_kernel<..., Q8>` is the fp16 GEMV with a different weight load. It covers all the decode
paths:

- the fused RMSNorm + gate/up + SwiGLU
- down + residual
- the fused QKV + bias + RoPE + KV-cache write
- o + residual
- the lm_head

The kernel works like this:

- Each lane loads 8 int8 weights (8 bytes) where the fp16 kernel loads 8 halves (16 bytes). It
  keeps twice as many loads in flight, so a lane has the same bytes outstanding as in fp16, and
  each load still lines up with one 16-byte chunk of the activation row (no shared-memory bank
  conflicts on the fused-norm path).
- Weights become fp32 in registers. On NVIDIA, each byte is XORed with 0x80 and placed into the
  low mantissa byte of 2^23 with one `__byte_perm`, and one subtraction of 2^23 + 128 leaves the
  exact value. That is two full-rate instructions per weight instead of an extract plus a
  quarter-rate int-to-float conversion. The portable version (`HI_AMD`, wave32 or wave64) uses
  sign-extending shifts and a conversion. The rest of the kernel (warp reductions, `HI_WARP`) is
  already wave-size generic.
- The fp32 dot product is `Σ q·x`. The row scale multiplies it once after the warp reduction,
  before the epilogue (bias, RoPE, residual, SwiGLU). Factoring the per-row scale out of the sum
  is exact algebra and saves a multiply per weight.

## Prefill: tiles dequantized in shared memory

The tensor-core GEMMs (portable WMMA `gemm_kernel` and NVIDIA `gemm_mma_kernel`) take a `Q8`
template flag:

- Each thread's W chunk is loaded as 8 raw int8 bytes. It stays in registers during the MMAs of
  the current tile, exactly where the fp16 chunk waited. At the shared-memory store it becomes
  `fp16(q * s)` (`q * s` is exact in fp32, so this is the correctly rounded dequantized weight).
- Each thread loads the same W rows for every K tile, so their scales are read once, before the
  K loop.
- From shared memory on, the kernels are unchanged: same MMAs, same epilogues, same split-K.
- int8 always uses the register-staged two-stage pipeline (the Turing path), also on sm_80+,
  because `cp.async` copies global to shared memory without passing through registers and cannot
  convert. Ampere int8 prefill therefore runs without the 3-stage pipeline (not measured here).
- The skinny weight-streaming kernels (batched decode, M ≤ 64) read fp16 weight fragments straight
  from global memory and have no int8 variant. For int8 weights the autotuner only considers the
  tiled configs. The autotune cache key includes int8, so fp16 and int8 shapes are tuned separately.

This is the simple version. Weights are still read as int8 from DRAM (half the bytes), but the
tensor cores do the same work as for fp16, so prefill should run at about fp16 speed, not faster.

## Checks

- **Kernel tests** (`tests/test_kernels.py`):
  - The exporter's rounding: max |q| = 127 per row, |w − q·s| ≤ s/2, zero rows stay zero.
  - The int8 GEMV for every epilogue and the fused norm, against fp32 math on q·s.
  - The fused QKV + RoPE + cache path.
  - The int8 GEMM for every tile config (skinny ones fall back), split-K and partial tiles, against
    fp32 math on fp16(q·s).
- **Token gates** (`tests/gate.py`):
  - `--ref int8` runs HF on the same int8 round-trip weights (`fake_quantize_hf`, which uses the
    exporter's own `quantize_rows`), so the engine must match it as strictly as fp16 matches fp16
    HF (near-ties only).
  - Against fp16 HF (the default), an int8 engine computes a different model and may diverge. The
    gate logs each divergence with its step and HF's top-2 gap, accepts it, and reports how many
    tokens matched first.
- **Perplexity** (`tests/ppl.py`): WikiText-2 against fp16 HF (the cost of quantization: perplexity
  change and mean KL) and against fake-quantized HF (`--ref int8`, the engine's own error).
- **Benchmarks** (`scripts/bench.py`): the bandwidth floor uses `hi_weight_bytes`, which counts
  int8 projections at one byte per weight plus their row scales.

`tests/test_int8_cpu.py` needs no GPU. It exports a tiny random Llama with `--int8` and parses the
file in the loader's order against `fake_quantize_hf`. It also checks the rounding, emulates the
byte-permute conversion bit for bit for all 256 byte values, and checks the tensor-parallel scale
algebra.

## What the bytes predict

The weight bytes one decode step reads (`hi_weight_bytes`; weights, scales and norms, no KV
cache), and the time to stream them at the 2080 Ti's 616 GB/s datasheet bandwidth. This is
arithmetic from the model configs, not a measurement:

| model | fp16 | int8 | fp16 floor | int8 floor |
|---|---|---|---|---|
| TinyLlama-1.1B | 2.07 GB | 1.04 GB | 3.36 ms | 1.68 ms |
| Llama-3.2-1B | 2.47 GB | 1.24 GB | 4.01 ms | 2.01 ms |
| Llama-3.2-3B | 6.43 GB | 3.22 GB | 10.43 ms | 5.22 ms |

Decode cannot reach a 2× speedup in practice:

- The KV-cache reads and fixed per-kernel costs stay the same. docs/profiling.md fits each GEMV at
  about 4–10 µs plus its bytes at ~580 GB/s, so small GEMVs gain less than half their time.
- Converting weights costs more instructions per byte than fp16 does, which may matter once
  several sequences share a GEMV.

Prefill should stay at about fp16 speed, since the tensor-core work is unchanged.

llama.cpp's Q8_0, the comparison format, is not the same format. It keeps one fp16 scale per
block of 32 weights (8.5 bits per weight, about 6 % more bytes than per-row int8). The finer
blocks should give it a slightly smaller quantization error than per-row scales; the perplexity
results will show how much that matters on these models.

## Results (RTX 2080 Ti)

From the merged tree on 2026-10-18 ([results/2080ti/MERGED_RUN.md](../results/2080ti/MERGED_RUN.md)).
TinyLlama and Llama-3.2-1B come from `results/2080ti/<model>/int8/`; Llama-3.2-3B's benchmarks come
from the cool-start rerun in `results/2080ti/rerun/`, its gates and perplexity from
`results/2080ti/llama32-3b/int8/`. The kernel tests (`results/2080ti/test_kernels_int8.log`) and the
AMD compile check pass.

**Accuracy.** Perplexity on WikiText-2, 16 × 1,024 tokens, and the token gates (20 runs each):

| model | PPL fp16 HF | PPL int8 hipinfer | change | mean KL vs fp16 HF | PPL vs int8 HF | KL vs int8 HF | gate vs int8 HF | gate vs fp16 HF |
|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B | 8.5151 | 8.5186 | +0.041 % | 9.9e-4 | −0.0003 % | 6.6e-6 | pass, 15/20 | 8/20 identical |
| Llama-3.2-1B | 10.4631 | 10.4871 | +0.229 % | 1.9e-3 | −0.003 % | 1.3e-5 | pass, 18/20 | 7/20 |
| Llama-3.2-3B | 8.2768 | 8.2930 | +0.195 % | 1.3e-3 | −0.001 % | 9.5e-6 | pass, 18/20 | 17/20 |

The "vs int8 HF" columns compare against HF running the exporter's own rounded weights
(`--ref int8`). They measure the kernels' error, which is fp16-sized. The "vs fp16 HF" columns
measure the cost of quantizing. The fp16 weights on the int8 build still pass their gates (20/20,
14/20 and 20/20 identical).

**Speed.** Batch-1 decode at a 128-token context, and prefill at 2,000 tokens, against llama.cpp
`a0e6dc7` on Q8_0 and F16 GGUFs in the same session:

| model | decode int8 | decode fp16 | int8 speed-up | llama.cpp Q8_0 | int8 vs Q8_0 | % of floor, int8 / fp16 | prefill int8 / fp16 / Q8_0 |
|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B | 2.54 ms | 4.15 ms | 1.63× | 3.21 ms | 1.26× | 67 / 81 % | 117 / 102 / 145 ms |
| Llama-3.2-1B | 2.73 ms | 4.72 ms | 1.73× | 3.32 ms | 1.22× | 74 / 85 % | 115 / 98 / 124 ms |
| Llama-3.2-3B | 6.71 ms | 12.01 ms | 1.79× | 7.94 ms | 1.18× | 78 / 87 % | 328 / 282 / 346 ms |

- **The bytes predicted a 2× decode speed-up; it reaches 1.6–1.8×.** The fixed per-kernel costs and
  the KV reads (above) stay, so the smallest model gains least.
- **Prefill is 1–18 % slower than fp16** (1–9 % at 128 tokens, 11–18 % at 512–2,000). Converting
  each tile on its way to shared memory costs more than the prediction above assumed. The int8
  prefill is still faster than llama.cpp's Q8_0 prefill at every length measured (1.05–1.72×).
- **llama.cpp gains less from Q8_0** (1.47–1.62× over its F16) than hipinfer does from int8
  (1.63–1.79× over its fp16).

Register spills, compared with main's build (`ptxas -warn-spills`, sm_75):

- **fp16 kernels:** unchanged, except `gemv_kernel<4, no norm, QKV>`, which now spills 8 bytes.
  The engine only uses it when the fused RMSNorm does not fit in shared memory, which none of the
  tested models hit.
- **int8 batch-1 GEMVs (MR = 1):** no spills.
- **int8, other instantiations:** a few spill 4–28 bytes, the same order as their fp16 twins:
  - QKV GEMVs at 2, 4 and 5 rows and SwiGLU at 6 and 8 rows
  - the 256×128 mma.sync tile
  - the 64×128×64 WMMA tile

`scripts/run_int8.sh <gpu_tag> [models]` runs every step above on the visible GPU;
`scripts/edmonds_all.sh` runs it on edmonds as part of the whole merged-tree check.
