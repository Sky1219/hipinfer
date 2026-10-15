# hipinfer

An LLM inference engine written from scratch in HIP: fused GEMV/GEMM, RMSNorm, RoPE and
FlashAttention-style kernels, a contiguous KV cache, and a graph-captured decode loop.
The one source tree builds with `hipcc` for AMD GPUs and with plain `nvcc` for NVIDIA GPUs.
It runs Llama-family models in fp16: Llama 2/3/3.1/3.2, TinyLlama, Mistral and Qwen2/2.5
(head_dim 64 or 128, any GQA group up to 8, tied or untied embeddings, Llama-3 RoPE scaling,
QKV bias).

**Status (2026-10-15):**

- **Verified on NVIDIA:** RTX 2080 Ti ×2 (Turing, sm_75) and RTX A6000 (Ampere, sm_86). All kernel
  tests pass on both architectures. Five models have full results on the 2080 Ti (below). The
  A6000 multi-model run, including Mistral-7B and Llama-3.1-8B, is pending; its numbers below
  are TinyLlama from the previous build.
- **New on 2026-10-15: a faster Turing prefill GEMM** ([below](#the-turing-prefill-gemm-2026-10-15)).
  After it, the kernel tests pass on the 2080 Ti (915 checks), and Llama-3.2-3B prefill was
  re-measured. **Not yet re-run on the new code:** the other models' benchmarks, the token gate and
  perplexity, the multi-GPU numbers, the A6000 and the AMD compile check. Every number below that
  predates the change is marked as such.
- **Every model matches HF transformers:** WikiText-2 perplexity within 0.003 %, and greedy tokens
  identical except where HF's own top-2 logits are tied. These were measured before the GEMM change.
- **Multi-GPU tensor parallelism:** verified on two RTX 2080 Tis without a peer-to-peer path. It gives
  1.55–1.75× faster decode, 1.34–1.65× faster prefill, and runs Llama-3.1-8B / Mistral-7B-v0.3, which do
  not fit one 11 GB card. See [Multi-GPU](#multi-gpu-tensor-parallelism). The prefill figures predate the
  GEMM change.
- **AMD is compile-checked only:** the build succeeds for MI200 (gfx90a), MI300 (gfx942) and RDNA3
  (gfx1100) with ROCm 7 hipcc and rocWMMA, but it has **not been run on AMD hardware yet**.

Author: Akash Dasarraju. The code was written with Claude (AI pair programmer); design
decisions, benchmarks and claims were reviewed by me.

## Headline numbers

RTX 2080 Ti (616 GB/s), fp16 weights, greedy, batch 1 unless noted. llama.cpp is `llama-bench` on
an F16 GGUF (no quantization), best of flash-attention on/off. HF = `transformers` eager.

| model | decode ms/token | % of bandwidth floor | vs llama.cpp | vs HF | prefill 2,000 tok vs llama.cpp | peak batched tok/s |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | **2.33** | 69 % | **1.48× faster** | 13.9× | **1.46× faster** | 11,800 (B=64) |
| TinyLlama-1.1B | **4.13** | 82 % | **1.15× faster** | 6.7× | **1.04× faster** | 7,550 (B=64) |
| Llama-3.2-1B | **4.70** | 86 % | **1.11× faster** | 4.3× | 1.08× slower | 7,470 (B=64) |
| Qwen2.5-1.5B | **6.17** | 81 % | **1.17× faster** | 6.0× | **1.05× faster** | 4,500 (B=64) |
| Llama-3.2-3B | **12.02** | 87 % | **1.06× faster** | (HF copy doesn't fit) | 1.04× slower (was 1.22×) | 906 (B=16) |

Only the Llama-3.2-3B prefill cell was re-measured after the 2026-10-15 GEMM change: 273.6 ms
against llama.cpp's 261.8 ms (it was 318.5 ms). The other prefill cells predate the change.

Batch-1 decode is memory-bound: every token streams all the weights once. The floor is the bytes
read per token (weights plus the KV cache) divided by peak bandwidth. **The engine reaches
81–87 % of it on every model of 1B parameters or more**, and stays there to 8k-token contexts
(86 % at 8,192 tokens on Llama-3.2-3B). Qwen2.5-0.5B is lower (69 %) because its matrices are
small enough that launch and synchronization overhead shows. llama.cpp reaches 47–83 % on the
same models.

The floor here counts only bytes actually read. Earlier versions of this README counted the
whole embedding table, which decode never reads; with that definition TinyLlama was "87 %" at
the same 4.1 ms.

### Against other engines on TinyLlama (same GPU, fp16 weights, batch 1)

| decode ms/token, ctx 128 → 1024 | RTX 2080 Ti | RTX A6000 (previous build) |
|---|---|---|
| **hipinfer** | **4.13 → 4.21** | **3.41 → 3.47** |
| llama.cpp `a0e6dc7`, F16 GGUF, CUDA, flash-attn | 4.77 → 4.79 | 3.76 → 3.76 |
| vLLM 0.31 (V1, CUDA graphs) | not run | 3.96 → 3.95 |
| torch.compile, gpt-fast-style static KV, `reduce-overhead` | 5.24 → 6.84 | 4.23 → 5.47 |
| HF transformers `StaticCache` + `torch.compile` | 5.51 → 7.09 | 4.34 → 5.56 |

| prefill ms, 128 / 512 / 2000 tokens | RTX 2080 Ti | RTX A6000 (previous build) |
|---|---|---|
| **hipinfer** | **9.3** / **28.6** / **111** | **6.4** / **17.7** / 67 |
| llama.cpp (best of flash-attn on/off, ubatch 512/2048) | 11.2 / 29.0 / 115 | 9.1 / 22.8 / 87 |
| torch.compile (inductor + cuBLAS) | 16.2 / 34.8 / 125 | 8.9 / 19.5 / **60** |

The other engines were measured in an earlier session on the same machine. Per-model llama.cpp
comparisons for the other four models are in [results/SUMMARY.md](results/SUMMARY.md).

What this says, honestly:

- **Batch-1 decode: hipinfer is the fastest engine measured on every model**, by 6–17 % over
  llama.cpp on the 1–3B models and 48 % on Qwen2.5-0.5B. The margin shrinks as models grow,
  because everything converges on the bandwidth floor. The gain comes from fewer launches (5 per
  layer, fused norm/RoPE/cache-write/SwiGLU epilogues) and from the GEMV's bandwidth. I have not
  profiled llama.cpp to attribute its extra time precisely.
- **Prefill: faster up to ~1.5B parameters, still slower at 3B.** At 2,000 tokens llama.cpp is
  1.04× faster on Llama-3.2-3B: 273.6 ms against 261.8 ms, 12 ms behind. The gap was 1.22× (318.5 ms)
  before the Turing GEMM change. llama.cpp is also 1.08× faster on Llama-3.2-1B, a number from
  before the change. Prefill of a 3B model is GEMM-bound, and the hand-written GEMM is now at
  0.92–0.95× cuBLAS (see below). I stopped here; the gap is not closed.
- These are batch-1 latency numbers on small models. vLLM and llama.cpp are built for much more:
  many concurrent requests, quantization, hundreds of architectures. Batch 1 is simply where a
  small engine can be checked against a hard floor.

Full tables, including decode latency vs context length, batched throughput with power draw and
energy per token, and GEMM vs cuBLAS at every model's shapes: [results/SUMMARY.md](results/SUMMARY.md).
SUMMARY.md predates the 2026-10-15 GEMM change.

## The Turing prefill GEMM (2026-10-15)

Llama-3.2-3B prefill on the 2080 Ti was the one headline row where llama.cpp won by a lot. About
90 % of that prefill is the four per-layer GEMMs, and those ran at 0.79–0.81× cuBLAS.

**What changed.** Turing has no `cp.async`, so the old main loop staged each K tile through
registers and paid for the shared-memory store, the barrier and the next tile's first fragment
loads one after another, with the MMAs idle. The new loop overlaps them:
- The global loads for the next tile are issued in two halves while the current tile's MMAs run.
- There is one `__syncthreads` per K tile.
- The next tile's first `ldmatrix` fragments are loaded before the last k16 step's MMAs, so those
  MMAs hide the load latency.

Three smaller changes:
- **A new 256×128 tile with 2×4 warps:** its 128×32 warp tiles can run the fused SwiGLU epilogue.
- **The autotuner times each candidate with the epilogue it will actually run** (plain, residual
  add or SwiGLU) and then re-times the three fastest back to back. Timed with a plain epilogue, it
  picked a slower config for gate/up.
- **The fallback choice for Turing** (used before tuning, or with `HIPINFER_GEMM_AUTOTUNE=0`) no
  longer picks big tiles for short prompts. The old one took 42.9 ms at 128 tokens; the new one
  takes 24.0 ms.

Accumulation is still fp32, and there is no vendor GEMM.

| Llama-3.2-3B, RTX 2080 Ti | before | after |
|---|---|---|
| prefill, 2,000 tokens | 316–318 ms | **273.6 ms** (llama.cpp: 261.8 ms) |
| GEMMs at 2,048 rows vs cuBLAS (qkv, o, gate/up, down, lm_head) | 0.79–0.81× | **0.92–0.95×** |

**How it was measured.**
- The 273.6 ms is the median of 5 prefills on an otherwise idle 2080 Ti.
- Under sustained load these cards power-cap: I logged SM clocks of 1.41–1.62 GHz during long
  runs, against about 1.77 GHz in short bursts. With autotuner picks that differ slightly between
  processes, repeated runs landed between 273 and 285 ms.
- GPU idle time between kernels, measured from a profiler trace, is only about 2.5 ms per prefill.
  The remaining time is inside the GEMMs.

**What didn't work.**
- **Where the rest of the gap is:** at 2,048 rows the 256×128 tiles run one block per SM, so each
  GEMM's last wave leaves some of the 68 SMs idle.
- **What I tried:** a stream-K variant that splits the tail tiles along K across the idle SMs. It
  was correct and deterministic, but 1.3–2.4× slower per GEMM on the 2080 Ti. Blocks that start
  mid-K seem to lose L2/DRAM locality, though I couldn't confirm that without profiler counters.
  It is not in the code.

**Cost:** the new 256×128 config spills 16 bytes of registers per thread on sm_75. I kept it
because it is the fastest SwiGLU config I measured.

## Multi-GPU: tensor parallelism

`Engine(path, tp=2)` (C: `hi_load_tp`) shards a model over several GPUs in one process, one
worker thread per GPU, with the same API as one GPU. The sharding is Megatron-style:
- QKV is column-parallel by heads, and each GPU keeps its heads' KV cache.
- O is row-parallel, followed by an all-reduce.
- gate/up are column-parallel, and down is row-parallel with an all-reduce.
- lm_head is split by vocabulary. Greedy argmax reduces one 64-bit key per row, made of the order-preserving logit bits and the inverted index, so ties break exactly as on one GPU.

The embedding and norms are replicated, so after each all-reduce every GPU holds the same residual stream.

**Measured on the worst kind of multi-GPU box:** two RTX 2080 Tis on different CPU sockets,
PCIe 3.0, **no peer-to-peer path** (`nvidia-smi topo`: SYS). Every byte between the GPUs crosses
host memory.

| model | decode ms/token, 1 GPU → 2 GPUs | prefill 2,000 tok, ms | batched B=64, tok/s |
|---|---|---|---|
| TinyLlama-1.1B | 4.14 → **2.67** (1.55×) | 117 → **88** (1.34×) | 7.4k → **8.5k** |
| Llama-3.2-1B | 4.71 → **2.79** (1.69×) | 115 → **76** (1.52×) | 7.5k → **9.5k** |
| Qwen2.5-1.5B | 6.18 → **3.85** (1.61×) | 159 → **110** (1.45×) | 4.5k → **5.4k** |
| Llama-3.2-3B | 11.99 → **6.84** (1.75×) | 329 → **199** (1.65×) | B=16 max on 1 GPU → **3.2k** at B=64 |
| Llama-3.1-8B | doesn't fit one card → **14.6** (83 % of the per-GPU floor) | **418** | **858** at B=16 |
| Mistral-7B-v0.3 | doesn't fit one card → **14.0** (83 %) | **422** | **1,406** at B=32 |

The prefill column was measured before the 2026-10-15 Turing GEMM change and has not been re-run.

What makes it work:

- **One-shot all-reduce kernel** ([src/kernels.cuh](src/kernels.cuh)) instead of NCCL for decode-sized messages:
  - **How it works:** each rank stages its chunk where every rank can read it (the peer's memory over P2P, or pinned host memory when there is no peer path, as here). It raises a system-scope release flag in every rank's flag array and waits on acquire. Then every rank sums all copies itself, in rank order and in fp32.
  - **Identical results on every GPU:** because every rank sums in the same order, all GPUs end bitwise identical, and NCCL and the one-shot kernel produce bitwise-identical model outputs.
  - **Channels:** every block is an independent channel with its own staging sub-buffers and call counter, so the grid can be sized to the message (one block for a 4 KB decode row) and can run in a CUDA graph.
  - **Speed:** 8 µs vs NCCL's 12 µs at 4 KB, and 58 vs 85 µs at 256 KB. Against NCCL alone that is 5–9 % lower batch-1 decode latency and 10–15 % more batched throughput at B=64.
  - **Backend choice:** `auto` uses the one-shot kernel up to 512 KB and NCCL above. NCCL is optional at build time; without it everything goes one-shot.
- **Prefill all-reduces overlap compute.** The prompt rows are split in two halves. Each half's all-reduce runs on a second stream while the other half computes; the attention of the second half reads the first half's freshly written KV cache. Without the overlap, PCIe makes 2-GPU prefill *slower* than 1 GPU (TinyLlama 124 ms vs 117); with it, 1.34–1.65× faster.
- **Per-GPU state is `thread_local`** (autotuned GEMM choices, split-K workspaces, kernel attributes), so each worker thread drives its GPU with the single-GPU code path unchanged.

**Correctness** ([tests/test_tp.py](tests/test_tp.py)), 2 GPUs against the same weights on one GPU, for every backend:
- **Greedy tokens:** 13 runs per model (5 prompts single-sequence plus a batch of 8). Tokens are identical except where the 1-GPU engine's own top-2 logits are within one fp16 step (gap 0.0078).
- **All-reduce self-test:** exact.
- **Backends:** one-shot and NCCL give bitwise-identical logits and tokens.
- **Llama-3.1-8B and Mistral-7B-v0.3** run only sharded, so they are checked against HF transformers spread over the same two GPUs, on WikiText-2 (8 windows × 1,024 tokens):

  | model (sharded x2) | PPL hipinfer | PPL HF | top-1 agreement | mean KL(HF, hipinfer) |
  |---|---|---|---|---|
  | Llama-3.1-8B | 8.1505 | 8.1511 | 99.89 % | 1.2e-5 |
  | Mistral-7B-v0.3 | 7.3269 | 7.3267 | 99.91 % | 6.7e-6 |

Testing TP also found a real single-GPU bug. The decode attention's split counters sat at an
offset that depended on the batch size, so a decode step at a *smaller* batch after a larger one
could read the larger one's partial sums as counters. TP's halved KV heads mean more splits per
launch, which is what exposed it. It is fixed, and `test_decode_work_reuse` covers it.

**Limits:** decode reaches 83 % of the per-GPU bandwidth floor on the 7–8B models, the same as one
GPU on the 1–3B models. On the 1–3B models sharded it reaches 63–77 %: each GPU's matrices are half
as big, but launches, the replicated norms and two all-reduces per layer are not, so small models
gain least. The P2P staging path is implemented but untested:
neither machine I have has GPUs with peer access. Tested at tp=2 only. Up to 8 GPUs are
supported in code, and heads, KV heads and the intermediate size must divide evenly.

## Correctness

| model | greedy runs token-identical | PPL hipinfer | PPL HF | top-1 agreement | mean KL(HF, hipinfer) |
|---|---|---|---|---|---|
| TinyLlama-1.1B | 20/20 | 8.5149 | 8.5151 | 99.90 % | 6.6e-6 |
| Qwen2.5-0.5B | 18/20 | 13.4851 | 13.4851 | 99.79 % | 1.6e-5 |
| Llama-3.2-1B | 9/20 | 10.4633 | 10.4631 | 99.81 % | 1.3e-5 |
| Qwen2.5-1.5B | 17/20 | 9.4536 | 9.4535 | 99.86 % | 1.6e-5 |

- **Greedy gate** ([tests/gate.py](tests/gate.py)): 7 prompts × 128 tokens, single-sequence with and
  without CUDA graphs, plus a batch of 6 sequences at different lengths (20 runs). **Every
  divergence happens at a step where HF's own top-2 logits are exactly tied (gap 0) or one fp16
  step apart (0.0156)**, so either token is a correct greedy choice. Llama-3.2-1B has three such
  near-ties in the first 128 tokens of its prompts, which is why it shows 9/20.
- **Perplexity** ([tests/ppl.py](tests/ppl.py)): WikiText-2 test, 16 windows of 1,024 tokens, both
  engines on the same tokens. This checks the full distribution at every position, not just the
  argmax.
- **Kernels** ([tests/test_kernels.py](tests/test_kernels.py)): every kernel against PyTorch
  references at head_dim 64 and 128, every GQA group from 1 to 8, all 27 GEMM configs and their
  split-K variants, ragged batches, chunked-prefill offsets and 4,000-token contexts. RMSNorm and
  RoPE are bit-exact against HF's fp16 op order.
- Llama-3.2-3B, Mistral-7B and Llama-3.1-8B don't fit next to an HF reference copy on an 11 GB
  card. Llama-3.2-3B passed the gate (20/20) and Llama-3.1-8B passed it on the A6000 in
  development runs; their recorded results are part of the pending A6000 run.

## What's inside

| Kernel | What it does | Notes |
|---|---|---|
| `gemv_kernel` (decode, batch ≤ 4) | `x @ Wᵀ` streaming weights with 16-byte loads | RMSNorm fused as a prologue; epilogues: residual add, SwiGLU (gate/up rows interleaved in groups of 16 so one warp owns a pair), and **QKV + bias + RoPE + KV-cache write** (a warp computes rows d and d+D/2 of a head and rotates them in registers). Grid sized from the occupancy API. 520–580 GB/s on a 2080 Ti. |
| `gemm_skinny_kernel` (NVIDIA, decode batch 5–64) | Weight-streaming tensor-core GEMM | Organized like the GEMV: each warp streams its weight rows from DRAM once, straight into `mma.sync` B fragments, with no shared-memory staging of weights. The k index inside each 32-wide window is permuted so a lane's 16-byte load *is* its B fragment; the activations get the same permutation. Up to 4 k-slices per block add loads in flight. |
| `gemm_mma_kernel` (NVIDIA, prefill) | Tensor-core GEMM on raw `mma.sync` + `ldmatrix` | 3-stage `cp.async` pipeline on Ampere, register-staged double buffering on Turing (no `cp.async` there): one barrier per K tile, and the next tile's first fragments are loaded before the last k16 step's MMAs. Fragment double-buffering, grouped tile rasterization so co-resident blocks share A and W rows in L2. 13 tile shapes up to 256×128, 4 or 8 warps. Epilogues straight from accumulator registers. |
| `gemm_kernel` (portable / AMD) | Same, on WMMA 16×16×16 (rocWMMA on AMD) | 6 tile configs. |
| All GEMMs | **Autotuned per GPU** | The first call for each shape times every valid (config × split-K factor) with the epilogue the caller uses, re-times the three fastest back to back, and caches the winner. Split-K reduces in the last block per tile (atomic counter), so it costs no extra launch. `HIPINFER_GEMM_VERBOSE=1` prints the choices. |
| `flash_prefill_mma_kernel` (NVIDIA) | Causal FlashAttention-2 with everything in registers | S lands in accumulator registers, the online softmax runs on them, P is repacked in registers as the A operand of P·V (two adjacent m16n8 key tiles *are* one m16n8k16 A fragment), O stays in registers. GQA, chunked prefill from the KV cache, K/V tiles software-pipelined. head_dim 64 and 128. Turing runs each m16n8k16 as two m16n8k8. |
| `decode_attn_mma_kernel` (NVIDIA) | Split-K ("flash-decoding") attention on tensor cores | The query heads of a GQA group are the rows of an m16 tile, so one pass over K/V serves the group. The split count adapts to the context length *on the device*, and the last split block merges the partials. |
| `flash_prefill_kernel`, `decode_attn_kernel` (portable / AMD) | Same algorithms on WMMA / scalar fp32 | Used on AMD; `HIPINFER_FLASH_WMMA=1` / `HIPINFER_DECODE_SCALAR=1` force them on NVIDIA. |
| `rmsnorm_kernel`, `rope_cache_kernel`, `embed_kernel`, `argmax_kernel` | standalone versions for the prefill and batched paths | argmax also advances the device-side position and token history |

**Engine** ([src/api.cu](src/api.cu)): loads one flat fp16 weight file written by
[scripts/export.py](scripts/export.py), which also stores HF's exact RoPE frequencies (so Llama-3
scaling needs no engine code). Q/K/V are fused into one matrix, and gate/up are interleaved.
Prefill runs chunked; decode runs over B sequence slots. A decode step keeps all its state on the
device (next token, position, history). It is captured once per batch size as a HIP/CUDA graph
and replayed, so generation needs no host round trip per token.

**Matching HF exactly.** Wherever PyTorch rounds an intermediate to fp16, the kernels round at
the same point: RMSNorm output, each RoPE product, projection outputs before the residual add,
SiLU. One trap: nvcc contracts `__hmul` + `__hsub` into an fp16 FMA, which moves RoPE by
1 ulp. The fix is to round each product explicitly.

**Portability** ([src/hip_compat.h](src/hip_compat.h)):

- **Toolchains:** `hipcc` uses the real HIP headers; plain `nvcc` maps the `hip*` API onto CUDA.
- **Wavefront width:** set per device target at compile time (64 on CDNA/gfx9, 32 on RDNA and NVIDIA). Every kernel derives its warps-per-block from it.
- **Matrix cores:** the portable kernels go through the WMMA API (`nvcuda::wmma` or `rocwmma`). The fastest NVIDIA kernels use `mma.sync` PTX directly and have no AMD equivalent yet.

## Where it is not fast (yet)

- **Prefill GEMM on Turing: 0.92–0.95× cuBLAS** at 2,048 rows on Llama-3.2-3B's shapes, up from
  0.79–0.81× before the 2026-10-15 change and 0.70× in the WMMA-only build. The other models'
  ratios in SUMMARY.md predate the change. That decides prefill on models of 3B parameters and up,
  and it is why Llama-3.2-3B is still 1.04× slower than llama.cpp. The remaining loss is mostly
  the last wave of each GEMM: one 256×128 block per SM on 68 SMs.
- **Prefill GEMM on Ampere: 0.71–0.85× cuBLAS** on the A6000 at the same shapes. I measured this
  on 2026-10-15, with the same result before and during the Turing work. The Ampere `cp.async`
  path was not part of that work.
- **Batched decode at 32–64 sequences:** the projections are ~1.15–1.2× slower than cuBLAS at
  these shapes. At 16 sequences the gap is ~1.1×.
- **Attention on Ampere is at parity, not ahead** (previous build: prefill 0.91–1.10×
  FlashAttention-2, decode 0.92–1.15× SDPA). On the 2080 Ti, where PyTorch has no flash kernel,
  it is 8–53× faster than SDPA.
- **AMD runs the portable kernels** (WMMA GEMM and attention, scalar decode attention). MFMA versions
  of the register-resident kernels are not written, and nothing has been run on AMD hardware.
- fp16 only (no quantization); contiguous per-slot KV cache (no paging); greedy sampling; no
  Mixture-of-Experts or sliding-window attention. A sliding-window model (Mistral-7B-v0.1: 4096
  tokens) loads only with `max_seq` within its window, where full attention is equivalent; the
  benchmarks cap its context there. Weight files exported before this check carry no window and
  should be re-exported.

## Run it

**Linux + NVIDIA (or AMD with ROCm PyTorch):**

```bash
pip install torch transformers datasets nvidia-ml-py
scripts/run_all.sh mygpu
```

`run_all.sh` builds for the detected GPU and runs the kernel tests once. Then, for each model that
fits in GPU memory, it exports the weights, runs the HF token gate, WikiText-2 perplexity and the
benchmarks, leaving logs and JSON in `results/mygpu/<model>/`. Pass model tags to pick a subset:

```bash
scripts/run_all.sh mygpu tinyllama llama32-1b qwen25-1.5b
```

Tags: `tinyllama`, `qwen25-0.5b`, `llama32-1b`, `qwen25-1.5b`, `llama32-3b`, `mistral-7b`,
`llama31-8b`. A model too large to sit next to an HF reference copy gets hipinfer-only benchmarks.
`GEMM_SWEEP=1` also times every GEMM config. Compare runs with `python scripts/summarize.py results`.
Engine baselines (llama.cpp, vLLM, torch.compile) come from [scripts/run_baselines.sh](scripts/run_baselines.sh).

**Multi-GPU:** with two or more GPUs visible, `run_all.sh` adds a tensor-parallel stage. For each model it runs the TP-vs-1-GPU test, the all-reduce/decode/prefill/batched benchmarks, and perplexity vs HF for models that only fit sharded. Results land in `results/<gpu>/<model>/tp2*.json`. The stage takes `TP=2` (default), `TP_MODELS="llama32-3b llama31-8b"` and `SKIP_TP=1`. It builds against NCCL from the pip wheel when present. To run it directly:

```bash
python tests/test_tp.py --weights llama32-3b.bin --tp 2
python scripts/bench_tp.py --weights llama31-8b.bin --tp 2 --no-single --out tp2.json
```

```python
eng = hipinfer.Engine("llama31-8b.bin", tp=2)   # devices=[0, 1] by default
eng.generate(prompt_ids, 128)
```

**Kaggle (free 2x T4):** [hipinfer_kaggle.ipynb](hipinfer_kaggle.ipynb) runs the single-GPU suite and then tensor parallelism over both T4s, including Llama-3.1-8B (16 GB of fp16 weights, which does not fit one 15 GB T4).

**Colab:** open [hipinfer_colab.ipynb](hipinfer_colab.ipynb) (sources are embedded, nothing to
clone), pick a GPU runtime, and run all cells. It downloads `hipinfer_results_<gpu>.zip` at the end.

**AMD:** the same `scripts/run_all.sh` on any ROCm 6/7 machine with PyTorch-ROCm and rocWMMA. To
only compile (no GPU needed), `pip install --index-url https://rocm.nightlies.amd.com/v2/gfx94X-dcgpu/ "rocm[libraries,devel]"`
and `make hip HIPCC=$(rocm-sdk path --root)/bin/hipcc`.

## Where to benchmark next

| Option | Arch / bandwidth | Why run it |
|---|---|---|
| **Colab T4** (free) | Turing sm_75, 320 GB/s | Run first: free, same arch as the 2080 Ti, and a lower-bandwidth point. Fits the 0.5–1.5B models. |
| **Colab A100** | Ampere sm_80, 1.5–2 TB/s | Most informative. The floor drops to ≈ 1–1.4 ms/token for 1B models, so launch latency shows up, and it fits Mistral-7B and Llama-3.1-8B. |
| **Colab L4** | Ada sm_89, 300 GB/s | Newest-but-cheap NVIDIA arch; checks the sm_89 build and autotuner; fits Llama-3.2-3B. |
| **Colab G4** | Blackwell (RTX PRO 6000) | Checks the forward-compat path (compute_80 PTX JIT if Colab's nvcc predates sm_120) |
| **Kaggle 2x T4** (free) | 2x Turing, PCIe | Second multi-GPU data point; checks whether its T4s have a peer path (the P2P all-reduce staging has not run anywhere yet). |
| Any 2-8 GPU NVLink box (e.g. 2x A100 / H100 hourly) | NVLink P2P | Where tensor parallelism should shine: P2P staging instead of host memory, and tp=4/8. |
| Colab CPU, TPU v5e/v6e | — | Can't run it: the kernels are GPU code; TPUs only run XLA |
| **AMD MI300X** (AMD Developer Cloud, or an hourly MI300X from a GPU cloud) | CDNA3 gfx942, 5.3 TB/s, wave64 | **The one that matters most for the HIP claim**: nothing has run on AMD hardware yet. One hour is enough for `run_all.sh`. |
