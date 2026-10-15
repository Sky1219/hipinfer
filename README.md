# hipinfer

An LLM inference engine I wrote from scratch in HIP. On one RTX 2080 Ti it runs Llama, Qwen and
TinyLlama models in fp16 **6–48 % faster than llama.cpp at batch-1 decode**, and its outputs match
Hugging Face transformers (WikiText-2 perplexity within 0.003 %).

The same source tree builds with `hipcc` for AMD GPUs and with plain `nvcc` for NVIDIA GPUs. There
are no vendor libraries in the hot path: the GEMV, GEMM, RMSNorm, RoPE and FlashAttention-style
kernels are all hand-written, the KV cache is contiguous, and the decode loop is captured once as a
CUDA/HIP graph and replayed.

**Charts and full results: [sky1219.github.io/hipinfer](https://sky1219.github.io/hipinfer/)** (rebuilt
from `results/` on every push by [scripts/build_site.py](scripts/build_site.py))

## What it runs, and where

Models: Llama 2 / 3 / 3.1 / 3.2, TinyLlama, Mistral and Qwen2 / 2.5. In practice that means head_dim
64 or 128, any GQA group up to 8, tied or untied embeddings, Llama-3 RoPE scaling and QKV bias.

Hardware:

- **NVIDIA, tested:** RTX 2080 Ti (Turing, sm_75), on one card and on two with tensor parallelism,
  and RTX A6000 (Ampere, sm_86). All kernel tests pass on both.
- **AMD, compile-checked only.** It builds for MI200 (gfx90a), MI300 (gfx942) and RDNA3 (gfx1100)
  with ROCm 7 and rocWMMA, but **it has never run on AMD hardware**. If you have an MI300X for an
  hour, see [Help wanted](#help-wanted).

> **How fresh are the numbers?** On 2026-10-15 I rewrote the Turing prefill GEMM
> ([write-up](docs/turing-prefill-gemm.md)). Since then I have re-run only the 2080 Ti kernel tests
> (915 checks, all passing) and Llama-3.2-3B prefill. Every other number here comes from the build
> before that change. Re-running the rest (other models' benchmarks, the HF token gate,
> perplexity, multi-GPU, the A6000 and the AMD compile) is next. The A6000 columns are older still:
> TinyLlama only, from an earlier build.

## Quick start

You need Linux, an NVIDIA GPU with the CUDA toolkit (or ROCm for AMD), and PyTorch.

```bash
pip install torch transformers
make                     # NVIDIA -> build/libhipinfer.so   (AMD: make hip)
python scripts/export.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --out tinyllama.bin
```

`export.py` converts a Hugging Face checkpoint into the one flat fp16 file the engine loads. Then:

```python
import sys; sys.path.insert(0, "scripts")
import hipinfer
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
eng = hipinfer.Engine("tinyllama.bin")            # tp=2 shards it over two GPUs
ids = eng.generate(tok("The capital of France is").input_ids, 32)
print(tok.decode(ids))
```

Generation is greedy only for now.

### The full suite

To reproduce everything on your own GPU:

```bash
pip install torch transformers datasets nvidia-ml-py
scripts/run_all.sh mygpu                                    # every model that fits
scripts/run_all.sh mygpu tinyllama llama32-1b qwen25-1.5b   # or a subset
```

For each model it exports the weights and runs the HF token gate, WikiText-2 perplexity and the
benchmarks, leaving logs and JSON in `results/mygpu/<model>/`. Before that it builds for the
detected GPU and runs the kernel tests once.

- **Model tags:** `tinyllama`, `qwen25-0.5b`, `llama32-1b`, `qwen25-1.5b`, `llama32-3b`,
  `mistral-7b`, `llama31-8b`. A model too big to sit next to an HF reference copy gets
  hipinfer-only benchmarks.
- **Options:** `GEMM_SWEEP=1` also times every GEMM config.
- **Comparing runs:** `python scripts/summarize.py results`. The llama.cpp / vLLM / torch.compile
  baselines come from [scripts/run_baselines.sh](scripts/run_baselines.sh).
- **Multi-GPU:** with two or more GPUs visible, the script adds a tensor-parallel stage (options
  below).

No GPU of your own? [hipinfer_colab.ipynb](hipinfer_colab.ipynb) has the sources embedded: pick a
GPU runtime, run all cells, and download `hipinfer_results_<gpu>.zip` at the end.
[hipinfer_kaggle.ipynb](hipinfer_kaggle.ipynb) does the same on Kaggle's free 2× T4, including
tensor parallelism and Llama-3.1-8B (16 GB of weights, which doesn't fit one 15 GB T4).

**AMD:** `scripts/run_all.sh` works the same on any ROCm 6/7 machine with PyTorch-ROCm and rocWMMA.
To compile without a GPU:
`pip install --index-url https://rocm.nightlies.amd.com/v2/gfx94X-dcgpu/ "rocm[libraries,devel]"`,
then `make hip HIPCC=$(rocm-sdk path --root)/bin/hipcc`.

## Results

RTX 2080 Ti (616 GB/s), fp16 weights, greedy, batch 1 unless noted. llama.cpp is `llama-bench` on an
F16 GGUF (no quantization), best of flash-attention on/off. HF is `transformers` in eager mode.

| model | decode ms/token | % of bandwidth floor | vs llama.cpp | vs HF | prefill 2,000 tok vs llama.cpp | peak batched tok/s |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | **2.33** | 69 % | **1.48× faster** | 13.9× | **1.46× faster** | 11,800 (B=64) |
| TinyLlama-1.1B | **4.13** | 82 % | **1.15× faster** | 6.7× | **1.04× faster** | 7,550 (B=64) |
| Llama-3.2-1B | **4.70** | 86 % | **1.11× faster** | 4.3× | 1.08× slower | 7,470 (B=64) |
| Qwen2.5-1.5B | **6.17** | 81 % | **1.17× faster** | 6.0× | **1.05× faster** | 4,500 (B=64) |
| Llama-3.2-3B | **12.02** | 87 % | **1.06× faster** | (HF copy doesn't fit) | 1.04× slower ✱ | 906 (B=16) |

✱ Re-measured after the GEMM change: 273.6 ms against llama.cpp's 261.8 ms. It was 318.5 ms (1.22× slower).

**Decode is close to the hardware limit.** At batch 1, every token has to read all the weights
once, so the fastest possible decode is the bytes read per token (weights plus KV cache) divided by
peak bandwidth. hipinfer reaches 81–87 % of that floor on every model of 1B parameters or more, and
holds it out to 8k-token contexts (86 % at 8,192 tokens on Llama-3.2-3B). llama.cpp reaches 47–83 %
on the same models. Qwen2.5-0.5B is lower, at 69 %: its matrices are small enough that launch and
synchronization overhead start to show.

(An earlier version of this README counted the whole embedding table in the floor, even though
decode never reads it. By that definition TinyLlama was "87 %" at the same 4.1 ms.)

### Against other engines on TinyLlama

Same GPU, fp16 weights, batch 1. The other engines were measured in an earlier session on the same
machine.

| decode ms/token, ctx 128 → 1024 | RTX 2080 Ti | RTX A6000 (older build) |
|---|---|---|
| **hipinfer** | **4.13 → 4.21** | **3.41 → 3.47** |
| llama.cpp `a0e6dc7`, F16 GGUF, CUDA, flash-attn | 4.77 → 4.79 | 3.76 → 3.76 |
| vLLM 0.31 (V1, CUDA graphs) | not run | 3.96 → 3.95 |
| torch.compile, gpt-fast-style static KV, `reduce-overhead` | 5.24 → 6.84 | 4.23 → 5.47 |
| HF transformers `StaticCache` + `torch.compile` | 5.51 → 7.09 | 4.34 → 5.56 |

| prefill ms, 128 / 512 / 2000 tokens | RTX 2080 Ti | RTX A6000 (older build) |
|---|---|---|
| **hipinfer** | **9.3** / **28.6** / **111** | **6.4** / **17.7** / 67 |
| llama.cpp (best of flash-attn on/off, ubatch 512/2048) | 11.2 / 29.0 / 115 | 9.1 / 22.8 / 87 |
| torch.compile (inductor + cuBLAS) | 16.2 / 34.8 / 125 | 8.9 / 19.5 / **60** |

### What I take from this

- **At batch-1 decode, hipinfer was the fastest engine I measured on every model.** The lead over
  llama.cpp is 6–17 % on the 1–3B models and 48 % on Qwen2.5-0.5B. It shrinks as models grow,
  because everyone converges on the same bandwidth floor. I think the gain comes from fewer kernel
  launches (5 per layer, with norm, RoPE, cache write and SwiGLU fused in) and from the GEMV's
  bandwidth. I haven't profiled llama.cpp, so I can't say exactly where its extra time goes.
- **Prefill is faster up to ~1.5B parameters, but not at 3B.** Prefill on a 3B model is bound by the
  GEMMs, and mine run at 0.92–0.95× cuBLAS. That leaves Llama-3.2-3B 12 ms (1.04×) behind
  llama.cpp, and Llama-3.2-1B is 1.08× behind on the older build. I stopped there, and the gap is
  not closed.
- **Keep the scope in mind.** These are batch-1 latency numbers on small models. vLLM and llama.cpp
  are built for far more: many concurrent requests, quantization, hundreds of architectures. Batch 1
  is just where a small engine can be checked against a hard floor.

The full tables are in [results/SUMMARY.md](results/SUMMARY.md) (from the build before the GEMM
change). They cover decode latency vs context length, batched throughput with power draw and energy
per token, GEMM vs cuBLAS at every model's shapes, and llama.cpp comparisons for each model.

## Multi-GPU: tensor parallelism

`Engine(path, tp=2)` (in C: `hi_load_tp`) shards a model across GPUs in one process, with one worker
thread per GPU and the same API as one GPU. The split is Megatron-style:

- **Attention:** QKV is split by heads, and each GPU keeps the KV cache for its own heads. O is
  split by rows, followed by an all-reduce.
- **MLP:** gate/up are split by columns, and down by rows with an all-reduce.
- **lm_head:** split by vocabulary. Greedy argmax reduces one 64-bit key per row (order-preserving
  logit bits plus the inverted index), so ties break exactly as they do on one GPU.
- **Replicated:** the embedding and the norms, so after each all-reduce every GPU holds the same
  residual stream.

I tested it on about the worst multi-GPU box you can build: two RTX 2080 Tis on different CPU
sockets, PCIe 3.0, and **no peer-to-peer path** (`nvidia-smi topo`: SYS). Every byte between the
GPUs goes through host memory.

| model | decode ms/token, 1 GPU → 2 GPUs | prefill 2,000 tok, ms | batched B=64, tok/s |
|---|---|---|---|
| TinyLlama-1.1B | 4.14 → **2.67** (1.55×) | 117 → **88** (1.34×) | 7.4k → **8.5k** |
| Llama-3.2-1B | 4.71 → **2.79** (1.69×) | 115 → **76** (1.52×) | 7.5k → **9.5k** |
| Qwen2.5-1.5B | 6.18 → **3.85** (1.61×) | 159 → **110** (1.45×) | 4.5k → **5.4k** |
| Llama-3.2-3B | 11.99 → **6.84** (1.75×) | 329 → **199** (1.65×) | B=16 max on 1 GPU → **3.2k** at B=64 |
| Llama-3.1-8B | doesn't fit one card → **14.6** (83 % of the per-GPU floor) | **418** | **858** at B=16 |
| Mistral-7B-v0.3 | doesn't fit one card → **14.0** (83 %) | **422** | **1,406** at B=32 |

Three things make it work:

- **A one-shot all-reduce kernel** ([src/kernels.cuh](src/kernels.cuh)) handles decode-sized
  messages instead of NCCL. Each GPU stages its chunk where the others can read it (in the peer's
  memory over P2P, or in pinned host memory when there's no peer path, as here). Then it raises a
  system-scope release flag on every GPU and waits on acquire. Finally every GPU sums all the copies
  itself, in the same order and in fp32. Because the order is fixed, every GPU ends up bitwise
  identical, and the NCCL and one-shot backends give bitwise-identical model outputs.
  - Each block is an independent channel with its own staging buffers and call counter. So the
    grid can be sized to the message (one block for a 4 KB decode row), and the whole thing can
    live inside a CUDA graph.
  - It takes 8 µs vs NCCL's 12 µs at 4 KB, and 58 vs 85 µs at 256 KB. That's 5–9 % lower batch-1
    decode latency than NCCL alone, and 10–15 % more batched throughput at B=64.
  - By default it handles messages up to 512 KB and NCCL takes the bigger ones. NCCL is optional
    at build time; without it, everything goes one-shot.
- **Prefill all-reduces overlap compute.** The prompt is split into two halves of rows. One half's
  all-reduce runs on a second stream while the other half computes, and the second half's attention
  reads the KV cache the first half just wrote. Without this, PCIe makes 2-GPU prefill *slower*
  than one GPU (TinyLlama: 124 ms vs 117); with it, prefill is 1.34–1.65× faster.
- **Per-GPU state is `thread_local`.** That covers autotuned GEMM choices, split-K workspaces and
  kernel attributes, so each worker thread drives its GPU through the unchanged single-GPU code path.

**Is it correct?** [tests/test_tp.py](tests/test_tp.py) runs 2 GPUs against the same weights on one
GPU, for every backend:

- **Greedy tokens:** 13 runs per model (5 single-sequence prompts plus a batch of 8) give identical
  tokens. The only exceptions are steps where the single-GPU engine's own top two logits are within
  one fp16 step (a gap of 0.0078).
- **Backends:** the all-reduce self-test is exact, and the one-shot and NCCL backends give
  bitwise-identical logits and tokens.
- **The 7–8B models,** which only run sharded, are checked against HF transformers spread over the
  same two GPUs, on WikiText-2 (8 windows × 1,024 tokens):

| model (sharded ×2) | PPL hipinfer | PPL HF | top-1 agreement | mean KL(HF, hipinfer) |
|---|---|---|---|---|
| Llama-3.1-8B | 8.1505 | 8.1511 | 99.89 % | 1.2e-5 |
| Mistral-7B-v0.3 | 7.3269 | 7.3267 | 99.91 % | 6.7e-6 |

Testing TP also turned up a real single-GPU bug. The decode attention's split counters sat at an
offset that depended on the batch size. So a decode step at a *smaller* batch, right after a larger
one, could read the larger step's partial sums as counters. TP halves the KV heads per GPU, which
means more splits per launch, and that's what exposed it. It's fixed, and `test_decode_work_reuse`
covers it.

**Limits:**

- **Small models gain least.** On the 7–8B models, sharded decode reaches 83 % of the per-GPU
  bandwidth floor, the same as one GPU on the 1–3B models. On the 1–3B models sharded it's 63–77 %:
  each GPU's matrices are half the size, but the launches, the replicated norms and the two
  all-reduces per layer aren't.
- **Untested paths:** the P2P staging path is written but has never run, because neither of my
  machines has GPUs with peer access. I've only tested tp=2. The code supports up to 8 GPUs, as long
  as heads, KV heads and the intermediate size divide evenly.

**Running it:** in `run_all.sh`, the tensor-parallel stage runs the TP-vs-1-GPU test, the
all-reduce, decode, prefill and batched benchmarks, and perplexity vs HF for models that only fit
sharded. Results land in `results/<gpu>/<model>/tp2*.json`. It takes `TP=2` (the default),
`TP_MODELS="llama32-3b llama31-8b"` and `SKIP_TP=1`, and builds against NCCL from the pip wheel when
present. To run it by hand:

```bash
python tests/test_tp.py --weights llama32-3b.bin --tp 2
python scripts/bench_tp.py --weights llama31-8b.bin --tp 2 --no-single --out tp2.json
```

## Correctness

Fast is only interesting if the answers are right, so every model is checked against HF transformers
in three ways. These results are from the build before the GEMM change.

| model | greedy runs token-identical | PPL hipinfer | PPL HF | top-1 agreement | mean KL(HF, hipinfer) |
|---|---|---|---|---|---|
| TinyLlama-1.1B | 20/20 | 8.5149 | 8.5151 | 99.90 % | 6.6e-6 |
| Qwen2.5-0.5B | 18/20 | 13.4851 | 13.4851 | 99.79 % | 1.6e-5 |
| Llama-3.2-1B | 9/20 | 10.4633 | 10.4631 | 99.81 % | 1.3e-5 |
| Qwen2.5-1.5B | 17/20 | 9.4536 | 9.4535 | 99.86 % | 1.6e-5 |

- **Greedy token gate** ([tests/gate.py](tests/gate.py)): 7 prompts × 128 tokens, single-sequence
  with and without CUDA graphs, plus a batch of 6 sequences at different lengths (20 runs in all).
  Every divergence happens at a step where HF's own top two logits are exactly tied (gap 0) or one
  fp16 step apart (0.0156). At such a step either token is a correct greedy choice. Llama-3.2-1B
  hits three of those near-ties in the first 128 tokens of its prompts, which is why it shows 9/20.
- **Perplexity** ([tests/ppl.py](tests/ppl.py)): WikiText-2 test, 16 windows of 1,024 tokens, both
  engines on the same tokens. This checks the whole distribution at every position, not just the
  top token.
- **Kernel tests** ([tests/test_kernels.py](tests/test_kernels.py)): every kernel against PyTorch
  references at head_dim 64 and 128, every GQA group from 1 to 8, all 27 GEMM configs and their
  split-K variants, ragged batches, chunked-prefill offsets and 4,000-token contexts. RMSNorm and
  RoPE are bit-exact against HF's fp16 order of operations.

Llama-3.2-3B, Mistral-7B and Llama-3.1-8B don't fit next to an HF reference copy on an 11 GB card.
In development runs, Llama-3.2-3B passed the gate 20/20, and Llama-3.1-8B passed it on the A6000.
Their recorded results will come with the A6000 run.

## How it works

### Kernels

| Kernel | What it does | Notes |
|---|---|---|
| `gemv_kernel` (decode, batch ≤ 4) | `x @ Wᵀ`, streaming weights with 16-byte loads | RMSNorm fused as a prologue. Epilogues: residual add, SwiGLU (gate/up rows interleaved in groups of 16 so one warp owns a pair), and **QKV + bias + RoPE + KV-cache write** (a warp computes rows d and d+D/2 of a head and rotates them in registers). Grid sized from the occupancy API. 520–580 GB/s on a 2080 Ti. |
| `gemm_skinny_kernel` (NVIDIA, decode batch 5–64) | Weight-streaming tensor-core GEMM | Organized like the GEMV: each warp streams its weight rows from DRAM once, straight into `mma.sync` B fragments, with no shared-memory staging of weights. The k index inside each 32-wide window is permuted so a lane's 16-byte load *is* its B fragment, and the activations get the same permutation. Up to 4 k-slices per block add loads in flight. |
| `gemm_mma_kernel` (NVIDIA, prefill) | Tensor-core GEMM on raw `mma.sync` + `ldmatrix` | 3-stage `cp.async` pipeline on Ampere. On Turing (no `cp.async`), register-staged double buffering with one barrier per K tile, and the next tile's first fragments loaded before the last k16 step's MMAs ([write-up](docs/turing-prefill-gemm.md)). Fragment double-buffering, and grouped tile rasterization so co-resident blocks share A and W rows in L2. 13 tile shapes up to 256×128, 4 or 8 warps. Epilogues straight from accumulator registers. |
| `gemm_kernel` (portable / AMD) | The same, on WMMA 16×16×16 (rocWMMA on AMD) | 6 tile configs. |
| All GEMMs | **Autotuned per GPU** | The first call for each shape times every valid config × split-K factor with the epilogue the caller uses, re-times the three fastest back to back, and caches the winner. Split-K reduces in the last block per tile (atomic counter), so it costs no extra launch. `HIPINFER_GEMM_VERBOSE=1` prints the choices. |
| `flash_prefill_mma_kernel` (NVIDIA) | Causal FlashAttention-2, entirely in registers | S lands in accumulator registers, the online softmax runs on them, and P is repacked in registers as the A operand of P·V (two adjacent m16n8 key tiles *are* one m16n8k16 A fragment). O stays in registers. GQA, chunked prefill from the KV cache, software-pipelined K/V tiles, head_dim 64 and 128. Turing runs each m16n8k16 as two m16n8k8. |
| `decode_attn_mma_kernel` (NVIDIA) | Split-K ("flash-decoding") attention on tensor cores | The query heads of a GQA group are the rows of an m16 tile, so one pass over K/V serves the whole group. The split count adapts to the context length *on the device*, and the last split block merges the partials. |
| `flash_prefill_kernel`, `decode_attn_kernel` (portable / AMD) | The same algorithms on WMMA / scalar fp32 | Used on AMD. `HIPINFER_FLASH_WMMA=1` / `HIPINFER_DECODE_SCALAR=1` force them on NVIDIA. |
| `rmsnorm_kernel`, `rope_cache_kernel`, `embed_kernel`, `argmax_kernel` | Standalone versions for the prefill and batched paths | argmax also advances the on-device position and token history. |

### The engine

[src/api.cu](src/api.cu) loads one flat fp16 weight file written by
[scripts/export.py](scripts/export.py). The export also stores HF's exact RoPE frequencies, so
Llama-3 scaling needs no engine code. Q/K/V are fused into one matrix, and gate/up are interleaved.

Prefill runs in chunks, and decode runs over B sequence slots. A decode step keeps all its state on
the device (next token, position, history), so it can be captured once per batch size as a
HIP/CUDA graph and replayed. Generation never has to round-trip to the host per token.

### Matching HF bit for bit

Wherever PyTorch rounds an intermediate to fp16, the kernels round at the same point: the RMSNorm
output, each RoPE product, the projection outputs before the residual add, and SiLU. One trap I hit:
nvcc contracts `__hmul` + `__hsub` into an fp16 FMA, which moves RoPE by 1 ulp. The fix is to round
each product explicitly.

### One source, two vendors

[src/hip_compat.h](src/hip_compat.h) makes the same source build for both vendors:

- **Compilers:** under `hipcc` the code uses the real HIP headers. Under plain `nvcc` it maps the
  `hip*` API onto CUDA.
- **Warp width:** the wavefront width is set per target at compile time (64 on CDNA/gfx9, 32 on RDNA
  and NVIDIA), and every kernel derives its warps per block from it.
- **Matrix units:** the portable kernels go through WMMA (`nvcuda::wmma` or `rocwmma`). The fastest
  NVIDIA kernels use `mma.sync` PTX directly and have no AMD equivalent yet.

## Where it's not fast (yet)

- **Prefill GEMM on Turing: 0.92–0.95× cuBLAS** at 2,048 rows on Llama-3.2-3B's shapes. That's up
  from 0.79–0.81× before the 2026-10-15 change and 0.70× in the WMMA-only build. This decides
  prefill on 3B-and-up models, and it's why Llama-3.2-3B is still 1.04× behind llama.cpp. Most of
  what's left is each GEMM's last wave: one 256×128 block per SM on 68 SMs. The other models'
  ratios in SUMMARY.md are from before the change.
- **Prefill GEMM on Ampere: 0.71–0.85× cuBLAS** on the A6000 at the same shapes (measured
  2026-10-15). The Ampere `cp.async` path wasn't part of the Turing work.
- **Batched decode at 32–64 sequences:** the projections are ~1.15–1.2× slower than cuBLAS at these
  shapes, and ~1.1× at 16 sequences.
- **Attention on Ampere is at parity, not ahead** (older build: prefill 0.91–1.10× FlashAttention-2,
  decode 0.92–1.15× SDPA). On the 2080 Ti, where PyTorch has no flash kernel, it's 8–53× faster than
  SDPA.
- **AMD runs only the portable kernels** (WMMA GEMM and attention, scalar decode attention). MFMA
  versions of the register-resident kernels aren't written yet, and nothing has run on AMD hardware.
- **Not supported:** quantization (it's fp16 only), a paged KV cache (each slot's cache is
  contiguous), sampling other than greedy, Mixture-of-Experts and sliding-window attention.
- **Sliding-window models:** Mistral-7B-v0.1 (4,096-token window) loads only with `max_seq` within
  its window, where full attention gives the same result, and the benchmarks cap its context there.
  Weight files exported before this check carry no window and should be re-exported.

## Help wanted

Benchmark runs on hardware I don't have would be the most useful contribution:

- **AMD MI300X (CDNA3, gfx942, 5.3 TB/s, wave64)** matters most. The whole point of HIP is that this runs on AMD, and
  it never has. One hour of `scripts/run_all.sh` on the AMD Developer Cloud or an hourly GPU cloud
  would settle it.
- **An NVLink box** (2× A100 / H100) is where tensor parallelism should shine: P2P staging instead
  of host memory, plus tp=4/8. The P2P path has never run anywhere.
- **Colab A100 (1.5–2 TB/s)** drops the decode floor to ≈ 1–1.4 ms/token for 1B models, which
  exposes launch latency. It also fits the 7–8B models on one GPU.
- **Cheap or free checks:** a Colab T4 (same arch as the 2080 Ti, but 320 GB/s; fits the 0.5–1.5B
  models), Colab L4 (the sm_89 build and autotuner; fits Llama-3.2-3B), Colab G4 / Blackwell (the
  forward-compatible path: compute_80 PTX JIT-compiled if Colab's nvcc predates sm_120), and
  Kaggle's 2× T4 (does it have a peer path?).

TPUs and CPUs can't run it: the kernels are GPU code.

## Author

Akash Dasarraju. I wrote the code with Claude as an AI pair programmer; I reviewed every design
decision, benchmark and claim.
