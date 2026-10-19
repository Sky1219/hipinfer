# hipinfer

[![CI](https://github.com/Sky1219/hipinfer/actions/workflows/ci.yml/badge.svg)](https://github.com/Sky1219/hipinfer/actions/workflows/ci.yml)

CI runs the CPU tests against HF transformers and compiles the engine for NVIDIA (nvcc) and AMD
(hipcc) on every commit.

An LLM inference engine I wrote from scratch in HIP. On one RTX 2080 Ti it runs Llama, Qwen and
TinyLlama models in fp16 **5–47 % faster than llama.cpp at batch-1 decode** (128-token context;
3–43 % at 1,024), and its outputs match Hugging Face transformers (WikiText-2 perplexity within
0.01 %). On an A100 it is 10–27 % faster than llama.cpp and 23–31 % faster than vLLM at batch-1
decode, measured in one session, though slower than vLLM and torch.compile on long-prompt prefill.
With int8 weights, decode on the 2080 Ti is 1.6–1.8× faster than fp16 and 16–26 % faster than
llama.cpp's Q8_0. Greedy speculative decoding gives up to 1.39× on Llama-3.2-3B with output
identical to plain greedy decoding.

The same source tree builds with `hipcc` for AMD GPUs and with plain `nvcc` for NVIDIA GPUs. There
are no vendor libraries in the hot path: the GEMV, GEMM, RMSNorm, RoPE and FlashAttention-style
kernels are all hand-written, the KV cache is contiguous, and the decode loop is captured once as a
CUDA/HIP graph and replayed.

**Charts and full results: [sky1219.github.io/hipinfer](https://sky1219.github.io/hipinfer/)**

## What it runs, and where

Models: Llama 2 / 3 / 3.1 / 3.2, TinyLlama, Mistral and Qwen2 / 2.5. In practice that means head_dim
64 or 128, any GQA group up to 8, tied or untied embeddings, Llama-3 RoPE scaling and QKV bias. It
also runs one Mixture-of-Experts model, Granite-3.1-1B-A400M (32 experts, top-8), on a single GPU
([design and results](docs/moe-design.md)).

Hardware:

- **NVIDIA, tested:** RTX 2080 Ti (Turing, sm_75), on one card and on two with tensor parallelism;
  RTX A6000 (Ampere, sm_86); and A100-SXM4-40GB (Ampere, sm_80, on Google Colab). The dense-model
  kernel tests pass on all three; the MoE kernel tests have run on the 2080 Ti only. The int8,
  speculative-decoding and eviction code has run on the 2080 Ti and the A6000.
- **AMD support:** Because this project was built in HIP, it has full AMD support, and compiles
  properly. However, I had no available AMD GPUs and the cloud solutions available were either at
  capacity or too expensive.

### Thermal throttling

Due to resource constraints and degrading GPUs, runs were at times throttled due to overheating.
All affected runtimes are listed below.

- **RTX A6000.** This card holds its 300 W limit for about a minute, then reaches
  90–93 °C with the fan at 90–100 % and throttles its SM clock (normally 1,800–1,920 MHz) to as low as
  210 MHz. Every benchmark started on an idle card below 60 °C, and each one's time window was checked
  against the GPU's temperature log ([results/a6000/THERMAL.md](results/a6000/THERMAL.md)). Affected
  benchmarks, marked ⚠ in [results/SUMMARY.md](results/SUMMARY.md):

  | model | share of the benchmark thermally throttled | SM clock min / median | peak |
  |---|---|---|---|
  | Llama-3.2-1B | 33 % | 1,485 / 1,605 MHz | 90 °C |
  | Qwen2.5-1.5B | 31 % | 1,410 / 1,590 MHz | 91 °C |
  | Llama-3.2-3B | 67 % | 420 / 1,410 MHz | 93 °C |
  | Mistral-7B-v0.3 | 78 % | 210 / 450 MHz | 93 °C |
  | Llama-3.1-8B | 84 % | 210 / 368 MHz | 93 °C |

  TinyLlama-1.1B and Qwen2.5-0.5B ran without throttling. The token gate and perplexity results don't
  depend on clock speed and are valid for every model. The engine comparisons (llama.cpp, vLLM) are
  only fair when every engine ran unthrottled, which on the A6000 holds only for TinyLlama; the other
  models' comparison runs are listed in THERMAL.md and not used anywhere.
- **RTX 2080 Ti.** Under sustained load their 250 W power cap holds the clock at 1.4–1.55 GHz.
  That affects Llama-3.2-3B prefill in the full suite: 289.7 ms against 273.6 ms in a short isolated
  run (marked ✱ in the results table). The int8 and speculative-decoding runs also logged the
  driver's throttle reasons. On the longer benchmarks the cards reached their ~84 °C thermal target
  too, so the comparisons where the two sides throttled unevenly were rerun from a cool card. The
  reruns agree with the first run within 2 % ([results/2080ti/MERGED_RUN.md](results/2080ti/MERGED_RUN.md)).

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
| Qwen2.5-0.5B | **2.34** | 69 % | **1.47× faster** | 11.5× | **1.56× faster** | 11,900 (B=64) |
| TinyLlama-1.1B | **4.14** | 81 % | **1.15× faster** | 7.6× | **1.14× faster** | 7,750 (B=64) |
| Llama-3.2-1B | **4.75** | 85 % | **1.10× faster** | 3.8× | even (101.1 vs 100.9 ms) | 7,320 (B=64) |
| Qwen2.5-1.5B | **6.21** | 81 % | **1.16× faster** | 6.5× | **1.11× faster** | 4,330 (B=64) |
| Llama-3.2-3B | **12.11** | 87 % | **1.05× faster** | (HF copy doesn't fit) | 1.11× slower ✱ | 890 (B=16) |

✱ 289.7 ms against llama.cpp's 261.8 ms inside the full benchmark suite, where the 2080 Ti's 250 W
power cap holds it at 1.4–1.55 GHz. A short isolated run on a cool GPU gives 273.6 ms (1.04×
slower).

**Decode is close to the hardware limit.** At batch 1, every token has to read all the weights
once, so the fastest possible decode is the bytes read per token (weights plus KV cache) divided by
peak bandwidth. hipinfer reaches 81–87 % of that floor on every model of 1B parameters or more, and
holds it out to 8k-token contexts (86 % at 8,192 tokens on Llama-3.2-3B). llama.cpp reaches 47–83 %
on the same models. Qwen2.5-0.5B is lower, at 69 %: its matrices are small enough that launch and
synchronization overhead start to show.

### On an A100

A100-SXM4-40GB (1,555 GB/s) on Google Colab, everything in one session with no thermal throttling
([THERMAL.md](results/a100/THERMAL.md)). hipinfer was re-run next to the other engines, so every
row compares runs a few minutes apart.

| batch-1 decode, ms/token, ctx 128 | hipinfer | % of floor | llama.cpp | vLLM 0.31 | torch.compile |
|---|---|---|---|---|---|
| TinyLlama-1.1B | **2.18** | 61 % | 2.57 (1.18× slower) | 2.83 (1.30×) | 3.14 (1.44×) |
| Llama-3.2-1B | **2.31** | 69 % | 2.63 (1.14×) | 2.86 (1.23×) | 3.08 (1.33×) |
| Qwen2.5-1.5B | **3.21** | 62 % | 4.08 (1.27×) | 4.19 (1.31×) | not valid ✝ |
| Llama-3.2-3B | **5.60** | 74 % | 6.17 (1.10×) | not run ✱ | 6.79 (1.21×) |

| prefill, 2,000 tokens, ms | hipinfer | llama.cpp | vLLM 0.31 (incl. 1 generated token) | torch.compile |
|---|---|---|---|---|
| TinyLlama-1.1B | 35.6 | 51.7 | **29.9** | 34.8 |
| Llama-3.2-1B | 33.3 | 47.7 | 30.4 | **30.2** |
| Qwen2.5-1.5B | 43.5 | 63.1 | **35.8** | not valid ✝ |
| Llama-3.2-3B | 87.0 | 100.8 | not run ✱ | **73.8** |

✝ My torch.compile baseline left out Qwen2's QKV bias, so its tokens didn't match HF (0/32) and
its numbers are excluded. The bias is fixed in `scripts/bench_baselines.py` but hasn't been re-run.
✱ The Colab disk filled before vLLM could load Llama-3.2-3B.

Mistral-7B-v0.3 also ran on the A100 (no other engines): 11.07 ms/token, 83 % of the floor.

**Decode on the A100 is further from the floor than on the 2080 Ti** (61–74 % on the 1–3B models,
against 81–87 %). The weights stream about 2.5× faster, but each kernel launch costs about the
same, so the fixed per-launch time is a bigger share of a token. On Mistral-7B, where each kernel
moves more data, it is back to 83 %. llama.cpp and vLLM sit further below the floor still (47–67 %).

### Against other engines on TinyLlama

Same GPU, fp16 weights, batch 1. On the A6000, hipinfer, llama.cpp and vLLM ran back to back in one
session, all unthrottled; torch.compile and HF transformers, and the 2080 Ti's engines, were measured
separately on the same machines.

| decode ms/token, ctx 128 → 1024 | RTX 2080 Ti | RTX A6000 |
|---|---|---|
| **hipinfer** | **4.14 → 4.22** | **3.42 → 3.49** |
| llama.cpp `a0e6dc7`, F16 GGUF, CUDA, flash-attn | 4.77 → 4.79 | 3.78 → 3.79 |
| vLLM 0.31 (V1, CUDA graphs) | not run | 3.98 → 3.98 |
| torch.compile, gpt-fast-style static KV, `reduce-overhead` | 5.24 → 6.84 | 4.23 → 5.47 |
| HF transformers `StaticCache` + `torch.compile` | 5.51 → 7.09 | 4.34 → 5.56 |

| prefill ms, 128 / 512 / 2000 tokens | RTX 2080 Ti | RTX A6000 |
|---|---|---|
| **hipinfer** | **8.3** / **26.6** / **101** | **6.3** / **16.7** / **57.9** |
| llama.cpp (best of flash-attn on/off, ubatch 512/2048) | 11.2 / 29.0 / 115 | 9.8 / 24.5 / 89.7 |
| torch.compile (inductor + cuBLAS) | 16.2 / 34.8 / 125 | 8.9 / 19.5 / 60 |

### What I take from this

- **At batch-1 decode, hipinfer was faster than llama.cpp on every model I measured**, on all
  three GPUs, and faster than vLLM and torch.compile on every model I ran them on (TinyLlama on
  the A6000; three or four models on the A100). At a 128-token context
  the lead over llama.cpp is 5–16 % on the 1–3B models and 47 % on Qwen2.5-0.5B; at 1,024 tokens it
  is 3.0–14.4 % and 43 %. It shrinks as models grow,
  because everyone converges on the same bandwidth floor. I think the gain comes from fewer kernel
  launches (5 per layer, with norm, RoPE, cache write and SwiGLU fused in) and from the GEMV's
  bandwidth. I haven't profiled llama.cpp, so I can't say exactly where its extra time goes.
- **Prefill is faster up to ~1.5B parameters (level at Llama-3.2-1B), and slower at 3B.** Prefill on a 3B model
  is bound by the GEMMs, and mine run at 0.92–0.95× cuBLAS. That leaves Llama-3.2-3B 1.04–1.11×
  behind llama.cpp at 2,000 tokens (isolated run vs full suite) and 1.17× behind at 512 (76.7 vs
  65.4 ms), with Llama-3.2-1B level with it. The gap is not closed.
- **On the A100, long prompts are where hipinfer loses.** It still beats llama.cpp at every prompt
  length, but at 2,000 tokens vLLM and torch.compile are 1.02–1.22× faster, and HF transformers
  itself pulls ahead on long prompts (from 512 to 4,096 tokens depending on the model, after which
  hipinfer runs at 0.79–0.95× HF). All three use cuBLAS and FlashAttention-2, and on the A100 my
  GEMMs run at only 0.49–0.72× cuBLAS.
- **Keep the scope in mind.** These are batch-1 latency numbers on small models. vLLM and llama.cpp
  are built for far more: many concurrent requests, quantization, hundreds of architectures. Batch 1
  is just where a small engine can be checked against a hard floor.

The full tables are in [results/SUMMARY.md](results/SUMMARY.md). They cover decode latency vs context length, batched throughput with power draw and energy
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
| Qwen2.5-0.5B | 2.33 → **1.76** (1.33×) | 44 → **42** (1.05×) | 12.0k → **13.2k** |
| TinyLlama-1.1B | 4.15 → **2.71** (1.53×) | 105 → **84** (1.26×) | 7.5k → **8.5k** |
| Llama-3.2-1B | 4.72 → **2.81** (1.68×) | 102 → **71** (1.45×) | 7.5k → **9.5k** |
| Qwen2.5-1.5B | 6.19 → **3.87** (1.60×) | 142 → **102** (1.39×) | 4.5k → **5.7k** |
| Llama-3.2-3B | 11.99 → **6.89** (1.74×) | 287 → **182** (1.57×) | B=16 max on 1 GPU → **3.3k** at B=64 |
| Llama-3.1-8B | doesn't fit one card → **14.7** (83 % of the per-GPU floor) | **371** | **864** at B=16 |
| Mistral-7B-v0.3 | doesn't fit one card → **14.0** (83 %) | **366** | **1,408** at B=32 |

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
  - It takes 8.5 µs vs NCCL's 12 µs at 4 KB, and 64 vs 85 µs at 256 KB. That's 1.5–15.2 % lower batch-1
    decode latency than NCCL alone, and 9–27 % more batched throughput at B=64.
  - By default it handles messages up to 512 KB and NCCL takes the bigger ones. NCCL is optional
    at build time; without it, everything goes one-shot.
- **Prefill all-reduces overlap compute.** The prompt is split into two halves of rows. One half's
  all-reduce runs on a second stream while the other half computes, and the second half's attention
  reads the KV cache the first half just wrote. Without this, PCIe makes 2-GPU prefill *slower*
  than one GPU (TinyLlama: 125 ms vs 105); with it, prefill is 1.26–1.57× faster on the 1–3B models.
- **Per-GPU state is `thread_local`.** That covers autotuned GEMM choices, split-K workspaces and
  kernel attributes, so each worker thread drives its GPU through the same code path as a single GPU.

**Is it correct?** [tests/test_tp.py](tests/test_tp.py) runs 2 GPUs against the same weights on one
GPU, for every backend:

- **Greedy tokens:** 13 runs per model (5 single-sequence prompts plus a batch of 8) give identical
  tokens. The only exceptions are steps where the single-GPU engine's own top two logits are within
  one fp16 step (a gap of 0.0078).
- **Backends:** the all-reduce self-test is exact, and the one-shot and NCCL backends give
  bitwise-identical logits and tokens.
- **The 7–8B models,** which only run sharded, are checked against HF transformers spread over the
  same two GPUs, on WikiText-2 (16 windows × 1,024 tokens):

| model (sharded ×2) | PPL hipinfer | PPL HF | top-1 agreement | mean KL(HF, hipinfer) |
|---|---|---|---|---|
| Llama-3.1-8B | 6.5467 | 6.5471 | 99.87 % | 1.2e-5 |
| Mistral-7B-v0.3 | 5.7773 | 5.7775 | 99.88 % | 6.5e-6 |

**Limits:**

- **Small models gain least.** On the 7–8B models, sharded decode reaches 83 % of the per-GPU
  bandwidth floor, the same as one GPU on the 1–3B models. On the 1–3B models sharded it's 62–76 %
  (46 % on Qwen2.5-0.5B):
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

## int8 weights (W8A16)

`scripts/export.py --int8` stores the projections and the lm_head as int8 with one fp16 scale per
output row (symmetric round-to-nearest). Activations, the KV cache, norms and the embedding stay
fp16. Batch-1 decode reads every weight once per token, so int8 halves the bytes the GEMVs stream.

- **Decode:** the GEMV converts the weights to fp32 in registers and applies the row scale once
  after the reduction.
- **Prefill:** each weight tile is dequantized to fp16 on its way into shared memory, and the
  existing tensor-core GEMM runs unchanged.
- **Tensor parallelism:** shards split the scales along with the rows.

Design: [docs/quantization.md](docs/quantization.md).

RTX 2080 Ti, batch 1, hipinfer's int8 and fp16 weights and llama.cpp's Q8_0 and F16 GGUFs measured
back to back:

| decode ms/token, ctx 128 | hipinfer int8 | hipinfer fp16 | int8 speed-up | llama.cpp Q8_0 | int8 vs Q8_0 |
|---|---|---|---|---|---|
| TinyLlama-1.1B | **2.54** | 4.15 | 1.63× | 3.21 | **1.26× faster** |
| Llama-3.2-1B | **2.73** | 4.72 | 1.73× | 3.32 | **1.22× faster** |
| Llama-3.2-3B | **6.71** | 12.01 | 1.79× | 7.94 | **1.18× faster** |

- **Decode is 1.6–1.8× faster than fp16, not 2×.** The weight bytes halve, but the KV-cache reads
  and each kernel's fixed cost don't, and the conversion costs instructions. int8 decode reaches
  66–78 % of its bandwidth floor, against 80–87 % for fp16. At a 1,024-token context it is
  1.61–1.75× faster than fp16 and 1.16–1.24× faster than Q8_0.
- **Prefill is 1–18 % slower than fp16** (2,000 tokens: 117 vs 102 ms on TinyLlama, 328 vs 282 ms
  on Llama-3.2-3B), because each weight tile is converted on its way to shared memory. It is still
  1.05–1.72× faster than llama.cpp's Q8_0 prefill.
- **Accuracy:** against fp16 HF, WikiText-2 perplexity rises by 0.04 % (TinyLlama), 0.23 %
  (Llama-3.2-1B) and 0.20 % (Llama-3.2-3B), which is the cost of rounding the weights. Against HF
  running the same rounded weights, it is within 0.003 % (mean KL 7e-6 to 1.3e-5), and the strict
  token gate passes with 15–18 of 20 runs identical. So the kernels add no error of their own
  beyond fp16 rounding.
- **Q8_0 is not the same format.** It keeps one scale per 32 weights instead of one per row, so it
  reads about 6 % more bytes and should round a little more accurately. I didn't measure
  llama.cpp's perplexity.

The 3B rows come from a rerun started on a cool card. In the first run, the fp16 benchmark had
throttled more than the int8 one. In the rerun, the two runs' median clocks are within 2 %
([results/2080ti/MERGED_RUN.md](results/2080ti/MERGED_RUN.md)).

## Speculative decoding

Greedy speculative decoding with two engines on one GPU. Llama-3.2-1B drafts `k` tokens with its
normal graph-captured decode steps, and Llama-3.2-3B checks all of them in one pass. The longest
agreeing prefix is accepted, plus the target's own next token. Both caches are then rolled back by
resetting their lengths, which is all a rollback needs because the KV cache is contiguous.

```python
target, draft = hi.Engine("llama32-3b.bin", max_batch=1), hi.Engine("llama32-1b.bin", max_batch=1)
tokens = target.generate_speculative(draft, prompt_ids, n_new=128, k=4)  # == target.generate(prompt_ids, 128)
```

The output is the same as plain greedy decoding **token for token**, and this holds without
needing any near-tie tolerance. The target doesn't verify through its prefill path, whose GEMMs
round differently from decode. It runs the `k + 1` rows through the B=1 decode kernels (GEMVs and
split-K decode attention), with each row's arithmetic independent of the row count. Every row's
logits are therefore bit-identical to the decode step they replace.
[docs/speculative.md](docs/speculative.md) explains the design and the cost model.

**Exact match:** [tests/test_spec.py](tests/test_spec.py) passes all 105 runs on a 2080 Ti and on
an A6000. Each run is one of the 7 gate prompts × 128 tokens at k = 2, 4, 6 or 8, plus k = 6
without CUDA graphs. That covers Llama-3.2-3B with a 1B draft, and Llama-3.2-1B and TinyLlama
drafting for themselves, where every draft token must be accepted.

**Speed on a 2080 Ti**, Llama-3.2-3B with a Llama-3.2-1B draft, 128 new tokens per prompt, against
plain greedy decoding of the same target (83.7 tok/s):

| k | 1 | 2 | 3 | 4 | **5** | 6 | 8 |
|---|---|---|---|---|---|---|---|
| tok/s | 89.7 | 102.1 | 111.8 | 115.0 | **116.7** | 116.1 | 111.7 |
| speed-up | 1.07× | 1.22× | 1.34× | 1.37× | **1.39×** | 1.39× | 1.33× |
| draft tokens accepted | 91 % | 87 % | 83 % | 79 % | 76 % | 72 % | 67 % |

- **The draft is not much cheaper than the target.** A 1B decode step costs 39 % of a 3B one
  (4.66 vs 11.85 ms), because both are weight-bandwidth bound. Gains level off around k = 5.
- **The cost model predicts every k within 2 %** ([docs/speculative.md](docs/speculative.md)).
- **Verification is cheap.** Checking 9 rows in one pass costs 1.31× a single 3B decode step
  (15.5 vs 11.8 ms), because the weights are read once for all rows.
- **Verifying through the prefill path would have broken exactness.** With prefill-path
  verification, the self-draft runs diverged from plain greedy output in 16 of 28 runs (none of the
  49 with the 3B target).

## KV-cache eviction scoring (H2O, WFL)

Eviction policies decide which cached keys to drop when a sequence outgrows its cache budget. H2O
keeps the keys that have received the most attention. WFL prune, from SubCache, removes the keys
that another kept key can best stand in for, judged by key-and-value similarity and attention
mass. Both need numbers the FlashAttention-style kernels don't produce, so with
`Engine.set_tracking(True)` the attention kernels also record them:

- **Decode:** the split-K kernels store each key's logit and each head's log-sum-exp, and a small
  kernel turns them into attention weights and adds them to a per-key score.
- **Prefill:** the flash kernel saves each row's log-sum-exp, and a second, key-parallel pass
  recomputes QKᵀ and sums each key's column, with no atomics.
- **Selection:** H2O is a radix-select top-k. WFL prune computes the key-and-value similarity on the
  fly (no n × n matrix) and rescans only the rows a removal can change.
- **`Engine.evict`** compacts the kept keys, values and scores to the front of the cache; decode
  continues at the true positions.

Tracking is off by default, and turning it on changes no output.

**Checked on a 2080 Ti and an A6000:** 475 kernel checks against PyTorch, and an engine test on
four models (two on the A6000). Attention outputs and greedy tokens with tracking on are bitwise
equal to tracking off, scores are within 2e-6 relative of a float64 reference, and the kept slots
equal PyTorch's choice from the engine's own state. The first GPU run found a bug the CPU
emulation couldn't: after an eviction, decode added each new key's weight on top of an evicted
key's stale score. Fixed and re-tested.

**Cost on the 2080 Ti:** tracking slows the decode-attention kernel by 5–33 %, but a whole
Llama-3.2-1B decode step at a 4k context by only 1.7 % (4.96 → 5.05 ms). Choosing 1,024 of 4,096
keys takes 0.1 ms with H2O and 0.9 s with full WFL (63 ms with WFL limited to a candidate pool and 8
rounds), plus 0.9 ms to compact the cache. The WFL kernels are 6–25× faster than the PyTorch
version of the same algorithm.

This shows the kernels compute the policies as specified. It says nothing about how well either
policy preserves a model's answers at a given budget. Details: [docs/eviction.md](docs/eviction.md).

## Correctness

Fast is only interesting if the answers are right, so every model is checked against HF transformers
in three ways.

| model | greedy runs token-identical | PPL hipinfer | PPL HF | top-1 agreement | mean KL(HF, hipinfer) |
|---|---|---|---|---|---|
| TinyLlama-1.1B | 20/20 | 8.5149 | 8.5151 | 99.90 % | 6.6e-6 |
| Qwen2.5-0.5B | 18/20 | 13.4851 | 13.4851 | 99.79 % | 1.6e-5 |
| Llama-3.2-1B | 13/20 | 10.4633 | 10.4631 | 99.81 % | 1.3e-5 |
| Qwen2.5-1.5B | 17/20 | 9.4536 | 9.4535 | 99.86 % | 1.6e-5 |
| Llama-3.2-3B | 20/20 | 8.2767 | 8.2768 | 99.82 % | 9.5e-6 |
| Granite-3.1-1B-A400M (MoE) | 12/20 ✝ | 7.7292 | 7.7294 | 99.41 % | 2.6e-4 |

- **Greedy token gate** ([tests/gate.py](tests/gate.py)): 7 prompts × 128 tokens, single-sequence
  with and without CUDA graphs, plus a batch of 6 sequences at different lengths (20 runs in all).
  Every divergence happens at a step where HF's own top two logits are exactly tied (gap 0) or one
  fp16 step apart (0.0156); one A6000 Llama-3.2-1B run also diverges at a gap of 0.0313. At such a step either token is a correct greedy choice. Llama-3.2-1B
  hits several of those near-ties in the first 128 tokens of its prompts, which is why it shows 13/20.
  ✝ The MoE model needs one more rule. Its router's top-8 is a second discrete choice that rounding
  can flip at a near-tie, and a different expert moves the logits more than rounding does. Without
  the rule the gate passed in 3 of 5 runs, depending on the GEMM configs the autotuner picked.
  [scripts/moe_router_trace.py](scripts/moe_router_trace.py) shows that every divergence found
  comes from router near-ties: once HF is given the engine's expert choice at the differing points
  (HF margins of at most 0.0042), HF produces the engine's tokens. The gate now accepts an MoE
  divergence only under that condition, with a router tolerance of 0.01. It passes in all 6 runs
  made with the rule: 3 autotuned, autotuning off, and split-K 2 and 4 forced on every GEMM
  ([details](docs/moe-design.md#router-near-ties-and-the-token-gate)). Its KL is
  also more than 10× the dense models'.
- **Perplexity** ([tests/ppl.py](tests/ppl.py)): WikiText-2 test, 16 windows of 1,024 tokens, both
  engines on the same tokens. This checks the whole distribution at every position, not just the
  top token. hipinfer's perplexity is within 0.003 % of HF's on the single-GPU 2080 Ti runs above,
  within 0.0062 % on the A6000, within 0.0052 % on the A100 and within 0.0098 % on sharded runs.
  On the A100 every model's token gate passes too.
- **Kernel tests** ([tests/test_kernels.py](tests/test_kernels.py)): every kernel against PyTorch
  references at head_dim 64 and 128, every GQA group from 1 to 8, all 27 GEMM configs and their
  split-K variants, ragged batches, chunked-prefill offsets and 4,000-token contexts. RoPE is
  bit-exact against HF's fp16 order of operations. RMSNorm is bit-exact at 3 of the 5 tested
  shapes; at the other two the max error is 2.0e-3 on the 2080 Ti and 3.9e-3 on the A6000, within
  the test's 2e-3 absolute + 2e-3 relative tolerance.

When a model doesn't match, [scripts/layer_diff.py](scripts/layer_diff.py) shows where: it compares
every sub-step of every layer against HF and names the first one that is off by more than fp16
rounding explains. On the four models I ran it on it finds nothing, and it names a planted 1 % RoPE
error exactly ([docs/layer-diff.md](docs/layer-diff.md)).

Llama-3.2-3B doesn't fit next to an HF reference copy on an 11 GB card, so its gate and perplexity
run HF first and the engine second (`--hf-first`). Llama-3.1-8B and Mistral-7B-v0.3 only fit
sharded; their perplexity is in the multi-GPU section above. On a 48 GB A6000 they also pass the
token gate with HF alongside ([results/a6000/](results/a6000/)).

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

- **Prefill GEMM on Turing: 0.92–0.95× cuBLAS** at 2,048 rows on Llama-3.2-3B's shapes. This decides
  prefill on 3B-and-up models, and it's why Llama-3.2-3B is still 1.04–1.11× behind llama.cpp. Most
  of what's left is each GEMM's last wave: one 256×128 block per SM on 68 SMs.
- **Prefill GEMM on Ampere: 0.71–0.85× cuBLAS** on the A6000 and **0.49–0.72×** on the A100 at the
  same shapes. The Ampere `cp.async` path is less tuned than the Turing one, and on the A100 it is
  the reason long-prompt prefill trails vLLM, torch.compile and HF.
- **Batched decode at 16–64 sequences:** the projections at these shapes haven't been timed against
  cuBLAS on the current build, so I have no number for them.
- **Attention on Ampere is at parity at best:** on the A6000, prefill attention runs at
  0.90–1.11× PyTorch SDPA (its FlashAttention backend) and decode attention at 0.87–1.09×; on the
  A100 it is behind, at 0.67–0.72× and 0.66–0.73×. On the 2080 Ti, where PyTorch has no flash kernel, it's 8–53× faster than
  SDPA.
- **AMD runs only the portable kernels** (WMMA GEMM and attention, scalar decode attention). MFMA
  versions of the register-resident kernels aren't written yet, and nothing has run on AMD hardware.
- **MoE decode runs at 63 % of the bandwidth floor** on Granite-3.1-1B-A400M (2.26 ms/token on the
  2080 Ti), against 81–87 % for the dense models. Each token reads 8 of 32 small experts, so the
  GEMVs are short and the reads scattered. MoE runs on one GPU only; there's no expert parallelism yet.
- **int8 prefill is slower than fp16** (1–18 %): the GEMM converts each weight tile on its way
  into shared memory instead of using integer tensor-core instructions.
- **Speculative decoding is greedy, batch 1, one GPU, dense models only,** and needs a draft model
  that shares the target's vocabulary.
- **Not supported:** quantization beyond int8 weights (no int4, no int8 activations or KV cache),
  a paged KV cache (each slot's cache is
  contiguous), sampling other than greedy, MoE models with shared experts or QK-norm (Qwen-MoE,
  DeepSeek, OLMoE) and sliding-window attention.
- **Sliding-window models:** Mistral-7B-v0.1 (4,096-token window) loads only with `max_seq` within
  its window, where full attention gives the same result, and the benchmarks cap its context there.
  Weight files exported before this check carry no window and should be re-exported.

## More detail

- [docs/claims.md](docs/claims.md): every claim in this README with its exact scope and the result
  file behind it.
- [docs/profiling.md](docs/profiling.md): where a decode step's time goes, kernel by kernel.
- [docs/porting-a-model.md](docs/porting-a-model.md): the weight format, what a new model family
  costs, and how each unsupported feature fails.
- [docs/quantization.md](docs/quantization.md), [docs/speculative.md](docs/speculative.md),
  [docs/eviction.md](docs/eviction.md): the int8, speculative-decoding and eviction designs, with
  their full results.
- [docs/moe-design.md](docs/moe-design.md), [docs/layer-diff.md](docs/layer-diff.md),
  [docs/turing-prefill-gemm.md](docs/turing-prefill-gemm.md).

## Author

Akash Dasarraju. I wrote the code with Claude as an AI pair programmer; I reviewed every design
decision, benchmark and claim.
