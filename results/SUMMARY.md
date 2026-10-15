## Models

| model | params (fp16 GB) | layers | hidden | heads (q/kv) | head_dim | vocab | notes |
|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B | 2.07 | 22 | 2048 | 32/4 | 64 | 32000 | GQA group 8 |
| Qwen2.5-0.5B | 0.99 | 24 | 896 | 14/2 | 64 | 151936 | tied embeddings, QKV bias, GQA group 7 |
| Llama-3.2-1B | 2.47 | 16 | 2048 | 32/8 | 64 | 128256 | tied embeddings, GQA group 4 |
| Qwen2.5-1.5B | 3.09 | 28 | 1536 | 12/2 | 128 | 151936 | tied embeddings, QKV bias, GQA group 6 |
| Llama-3.2-3B | 6.43 | 28 | 3072 | 24/8 | 128 | 128256 | tied embeddings, GQA group 3 |

## Correctness vs HF transformers (fp16)

Greedy gate: 7 prompts x 128 tokens, single-sequence with and without CUDA graphs, plus a 6-sequence batch (20 runs). A run that diverges only where HF's own top-2 logits are within 0.05 counts as a near-tie. Perplexity: WikiText-2 test, non-overlapping 1024-token windows, both engines on the same tokens.

| model | GPU | gate | runs token-identical | PPL hipinfer | PPL HF | top-1 agreement | mean KL(HF, hipinfer) | max abs logit diff |
|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B | RTX 2080 Ti | PASS | 20/20 | 8.5149 | 8.5151 | 99.90% | 6.6e-06 | 0.193 |
| TinyLlama-1.1B | RTX A6000 | PASS | 20/20 | 8.5149 | 8.5150 | 99.88% | 6.6e-06 | 0.111 |
| Qwen2.5-0.5B | RTX 2080 Ti | PASS | 18/20 | 13.4851 | 13.4851 | 99.79% | 1.6e-05 | 0.129 |
| Qwen2.5-0.5B | RTX A6000 | PASS | 19/20 | 13.4846 | 13.4849 | 99.74% | 1.5e-05 | 0.175 |
| Llama-3.2-1B | RTX 2080 Ti | PASS | 9/20 | 10.4633 | 10.4631 | 99.81% | 1.3e-05 | 0.131 |
| Qwen2.5-1.5B | RTX 2080 Ti | PASS | 17/20 | 9.4536 | 9.4535 | 99.86% | 1.6e-05 | 0.188 |
| Qwen2.5-1.5B | RTX A6000 | PASS | 17/20 | 9.4542 | 9.4539 | 99.85% | 1.6e-05 | 0.348 |
| Llama-3.2-3B | RTX A6000 | PASS | 20/20 | 8.2765 | 8.2765 | 99.87% | 9.8e-06 | 1.219 |
| Mistral-7B-v0.3 | RTX A6000 | PASS | 20/20 | 5.7775 | 5.7774 | 99.88% | 4.4e-06 | 1.047 |
| Mistral-7B-v0.1 (window ignored) | RTX A6000 | PASS | 20/20 | 5.6893 | 5.6893 | 99.84% | 4.6e-06 | 0.506 |
| Llama-3.1-8B | RTX A6000 | PASS | 14/20 | 6.5469 | 6.5469 | 99.87% | 1.1e-05 | 2.405 |

## Decode, batch 1 (ms/token, greedy, 256 new tokens after a 128-token prompt)

Floor = (weight bytes + KV bytes read) / peak DRAM bandwidth. HF = transformers eager with a DynamicCache.

| model | RTX 2080 Ti | RTX A6000 |
|---|---|---|
| peak DRAM BW (GB/s) | 616 | - |
| TinyLlama-1.1B | **4.13** (82% BW; HF 27.7, 6.7x) | - |
| Qwen2.5-0.5B | **2.33** (69% BW; HF 32.3, 13.9x) | - |
| Llama-3.2-1B | **4.70** (86% BW; HF 20.4, 4.3x) | - |
| Qwen2.5-1.5B | **6.17** (81% BW; HF 37.1, 6.0x) | - |
| Llama-3.2-3B | **12.02** (87% BW; HF copy does not fit) | - |

## Decode throughput, batched (tok/s; ctx 512 + 128 new tokens; largest batch that fits)

| model | RTX 2080 Ti | RTX A6000 |
|---|---|---|
| TinyLlama-1.1B | B=8: 1618; B=64: **7553** (8.5 ms/step, 33 mJ/tok at 247 W) | - |
| Qwen2.5-0.5B | B=8: 2449; B=64: **11.8k** (5.4 ms/step, 21 mJ/tok at 245 W) | - |
| Llama-3.2-1B | B=8: 1486; B=64: **7467** (8.6 ms/step, 33 mJ/tok at 247 W) | - |
| Qwen2.5-1.5B | B=8: 877; B=64: **4498** (14.2 ms/step, 55 mJ/tok at 248 W) | - |
| Llama-3.2-3B | B=8: 487; B=16: **906** (17.7 ms/step, 273 mJ/tok at 248 W) | - |

## Prefill (prompt tokens/s; ms in parentheses; HF = transformers eager, SDPA attention)

| model | T | RTX 2080 Ti | RTX A6000 |
|---|---|---|---|
| TinyLlama-1.1B | 128 | 13.8k (9.3 ms; HF 30.7 ms, 3.30x) | - |
| TinyLlama-1.1B | 512 | 17.9k (28.6 ms; HF 51.0 ms, 1.79x) | - |
| TinyLlama-1.1B | 1024 | 18.5k (55.3 ms; HF 138.3 ms, 2.50x) | - |
| TinyLlama-1.1B | 2000 | 18.1k (110.8 ms; HF 425.6 ms, 3.84x) | - |
| Qwen2.5-0.5B | 128 | 24.5k (5.2 ms; HF 35.9 ms, 6.86x) | - |
| Qwen2.5-0.5B | 512 | 40.2k (12.7 ms; HF 36.0 ms, 2.83x) | - |
| Qwen2.5-0.5B | 1024 | 42.7k (24.0 ms; HF 63.4 ms, 2.64x) | - |
| Qwen2.5-0.5B | 2000 | 43.2k (46.2 ms; HF 199.4 ms, 4.31x) | - |
| Qwen2.5-0.5B | 4096 | 41.0k (100.0 ms; HF 679.1 ms, 6.79x) | - |
| Llama-3.2-1B | 128 | 13.8k (9.3 ms; HF 23.7 ms, 2.55x) | - |
| Llama-3.2-1B | 512 | 18.5k (27.7 ms; HF 41.3 ms, 1.49x) | - |
| Llama-3.2-1B | 1024 | 18.8k (54.4 ms; HF 113.4 ms, 2.08x) | - |
| Llama-3.2-1B | 2000 | 18.3k (109.4 ms; HF 331.5 ms, 3.03x) | - |
| Llama-3.2-1B | 4096 | 17.8k (230.7 ms; HF 1080.8 ms, 4.69x) | - |
| Qwen2.5-1.5B | 128 | 8983 (14.2 ms; HF 41.3 ms, 2.90x) | - |
| Qwen2.5-1.5B | 512 | 13.4k (38.3 ms; HF 54.3 ms, 1.42x) | - |
| Qwen2.5-1.5B | 1024 | 13.2k (77.7 ms; HF 128.8 ms, 1.66x) | - |
| Qwen2.5-1.5B | 2000 | 13.4k (149.7 ms; HF 329.4 ms, 2.20x) | - |
| Qwen2.5-1.5B | 4096 | 12.9k (317.6 ms; HF 961.7 ms, 3.03x) | - |
| Llama-3.2-3B | 128 | 4971 (25.8 ms) | - |
| Llama-3.2-3B | 512 | 6230 (82.2 ms) | - |
| Llama-3.2-3B | 1024 | 6553 (156.3 ms) | - |
| Llama-3.2-3B | 2000 | 6280 (318.5 ms) | - |
| Llama-3.2-3B | 4096 | 6014 (681.1 ms) | - |

## Decode latency vs context length (batch 1, ms/token; % of the bandwidth floor incl. KV reads)

| model | ctx | RTX 2080 Ti | RTX A6000 |
|---|---|---|---|
| TinyLlama-1.1B | 128 | 4.13 (82%) | - |
| TinyLlama-1.1B | 512 | 4.16 (81%) | - |
| TinyLlama-1.1B | 1024 | 4.21 (81%) | - |
| Qwen2.5-0.5B | 128 | 2.31 (69%) | - |
| Qwen2.5-0.5B | 512 | 2.35 (69%) | - |
| Qwen2.5-0.5B | 1024 | 2.39 (68%) | - |
| Qwen2.5-0.5B | 2048 | 2.47 (67%) | - |
| Qwen2.5-0.5B | 4096 | 2.62 (64%) | - |
| Qwen2.5-0.5B | 8192 | 2.70 (66%) | - |
| Llama-3.2-1B | 128 | 4.70 (86%) | - |
| Llama-3.2-1B | 512 | 4.74 (85%) | - |
| Llama-3.2-1B | 1024 | 4.78 (85%) | - |
| Llama-3.2-1B | 2048 | 4.83 (85%) | - |
| Llama-3.2-1B | 4096 | 4.95 (86%) | - |
| Llama-3.2-1B | 8192 | 5.18 (86%) | - |
| Qwen2.5-1.5B | 128 | 6.16 (81%) | - |
| Qwen2.5-1.5B | 512 | 6.20 (81%) | - |
| Qwen2.5-1.5B | 1024 | 6.26 (81%) | - |
| Qwen2.5-1.5B | 2048 | 6.41 (80%) | - |
| Qwen2.5-1.5B | 4096 | 6.79 (77%) | - |
| Qwen2.5-1.5B | 8192 | 6.98 (77%) | - |
| Llama-3.2-3B | 128 | 12.06 (87%) | - |
| Llama-3.2-3B | 512 | 12.24 (86%) | - |
| Llama-3.2-3B | 1024 | 12.47 (85%) | - |
| Llama-3.2-3B | 2048 | 12.62 (86%) | - |
| Llama-3.2-3B | 4096 | 13.07 (86%) | - |
| Llama-3.2-3B | 8192 | 13.90 (86%) | - |

## Kernels vs PyTorch (speedup = torch time / hipinfer time; graph-replayed; TinyLlama shapes)

| | RTX 2080 Ti | RTX A6000 |
|---|---|---|
| GEMM qkv M=2048 vs cuBLAS | 0.81x (41 TF) | - |
| GEMM gate_up M=2048 vs cuBLAS | 0.82x (42 TF) | - |
| GEMM down M=512 vs cuBLAS | 0.86x (41 TF) | - |
| GEMM o M=128 vs cuBLAS | 0.78x (25 TF) | - |
| GEMV norm+gate_up+SwiGLU, M=1 | 1.40x (555 GB/s) | - |
| GEMV down+residual, M=1 | 0.99x (516 GB/s) | - |
| GEMV norm+lm_head, M=8 | 1.00x (506 GB/s) | - |
| RMSNorm 4096 rows vs F.rms_norm | 1.07x (516 GB/s) | - |
| RoPE+KV write, 512 tok vs HF | 10.91x | - |
| causal attention T=1024 vs SDPA | 28.97x (32 TF) | - |
| causal attention T=2048 vs SDPA | 28.51x (34 TF) | - |
| decode attention B=1 L=4096 vs SDPA | 29.29x (269 GB/s) | - |
| decode attention B=8 L=4096 vs SDPA | 43.63x (430 GB/s) | - |

## GEMM vs cuBLAS at every model's shapes (TFLOPs, M=2048 prefill rows; ratio = hipinfer / cuBLAS)

| model | shape | RTX 2080 Ti | RTX A6000 |
|---|---|---|---|
| TinyLlama-1.1B | qkv | 41 / 50 (0.81) | - |
| TinyLlama-1.1B | o | 40 / 49 (0.81) | - |
| TinyLlama-1.1B | gate_up | 42 / 51 (0.82) | - |
| TinyLlama-1.1B | down | 41 / 52 (0.79) | - |
| TinyLlama-1.1B | lm_head | 42 / 53 (0.80) | - |
| Qwen2.5-0.5B | qkv | 35 / 28 (1.23) | - |
| Qwen2.5-0.5B | o | 38 / 46 (0.83) | - |
| Qwen2.5-0.5B | gate_up | 39 / 50 (0.79) | - |
| Qwen2.5-0.5B | down | 37 / 46 (0.82) | - |
| Qwen2.5-0.5B | lm_head | 39 / 51 (0.77) | - |
| Llama-3.2-1B | qkv | 40 / 50 (0.80) | - |
| Llama-3.2-1B | o | 40 / 49 (0.80) | - |
| Llama-3.2-1B | gate_up | 42 / 51 (0.82) | - |
| Llama-3.2-1B | down | 42 / 52 (0.80) | - |
| Llama-3.2-1B | lm_head | 41 / 52 (0.79) | - |
| Qwen2.5-1.5B | qkv | 40 / 48 (0.85) | - |
| Qwen2.5-1.5B | o | 39 / 46 (0.86) | - |
| Qwen2.5-1.5B | gate_up | 41 / 52 (0.79) | - |
| Qwen2.5-1.5B | down | 41 / 42 (0.98) | - |
| Qwen2.5-1.5B | lm_head | 42 / 52 (0.80) | - |
| Llama-3.2-3B | qkv | 40 / 51 (0.79) | - |
| Llama-3.2-3B | o | 40 / 50 (0.79) | - |
| Llama-3.2-3B | gate_up | 41 / 51 (0.80) | - |
| Llama-3.2-3B | down | 41 / 52 (0.80) | - |
| Llama-3.2-3B | lm_head | 40 / 51 (0.79) | - |

## hipinfer vs other engines (fp16 weights, batch 1, same GPU)

Decode: 256 greedy tokens after a ctx-token prompt, ms/token (% BW = weight+KV bytes / peak BW / time). Prefill: one prompt of T random tokens, ms. llama.cpp: `llama-bench` (pp / tg256 at depth ctx), F16 GGUF, best of flash-attn on/off and ubatch 512/2048. torch.compile / HF static: decode step in a CUDA graph, KV cache sized ctx+256. vLLM: decode = (generate 257 tokens - generate 1 token) / 256. The TinyLlama rows were measured in one session by scripts/run_baselines.sh; the other models' hipinfer rows come from the main run (results/<gpu>/<model>/bench.json). Raw JSON/logs: results/baselines/.


### TinyLlama-1.1B

| | RTX 2080 Ti | RTX A6000 |
|---|---|---|
| decode ms/token, decode 128: **hipinfer** | 4.13 ms, 82% BW | - |
| decode ms/token, decode 128: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 4.77 ms, 71% BW (hipinfer 1.15x faster) | 3.76 ms |
| decode ms/token, decode 128: torch.compile, gpt-fast style (reduce-overhead + static KV) | 5.24 ms, 64% BW (hipinfer 1.27x faster) | 4.23 ms |
| decode ms/token, decode 128: HF transformers, StaticCache + torch.compile | 5.51 ms, 61% BW (hipinfer 1.33x faster) | 4.34 ms |
| decode ms/token, decode 128: vLLM (V1 engine, CUDA graphs) | - | 3.96 ms |
| decode ms/token, decode 1024: **hipinfer** | 4.21 ms, 81% BW | - |
| decode ms/token, decode 1024: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 4.79 ms, 71% BW (hipinfer 1.14x faster) | 3.76 ms |
| decode ms/token, decode 1024: torch.compile, gpt-fast style (reduce-overhead + static KV) | 6.84 ms, 50% BW (hipinfer 1.62x faster) | 5.47 ms |
| decode ms/token, decode 1024: HF transformers, StaticCache + torch.compile | 7.09 ms, 48% BW (hipinfer 1.68x faster) | 5.56 ms |
| decode ms/token, decode 1024: vLLM (V1 engine, CUDA graphs) | - | 3.95 ms |

| | RTX 2080 Ti | RTX A6000 |
|---|---|---|
| prefill ms, prefill 128: **hipinfer** | 9.3 ms | - |
| prefill ms, prefill 128: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 11.2 ms (hipinfer 1.20x faster) | 9.1 ms |
| prefill ms, prefill 128: torch.compile, gpt-fast style (reduce-overhead + static KV) | 16.2 ms (hipinfer 1.74x faster) | 8.9 ms |
| prefill ms, prefill 512: **hipinfer** | 28.6 ms | - |
| prefill ms, prefill 512: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 29.0 ms (hipinfer 1.02x faster) | 22.8 ms |
| prefill ms, prefill 512: torch.compile, gpt-fast style (reduce-overhead + static KV) | 34.8 ms (hipinfer 1.22x faster) | 19.5 ms |
| prefill ms, prefill 2000: **hipinfer** | 110.8 ms | - |
| prefill ms, prefill 2000: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 114.8 ms (hipinfer 1.04x faster) | 87.3 ms |
| prefill ms, prefill 2000: torch.compile, gpt-fast style (reduce-overhead + static KV) | 125.4 ms (hipinfer 1.13x faster) | 59.6 ms |


### Qwen2.5-0.5B

| | RTX 2080 Ti |
|---|---|
| decode ms/token, decode 128: **hipinfer** | 2.33 ms, 69% BW |
| decode ms/token, decode 128: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 3.45 ms, 47% BW (hipinfer 1.48x faster) |
| decode ms/token, decode 1024: **hipinfer** | 2.39 ms, 68% BW |
| decode ms/token, decode 1024: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 3.46 ms, 47% BW (hipinfer 1.44x faster) |

| | RTX 2080 Ti |
|---|---|
| prefill ms, prefill 128: **hipinfer** | 5.2 ms |
| prefill ms, prefill 128: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 8.4 ms (hipinfer 1.61x faster) |
| prefill ms, prefill 512: **hipinfer** | 12.7 ms |
| prefill ms, prefill 512: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 18.5 ms (hipinfer 1.45x faster) |
| prefill ms, prefill 2000: **hipinfer** | 46.2 ms |
| prefill ms, prefill 2000: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 67.3 ms (hipinfer 1.46x faster) |


### Llama-3.2-1B

| | RTX 2080 Ti |
|---|---|
| decode ms/token, decode 128: **hipinfer** | 4.70 ms, 86% BW |
| decode ms/token, decode 128: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 5.23 ms, 77% BW (hipinfer 1.11x faster) |
| decode ms/token, decode 1024: **hipinfer** | 4.78 ms, 85% BW |
| decode ms/token, decode 1024: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 5.27 ms, 77% BW (hipinfer 1.10x faster) |

| | RTX 2080 Ti |
|---|---|
| prefill ms, prefill 128: **hipinfer** | 9.3 ms |
| prefill ms, prefill 128: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 10.4 ms (hipinfer 1.12x faster) |
| prefill ms, prefill 512: **hipinfer** | 27.7 ms |
| prefill ms, prefill 512: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 27.0 ms (**1.03x faster than hipinfer**) |
| prefill ms, prefill 2000: **hipinfer** | 109.4 ms |
| prefill ms, prefill 2000: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 100.9 ms (**1.08x faster than hipinfer**) |


### Qwen2.5-1.5B

| | RTX 2080 Ti |
|---|---|
| decode ms/token, decode 128: **hipinfer** | 6.17 ms, 81% BW |
| decode ms/token, decode 128: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 7.19 ms, 70% BW (hipinfer 1.17x faster) |
| decode ms/token, decode 1024: **hipinfer** | 6.26 ms, 81% BW |
| decode ms/token, decode 1024: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 7.22 ms, 70% BW (hipinfer 1.15x faster) |

| | RTX 2080 Ti |
|---|---|
| prefill ms, prefill 128: **hipinfer** | 14.2 ms |
| prefill ms, prefill 128: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 17.3 ms (hipinfer 1.21x faster) |
| prefill ms, prefill 512: **hipinfer** | 38.3 ms |
| prefill ms, prefill 512: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 40.1 ms (hipinfer 1.05x faster) |
| prefill ms, prefill 2000: **hipinfer** | 149.7 ms |
| prefill ms, prefill 2000: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 156.7 ms (hipinfer 1.05x faster) |


### Llama-3.2-3B

| | RTX 2080 Ti |
|---|---|
| decode ms/token, decode 128: **hipinfer** | 12.02 ms, 87% BW |
| decode ms/token, decode 128: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 12.70 ms, 83% BW (hipinfer 1.06x faster) |
| decode ms/token, decode 1024: **hipinfer** | 12.49 ms, 85% BW |
| decode ms/token, decode 1024: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 12.92 ms, 82% BW (hipinfer 1.03x faster) |

| | RTX 2080 Ti |
|---|---|
| prefill ms, prefill 128: **hipinfer** | 25.8 ms |
| prefill ms, prefill 128: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 25.8 ms (hipinfer 1.00x faster) |
| prefill ms, prefill 512: **hipinfer** | 82.2 ms |
| prefill ms, prefill 512: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 65.4 ms (**1.26x faster than hipinfer**) |
| prefill ms, prefill 2000: **hipinfer** | 318.5 ms |
| prefill ms, prefill 2000: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 261.8 ms (**1.22x faster than hipinfer**) |


## Tensor parallelism (multi-GPU)

Megatron-style sharding over N GPUs in one process (one worker thread per GPU): QKV/gate/up column-parallel, O/down row-parallel with an all-reduce after each, lm_head vocabulary-parallel. All-reduce `auto` = hipinfer's one-shot kernel for messages up to 512 KB (decode), NCCL above (prefill); `NCCL` = NCCL only. Prefill all-reduces overlap with compute (two row halves on two streams) unless noted. % floor = this GPU's shard of the weights + its KV share / peak BW / time.


### RTX 2080 Ti + RTX 2080 Ti (`2080ti`), tp=2

Peer-to-peer access between the GPUs: no (one-shot all-reduce stages through pinned host memory); NCCL built in: yes.

All-reduce latency, fp16, in place (TinyLlama-1.1B hidden size 2048):

| message | size | one-shot kernel | NCCL | one-shot speedup |
|---|---|---|---|---|
| decode B=1 | 4 KB | 8.0 us | 11.9 us | 1.48x |
| decode B=8 | 32 KB | 16.7 us | 23.8 us | 1.43x |
| decode B=64 | 256 KB | 59.0 us | 83.5 us | 1.42x |
| prefill T=512 | 2048 KB | 395.1 us | 428.2 us | 1.08x |
| prefill T=2000 | 8000 KB | 1490.2 us | 1588.2 us | 1.07x |

Batch-1 decode, ms/token (1 GPU vs tp=2):

| model | ctx | 1 GPU | tp=2 auto | tp=2 NCCL only | speedup | % floor (tp) |
|---|---|---|---|---|---|---|
| TinyLlama-1.1B | 128 | 4.14 | **2.67** | 2.88 | 1.55x | 63 % |
| TinyLlama-1.1B | 1024 | 4.23 | **2.75** | 2.97 | 1.54x | 62 % |
| Llama-3.2-1B | 128 | 4.71 | **2.79** | 2.95 | 1.69x | 72 % |
| Llama-3.2-1B | 1024 | 4.79 | **2.83** | 2.99 | 1.69x | 72 % |
| Llama-3.2-1B | 4096 | 4.95 | **2.94** | 3.09 | 1.69x | 72 % |
| Qwen2.5-1.5B | 128 | 6.18 | **3.85** | 4.21 | 1.61x | 65 % |
| Qwen2.5-1.5B | 1024 | 6.28 | **3.93** | 4.31 | 1.60x | 64 % |
| Qwen2.5-1.5B | 4096 | 6.83 | **4.26** | 4.63 | 1.60x | 61 % |
| Llama-3.2-3B | 128 | 11.99 | **6.84** | 7.25 | 1.75x | 77 % |
| Llama-3.2-3B | 1024 | 12.37 | **7.00** | 7.41 | 1.77x | 76 % |
| Llama-3.2-3B | 4096 | 12.97 | **7.46** | 7.88 | 1.74x | 75 % |
| Mistral-7B-v0.3 | 128 | does not fit | **13.97** | 14.24 | - | 83 % |
| Mistral-7B-v0.3 | 1024 | does not fit | **14.16** | 14.42 | - | 82 % |
| Mistral-7B-v0.3 | 4096 | does not fit | **14.68** | 14.98 | - | 82 % |
| Mistral-7B-v0.1 (window ignored) | 128 | does not fit | **13.96** | 14.24 | - | 83 % |
| Mistral-7B-v0.1 (window ignored) | 1024 | does not fit | **14.15** | 14.44 | - | 82 % |
| Mistral-7B-v0.1 (window ignored) | 4096 | does not fit | **14.72** | 15.02 | - | 81 % |
| Llama-3.1-8B | 128 | does not fit | **14.63** | 14.94 | - | 83 % |
| Llama-3.1-8B | 1024 | does not fit | **14.82** | 15.11 | - | 83 % |
| Llama-3.1-8B | 4096 | does not fit | **15.38** | 15.68 | - | 82 % |

Prefill, ms (1 GPU vs tp=2):

| model | tokens | 1 GPU | tp=2 auto, overlapped | tp=2 auto, no overlap | tp=2 NCCL only | speedup |
|---|---|---|---|---|---|---|
| TinyLlama-1.1B | 512 | 29.8 | **26.2** | 32.7 | 26.3 | 1.14x |
| TinyLlama-1.1B | 2000 | 117.2 | **87.6** | 123.9 | 88.0 | 1.34x |
| Llama-3.2-1B | 512 | 29.3 | **21.6** | 27.2 | 21.7 | 1.35x |
| Llama-3.2-1B | 2000 | 115.1 | **75.8** | 104.1 | 76.2 | 1.52x |
| Qwen2.5-1.5B | 512 | 40.9 | **35.4** | 39.6 | 35.3 | 1.15x |
| Qwen2.5-1.5B | 2000 | 159.2 | **109.7** | 141.0 | 110.2 | 1.45x |
| Llama-3.2-3B | 512 | 85.7 | **56.4** | 75.5 | 57.1 | 1.52x |
| Llama-3.2-3B | 2000 | 329.1 | **199.0** | 285.6 | 202.1 | 1.65x |
| Mistral-7B-v0.3 | 512 | does not fit | **115.3** | 140.1 | 115.6 | - |
| Mistral-7B-v0.3 | 2000 | does not fit | **421.6** | 542.6 | 424.9 | - |
| Mistral-7B-v0.1 (window ignored) | 512 | does not fit | **116.1** | 141.6 | 124.9 | - |
| Mistral-7B-v0.1 (window ignored) | 2000 | does not fit | **425.3** | 547.6 | 463.7 | - |
| Llama-3.1-8B | 512 | does not fit | **113.5** | 139.9 | 115.5 | - |
| Llama-3.1-8B | 2000 | does not fit | **418.5** | 540.7 | 425.5 | - |

Batched decode throughput, tok/s (ctx 512 + 128 new tokens):

| model | B | 1 GPU | tp=2 auto | tp=2 NCCL only | speedup |
|---|---|---|---|---|---|
| TinyLlama-1.1B | 1 | 240 | **369** | 340 | 1.54x |
| TinyLlama-1.1B | 4 | 862 | **1237** | 1148 | 1.44x |
| TinyLlama-1.1B | 8 | 1650 | **2187** | 1963 | 1.33x |
| TinyLlama-1.1B | 16 | 3095 | **3912** | 3314 | 1.26x |
| TinyLlama-1.1B | 32 | 4724 | **5836** | 4962 | 1.24x |
| TinyLlama-1.1B | 64 | 7385 | **8529** | 7382 | 1.15x |
| Llama-3.2-1B | 1 | 211 | **357** | 337 | 1.69x |
| Llama-3.2-1B | 4 | 773 | **1228** | 1163 | 1.59x |
| Llama-3.2-1B | 8 | 1521 | **2275** | 2091 | 1.50x |
| Llama-3.2-1B | 16 | 2832 | **4068** | 3579 | 1.44x |
| Llama-3.2-1B | 32 | 4680 | **6196** | 5452 | 1.32x |
| Llama-3.2-1B | 64 | 7454 | **9494** | 8399 | 1.27x |
| Qwen2.5-1.5B | 1 | 161 | **258** | 234 | 1.60x |
| Qwen2.5-1.5B | 4 | 578 | **886** | 796 | 1.53x |
| Qwen2.5-1.5B | 8 | 852 | **1307** | 1196 | 1.53x |
| Qwen2.5-1.5B | 16 | 1638 | **2355** | 2113 | 1.44x |
| Qwen2.5-1.5B | 32 | 2719 | **3633** | 3131 | 1.34x |
| Qwen2.5-1.5B | 64 | 4457 | **5425** | 4870 | 1.22x |
| Llama-3.2-3B | 1 | 82 | **145** | 136 | 1.76x |
| Llama-3.2-3B | 4 | 298 | **496** | 463 | 1.67x |
| Llama-3.2-3B | 8 | 489 | **805** | 743 | 1.65x |
| Llama-3.2-3B | 16 | 910 | **1447** | 1280 | 1.59x |
| Llama-3.2-3B | 32 | - | **2193** | 1973 | - |
| Llama-3.2-3B | 64 | - | **3226** | 2945 | - |
| Mistral-7B-v0.3 | 1 | - | **71** | 70 | - |
| Mistral-7B-v0.3 | 4 | - | **255** | 247 | - |
| Mistral-7B-v0.3 | 8 | - | **488** | 463 | - |
| Mistral-7B-v0.3 | 16 | - | **894** | 835 | - |
| Mistral-7B-v0.3 | 32 | - | **1406** | 1275 | - |
| Mistral-7B-v0.1 (window ignored) | 1 | - | **71** | 70 | - |
| Mistral-7B-v0.1 (window ignored) | 4 | - | **255** | 244 | - |
| Mistral-7B-v0.1 (window ignored) | 8 | - | **490** | 457 | - |
| Mistral-7B-v0.1 (window ignored) | 16 | - | **894** | 815 | - |
| Mistral-7B-v0.1 (window ignored) | 32 | - | **1339** | 1271 | - |
| Llama-3.1-8B | 1 | - | **68** | 67 | - |
| Llama-3.1-8B | 4 | - | **245** | 236 | - |
| Llama-3.1-8B | 8 | - | **471** | 445 | - |
| Llama-3.1-8B | 16 | - | **858** | 798 | - |

Correctness of the sharded engine (tests/test_tp.py: tp=2 vs the same weights on 1 GPU; tests/ppl.py --tp for models that only fit sharded):

| model | greedy runs token-identical to 1 GPU | prefill top-1 agreement | max abs logit diff | backends bitwise equal | all-reduce self-test | PPL sharded / HF |
|---|---|---|---|---|---|---|
| TinyLlama-1.1B | 13/13 | 99.84 % | 0.789 | yes | exact | - |
| Qwen2.5-0.5B | 11/13 (rest at near-ties) | 99.38 % | 0.475 | yes | exact | - |
| Llama-3.2-1B | 11/13 (rest at near-ties) | 100.00 % | 0.064 | yes | exact | - |
| Qwen2.5-1.5B | 13/13 | 98.75 % | 1.203 | yes | exact | - |
| Llama-3.2-3B | 13/13 | 99.84 % | 0.320 | yes | exact | - |
| Mistral-7B-v0.3 | no 1-GPU reference | - | - | yes | exact | 7.3269 / 7.3267 |
| Mistral-7B-v0.1 (window ignored) | no 1-GPU reference | - | - | yes | exact | 7.1826 / 7.1833 |
| Llama-3.1-8B | no 1-GPU reference | - | - | yes | exact | 8.1505 / 8.1511 |

