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
| Llama-3.2-1B | RTX 2080 Ti | PASS | 13/20 | 10.4633 | 10.4631 | 99.81% | 1.3e-05 | 0.131 |
| Qwen2.5-1.5B | RTX 2080 Ti | PASS | 17/20 | 9.4536 | 9.4535 | 99.86% | 1.6e-05 | 0.188 |
| Qwen2.5-1.5B | RTX A6000 | PASS | 17/20 | 9.4542 | 9.4539 | 99.85% | 1.6e-05 | 0.348 |
| Llama-3.2-3B | RTX 2080 Ti | PASS | 20/20 | 8.2767 | 8.2768 | 99.82% | 9.5e-06 | 0.367 |
| Llama-3.2-3B | RTX A6000 | PASS | 20/20 | 8.2765 | 8.2765 | 99.87% | 9.8e-06 | 1.219 |
| Mistral-7B-v0.3 | RTX A6000 | PASS | 20/20 | 5.7775 | 5.7774 | 99.88% | 4.4e-06 | 1.047 |
| Mistral-7B-v0.1 (window ignored) | RTX A6000 | PASS | 20/20 | 5.6893 | 5.6893 | 99.84% | 4.6e-06 | 0.506 |
| Llama-3.1-8B | RTX A6000 | PASS | 14/20 | 6.5469 | 6.5469 | 99.87% | 1.1e-05 | 2.405 |

## Decode, batch 1 (ms/token, greedy, 256 new tokens after a 128-token prompt)

Floor = (weight bytes + KV bytes read) / peak DRAM bandwidth. HF = transformers eager with a DynamicCache.

| model | RTX 2080 Ti | RTX A6000 |
|---|---|---|
| peak DRAM BW (GB/s) | 616 | - |
| TinyLlama-1.1B | **4.14** (81% BW; HF 31.3, 7.6x) | - |
| Qwen2.5-0.5B | **2.34** (69% BW; HF 27.0, 11.5x) | - |
| Llama-3.2-1B | **4.75** (85% BW; HF 18.2, 3.8x) | - |
| Qwen2.5-1.5B | **6.21** (81% BW; HF 40.6, 6.5x) | - |
| Llama-3.2-3B | **12.11** (87% BW; HF copy does not fit) | - |

## Decode throughput, batched (tok/s; ctx 512 + 128 new tokens; largest batch that fits)

| model | RTX 2080 Ti | RTX A6000 |
|---|---|---|
| TinyLlama-1.1B | B=8: 1639; B=64: **7745** (8.3 ms/step, 32 mJ/tok at 248 W) | - |
| Qwen2.5-0.5B | B=8: 2421; B=64: **11.9k** (5.4 ms/step, 20 mJ/tok at 244 W) | - |
| Llama-3.2-1B | B=8: 1493; B=64: **7322** (8.7 ms/step, 31 mJ/tok at 228 W) | - |
| Qwen2.5-1.5B | B=8: 866; B=64: **4332** (14.8 ms/step, 51 mJ/tok at 219 W) | - |
| Llama-3.2-3B | B=8: 479; B=16: **888** (18.0 ms/step, 249 mJ/tok at 221 W) | - |

## Prefill (prompt tokens/s; ms in parentheses; HF = transformers eager, SDPA attention)

| model | T | RTX 2080 Ti | RTX A6000 |
|---|---|---|---|
| TinyLlama-1.1B | 128 | 15.4k (8.3 ms; HF 34.0 ms, 4.07x) | - |
| TinyLlama-1.1B | 512 | 19.2k (26.6 ms; HF 51.6 ms, 1.94x) | - |
| TinyLlama-1.1B | 1024 | 20.2k (50.6 ms; HF 138.5 ms, 2.74x) | - |
| TinyLlama-1.1B | 2000 | 19.9k (100.5 ms; HF 426.6 ms, 4.24x) | - |
| Qwen2.5-0.5B | 128 | 24.9k (5.1 ms; HF 29.9 ms, 5.81x) | - |
| Qwen2.5-0.5B | 512 | 42.3k (12.1 ms; HF 31.4 ms, 2.60x) | - |
| Qwen2.5-0.5B | 1024 | 44.1k (23.2 ms; HF 64.5 ms, 2.77x) | - |
| Qwen2.5-0.5B | 2000 | 46.4k (43.1 ms; HF 201.4 ms, 4.68x) | - |
| Qwen2.5-0.5B | 4096 | 43.1k (95.0 ms; HF 684.8 ms, 7.21x) | - |
| Llama-3.2-1B | 128 | 14.9k (8.6 ms; HF 20.6 ms, 2.40x) | - |
| Llama-3.2-1B | 512 | 19.6k (26.1 ms; HF 42.2 ms, 1.62x) | - |
| Llama-3.2-1B | 1024 | 20.5k (49.9 ms; HF 115.2 ms, 2.31x) | - |
| Llama-3.2-1B | 2000 | 19.8k (101.1 ms; HF 335.4 ms, 3.32x) | - |
| Llama-3.2-1B | 4096 | 19.0k (215.8 ms; HF 1094.8 ms, 5.07x) | - |
| Qwen2.5-1.5B | 128 | 9575 (13.4 ms; HF 44.5 ms, 3.33x) | - |
| Qwen2.5-1.5B | 512 | 13.6k (37.8 ms; HF 56.3 ms, 1.49x) | - |
| Qwen2.5-1.5B | 1024 | 14.6k (70.0 ms; HF 132.0 ms, 1.88x) | - |
| Qwen2.5-1.5B | 2000 | 14.2k (140.9 ms; HF 337.0 ms, 2.39x) | - |
| Qwen2.5-1.5B | 4096 | 14.1k (291.5 ms; HF 985.6 ms, 3.38x) | - |
| Llama-3.2-3B | 128 | 5273 (24.3 ms) | - |
| Llama-3.2-3B | 512 | 6673 (76.7 ms) | - |
| Llama-3.2-3B | 1024 | 6863 (149.2 ms) | - |
| Llama-3.2-3B | 2000 | 6905 (289.7 ms) | - |
| Llama-3.2-3B | 4096 | 6459 (634.1 ms) | - |

## Decode latency vs context length (batch 1, ms/token; % of the bandwidth floor incl. KV reads)

| model | ctx | RTX 2080 Ti | RTX A6000 |
|---|---|---|---|
| TinyLlama-1.1B | 128 | 4.13 (81%) | - |
| TinyLlama-1.1B | 512 | 4.17 (81%) | - |
| TinyLlama-1.1B | 1024 | 4.22 (81%) | - |
| Qwen2.5-0.5B | 128 | 2.33 (69%) | - |
| Qwen2.5-0.5B | 512 | 2.38 (68%) | - |
| Qwen2.5-0.5B | 1024 | 2.42 (67%) | - |
| Qwen2.5-0.5B | 2048 | 2.50 (66%) | - |
| Qwen2.5-0.5B | 4096 | 2.64 (64%) | - |
| Qwen2.5-0.5B | 8192 | 2.73 (65%) | - |
| Llama-3.2-1B | 128 | 4.77 (84%) | - |
| Llama-3.2-1B | 512 | 4.80 (84%) | - |
| Llama-3.2-1B | 1024 | 4.84 (84%) | - |
| Llama-3.2-1B | 2048 | 4.90 (84%) | - |
| Llama-3.2-1B | 4096 | 5.02 (84%) | - |
| Llama-3.2-1B | 8192 | 5.26 (85%) | - |
| Qwen2.5-1.5B | 128 | 6.22 (81%) | - |
| Qwen2.5-1.5B | 512 | 6.27 (80%) | - |
| Qwen2.5-1.5B | 1024 | 6.32 (80%) | - |
| Qwen2.5-1.5B | 2048 | 6.48 (79%) | - |
| Qwen2.5-1.5B | 4096 | 6.90 (75%) | - |
| Qwen2.5-1.5B | 8192 | 7.11 (76%) | - |
| Llama-3.2-3B | 128 | 12.11 (86%) | - |
| Llama-3.2-3B | 512 | 12.29 (86%) | - |
| Llama-3.2-3B | 1024 | 12.53 (85%) | - |
| Llama-3.2-3B | 2048 | 12.68 (85%) | - |
| Llama-3.2-3B | 4096 | 13.13 (85%) | - |
| Llama-3.2-3B | 8192 | 13.99 (86%) | - |

## Kernels vs PyTorch (speedup = torch time / hipinfer time; graph-replayed; TinyLlama shapes)

| | RTX 2080 Ti | RTX A6000 |
|---|---|---|
| GEMM qkv M=2048 vs cuBLAS | 0.84x (41 TF) | - |
| GEMM gate_up M=2048 vs cuBLAS | 0.92x (47 TF) | - |
| GEMM down M=512 vs cuBLAS | 0.94x (44 TF) | - |
| GEMM o M=128 vs cuBLAS | 0.87x (26 TF) | - |
| GEMV norm+gate_up+SwiGLU, M=1 | 1.40x (556 GB/s) | - |
| GEMV down+residual, M=1 | 0.99x (519 GB/s) | - |
| GEMV norm+lm_head, M=8 | 1.03x (520 GB/s) | - |
| RMSNorm 4096 rows vs F.rms_norm | 1.07x (519 GB/s) | - |
| RoPE+KV write, 512 tok vs HF | 11.02x | - |
| causal attention T=1024 vs SDPA | 29.23x (32 TF) | - |
| causal attention T=2048 vs SDPA | 27.71x (33 TF) | - |
| decode attention B=1 L=4096 vs SDPA | 29.69x (273 GB/s) | - |
| decode attention B=8 L=4096 vs SDPA | 44.15x (435 GB/s) | - |

## GEMM vs cuBLAS at every model's shapes (TFLOPs, M=2048 prefill rows; ratio = hipinfer / cuBLAS)

| model | shape | RTX 2080 Ti | RTX A6000 |
|---|---|---|---|
| TinyLlama-1.1B | qkv | 41 / 49 (0.84) | - |
| TinyLlama-1.1B | o | 47 / 51 (0.93) | - |
| TinyLlama-1.1B | gate_up | 47 / 51 (0.92) | - |
| TinyLlama-1.1B | down | 48 / 51 (0.95) | - |
| TinyLlama-1.1B | lm_head | 48 / 52 (0.93) | - |
| Qwen2.5-0.5B | qkv | 37 / 27 (1.40) | - |
| Qwen2.5-0.5B | o | 39 / 42 (0.91) | - |
| Qwen2.5-0.5B | gate_up | 44 / 48 (0.90) | - |
| Qwen2.5-0.5B | down | 42 / 45 (0.95) | - |
| Qwen2.5-0.5B | lm_head | 44 / 49 (0.90) | - |
| Llama-3.2-1B | qkv | 45 / 49 (0.93) | - |
| Llama-3.2-1B | o | 46 / 48 (0.95) | - |
| Llama-3.2-1B | gate_up | 46 / 49 (0.93) | - |
| Llama-3.2-1B | down | 47 / 50 (0.95) | - |
| Llama-3.2-1B | lm_head | 47 / 50 (0.94) | - |
| Qwen2.5-1.5B | qkv | 45 / 48 (0.93) | - |
| Qwen2.5-1.5B | o | 40 / 46 (0.86) | - |
| Qwen2.5-1.5B | gate_up | 45 / 50 (0.91) | - |
| Qwen2.5-1.5B | down | 42 / 40 (1.03) | - |
| Qwen2.5-1.5B | lm_head | 45 / 49 (0.92) | - |
| Llama-3.2-3B | qkv | 45 / 48 (0.94) | - |
| Llama-3.2-3B | o | 45 / 48 (0.93) | - |
| Llama-3.2-3B | gate_up | 45 / 49 (0.93) | - |
| Llama-3.2-3B | down | 46 / 49 (0.94) | - |
| Llama-3.2-3B | lm_head | 45 / 48 (0.94) | - |

## hipinfer vs other engines (fp16 weights, batch 1, same GPU)

Decode: 256 greedy tokens after a ctx-token prompt, ms/token (% BW = weight+KV bytes / peak BW / time). Prefill: one prompt of T random tokens, ms. llama.cpp: `llama-bench` (pp / tg256 at depth ctx), F16 GGUF, best of flash-attn on/off and ubatch 512/2048. torch.compile / HF static: decode step in a CUDA graph, KV cache sized ctx+256. vLLM: decode = (generate 257 tokens - generate 1 token) / 256. The TinyLlama rows were measured in one session by scripts/run_baselines.sh; the other models' hipinfer rows come from the main run (results/<gpu>/<model>/bench.json). Raw JSON/logs: results/baselines/.


### TinyLlama-1.1B

| | RTX 2080 Ti | RTX A6000 |
|---|---|---|
| decode ms/token, decode 128: **hipinfer** | 4.14 ms, 81% BW | - |
| decode ms/token, decode 128: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 4.77 ms, 71% BW (hipinfer 1.15x faster) | 3.76 ms |
| decode ms/token, decode 128: torch.compile, gpt-fast style (reduce-overhead + static KV) | 5.24 ms, 64% BW (hipinfer 1.27x faster) | 4.23 ms |
| decode ms/token, decode 128: HF transformers, StaticCache + torch.compile | 5.51 ms, 61% BW (hipinfer 1.33x faster) | 4.34 ms |
| decode ms/token, decode 128: vLLM (V1 engine, CUDA graphs) | - | 3.96 ms |
| decode ms/token, decode 1024: **hipinfer** | 4.22 ms, 81% BW | - |
| decode ms/token, decode 1024: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 4.79 ms, 71% BW (hipinfer 1.14x faster) | 3.76 ms |
| decode ms/token, decode 1024: torch.compile, gpt-fast style (reduce-overhead + static KV) | 6.84 ms, 50% BW (hipinfer 1.62x faster) | 5.47 ms |
| decode ms/token, decode 1024: HF transformers, StaticCache + torch.compile | 7.09 ms, 48% BW (hipinfer 1.68x faster) | 5.56 ms |
| decode ms/token, decode 1024: vLLM (V1 engine, CUDA graphs) | - | 3.95 ms |

| | RTX 2080 Ti | RTX A6000 |
|---|---|---|
| prefill ms, prefill 128: **hipinfer** | 8.3 ms | - |
| prefill ms, prefill 128: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 11.2 ms (hipinfer 1.34x faster) | 9.1 ms |
| prefill ms, prefill 128: torch.compile, gpt-fast style (reduce-overhead + static KV) | 16.2 ms (hipinfer 1.94x faster) | 8.9 ms |
| prefill ms, prefill 512: **hipinfer** | 26.6 ms | - |
| prefill ms, prefill 512: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 29.0 ms (hipinfer 1.09x faster) | 22.8 ms |
| prefill ms, prefill 512: torch.compile, gpt-fast style (reduce-overhead + static KV) | 34.8 ms (hipinfer 1.31x faster) | 19.5 ms |
| prefill ms, prefill 2000: **hipinfer** | 100.5 ms | - |
| prefill ms, prefill 2000: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 114.8 ms (hipinfer 1.14x faster) | 87.3 ms |
| prefill ms, prefill 2000: torch.compile, gpt-fast style (reduce-overhead + static KV) | 125.4 ms (hipinfer 1.25x faster) | 59.6 ms |


### Qwen2.5-0.5B

| | RTX 2080 Ti |
|---|---|
| decode ms/token, decode 128: **hipinfer** | 2.34 ms, 69% BW |
| decode ms/token, decode 128: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 3.45 ms, 47% BW (hipinfer 1.47x faster) |
| decode ms/token, decode 1024: **hipinfer** | 2.42 ms, 67% BW |
| decode ms/token, decode 1024: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 3.46 ms, 47% BW (hipinfer 1.43x faster) |

| | RTX 2080 Ti |
|---|---|
| prefill ms, prefill 128: **hipinfer** | 5.1 ms |
| prefill ms, prefill 128: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 8.4 ms (hipinfer 1.64x faster) |
| prefill ms, prefill 512: **hipinfer** | 12.1 ms |
| prefill ms, prefill 512: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 18.5 ms (hipinfer 1.53x faster) |
| prefill ms, prefill 2000: **hipinfer** | 43.1 ms |
| prefill ms, prefill 2000: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 67.3 ms (hipinfer 1.56x faster) |


### Llama-3.2-1B

| | RTX 2080 Ti |
|---|---|
| decode ms/token, decode 128: **hipinfer** | 4.75 ms, 85% BW |
| decode ms/token, decode 128: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 5.23 ms, 77% BW (hipinfer 1.10x faster) |
| decode ms/token, decode 1024: **hipinfer** | 4.84 ms, 84% BW |
| decode ms/token, decode 1024: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 5.27 ms, 77% BW (hipinfer 1.09x faster) |

| | RTX 2080 Ti |
|---|---|
| prefill ms, prefill 128: **hipinfer** | 8.6 ms |
| prefill ms, prefill 128: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 10.4 ms (hipinfer 1.21x faster) |
| prefill ms, prefill 512: **hipinfer** | 26.1 ms |
| prefill ms, prefill 512: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 27.0 ms (hipinfer 1.03x faster) |
| prefill ms, prefill 2000: **hipinfer** | 101.1 ms |
| prefill ms, prefill 2000: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 100.9 ms (**1.00x faster than hipinfer**) |


### Qwen2.5-1.5B

| | RTX 2080 Ti |
|---|---|
| decode ms/token, decode 128: **hipinfer** | 6.21 ms, 81% BW |
| decode ms/token, decode 128: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 7.19 ms, 70% BW (hipinfer 1.16x faster) |
| decode ms/token, decode 1024: **hipinfer** | 6.32 ms, 80% BW |
| decode ms/token, decode 1024: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 7.22 ms, 70% BW (hipinfer 1.14x faster) |

| | RTX 2080 Ti |
|---|---|
| prefill ms, prefill 128: **hipinfer** | 13.4 ms |
| prefill ms, prefill 128: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 17.3 ms (hipinfer 1.29x faster) |
| prefill ms, prefill 512: **hipinfer** | 37.8 ms |
| prefill ms, prefill 512: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 40.1 ms (hipinfer 1.06x faster) |
| prefill ms, prefill 2000: **hipinfer** | 140.9 ms |
| prefill ms, prefill 2000: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 156.7 ms (hipinfer 1.11x faster) |


### Llama-3.2-3B

| | RTX 2080 Ti |
|---|---|
| decode ms/token, decode 128: **hipinfer** | 12.11 ms, 87% BW |
| decode ms/token, decode 128: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 12.70 ms, 83% BW (hipinfer 1.05x faster) |
| decode ms/token, decode 1024: **hipinfer** | 12.54 ms, 85% BW |
| decode ms/token, decode 1024: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 12.92 ms, 82% BW (hipinfer 1.03x faster) |

| | RTX 2080 Ti |
|---|---|
| prefill ms, prefill 128: **hipinfer** | 24.3 ms |
| prefill ms, prefill 128: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 25.8 ms (hipinfer 1.06x faster) |
| prefill ms, prefill 512: **hipinfer** | 76.7 ms |
| prefill ms, prefill 512: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 65.4 ms (**1.17x faster than hipinfer**) |
| prefill ms, prefill 2000: **hipinfer** | 289.7 ms |
| prefill ms, prefill 2000: llama.cpp (F16 GGUF, CUDA, best of FA on/off) | 261.8 ms (**1.11x faster than hipinfer**) |


## Tensor parallelism (multi-GPU)

Megatron-style sharding over N GPUs in one process (one worker thread per GPU): QKV/gate/up column-parallel, O/down row-parallel with an all-reduce after each, lm_head vocabulary-parallel. All-reduce `auto` = hipinfer's one-shot kernel for messages up to 512 KB (decode), NCCL above (prefill); `NCCL` = NCCL only. Prefill all-reduces overlap with compute (two row halves on two streams) unless noted. % floor = this GPU's shard of the weights + its KV share / peak BW / time.


### RTX 2080 Ti + RTX 2080 Ti (`2080ti`), tp=2

Peer-to-peer access between the GPUs: no (one-shot all-reduce stages through pinned host memory); NCCL built in: yes.

All-reduce latency, fp16, in place (TinyLlama-1.1B hidden size 2048):

| message | size | one-shot kernel | NCCL | one-shot speedup |
|---|---|---|---|---|
| decode B=1 | 4 KB | 8.5 us | 12.2 us | 1.44x |
| decode B=8 | 32 KB | 17.7 us | 24.9 us | 1.41x |
| decode B=64 | 256 KB | 63.6 us | 85.4 us | 1.34x |
| prefill T=512 | 2048 KB | 425.0 us | 426.0 us | 1.00x |
| prefill T=2000 | 8000 KB | 1574.4 us | 1554.9 us | 0.99x |

Batch-1 decode, ms/token (1 GPU vs tp=2):

| model | ctx | 1 GPU | tp=2 auto | tp=2 NCCL only | speedup | % floor (tp) |
|---|---|---|---|---|---|---|
| TinyLlama-1.1B | 128 | 4.15 | **2.71** | 3.04 | 1.53x | 62 % |
| TinyLlama-1.1B | 1024 | 4.23 | **2.83** | 3.12 | 1.49x | 60 % |
| Qwen2.5-0.5B | 128 | 2.33 | **1.76** | 2.07 | 1.33x | 46 % |
| Qwen2.5-0.5B | 1024 | 2.42 | **1.79** | 2.12 | 1.35x | 45 % |
| Qwen2.5-0.5B | 4096 | 2.63 | **1.98** | 2.33 | 1.33x | 43 % |
| Llama-3.2-1B | 128 | 4.72 | **2.81** | 2.97 | 1.68x | 72 % |
| Llama-3.2-1B | 1024 | 4.80 | **2.86** | 3.02 | 1.68x | 71 % |
| Llama-3.2-1B | 4096 | 4.97 | **2.97** | 3.13 | 1.68x | 71 % |
| Qwen2.5-1.5B | 128 | 6.19 | **3.87** | 4.21 | 1.60x | 65 % |
| Qwen2.5-1.5B | 1024 | 6.29 | **3.95** | 4.31 | 1.59x | 64 % |
| Qwen2.5-1.5B | 4096 | 6.83 | **4.28** | 4.62 | 1.60x | 61 % |
| Llama-3.2-3B | 128 | 11.99 | **6.89** | 7.27 | 1.74x | 76 % |
| Llama-3.2-3B | 1024 | 12.37 | **7.03** | 7.42 | 1.76x | 76 % |
| Llama-3.2-3B | 4096 | 12.98 | **7.51** | 7.87 | 1.73x | 75 % |
| Mistral-7B-v0.3 | 128 | does not fit | **13.99** | 14.24 | - | 83 % |
| Mistral-7B-v0.3 | 1024 | does not fit | **14.18** | 14.41 | - | 82 % |
| Mistral-7B-v0.3 | 4096 | does not fit | **14.72** | 14.97 | - | 81 % |
| Mistral-7B-v0.1 (window ignored) | 128 | does not fit | **13.96** | 14.24 | - | 83 % |
| Mistral-7B-v0.1 (window ignored) | 1024 | does not fit | **14.15** | 14.44 | - | 82 % |
| Mistral-7B-v0.1 (window ignored) | 4096 | does not fit | **14.72** | 15.02 | - | 81 % |
| Llama-3.1-8B | 128 | does not fit | **14.68** | 14.96 | - | 83 % |
| Llama-3.1-8B | 1024 | does not fit | **14.89** | 15.15 | - | 83 % |
| Llama-3.1-8B | 4096 | does not fit | **15.43** | 15.70 | - | 82 % |

Prefill, ms (1 GPU vs tp=2):

| model | tokens | 1 GPU | tp=2 auto, overlapped | tp=2 auto, no overlap | tp=2 NCCL only | speedup |
|---|---|---|---|---|---|---|
| TinyLlama-1.1B | 512 | 27.9 | **24.7** | 33.9 | 24.7 | 1.13x |
| TinyLlama-1.1B | 2000 | 105.3 | **83.8** | 125.5 | 83.0 | 1.26x |
| Qwen2.5-0.5B | 512 | 12.2 | **16.3** | 16.4 | 14.3 | 0.75x |
| Qwen2.5-0.5B | 2000 | 43.8 | **41.6** | 57.5 | 41.1 | 1.05x |
| Llama-3.2-1B | 512 | 26.6 | **20.7** | 26.6 | 20.8 | 1.29x |
| Llama-3.2-1B | 2000 | 102.2 | **70.6** | 99.0 | 71.7 | 1.45x |
| Qwen2.5-1.5B | 512 | 38.4 | **31.3** | 37.7 | 30.9 | 1.23x |
| Qwen2.5-1.5B | 2000 | 141.5 | **101.5** | 135.5 | 101.8 | 1.39x |
| Llama-3.2-3B | 512 | 76.4 | **53.0** | 70.6 | 53.8 | 1.44x |
| Llama-3.2-3B | 2000 | 286.7 | **182.2** | 267.2 | 184.1 | 1.57x |
| Mistral-7B-v0.3 | 512 | does not fit | **101.0** | 129.5 | 102.9 | - |
| Mistral-7B-v0.3 | 2000 | does not fit | **366.1** | 493.4 | 371.5 | - |
| Mistral-7B-v0.1 (window ignored) | 512 | does not fit | **116.1** | 141.6 | 124.9 | - |
| Mistral-7B-v0.1 (window ignored) | 2000 | does not fit | **425.3** | 547.6 | 463.7 | - |
| Llama-3.1-8B | 512 | does not fit | **103.2** | 131.6 | 104.4 | - |
| Llama-3.1-8B | 2000 | does not fit | **371.1** | 498.9 | 375.0 | - |

Batched decode throughput, tok/s (ctx 512 + 128 new tokens):

| model | B | 1 GPU | tp=2 auto | tp=2 NCCL only | speedup |
|---|---|---|---|---|---|
| TinyLlama-1.1B | 1 | 239 | **356** | 322 | 1.49x |
| TinyLlama-1.1B | 4 | 857 | **1181** | 1090 | 1.38x |
| TinyLlama-1.1B | 8 | 1643 | **2121** | 1841 | 1.29x |
| TinyLlama-1.1B | 16 | 3076 | **3823** | 3122 | 1.24x |
| TinyLlama-1.1B | 32 | 4733 | **5581** | 4559 | 1.18x |
| TinyLlama-1.1B | 64 | 7459 | **8464** | 7062 | 1.13x |
| Qwen2.5-0.5B | 1 | 421 | **565** | 477 | 1.34x |
| Qwen2.5-0.5B | 4 | 1451 | **1819** | 1544 | 1.25x |
| Qwen2.5-0.5B | 8 | 2424 | **2724** | 2246 | 1.12x |
| Qwen2.5-0.5B | 16 | 4589 | **4915** | 4127 | 1.07x |
| Qwen2.5-0.5B | 32 | 7294 | **8266** | 6590 | 1.13x |
| Qwen2.5-0.5B | 64 | 12.0k | **13.2k** | 10.4k | 1.10x |
| Llama-3.2-1B | 1 | 211 | **354** | 331 | 1.68x |
| Llama-3.2-1B | 4 | 770 | **1210** | 1137 | 1.57x |
| Llama-3.2-1B | 8 | 1512 | **2253** | 2020 | 1.49x |
| Llama-3.2-1B | 16 | 2831 | **4023** | 3457 | 1.42x |
| Llama-3.2-1B | 32 | 4666 | **6181** | 5247 | 1.32x |
| Llama-3.2-1B | 64 | 7530 | **9454** | 8190 | 1.26x |
| Qwen2.5-1.5B | 1 | 161 | **256** | 233 | 1.59x |
| Qwen2.5-1.5B | 4 | 578 | **880** | 799 | 1.52x |
| Qwen2.5-1.5B | 8 | 875 | **1305** | 1197 | 1.49x |
| Qwen2.5-1.5B | 16 | 1686 | **2344** | 2109 | 1.39x |
| Qwen2.5-1.5B | 32 | 2723 | **3632** | 3133 | 1.33x |
| Qwen2.5-1.5B | 64 | 4476 | **5675** | 4999 | 1.27x |
| Llama-3.2-3B | 1 | 82 | **144** | 136 | 1.75x |
| Llama-3.2-3B | 4 | 297 | **492** | 465 | 1.66x |
| Llama-3.2-3B | 8 | 488 | **800** | 747 | 1.64x |
| Llama-3.2-3B | 16 | 910 | **1435** | 1282 | 1.58x |
| Llama-3.2-3B | 32 | - | **2196** | 1982 | - |
| Llama-3.2-3B | 64 | - | **3298** | 3025 | - |
| Mistral-7B-v0.3 | 1 | - | **71** | 70 | - |
| Mistral-7B-v0.3 | 4 | - | **255** | 246 | - |
| Mistral-7B-v0.3 | 8 | - | **492** | 464 | - |
| Mistral-7B-v0.3 | 16 | - | **902** | 828 | - |
| Mistral-7B-v0.3 | 32 | - | **1408** | 1267 | - |
| Mistral-7B-v0.1 (window ignored) | 1 | - | **71** | 70 | - |
| Mistral-7B-v0.1 (window ignored) | 4 | - | **255** | 244 | - |
| Mistral-7B-v0.1 (window ignored) | 8 | - | **490** | 457 | - |
| Mistral-7B-v0.1 (window ignored) | 16 | - | **894** | 815 | - |
| Mistral-7B-v0.1 (window ignored) | 32 | - | **1339** | 1271 | - |
| Llama-3.1-8B | 1 | - | **68** | 67 | - |
| Llama-3.1-8B | 4 | - | **244** | 236 | - |
| Llama-3.1-8B | 8 | - | **471** | 446 | - |
| Llama-3.1-8B | 16 | - | **864** | 807 | - |

Correctness of the sharded engine (tests/test_tp.py: tp=2 vs the same weights on 1 GPU; tests/ppl.py --tp for models that only fit sharded):

| model | greedy runs token-identical to 1 GPU | prefill top-1 agreement | max abs logit diff | backends bitwise equal | all-reduce self-test | PPL sharded / HF |
|---|---|---|---|---|---|---|
| TinyLlama-1.1B | 13/13 | 99.84 % | 0.781 | yes | exact | - |
| Qwen2.5-0.5B | 11/13 (rest at near-ties) | 99.69 % | 0.482 | yes | exact | - |
| Llama-3.2-1B | 11/13 (rest at near-ties) | 100.00 % | 0.064 | yes | exact | - |
| Qwen2.5-1.5B | 13/13 | 98.91 % | 1.152 | yes | exact | - |
| Llama-3.2-3B | 13/13 | 99.84 % | 0.316 | yes | exact | - |
| Mistral-7B-v0.3 | no 1-GPU reference | - | - | yes | exact | 5.7773 / 5.7775 |
| Mistral-7B-v0.1 (window ignored) | no 1-GPU reference | - | - | yes | exact | 7.1826 / 7.1833 |
| Llama-3.1-8B | no 1-GPU reference | - | - | yes | exact | 6.5467 / 6.5471 |

