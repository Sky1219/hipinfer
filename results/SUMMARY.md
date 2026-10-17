## Models

| model | params (fp16 GB) | layers | hidden | heads (q/kv) | head_dim | vocab | notes |
|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B | 2.07 | 22 | 2048 | 32/4 | 64 | 32000 | GQA group 8 |
| Qwen2.5-0.5B | 0.99 | 24 | 896 | 14/2 | 64 | 151936 | tied embeddings, QKV bias, GQA group 7 |
| Llama-3.2-1B | 2.47 | 16 | 2048 | 32/8 | 64 | 128256 | tied embeddings, GQA group 4 |
| Qwen2.5-1.5B | 3.09 | 28 | 1536 | 12/2 | 128 | 151936 | tied embeddings, QKV bias, GQA group 6 |
| Llama-3.2-3B | 6.43 | 28 | 3072 | 24/8 | 128 | 128256 | tied embeddings, GQA group 3 |
| Mistral-7B-v0.3 | 14.23 | 32 | 4096 | 32/8 | 128 | 32768 | GQA group 4 |
| Llama-3.1-8B | 15.01 | 32 | 4096 | 32/8 | 128 | 128256 | GQA group 4 |
| Granite-3.1-1B-A400M (MoE) | 2.6 (0.86 read per token) | 24 | 1024 | 16/8 | 64 | 49152 | GQA group 2, MoE, 32 experts, top-8 |

## Correctness vs HF transformers (fp16)

Greedy gate: 7 prompts x 128 tokens, single-sequence with and without CUDA graphs, plus a 6-sequence batch (20 runs). A run that diverges only where HF's own top-2 logits are within 0.05 counts as a near-tie. Perplexity: WikiText-2 test, non-overlapping 1024-token windows, both engines on the same tokens.

| model | GPU | gate | runs token-identical | PPL hipinfer | PPL HF | top-1 agreement | mean KL(HF, hipinfer) | max abs logit diff |
|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B | RTX 2080 Ti | PASS | 20/20 | 8.5149 | 8.5151 | 99.90% | 6.6e-06 | 0.193 |
| TinyLlama-1.1B | A100-SXM4-40GB | PASS | 20/20 | 10.7326 | 10.7326 | 99.85% | 6.3e-06 | 0.119 |
| TinyLlama-1.1B | RTX A6000 | PASS | 20/20 | 8.5149 | 8.5150 | 99.88% | 6.6e-06 | 0.111 |
| Qwen2.5-0.5B | RTX 2080 Ti | PASS | 18/20 | 13.4851 | 13.4851 | 99.79% | 1.6e-05 | 0.129 |
| Qwen2.5-0.5B | A100-SXM4-40GB | PASS | 20/20 | 17.2227 | 17.2236 | 99.74% | 1.6e-05 | 0.125 |
| Qwen2.5-0.5B | RTX A6000 | PASS | 19/20 | 13.4846 | 13.4849 | 99.74% | 1.5e-05 | 0.175 |
| Llama-3.2-1B | RTX 2080 Ti | PASS | 13/20 | 10.4633 | 10.4631 | 99.81% | 1.3e-05 | 0.131 |
| Llama-3.2-1B | A100-SXM4-40GB | PASS | 13/20 | 12.5618 | 12.5619 | 99.87% | 1.3e-05 | 0.156 |
| Llama-3.2-1B | RTX A6000 | PASS | 11/20 | 10.4631 | 10.4633 | 99.86% | 1.3e-05 | 0.129 |
| Qwen2.5-1.5B | RTX 2080 Ti | PASS | 17/20 | 9.4536 | 9.4535 | 99.86% | 1.6e-05 | 0.188 |
| Qwen2.5-1.5B | A100-SXM4-40GB | PASS | 16/20 | 11.9787 | 11.9787 | 99.77% | 1.6e-05 | 0.216 |
| Qwen2.5-1.5B | RTX A6000 | PASS | 17/20 | 9.4534 | 9.4539 | 99.77% | 1.6e-05 | 0.285 |
| Llama-3.2-3B | RTX 2080 Ti | PASS | 20/20 | 8.2767 | 8.2768 | 99.82% | 9.5e-06 | 0.367 |
| Llama-3.2-3B | A100-SXM4-40GB | PASS | 20/20 | 10.2217 | 10.2214 | 99.90% | 9.5e-06 | 0.723 |
| Llama-3.2-3B | RTX A6000 | PASS | 20/20 | 8.2770 | 8.2765 | 99.88% | 9.8e-06 | 1.320 |
| Mistral-7B-v0.3 | A100-SXM4-40GB | PASS | 20/20 | 7.3269 | 7.3267 | 99.88% | 4.2e-06 | 0.177 |
| Mistral-7B-v0.3 | RTX A6000 | PASS | 20/20 | 5.7774 | 5.7774 | 99.91% | 4.6e-06 | 1.939 |
| Mistral-7B-v0.1 (window ignored) | RTX A6000 | PASS | 20/20 | 5.6893 | 5.6893 | 99.84% | 4.6e-06 | 0.506 |
| Llama-3.1-8B | RTX A6000 | PASS | 14/20 | 6.5467 | 6.5469 | 99.89% | 1.1e-05 | 3.141 |
| Granite-3.1-1B-A400M (MoE) | RTX 2080 Ti | PASS | 12/20 | 7.7292 | 7.7294 | 99.41% | 2.6e-04 | 3.594 |

## Decode, batch 1 (ms/token, greedy, 256 new tokens after a 128-token prompt)

Floor = (weight bytes + KV bytes read) / peak DRAM bandwidth. HF = transformers eager with a DynamicCache.

⚠ = RTX A6000 benchmark ran partly thermally throttled; this and every speed table below flags it. Details: results/a6000/THERMAL.md.

| model | RTX 2080 Ti | A100-SXM4-40GB | RTX A6000 |
|---|---|---|---|
| peak DRAM BW (GB/s) | 616 | 1555 | 768 |
| TinyLlama-1.1B | **4.14** (81% BW; HF 31.3, 7.6x) | **2.17** (62% BW; HF 23.5, 10.8x) | **3.43** (79% BW; HF 19.3, 5.6x) |
| Qwen2.5-0.5B | **2.34** (69% BW; HF 27.0, 11.5x) | **1.49** (43% BW; HF 25.2, 16.9x) | **1.92** (67% BW; HF 21.2, 11.1x) |
| Llama-3.2-1B | **4.75** (85% BW; HF 18.2, 3.8x) | **2.33** (68% BW; HF 17.5, 7.5x) | **3.95** (82% BW; HF 14.6, 3.7x) ⚠ |
| Qwen2.5-1.5B | **6.21** (81% BW; HF 40.6, 6.5x) | **3.20** (62% BW; HF 30.8, 9.6x) | **5.14** (78% BW; HF 34.9, 6.8x) ⚠ |
| Llama-3.2-3B | **12.11** (87% BW; HF copy does not fit) | **5.61** (74% BW; HF 30.0, 5.4x) | **9.97** (84% BW; HF 33.2, 3.3x) ⚠ |
| Mistral-7B-v0.3 | - | **11.07** (83% BW; HF 36.0, 3.2x) | **21.36** (87% BW; HF 37.1, 1.7x) ⚠ |
| Llama-3.1-8B | - | - | **24.87** (79% BW; HF 37.0, 1.5x) ⚠ |
| Granite-3.1-1B-A400M (MoE) | **2.26** (63% BW; HF 67.2, 29.8x) | - | - |

## Decode throughput, batched (tok/s; ctx 512 + 128 new tokens; largest batch that fits)

| model | RTX 2080 Ti | A100-SXM4-40GB | RTX A6000 |
|---|---|---|---|
| TinyLlama-1.1B | B=8: 1639; B=64: **7745** (8.3 ms/step, 32 mJ/tok at 248 W) | B=8: 2556; B=64: **14.0k** (4.6 ms/step, 19 mJ/tok at 262 W) | B=8: 1991; B=64: **9996** (6.4 ms/step, 30 mJ/tok at 299 W) |
| Qwen2.5-0.5B | B=8: 2421; B=64: **11.9k** (5.4 ms/step, 20 mJ/tok at 244 W) | B=8: 3390; B=64: **19.8k** (3.2 ms/step, 11 mJ/tok at 216 W) | B=8: 3354; B=64: **18.0k** (3.5 ms/step, 16 mJ/tok at 289 W) |
| Llama-3.2-1B | B=8: 1493; B=64: **7322** (8.7 ms/step, 31 mJ/tok at 228 W) | B=8: 2641; B=64: **13.9k** (4.6 ms/step, 21 mJ/tok at 287 W) | B=8: 1786; B=64: **9422** (6.8 ms/step, 30 mJ/tok at 278 W) ⚠ |
| Qwen2.5-1.5B | B=8: 866; B=64: **4332** (14.8 ms/step, 51 mJ/tok at 219 W) | B=8: 1839; B=64: **10.3k** (6.2 ms/step, 24 mJ/tok at 244 W) | B=8: 1350; B=64: **7439** (8.6 ms/step, 38 mJ/tok at 284 W) ⚠ |
| Llama-3.2-3B | B=8: 479; B=16: **888** (18.0 ms/step, 249 mJ/tok at 221 W) | B=8: 1116; B=64: **5565** (11.5 ms/step, 51 mJ/tok at 284 W) | B=8: 674; B=64: **1920** (33.3 ms/step, 93 mJ/tok at 179 W) ⚠ |
| Mistral-7B-v0.3 | - | B=8: 600; B=64: **3221** (19.9 ms/step, 95 mJ/tok at 306 W) | B=8: 181; B=64: **1082** (59.2 ms/step, 168 mJ/tok at 182 W) ⚠ |
| Llama-3.1-8B | - | - | B=8: 189; B=64: **908** (70.5 ms/step, 198 mJ/tok at 180 W) ⚠ |
| Granite-3.1-1B-A400M (MoE) | B=8: 1561; B=64: **5879** (10.9 ms/step, 42 mJ/tok at 249 W) | - | - |

## Prefill (prompt tokens/s; ms in parentheses; HF = transformers eager, SDPA attention)

| model | T | RTX 2080 Ti | A100-SXM4-40GB | RTX A6000 |
|---|---|---|---|---|
| TinyLlama-1.1B | 128 | 15.4k (8.3 ms; HF 34.0 ms, 4.07x) | 22.7k (5.7 ms; HF 24.4 ms, 4.32x) | 20.4k (6.3 ms; HF 19.2 ms, 3.06x) |
| TinyLlama-1.1B | 512 | 19.2k (26.6 ms; HF 51.6 ms, 1.94x) | 45.9k (11.2 ms; HF 24.6 ms, 2.21x) | 30.7k (16.7 ms; HF 20.5 ms, 1.23x) |
| TinyLlama-1.1B | 1024 | 20.2k (50.6 ms; HF 138.5 ms, 2.74x) | 55.4k (18.5 ms; HF 25.6 ms, 1.38x) | 36.4k (28.2 ms; HF 33.2 ms, 1.18x) |
| TinyLlama-1.1B | 2000 | 19.9k (100.5 ms; HF 426.6 ms, 4.24x) | 53.4k (37.5 ms; HF 32.9 ms, 0.88x) | 33.5k (59.6 ms; HF 61.4 ms, 1.03x) |
| Qwen2.5-0.5B | 128 | 24.9k (5.1 ms; HF 29.9 ms, 5.81x) | 30.2k (4.2 ms; HF 26.4 ms, 6.23x) | 38.1k (3.4 ms; HF 22.2 ms, 6.62x) |
| Qwen2.5-0.5B | 512 | 42.3k (12.1 ms; HF 31.4 ms, 2.60x) | 65.0k (7.9 ms; HF 26.9 ms, 3.41x) | 70.6k (7.3 ms; HF 22.4 ms, 3.09x) |
| Qwen2.5-0.5B | 1024 | 44.1k (23.2 ms; HF 64.5 ms, 2.77x) | 105.9k (9.7 ms; HF 27.7 ms, 2.86x) | 75.7k (13.5 ms; HF 23.2 ms, 1.72x) |
| Qwen2.5-0.5B | 2000 | 46.4k (43.1 ms; HF 201.4 ms, 4.68x) | 115.9k (17.3 ms; HF 27.3 ms, 1.58x) | 80.3k (24.9 ms; HF 30.2 ms, 1.21x) |
| Qwen2.5-0.5B | 4096 | 43.1k (95.0 ms; HF 684.8 ms, 7.21x) | 118.3k (34.6 ms; HF 32.8 ms, 0.95x) | 78.5k (52.2 ms; HF 61.5 ms, 1.18x) |
| Llama-3.2-1B | 128 | 14.9k (8.6 ms; HF 20.6 ms, 2.40x) | 26.3k (4.9 ms; HF 18.1 ms, 3.72x) | 20.1k (6.4 ms; HF 15.5 ms, 2.44x) ⚠ |
| Llama-3.2-1B | 512 | 19.6k (26.1 ms; HF 42.2 ms, 1.62x) | 51.2k (10.0 ms; HF 18.1 ms, 1.81x) | 30.9k (16.6 ms; HF 17.3 ms, 1.04x) ⚠ |
| Llama-3.2-1B | 1024 | 20.5k (49.9 ms; HF 115.2 ms, 2.31x) | 58.4k (17.5 ms; HF 19.1 ms, 1.09x) | 34.1k (30.0 ms; HF 30.3 ms, 1.01x) ⚠ |
| Llama-3.2-1B | 2000 | 19.8k (101.1 ms; HF 335.4 ms, 3.32x) | 58.2k (34.4 ms; HF 28.9 ms, 0.84x) | 33.7k (59.3 ms; HF 56.9 ms, 0.96x) ⚠ |
| Llama-3.2-1B | 4096 | 19.0k (215.8 ms; HF 1094.8 ms, 5.07x) | 60.2k (68.0 ms; HF 62.0 ms, 0.91x) | 34.0k (120.5 ms; HF 119.6 ms, 0.99x) ⚠ |
| Qwen2.5-1.5B | 128 | 9575 (13.4 ms; HF 44.5 ms, 3.33x) | 19.5k (6.6 ms; HF 30.7 ms, 4.68x) | 14.2k (9.0 ms; HF 34.0 ms, 3.78x) ⚠ |
| Qwen2.5-1.5B | 512 | 13.6k (37.8 ms; HF 56.3 ms, 1.49x) | 37.0k (13.8 ms; HF 30.6 ms, 2.21x) | 22.5k (22.8 ms; HF 35.0 ms, 1.54x) ⚠ |
| Qwen2.5-1.5B | 1024 | 14.6k (70.0 ms; HF 132.0 ms, 1.88x) | 43.4k (23.6 ms; HF 32.0 ms, 1.36x) | 24.9k (41.1 ms; HF 41.4 ms, 1.01x) ⚠ |
| Qwen2.5-1.5B | 2000 | 14.2k (140.9 ms; HF 337.0 ms, 2.39x) | 45.8k (43.7 ms; HF 40.0 ms, 0.92x) | 24.6k (81.4 ms; HF 77.1 ms, 0.95x) ⚠ |
| Qwen2.5-1.5B | 4096 | 14.1k (291.5 ms; HF 985.6 ms, 3.38x) | 45.1k (90.8 ms; HF 82.6 ms, 0.91x) | 24.2k (169.6 ms; HF 158.7 ms, 0.94x) ⚠ |
| Llama-3.2-3B | 128 | 5273 (24.3 ms) | 9546 (13.4 ms; HF 30.1 ms, 2.25x) | 7777 (16.5 ms; HF 33.4 ms, 2.03x) ⚠ |
| Llama-3.2-3B | 512 | 6673 (76.7 ms) | 21.1k (24.3 ms; HF 30.3 ms, 1.25x) | 10.9k (46.9 ms; HF 42.6 ms, 0.91x) ⚠ |
| Llama-3.2-3B | 1024 | 6863 (149.2 ms) | 22.6k (45.3 ms; HF 39.6 ms, 0.87x) | 12.1k (84.9 ms; HF 81.8 ms, 0.96x) ⚠ |
| Llama-3.2-3B | 2000 | 6905 (289.7 ms) | 22.9k (87.3 ms; HF 78.0 ms, 0.89x) | 12.3k (163.2 ms; HF 151.7 ms, 0.93x) ⚠ |
| Llama-3.2-3B | 4096 | 6459 (634.1 ms) | 22.2k (184.4 ms; HF 160.7 ms, 0.87x) | 11.8k (345.8 ms; HF 315.0 ms, 0.91x) ⚠ |
| Mistral-7B-v0.3 | 128 | - | 6058 (21.1 ms; HF 36.0 ms, 1.70x) | 3818 (33.5 ms; HF 36.7 ms, 1.10x) ⚠ |
| Mistral-7B-v0.3 | 512 | - | 9300 (55.1 ms; HF 45.3 ms, 0.82x) | 5046 (101.5 ms; HF 89.1 ms, 0.88x) ⚠ |
| Mistral-7B-v0.3 | 1024 | - | 9732 (105.2 ms; HF 83.5 ms, 0.79x) | 5168 (198.1 ms; HF 171.1 ms, 0.86x) ⚠ |
| Mistral-7B-v0.3 | 2000 | - | 9855 (202.9 ms; HF 160.8 ms, 0.79x) | 5158 (387.8 ms; HF 331.7 ms, 0.86x) ⚠ |
| Mistral-7B-v0.3 | 4096 | - | 9896 (413.9 ms; HF 326.2 ms, 0.79x) | 5073 (807.4 ms; HF 673.3 ms, 0.83x) ⚠ |
| Llama-3.1-8B | 128 | - | - | 3635 (35.2 ms; HF 39.3 ms, 1.12x) ⚠ |
| Llama-3.1-8B | 512 | - | - | 4899 (104.5 ms; HF 91.6 ms, 0.88x) ⚠ |
| Llama-3.1-8B | 1024 | - | - | 4983 (205.5 ms; HF 174.7 ms, 0.85x) ⚠ |
| Llama-3.1-8B | 2000 | - | - | 5041 (396.8 ms; HF 339.9 ms, 0.86x) ⚠ |
| Llama-3.1-8B | 4096 | - | - | 2494 (1642.2 ms; HF 718.9 ms, 0.44x) ⚠ |
| Granite-3.1-1B-A400M (MoE) | 128 | 7202 (17.8 ms; HF 105.8 ms, 5.95x) | - | - |
| Granite-3.1-1B-A400M (MoE) | 512 | 18.8k (27.2 ms; HF 107.0 ms, 3.94x) | - | - |
| Granite-3.1-1B-A400M (MoE) | 1024 | 24.4k (42.0 ms; HF 135.2 ms, 3.22x) | - | - |
| Granite-3.1-1B-A400M (MoE) | 2000 | 28.4k (70.4 ms; HF 274.9 ms, 3.90x) | - | - |
| Granite-3.1-1B-A400M (MoE) | 4096 | 30.3k (135.2 ms; HF 824.3 ms, 6.10x) | - | - |

## Decode latency vs context length (batch 1, ms/token; % of the bandwidth floor incl. KV reads)

| model | ctx | RTX 2080 Ti | A100-SXM4-40GB | RTX A6000 |
|---|---|---|---|---|
| TinyLlama-1.1B | 128 | 4.13 (81%) | 2.16 (62%) | 3.43 (79%) |
| TinyLlama-1.1B | 512 | 4.17 (81%) | 2.18 (62%) | 3.45 (79%) |
| TinyLlama-1.1B | 1024 | 4.22 (81%) | 2.23 (60%) | 3.49 (78%) |
| Qwen2.5-0.5B | 128 | 2.33 (69%) | 1.48 (43%) | 1.91 (67%) |
| Qwen2.5-0.5B | 512 | 2.38 (68%) | 1.50 (43%) | 1.93 (67%) |
| Qwen2.5-0.5B | 1024 | 2.42 (67%) | 1.53 (42%) | 1.96 (66%) |
| Qwen2.5-0.5B | 2048 | 2.50 (66%) | 1.61 (41%) | 2.05 (64%) |
| Qwen2.5-0.5B | 4096 | 2.64 (64%) | 1.78 (38%) | 2.18 (62%) |
| Qwen2.5-0.5B | 8192 | 2.73 (65%) | 1.86 (38%) | 2.26 (63%) |
| Llama-3.2-1B | 128 | 4.77 (84%) | 2.33 (68%) | 3.95 (82%) ⚠ |
| Llama-3.2-1B | 512 | 4.80 (84%) | 2.34 (68%) | 3.97 (82%) ⚠ |
| Llama-3.2-1B | 1024 | 4.84 (84%) | 2.36 (68%) | 4.01 (81%) ⚠ |
| Llama-3.2-1B | 2048 | 4.90 (84%) | 2.41 (68%) | 4.07 (81%) ⚠ |
| Llama-3.2-1B | 4096 | 5.02 (84%) | 2.48 (68%) | 4.17 (81%) ⚠ |
| Llama-3.2-1B | 8192 | 5.26 (85%) | 2.61 (68%) | 4.37 (82%) ⚠ |
| Qwen2.5-1.5B | 128 | 6.22 (81%) | 3.19 (62%) | 5.15 (78%) ⚠ |
| Qwen2.5-1.5B | 512 | 6.27 (80%) | 3.21 (62%) | 5.18 (78%) ⚠ |
| Qwen2.5-1.5B | 1024 | 6.32 (80%) | 3.27 (61%) | 5.23 (78%) ⚠ |
| Qwen2.5-1.5B | 2048 | 6.48 (79%) | 3.41 (59%) | 5.39 (76%) ⚠ |
| Qwen2.5-1.5B | 4096 | 6.90 (75%) | 3.71 (56%) | 5.63 (74%) ⚠ |
| Qwen2.5-1.5B | 8192 | 7.11 (76%) | 3.81 (56%) | 5.81 (74%) ⚠ |
| Llama-3.2-3B | 128 | 12.11 (86%) | 5.60 (74%) | 10.10 (83%) ⚠ |
| Llama-3.2-3B | 512 | 12.29 (86%) | 5.64 (74%) | 10.23 (83%) ⚠ |
| Llama-3.2-3B | 1024 | 12.53 (85%) | 5.70 (74%) | 10.33 (83%) ⚠ |
| Llama-3.2-3B | 2048 | 12.68 (85%) | 5.86 (73%) | 10.56 (82%) ⚠ |
| Llama-3.2-3B | 4096 | 13.13 (85%) | 6.03 (74%) | 11.06 (81%) ⚠ |
| Llama-3.2-3B | 8192 | 13.99 (86%) | 6.37 (74%) | 11.95 (80%) ⚠ |
| Mistral-7B-v0.3 | 128 | - | 11.06 (83%) | 32.94 (56%) ⚠ |
| Mistral-7B-v0.3 | 512 | - | 11.10 (83%) | 33.09 (56%) ⚠ |
| Mistral-7B-v0.3 | 1024 | - | 11.18 (83%) | 28.41 (66%) ⚠ |
| Mistral-7B-v0.3 | 2048 | - | 11.38 (82%) | 22.85 (83%) ⚠ |
| Mistral-7B-v0.3 | 4096 | - | 11.57 (82%) | 22.42 (86%) ⚠ |
| Mistral-7B-v0.3 | 8192 | - | 11.96 (82%) | 23.66 (84%) ⚠ |
| Llama-3.1-8B | 128 | - | - | 43.77 (45%) ⚠ |
| Llama-3.1-8B | 512 | - | - | 43.83 (45%) ⚠ |
| Llama-3.1-8B | 1024 | - | - | 36.45 (54%) ⚠ |
| Llama-3.1-8B | 2048 | - | - | 33.46 (59%) ⚠ |
| Llama-3.1-8B | 4096 | - | - | 33.90 (60%) ⚠ |
| Llama-3.1-8B | 8192 | - | - | 29.48 (71%) ⚠ |
| Granite-3.1-1B-A400M (MoE) | 128 | 2.25 (62%) | - | - |
| Granite-3.1-1B-A400M (MoE) | 512 | 2.30 (63%) | - | - |
| Granite-3.1-1B-A400M (MoE) | 1024 | 2.36 (63%) | - | - |
| Granite-3.1-1B-A400M (MoE) | 2048 | 2.43 (64%) | - | - |

## Kernels vs PyTorch (speedup = torch time / hipinfer time; graph-replayed; TinyLlama shapes)

| | RTX 2080 Ti | A100-SXM4-40GB | RTX A6000 |
|---|---|---|---|
| GEMM qkv M=2048 vs cuBLAS | 0.84x (41 TF) | 0.58x (120 TF) | 0.79x (86 TF) |
| GEMM gate_up M=2048 vs cuBLAS | 0.92x (47 TF) | 0.64x (150 TF) | 0.79x (85 TF) |
| GEMM down M=512 vs cuBLAS | 0.94x (44 TF) | 0.44x (82 TF) | 0.94x (63 TF) |
| GEMM o M=128 vs cuBLAS | 0.87x (26 TF) | 0.46x (37 TF) | 0.54x (33 TF) |
| GEMV norm+gate_up+SwiGLU, M=1 | 1.40x (556 GB/s) | 1.99x (1205 GB/s) | 1.40x (672 GB/s) |
| GEMV down+residual, M=1 | 0.99x (519 GB/s) | 1.60x (1746 GB/s) | 1.04x (563 GB/s) |
| GEMV norm+lm_head, M=8 | 1.03x (520 GB/s) | 0.95x (1020 GB/s) | 1.13x (654 GB/s) |
| RMSNorm 4096 rows vs F.rms_norm | 1.07x (519 GB/s) | 1.10x (1400 GB/s) | 1.23x (619 GB/s) |
| RoPE+KV write, 512 tok vs HF | 11.02x | 8.80x | 9.05x |
| causal attention T=1024 vs SDPA | 29.23x (32 TF) | 0.72x (61 TF) | 0.90x (57 TF) |
| causal attention T=2048 vs SDPA | 27.71x (33 TF) | 0.67x (80 TF) | 0.92x (69 TF) |
| decode attention B=1 L=4096 vs SDPA | 29.69x (273 GB/s) | 0.66x (181 GB/s) | 0.87x (273 GB/s) |
| decode attention B=8 L=4096 vs SDPA | 44.15x (435 GB/s) | 0.73x (755 GB/s) | 1.01x (590 GB/s) |

## GEMM vs cuBLAS at every model's shapes (TFLOPs, M=2048 prefill rows; ratio = hipinfer / cuBLAS)

| model | shape | RTX 2080 Ti | A100-SXM4-40GB | RTX A6000 |
|---|---|---|---|---|
| TinyLlama-1.1B | qkv | 41 / 49 (0.84) | 120 / 206 (0.58) | 86 / 110 (0.79) |
| TinyLlama-1.1B | o | 47 / 51 (0.93) | 117 / 241 (0.49) | 70 / 101 (0.69) |
| TinyLlama-1.1B | gate_up | 47 / 51 (0.92) | 150 / 236 (0.64) | 85 / 108 (0.79) |
| TinyLlama-1.1B | down | 48 / 51 (0.95) | 128 / 216 (0.59) | 66 / 103 (0.64) |
| TinyLlama-1.1B | lm_head | 48 / 52 (0.93) | 158 / 240 (0.66) | 85 / 115 (0.74) |
| Qwen2.5-0.5B | qkv | 37 / 27 (1.40) | 98 / 178 (0.55) | 76 / 106 (0.72) |
| Qwen2.5-0.5B | o | 39 / 42 (0.91) | 78 / 152 (0.51) | 69 / 83 (0.83) |
| Qwen2.5-0.5B | gate_up | 44 / 48 (0.90) | 136 / 226 (0.60) | 81 / 102 (0.79) |
| Qwen2.5-0.5B | down | 42 / 45 (0.95) | 100 / 199 (0.50) | 67 / 110 (0.61) |
| Qwen2.5-0.5B | lm_head | 44 / 49 (0.90) | 144 / 226 (0.64) | 83 / 108 (0.77) |
| Llama-3.2-1B | qkv | 45 / 49 (0.93) | 148 / 235 (0.63) | 79 / 107 (0.74) ⚠ |
| Llama-3.2-1B | o | 46 / 48 (0.95) | 121 / 243 (0.50) | 70 / 100 (0.70) ⚠ |
| Llama-3.2-1B | gate_up | 46 / 49 (0.93) | 154 / 241 (0.64) | 84 / 111 (0.76) ⚠ |
| Llama-3.2-1B | down | 47 / 50 (0.95) | 132 / 227 (0.58) | 67 / 104 (0.64) ⚠ |
| Llama-3.2-1B | lm_head | 47 / 50 (0.94) | 160 / 245 (0.65) | 86 / 114 (0.76) ⚠ |
| Qwen2.5-1.5B | qkv | 45 / 48 (0.93) | 116 / 232 (0.50) | 73 / 97 (0.75) ⚠ |
| Qwen2.5-1.5B | o | 40 / 46 (0.86) | 140 / 193 (0.72) | 73 / 100 (0.73) ⚠ |
| Qwen2.5-1.5B | gate_up | 45 / 50 (0.91) | 151 / 230 (0.66) | 82 / 108 (0.76) ⚠ |
| Qwen2.5-1.5B | down | 42 / 40 (1.03) | 148 / 245 (0.60) | 68 / 113 (0.61) ⚠ |
| Qwen2.5-1.5B | lm_head | 45 / 49 (0.92) | 156 / 239 (0.65) | 83 / 111 (0.74) ⚠ |
| Llama-3.2-3B | qkv | 45 / 48 (0.94) | 150 / 256 (0.59) | 81 / 95 (0.85) ⚠ |
| Llama-3.2-3B | o | 45 / 48 (0.93) | 149 / 244 (0.61) | 77 / 106 (0.73) ⚠ |
| Llama-3.2-3B | gate_up | 45 / 49 (0.93) | 158 / 226 (0.70) | 83 / 108 (0.77) ⚠ |
| Llama-3.2-3B | down | 46 / 49 (0.94) | 150 / 240 (0.62) | 80 / 111 (0.72) ⚠ |
| Llama-3.2-3B | lm_head | 45 / 48 (0.94) | 163 / 251 (0.65) | 85 / 116 (0.73) ⚠ |
| Mistral-7B-v0.3 | qkv | - | 156 / 241 (0.65) | 81 / 107 (0.76) ⚠ |
| Mistral-7B-v0.3 | o | - | 146 / 248 (0.59) | 78 / 102 (0.76) ⚠ |
| Mistral-7B-v0.3 | gate_up | - | 164 / 236 (0.69) | 87 / 116 (0.75) ⚠ |
| Mistral-7B-v0.3 | down | - | 151 / 248 (0.61) | 75 / 103 (0.73) ⚠ |
| Mistral-7B-v0.3 | lm_head | - | 163 / 252 (0.65) | 86 / 116 (0.75) ⚠ |
| Llama-3.1-8B | qkv | - | - | 80 / 107 (0.75) ⚠ |
| Llama-3.1-8B | o | - | - | 77 / 102 (0.76) ⚠ |
| Llama-3.1-8B | gate_up | - | - | 87 / 116 (0.75) ⚠ |
| Llama-3.1-8B | down | - | - | 75 / 103 (0.73) ⚠ |
| Llama-3.1-8B | lm_head | - | - | 87 / 115 (0.76) ⚠ |
<!-- skipped results/baselines/a100/qwen25-1.5b/torch_compile.json: greedy tokens vs HF 0/32 -->

## hipinfer vs other engines (fp16 weights, batch 1, same GPU)

Decode: 256 greedy tokens after a ctx-token prompt, ms/token (% BW = weight+KV bytes / peak BW / time). Prefill: one prompt of T random tokens, ms. llama.cpp: `llama-bench` (pp / tg256 at depth ctx), F16 GGUF, best of flash-attn on/off and ubatch 512/2048. torch.compile / HF static: decode step in a CUDA graph, KV cache sized ctx+256. vLLM: decode = (generate 257 tokens - generate 1 token) / 256. The TinyLlama rows were measured in one session by scripts/run_baselines.sh; the other models' hipinfer rows come from the main run (results/<gpu>/<model>/bench.json). Raw JSON/logs: results/baselines/.


### TinyLlama-1.1B

| | RTX 2080 Ti | A100-SXM4-40GB | RTX A6000 |
|---|---|---|---|
| decode ms/token, decode 128: **hipinfer** | 4.14 ms, 81% BW | 2.18 ms, 61% BW | 3.42 ms, 79% BW |
| decode ms/token, decode 128: llama.cpp (F16 GGUF, CUDA or HIP backend, best of FA on/off) | 4.77 ms, 71% BW (hipinfer 1.15x faster) | 2.57 ms, 52% BW (hipinfer 1.18x faster) | 3.78 ms, 72% BW (hipinfer 1.10x faster) |
| decode ms/token, decode 128: torch.compile, gpt-fast style (reduce-overhead + static KV) | 5.24 ms, 64% BW (hipinfer 1.27x faster) | 3.14 ms, 43% BW (hipinfer 1.44x faster) | 4.23 ms, 64% BW (hipinfer 1.24x faster) |
| decode ms/token, decode 128: HF transformers, StaticCache + torch.compile | 5.51 ms, 61% BW (hipinfer 1.33x faster) | 3.39 ms, 39% BW (hipinfer 1.56x faster) | 4.34 ms, 62% BW (hipinfer 1.27x faster) |
| decode ms/token, decode 128: vLLM (V1 engine, GPU graphs) | - | 2.83 ms, 47% BW (hipinfer 1.30x faster) | 3.98 ms, 68% BW (hipinfer 1.16x faster) |
| decode ms/token, decode 1024: **hipinfer** | 4.22 ms, 81% BW | 2.25 ms, 60% BW | 3.49 ms, 78% BW |
| decode ms/token, decode 1024: llama.cpp (F16 GGUF, CUDA or HIP backend, best of FA on/off) | 4.79 ms, 71% BW (hipinfer 1.14x faster) | 2.55 ms, 53% BW (hipinfer 1.14x faster) | 3.79 ms, 72% BW (hipinfer 1.09x faster) |
| decode ms/token, decode 1024: torch.compile, gpt-fast style (reduce-overhead + static KV) | 6.84 ms, 50% BW (hipinfer 1.62x faster) | 4.56 ms, 30% BW (hipinfer 2.03x faster) | 5.47 ms, 50% BW (hipinfer 1.57x faster) |
| decode ms/token, decode 1024: HF transformers, StaticCache + torch.compile | 7.09 ms, 48% BW (hipinfer 1.68x faster) | 4.84 ms, 28% BW (hipinfer 2.15x faster) | 5.56 ms, 49% BW (hipinfer 1.59x faster) |
| decode ms/token, decode 1024: vLLM (V1 engine, GPU graphs) | - | 2.83 ms, 48% BW (hipinfer 1.26x faster) | 3.98 ms, 69% BW (hipinfer 1.14x faster) |

| | RTX 2080 Ti | A100-SXM4-40GB | RTX A6000 |
|---|---|---|---|
| prefill ms, prefill 128: **hipinfer** | 8.3 ms | 5.1 ms | 6.3 ms |
| prefill ms, prefill 128: llama.cpp (F16 GGUF, CUDA or HIP backend, best of FA on/off) | 11.2 ms (hipinfer 1.34x faster) | 6.5 ms (hipinfer 1.26x faster) | 9.8 ms (hipinfer 1.56x faster) |
| prefill ms, prefill 128: torch.compile, gpt-fast style (reduce-overhead + static KV) | 16.2 ms (hipinfer 1.94x faster) | 8.6 ms (hipinfer 1.68x faster) | 8.9 ms (hipinfer 1.42x faster) |
| prefill ms, prefill 512: **hipinfer** | 26.6 ms | 11.2 ms | 16.7 ms |
| prefill ms, prefill 512: llama.cpp (F16 GGUF, CUDA or HIP backend, best of FA on/off) | 29.0 ms (hipinfer 1.09x faster) | 13.1 ms (hipinfer 1.17x faster) | 24.5 ms (hipinfer 1.46x faster) |
| prefill ms, prefill 512: torch.compile, gpt-fast style (reduce-overhead + static KV) | 34.8 ms (hipinfer 1.31x faster) | 14.9 ms (hipinfer 1.33x faster) | 19.5 ms (hipinfer 1.16x faster) |
| prefill ms, prefill 2000: **hipinfer** | 100.5 ms | 35.6 ms | 57.9 ms |
| prefill ms, prefill 2000: llama.cpp (F16 GGUF, CUDA or HIP backend, best of FA on/off) | 114.8 ms (hipinfer 1.14x faster) | 51.7 ms (hipinfer 1.45x faster) | 89.7 ms (hipinfer 1.55x faster) |
| prefill ms, prefill 2000: torch.compile, gpt-fast style (reduce-overhead + static KV) | 125.4 ms (hipinfer 1.25x faster) | 34.8 ms (**1.02x faster than hipinfer**) | 59.6 ms (hipinfer 1.03x faster) |


### Qwen2.5-0.5B

| | RTX 2080 Ti |
|---|---|
| decode ms/token, decode 128: **hipinfer** | 2.34 ms, 69% BW |
| decode ms/token, decode 128: llama.cpp (F16 GGUF, CUDA or HIP backend, best of FA on/off) | 3.45 ms, 47% BW (hipinfer 1.47x faster) |
| decode ms/token, decode 1024: **hipinfer** | 2.42 ms, 67% BW |
| decode ms/token, decode 1024: llama.cpp (F16 GGUF, CUDA or HIP backend, best of FA on/off) | 3.46 ms, 47% BW (hipinfer 1.43x faster) |

| | RTX 2080 Ti |
|---|---|
| prefill ms, prefill 128: **hipinfer** | 5.1 ms |
| prefill ms, prefill 128: llama.cpp (F16 GGUF, CUDA or HIP backend, best of FA on/off) | 8.4 ms (hipinfer 1.64x faster) |
| prefill ms, prefill 512: **hipinfer** | 12.1 ms |
| prefill ms, prefill 512: llama.cpp (F16 GGUF, CUDA or HIP backend, best of FA on/off) | 18.5 ms (hipinfer 1.53x faster) |
| prefill ms, prefill 2000: **hipinfer** | 43.1 ms |
| prefill ms, prefill 2000: llama.cpp (F16 GGUF, CUDA or HIP backend, best of FA on/off) | 67.3 ms (hipinfer 1.56x faster) |


### Llama-3.2-1B

| | RTX 2080 Ti | A100-SXM4-40GB |
|---|---|---|
| decode ms/token, decode 128: **hipinfer** | 4.75 ms, 85% BW | 2.31 ms, 69% BW |
| decode ms/token, decode 128: llama.cpp (F16 GGUF, CUDA or HIP backend, best of FA on/off) | 5.23 ms, 77% BW (hipinfer 1.10x faster) | 2.63 ms, 61% BW (hipinfer 1.14x faster) |
| decode ms/token, decode 128: torch.compile, gpt-fast style (reduce-overhead + static KV) | - | 3.08 ms, 52% BW (hipinfer 1.33x faster) |
| decode ms/token, decode 128: HF transformers, StaticCache + torch.compile | - | 3.23 ms, 49% BW (hipinfer 1.40x faster) |
| decode ms/token, decode 128: vLLM (V1 engine, GPU graphs) | - | 2.86 ms, 56% BW (hipinfer 1.23x faster) |
| decode ms/token, decode 1024: **hipinfer** | 4.84 ms, 84% BW | 2.35 ms, 69% BW |
| decode ms/token, decode 1024: llama.cpp (F16 GGUF, CUDA or HIP backend, best of FA on/off) | 5.27 ms, 77% BW (hipinfer 1.09x faster) | 2.64 ms, 61% BW (hipinfer 1.13x faster) |
| decode ms/token, decode 1024: torch.compile, gpt-fast style (reduce-overhead + static KV) | - | 4.12 ms, 39% BW (hipinfer 1.75x faster) |
| decode ms/token, decode 1024: HF transformers, StaticCache + torch.compile | - | 4.25 ms, 38% BW (hipinfer 1.81x faster) |
| decode ms/token, decode 1024: vLLM (V1 engine, GPU graphs) | - | 2.85 ms, 57% BW (hipinfer 1.21x faster) |

| | RTX 2080 Ti | A100-SXM4-40GB |
|---|---|---|
| prefill ms, prefill 128: **hipinfer** | 8.6 ms | 4.9 ms |
| prefill ms, prefill 128: llama.cpp (F16 GGUF, CUDA or HIP backend, best of FA on/off) | 10.4 ms (hipinfer 1.21x faster) | 6.4 ms (hipinfer 1.31x faster) |
| prefill ms, prefill 128: torch.compile, gpt-fast style (reduce-overhead + static KV) | - | 7.7 ms (hipinfer 1.58x faster) |
| prefill ms, prefill 512: **hipinfer** | 26.1 ms | 10.0 ms |
| prefill ms, prefill 512: llama.cpp (F16 GGUF, CUDA or HIP backend, best of FA on/off) | 27.0 ms (hipinfer 1.03x faster) | 12.7 ms (hipinfer 1.27x faster) |
| prefill ms, prefill 512: torch.compile, gpt-fast style (reduce-overhead + static KV) | - | 11.0 ms (hipinfer 1.09x faster) |
| prefill ms, prefill 2000: **hipinfer** | 101.1 ms | 33.3 ms |
| prefill ms, prefill 2000: llama.cpp (F16 GGUF, CUDA or HIP backend, best of FA on/off) | 100.9 ms (**1.00x faster than hipinfer**) | 47.7 ms (hipinfer 1.43x faster) |
| prefill ms, prefill 2000: torch.compile, gpt-fast style (reduce-overhead + static KV) | - | 30.2 ms (**1.10x faster than hipinfer**) |


### Qwen2.5-1.5B

| | RTX 2080 Ti | A100-SXM4-40GB |
|---|---|---|
| decode ms/token, decode 128: **hipinfer** | 6.21 ms, 81% BW | 3.21 ms, 62% BW |
| decode ms/token, decode 128: llama.cpp (F16 GGUF, CUDA or HIP backend, best of FA on/off) | 7.19 ms, 70% BW (hipinfer 1.16x faster) | 4.08 ms, 49% BW (hipinfer 1.27x faster) |
| decode ms/token, decode 128: HF transformers, StaticCache + torch.compile | - | 4.81 ms, 41% BW (hipinfer 1.50x faster) |
| decode ms/token, decode 128: vLLM (V1 engine, GPU graphs) | - | 4.19 ms, 47% BW (hipinfer 1.31x faster) |
| decode ms/token, decode 1024: **hipinfer** | 6.32 ms, 80% BW | 3.29 ms, 61% BW |
| decode ms/token, decode 1024: llama.cpp (F16 GGUF, CUDA or HIP backend, best of FA on/off) | 7.22 ms, 70% BW (hipinfer 1.14x faster) | 4.06 ms, 49% BW (hipinfer 1.23x faster) |
| decode ms/token, decode 1024: HF transformers, StaticCache + torch.compile | - | 6.27 ms, 32% BW (hipinfer 1.90x faster) |
| decode ms/token, decode 1024: vLLM (V1 engine, GPU graphs) | - | 4.19 ms, 48% BW (hipinfer 1.27x faster) |

| | RTX 2080 Ti | A100-SXM4-40GB |
|---|---|---|
| prefill ms, prefill 128: **hipinfer** | 13.4 ms | 6.5 ms |
| prefill ms, prefill 128: llama.cpp (F16 GGUF, CUDA or HIP backend, best of FA on/off) | 17.3 ms (hipinfer 1.29x faster) | 9.4 ms (hipinfer 1.45x faster) |
| prefill ms, prefill 512: **hipinfer** | 37.8 ms | 13.8 ms |
| prefill ms, prefill 512: llama.cpp (F16 GGUF, CUDA or HIP backend, best of FA on/off) | 40.1 ms (hipinfer 1.06x faster) | 17.5 ms (hipinfer 1.27x faster) |
| prefill ms, prefill 2000: **hipinfer** | 140.9 ms | 43.5 ms |
| prefill ms, prefill 2000: llama.cpp (F16 GGUF, CUDA or HIP backend, best of FA on/off) | 156.7 ms (hipinfer 1.11x faster) | 63.1 ms (hipinfer 1.45x faster) |


### Llama-3.2-3B

| | RTX 2080 Ti | A100-SXM4-40GB |
|---|---|---|
| decode ms/token, decode 128: **hipinfer** | 12.11 ms, 87% BW | 5.60 ms, 74% BW |
| decode ms/token, decode 128: llama.cpp (F16 GGUF, CUDA or HIP backend, best of FA on/off) | 12.70 ms, 83% BW (hipinfer 1.05x faster) | 6.17 ms, 67% BW (hipinfer 1.10x faster) |
| decode ms/token, decode 128: torch.compile, gpt-fast style (reduce-overhead + static KV) | - | 6.79 ms, 61% BW (hipinfer 1.21x faster) |
| decode ms/token, decode 128: HF transformers, StaticCache + torch.compile | - | 7.18 ms, 58% BW (hipinfer 1.28x faster) |
| decode ms/token, decode 1024: **hipinfer** | 12.54 ms, 85% BW | 5.71 ms, 74% BW |
| decode ms/token, decode 1024: llama.cpp (F16 GGUF, CUDA or HIP backend, best of FA on/off) | 12.92 ms, 82% BW (hipinfer 1.03x faster) | 6.25 ms, 68% BW (hipinfer 1.09x faster) |
| decode ms/token, decode 1024: torch.compile, gpt-fast style (reduce-overhead + static KV) | - | 8.51 ms, 50% BW (hipinfer 1.49x faster) |
| decode ms/token, decode 1024: HF transformers, StaticCache + torch.compile | - | 8.85 ms, 48% BW (hipinfer 1.55x faster) |

| | RTX 2080 Ti | A100-SXM4-40GB |
|---|---|---|
| prefill ms, prefill 128: **hipinfer** | 24.3 ms | 10.7 ms |
| prefill ms, prefill 128: llama.cpp (F16 GGUF, CUDA or HIP backend, best of FA on/off) | 25.8 ms (hipinfer 1.06x faster) | 13.0 ms (hipinfer 1.21x faster) |
| prefill ms, prefill 128: torch.compile, gpt-fast style (reduce-overhead + static KV) | - | 13.1 ms (hipinfer 1.22x faster) |
| prefill ms, prefill 512: **hipinfer** | 76.7 ms | 24.2 ms |
| prefill ms, prefill 512: llama.cpp (F16 GGUF, CUDA or HIP backend, best of FA on/off) | 65.4 ms (**1.17x faster than hipinfer**) | 26.7 ms (hipinfer 1.10x faster) |
| prefill ms, prefill 512: torch.compile, gpt-fast style (reduce-overhead + static KV) | - | 20.4 ms (**1.19x faster than hipinfer**) |
| prefill ms, prefill 2000: **hipinfer** | 289.7 ms | 87.0 ms |
| prefill ms, prefill 2000: llama.cpp (F16 GGUF, CUDA or HIP backend, best of FA on/off) | 261.8 ms (**1.11x faster than hipinfer**) | 100.8 ms (hipinfer 1.16x faster) |
| prefill ms, prefill 2000: torch.compile, gpt-fast style (reduce-overhead + static KV) | - | 73.8 ms (**1.18x faster than hipinfer**) |


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

