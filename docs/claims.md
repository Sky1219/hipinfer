# hipinfer: claims, their exact scope, and the evidence

This is the list of things hipinfer can be said to do, each with the conditions it was measured
under and the file that backs it. Every number here was checked against the result files on
2026-10-16. Where the README rounds or generalizes, the precise figure is given. The last two
sections list claims to avoid, and replacement wording for resume lines.

Hardware abbreviations: **2080 Ti** = NVIDIA GeForce RTX 2080 Ti (Turing, sm_75, 11 GB, 616 GB/s
datasheet). **A6000** = NVIDIA RTX A6000 (Ampere, sm_86, 48 GB, 768 GB/s datasheet). All weights
are fp16 and all generation is greedy.

---

## 1. Platforms

| claim | exact scope | evidence |
|---|---|---|
| Runs and passes all kernel tests on NVIDIA | 2080 Ti, A6000 and A100-SXM4-40GB (Colab). The 899 dense-model checks pass with 0 failures on each; the MoE kernel checks have run on the 2080 Ti only (on the A100 run they were skipped for a missing file). | [results/2080ti/test_kernels.log](../results/2080ti/test_kernels.log), [results/a6000/test_kernels.log](../results/a6000/test_kernels.log), [results/a100/test_kernels.log](../results/a100/test_kernels.log) |
| One source tree for NVIDIA (`nvcc`) and AMD (`hipcc`) | `src/hip_compat.h` maps the `hip*` API onto CUDA under `nvcc`. The NVIDIA fast paths (`mma.sync`, `ldmatrix`, `cp.async`) are NVIDIA-only; AMD gets the portable WMMA / rocWMMA and scalar kernels. | [src/hip_compat.h](../src/hip_compat.h), [Makefile](../Makefile) |
| **AMD: compile-checked only** | `make hip` builds for gfx90a (MI200), gfx942 (MI300) and gfx1100 (RDNA3) with ROCm 7 + rocWMMA, RCCL optional. **It has never run on AMD hardware**: no kernel test, correctness or performance result exists for any AMD GPU. The MoE kernels were added after that build check and haven't been compiled for AMD. No build log is committed; the evidence is the commit message of 7370172 ("The AMD build ... compiles on this code"). | README "What it runs, and where"; `git show 7370172` |

## 2. Batch-1 decode against the hardware limit

| claim | exact scope | evidence |
|---|---|---|
| **Decode reaches 81–87 % of the memory-bandwidth floor** | 2080 Ti, batch 1, 256 tokens after a 128-token prompt. The four models of 1–3B parameters: TinyLlama-1.1B 81.4 %, Llama-3.2-1B 84.8 %, Qwen2.5-1.5B 80.8 %, Llama-3.2-3B 86.5 % (so 80.8–86.5 %, which rounds to 81–87 %). Qwen2.5-0.5B: 68.6 %. Floor = (weight bytes + KV bytes read) / **datasheet** peak bandwidth, not a measured copy bandwidth. | `results/2080ti/<model>/bench.json` → `e2e.decode_b1[ctx=128].bw_util` |
| Holds as the context grows | 2080 Ti, 128–8,192 tokens: Llama-3.2-1B 84–85 %, Llama-3.2-3B 85–86 %, Qwen2.5-1.5B falls to 75–76 % at 4–8k, Qwen2.5-0.5B 64–69 %. TinyLlama is measured only to 1,024 (81 %). | `e2e.decode_ctx` in the same files |
| A6000 | TinyLlama 79 %, Qwen2.5-0.5B 67 % (both unthrottled). The 78–87 % figures for the other models come from partly **throttled** runs (§8). | `results/a6000/<model>/bench.json` |
| Where the remaining 13–19 % goes | A decomposition from microbenchmarks, not a profiler trace. GEMVs stream at 575–583 GB/s plus a 4–10 µs fixed cost per call. The fixed cost is 44–59 % of the gap, the streaming shortfall 14–33 %, and latency-bound batch-1 attention 12–29 %. | [docs/profiling.md](profiling.md) |

## 3. Against llama.cpp

**Scope of every row:** 2080 Ti, fp16 weights (F16 GGUF, no quantization), batch 1. llama.cpp
commit `a0e6dc7` with CUDA, measured with `llama-bench` (5 repetitions, mean; run-to-run spread
under 0.5 %), taking the best of flash-attention on/off and ubatch 512/2048. hipinfer's rows come
from the main 2026-10-15 suite, and llama.cpp's from a separate session on 2026-10-14/15 on the same machine.

| claim | exact numbers | evidence |
|---|---|---|
| **5–16 % faster decode on 1–3B models** | ms/token at a 128-token context, hipinfer vs llama.cpp: Llama-3.2-3B 12.11 vs 12.70 (1.05×), Llama-3.2-1B 4.75 vs 5.23 (1.10×), TinyLlama 4.14 vs 4.77 (1.15×), Qwen2.5-1.5B 6.21 vs 7.19 (1.16×). **At a 1,024-token context the lead is 3.0–14.4 %** (3B: 12.54 vs 12.92). | `results/baselines/2080ti/<model>/llamacpp.json` (TinyLlama: `results/baselines/2080ti/llamacpp.json`) vs `results/2080ti/<model>/bench.json` |
| **47 % faster decode on Qwen2.5-0.5B** | 2.34 vs 3.45 ms (1.47×) at ctx 128; 2.42 vs 3.46 (1.43×) at ctx 1,024 | same |
| Prefill faster on the small models | 2,000 tokens: Qwen2.5-0.5B 1.56×, TinyLlama 1.14×, Qwen2.5-1.5B 1.11×, Llama-3.2-1B level (101.1 vs 100.9 ms). At 128 tokens hipinfer is faster on all five (1.06–1.64×). | same |
| **Slower at 3B prefill** | Llama-3.2-3B, 2,000 tokens: 289.7 vs 261.8 ms (**1.11× slower**) in the full suite, where the 2080 Ti's 250 W power cap holds it at 1.4–1.55 GHz. A short isolated run gives 273.6 ms (1.04× slower). **Also slower at 512 tokens: 76.7 vs 65.4 ms (1.17×).** The cause is GEMMs at 0.93–0.94× cuBLAS (§6). | same; [docs/turing-prefill-gemm.md](turing-prefill-gemm.md) |
| A6000, TinyLlama only | Decode 3.42 vs 3.78 ms (1.11×) at ctx 128, 3.49 vs 3.79 (1.09×) at 1,024. Prefill 6.3 / 16.7 / 57.9 vs 9.8 / 24.5 / 89.7 ms at 128 / 512 / 2,000 tokens. hipinfer, llama.cpp and vLLM ran back to back in one session, all unthrottled ([THERMAL.md](../results/a6000/THERMAL.md)). The other models' A6000 llama.cpp runs were throttled and are not used. | `results/baselines/a6000/tinyllama/{hipinfer,llamacpp}.json` |

| A100, four models, one session | Decode at ctx 128, hipinfer vs llama.cpp: TinyLlama 2.18 vs 2.57 ms (1.18×), Llama-3.2-1B 2.32 vs 2.63 (1.14×), Qwen2.5-1.5B 3.21 vs 4.08 (1.27×), Llama-3.2-3B 5.60 vs 6.17 (1.10×). Prefill at 2,000 tokens: hipinfer faster on all four (1.16–1.45×). No throttling ([THERMAL.md](../results/a100/THERMAL.md)). | `results/baselines/a100/<model>/{hipinfer,llamacpp}.json` |

Not compared: batch > 1, quantized formats (llama.cpp's main use), CPU offload, any model above 3B.

## 4. Against vLLM, torch.compile and HF transformers

| claim | exact scope | evidence |
|---|---|---|
| **Faster batch-1 decode than vLLM** | **TinyLlama-1.1B only, batch 1, A6000 only.** vLLM 0.31 (V1 engine, CUDA graphs, fp16): 3.98 / 3.98 ms/token at ctx 128 / 1,024, measured as (generate 257 tokens − generate 1 token) / 256, so its per-step scheduler overhead is included. hipinfer in the same session: 3.42 / 3.49 ms (1.16× / 1.14×), all unthrottled. vLLM was not run on the 2080 Ti, on any other model unthrottled, or at any batch size above 1. | `results/baselines/a6000/tinyllama/{hipinfer,vllm}.json` |
| Faster than torch.compile (gpt-fast style) | TinyLlama only, batch 1, both GPUs. 2080 Ti: 4.14 vs 5.24 ms (1.27×) at ctx 128, 4.22 vs 6.84 (1.62×) at 1,024. A6000: 1.23× / 1.57×. Prefill at 2,000 tokens on the A6000 is level (59.6 vs 59.6 ms). | `results/baselines/{2080ti,a6000}/torch_compile.json` |
| A100: faster batch-1 decode than vLLM and torch.compile | vLLM 0.31 at ctx 128: TinyLlama 2.83 vs 2.18 ms (1.30×), Llama-3.2-1B 2.86 vs 2.32 (1.23×), Qwen2.5-1.5B 4.19 vs 3.21 (1.31×); Llama-3.2-3B not run (disk). torch.compile: 1.44× / 1.33× / 1.21× on TinyLlama / Llama-3.2-1B / Llama-3.2-3B; its Qwen2.5-1.5B run is invalid (the baseline dropped the QKV bias, 0/32 tokens matched HF). **Prefill at 2,000 tokens is the other way round:** vLLM (its time includes one generated token) and torch.compile are 1.02–1.22× faster than hipinfer. Same session, no throttling. | `results/baselines/a100/<model>/*.json` |
| 3.8–11.5× faster than HF transformers | 2080 Ti, batch-1 decode, against HF **eager** with a DynamicCache. That is a weak baseline, so quote it with that qualifier or not at all. | `e2e.decode_b1[].hf_ms` |

## 5. Correctness against HF transformers

| claim | exact scope | evidence |
|---|---|---|
| Greedy tokens match HF | `tests/gate.py`, 7 prompts × 128 tokens: single-sequence with and without CUDA graphs, plus a batch of 6 (20 runs). PASS on all 13 model/GPU combinations. Token-identical runs range from 11/20 (Llama-3.2-1B, A6000) to 20/20. Every divergence is at a step where HF's own top-2 logit gap is 0, 0.0156 or 0.0313 (tolerance 0.05). At such a step either token is a correct greedy choice. | `results/{2080ti,a6000}/<model>/gate.json` |
| **Perplexity matches HF within 0.01 %** | WikiText-2 test, 16 × 1,024-token windows, same tokens for both engines. Single-GPU 2080 Ti: ≤ 0.0027 %. A6000: ≤ 0.0062 % (Llama-3.2-3B). Sharded over 2 GPUs: ≤ 0.0098 % (Mistral-7B-v0.1, 8 windows). The README's "within 0.003 %" holds for the single-GPU 2080 Ti runs only. Top-1 agreement 99.74–99.91 %; mean KL(HF ‖ hipinfer) 4.6e-6 to 1.6e-5. | `results/*/<model>/ppl*.json` |
| Kernels match PyTorch references | 899 checks per GPU. **RoPE is bit-exact** against HF's fp16 order (tolerance 0). **RMSNorm is bit-exact at 3 of 5 tested shapes**; at H = 4,096 and H = 3,072 the max error is 1.95e-3 and 9.8e-4 on the 2080 Ti, and 3.9e-3 and 2.4e-4 on the A6000. All of them pass the test's tolerance of 2e-3 + 2e-3·|ref|. The README's "RMSNorm and RoPE are bit-exact" overstates the RMSNorm half. | `results/{2080ti,a6000}/test_kernels.log` |
| Not bit-identical to HF end to end | Max absolute logit difference against HF is 0.11–2.4 depending on model (fp16 summation order). Say "matches", not "bit-exact". | `results/SUMMARY.md` correctness table |

## 6. Kernels against vendor libraries

| claim | exact scope | evidence |
|---|---|---|
| Prefill GEMM close to cuBLAS on Turing | 2080 Ti, M = 2,048 rows: **0.84–0.95× cuBLAS** on the per-layer shapes of all five models (Llama-3.2-3B 0.93–0.94×). Two shapes are faster than cuBLAS: Qwen2.5-0.5B QKV (1.40×, where cuBLAS is slow at N = 1,152) and Qwen2.5-1.5B down (1.03×). | `bench.json` → `kernels[kernel=gemm, M=2048]` |
| Prefill GEMM on Ampere | **0.61–0.85× cuBLAS** on the A6000 at the same shapes. This is a known weakness. | `results/a6000/*/bench.json` |
| Decode GEMV | 2080 Ti, M = 1: 355–579 GB/s depending on matrix size (58–94 % of datasheet). The large matrices (gate/up, lm_head) reach 510–579 GB/s; the README's "520–580 GB/s" describes those, not the QKV/O GEMVs. Against `F.linear`: 0.95–1.01× on the plain residual GEMVs; the RMSNorm-fused ones are faster because PyTorch runs the norm separately. | `kernels[kernel=gemv]` |
| Attention | 2080 Ti: 7.7–52.7× PyTorch SDPA, but SDPA has no flash kernel on Turing, so this shows little. A6000 (TinyLlama, unthrottled): prefill 0.90–1.11× SDPA's FlashAttention backend, decode 0.87–1.09×, i.e. parity, not faster. | `kernels[kernel=flash_prefill / decode_attn]` |

## 7. Multi-GPU tensor parallelism and the all-reduce

**Scope of every row:** tp = 2, two RTX 2080 Tis on different CPU sockets, PCIe 3.0, **no
peer-to-peer path** (`tp_info.p2p: false`; `nvidia-smi topo` SYS). The one-shot all-reduce
therefore stages through pinned host memory. The P2P staging path is written but has never run,
and tp > 2 has never run.

| claim | exact numbers | evidence |
|---|---|---|
| **One-shot all-reduce faster than NCCL for decode-sized messages** | fp16, in place, graph-replayed loop, hidden size 2,048: **8.5 vs 12.2 µs at 4 KB** (1.44×), 17.7 vs 24.9 µs at 32 KB, **63.6 vs 85.4 µs at 256 KB** (1.34×). At 2 MB it is level (425.0 vs 426.0 µs) and at 8 MB slightly slower (1,574 vs 1,555 µs), so `auto` hands messages over 512 KB to NCCL. | `results/2080ti/tinyllama/tp2.json` → `allreduce` |
| End-to-end effect | Batch-1 decode **1.5–15.2 % lower** latency than NCCL-only across all 8 sharded runs (the README rounds to 2–15 %). Batched throughput at B = 64: **+9.0 to +27.0 %** on the five 1–3B models. | `results/2080ti/*/tp2.json` → `tp_rows` |
| 2 GPUs faster than 1 | Batch-1 decode 1.33–1.74× (Qwen2.5-0.5B to Llama-3.2-3B). 2,000-token prefill 1.05–1.57× with compute/all-reduce overlap; without the overlap, TinyLlama prefill is *slower* than one GPU (125.5 vs 105.3 ms). Exception: Qwen2.5-0.5B 512-token prefill is slower sharded (16.3 vs 12.2 ms). | same |
| Runs 7–8B models that don't fit one card | Llama-3.1-8B 14.68 ms/token and Mistral-7B-v0.3 13.99 ms/token, each at 83 % of the per-GPU bandwidth floor. The 1–3B models reach only 62–76 % sharded (Qwen2.5-0.5B 46 %). | same |
| Sharded output is correct | `tests/test_tp.py` passes for all 8 models. Greedy tokens match one GPU except at 1-GPU near-ties (gap 0.0078). The all-reduce self-test is exact, and the one-shot and NCCL backends give **bitwise-identical** logits and tokens. 7–8B perplexity matches HF spread over the same two GPUs (6.5467 vs 6.5471; 5.7773 vs 5.7775). | `results/2080ti/*/tp2_test.json`, `ppl_tp2.json` |

## 8. Measurement conditions that limit the numbers

- **A6000 runs were partly thermally throttled** ([results/a6000/THERMAL.md](../results/a6000/THERMAL.md)).
  The card reaches 90–93 °C and throttles its SM clock to as low as 210 MHz. Share of each benchmark
  spent throttled: Llama-3.2-1B 33 %, Qwen2.5-1.5B 31 %, Llama-3.2-3B 67 %, Mistral-7B-v0.3 78 %
  (median 450 MHz). TinyLlama and Qwen2.5-0.5B were clean.
  - Only the *speed* numbers of the throttled models are affected. Gate and perplexity results are
    valid for all of them.
  - Llama-3.1-8B's A6000 gate/ppl come from an earlier build (2026-10-14).
  - The A6000 llama.cpp / vLLM baselines (2026-10-14) predate the thermal logging.
- **2080 Ti power cap:** under sustained load the 250 W cap holds the clock at 1.4–1.55 GHz. This is
  why Llama-3.2-3B prefill reads 289.7 ms in the suite and 273.6 ms in isolation. The batched runs draw
  218–249 W.
- **A contended A6000 run (2026-10-14/15)** shared the GPU with another job and was set aside entirely
  ([results/a6000_contended_2026-10-14/README.md](../results/a6000_contended_2026-10-14/README.md)).
- **Tools:** all timings are CUDA/HIP events (or wall clock for HF decode), and power comes from NVML. There are
  **no Nsight Systems, Nsight Compute or rocprof captures**. Bandwidth and FLOP rates are derived from
  timings, not counters ([docs/profiling.md](profiling.md)).

## 9. Model support

| claim | exact scope | evidence |
|---|---|---|
| Families with results | TinyLlama-1.1B (Llama-2 architecture), Llama-3.2-1B / 3B, Llama-3.1-8B, Qwen2.5-0.5B / 1.5B, Mistral-7B-v0.3, and Mistral-7B-v0.1 within its 4,096-token window. **Llama 2 and Qwen2 (non-2.5) themselves have no result files**; they are claimed because their architectures match tested models. | `results/` |
| Config features | head_dim 64 / 128, GQA groups 1–8, tied embeddings, Llama-3 RoPE scaling (no engine code; HF's `inv_freq` is exported), QKV bias. Costs per feature are in [docs/porting-a-model.md](porting-a-model.md). | `scripts/export.py`, `src/api.cu` |
| Mixture-of-Experts | **Granite-3.1-1B-A400M only, one GPU, 2080 Ti only.** Kernel tests pass. **Token gate: passes only with an MoE-specific router near-tie rule**, in all 6 runs made with it, same weights (3 autotuned: 12/20 token-identical each; autotuning off: 15/20, 2 accepted by the router rule; forced split-K 2: 13/20; forced split-K 4: 9/20, 3 accepted by the router rule). The engine's router kernel fed HF's own inputs differs from HF only at fp16 ties (5 of 40,392 evaluations, margins ≤ 0.0039), so there is no engine rounding mismatch to fix. Without the rule it passed in 3 of 5 runs, depending on GEMM configs. The two failures (gaps 0.0625 / 0.1094) were traced to router top-8 near-ties (`scripts/moe_router_trace.py`). Every adopted router difference has an HF margin ≤ 0.0042, against a tolerance of 0.01 and rounding noise of up to 0.0098. A planted 2 % fault in one layer is rejected. The prompt 3 failure was reproduced with forced split-K 4, not with the original autotuner picks; the three autotuned runs all gave the same gate results. Don't say it "matches HF" without saying that the router rule is part of the gate; perplexity 7.7292 vs 7.7294; decode 2.26 ms/token at ctx 128, 63 % of the weight-read floor. Mixtral and OLMoE export but have no GPU result (OLMoE's QK-norm is rejected at load). No MoE with tensor or expert parallelism, no MoE on AMD (not even compiled). | `results/2080ti/granite-moe-1b/` (`gate_runs/`, `router_trace/`), [docs/moe-design.md](moe-design.md) |
| Not supported | Quantization (fp16 only), paged KV cache, sampling other than greedy, MoE with shared experts or QK-norm, sliding-window attention beyond the window, norms other than RMSNorm, activations other than SiLU, o-proj/MLP biases, head_dim ∉ {64, 128}. Some of these export **silently wrong** rather than failing (see porting doc). | README "Not supported"; [porting-a-model.md](porting-a-model.md) |

## 10. README statements with weak or missing evidence

These may well be true, but no committed file shows them on the current build. Don't repeat them
in an interview without saying so.

1. "Batched decode at 32–64 sequences: projections ~1.15–1.2× slower than cuBLAS, ~1.1× at 16."
   `bench.json` times GEMMs only at M = 128 / 512 / 2,048. The only M ≤ 64 data is
   `results/2080ti/gemm_sweep_*.json`, which predates the current build (no cfg 26).
2. "A profiler trace shows only about 2.5 ms of GPU idle time per prefill." The `prof_prefill.py`
   output is not committed.
3. "RMSNorm ... bit-exact": true at 3 of 5 tested shapes (§5).
4. "Perplexity within 0.003 %": true for the single-GPU 2080 Ti runs; 0.0062 % on the A6000 (§5).
5. "Every divergence ... gap 0 or 0.0156": true on the 2080 Ti; one A6000 run diverges at 0.0313 (§5).
6. "2–15 % lower batch-1 decode latency than NCCL alone": 1.5–15.2 % (§7).
7. "5–16 % faster than llama.cpp": true at a 128-token context; 3.0–14.4 % at 1,024 (§3).
8. "At batch-1 decode, hipinfer was the fastest engine I measured on every model": llama.cpp
   was measured on all five 2080 Ti models, but vLLM and torch.compile only on TinyLlama. The claim
   is "faster than llama.cpp on every model tested, and faster than vLLM and torch.compile on the one model where
   they were measured".

## 11. Claims to avoid

| don't say | because |
|---|---|
| "runs on AMD" / "MI300 support" / "cross-vendor performance" | compile-checked only, never executed on AMD |
| "faster than vLLM" (unqualified) | one model, batch 1, one GPU. vLLM is built for throughput under concurrency, which wasn't measured. |
| "faster than llama.cpp" (unqualified) | fp16 batch 1 on one GPU; slower at 3B prefill; quantized llama.cpp (its normal mode) not compared |
| "faster than NCCL" (unqualified) | only for messages ≤ 256 KB, only on a no-P2P PCIe pair; NCCL is level or faster at ≥ 2 MB |
| "scales to N GPUs" / "multi-node" | only tp = 2 has run; the P2P path has never run |
| "near-peak bandwidth" / "at the roofline" | 81–87 % of a datasheet-bandwidth floor, and only for 1–3B models at batch 1 |
| "bit-exact with HF" | logits differ by up to 0.11–2.4 (max abs); "matches HF perplexity within 0.01 %" is accurate |
| "beats cuBLAS" | 0.84–0.95× on Turing prefill shapes, 0.61–0.85× on Ampere; faster on 2 shapes only |
| "production-ready" / "serving engine" | no scheduler, no paged KV, greedy only, fp16 only |
| "profiled with Nsight" | no Nsight or rocprof captures exist |

## 12. Replacement resume lines

Each line below is backed by the sections above and can be defended number by number.

- **Engine:** "Wrote an LLM inference engine from scratch in HIP: hand-written GEMV, tensor-core GEMM,
  FlashAttention-style prefill and split-K decode attention, and a CUDA/HIP-graph decode loop. Tested on
  NVIDIA RTX 2080 Ti and RTX A6000. The AMD build (MI200/MI300/RDNA3) compiles but has not been run
  on AMD hardware."
- **Speed:** "Batch-1 fp16 decode at 81–87 % of the memory-bandwidth floor on 1–3B models
  (RTX 2080 Ti), 5–16 % faster than llama.cpp on those models and 47 % faster on Qwen2.5-0.5B. 3B
  prefill is still 4–11 % behind llama.cpp, because the GEMMs run at 0.93× cuBLAS."
- **vLLM (if used at all):** "1.15× faster than vLLM 0.31 at batch-1 decode on TinyLlama-1.1B
  (RTX A6000)."
- **Multi-GPU:** "Megatron-style tensor parallelism with a custom one-shot all-reduce: 8.5 vs 12.2 µs
  (4 KB) and 64 vs 85 µs (256 KB) against NCCL on 2× RTX 2080 Ti over PCIe 3.0 without
  peer-to-peer (staged through pinned host memory). 2-GPU decode is 1.33–1.74× faster than 1 GPU, and
  sharded outputs are bitwise-identical across all-reduce backends."
- **Model porting and testing:** "Ported Llama 3.x, TinyLlama, Mistral and Qwen2.5 through one weight
  exporter and loader: Qwen's QKV bias took ~20 lines, tied embeddings 4, Llama-3 RoPE scaling no
  engine code. Each model is gated by a 4-level test ladder: 899 kernel checks, greedy-token match
  against HF with near-tie tolerance, WikiText-2 perplexity within 0.01 % of HF, and 2-GPU-vs-1-GPU
  equivalence."
- **Profiling (once a trace exists, upgrade the tool name):** "Decomposed the gap between batch-1
  decode and the bandwidth floor from CUDA-event microbenchmarks. GEMVs stream at ~94 % of peak, and
  the rest is fixed per-kernel cost and latency-bound attention."
