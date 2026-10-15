# Close the Llama-3.2-3B prefill gap with llama.cpp: design

Date: 2026-10-15

## Intent

On the RTX 2080 Ti, llama.cpp prefills a 2,000-token Llama-3.2-3B prompt 1.22× faster than
hipinfer (about 261 ms against our 318.45 ms in `results/2080ti/llama32-3b/bench.log`). Every
other headline row is a hipinfer win or close to one. The goal is to flip this row: hipinfer
at least ties llama.cpp, ideally beats it, without giving up anything the project stands on.

## Why the gap exists (measured, 2080 Ti)

- The four per-layer GEMMs at M=2048 cost 1.60 (qkv) + 0.97 (o) + 5.01 (gate_up) + 2.49 (down)
  = 10.06 ms per layer, ×28 layers ≈ 282 ms. That is about 88 % of the 318 ms prefill.
- Prefill attention is about 0.84 ms per layer at T=2048 (≈ 23 ms total) and already fast.
- Our GEMM runs at 0.79–0.80× cuBLAS at M=2048 (≈ 40–41 TFLOPs vs 50–52). At cuBLAS speed the
  GEMMs would total ≈ 225 ms and prefill ≈ 261 ms, which is llama.cpp's time. llama.cpp uses
  cuBLAS for F16 prefill matmuls, so **the gap is our GEMM vs cuBLAS**.
- gate_up is half of the GEMM time, so it carries most of the win.
- The 26-entry tile-config sweep (`results/2080ti/gemm_sweep_llama32-3b.log`) peaks at about
  35–38 TFLOPs at M=2048. Retuning tile sizes alone will not get there; the kernel itself must
  get faster.
- On Turing (sm_75), `gemm_mma_kernel` (`src/kernels.cuh`) uses its STAGES == 2 path: register
  staged double buffering (global → registers during compute, registers → shared after, one
  `__syncthreads` per K tile), because sm_75 has no `cp.async`.

## Goals

1. Llama-3.2-3B prefill at T=2000 on the 2080 Ti at or below ~261 ms (tie llama.cpp); below
   that is a win.
2. Any other change that measurably lowers prefill time is in scope (user: "if it would
   improve prefill time do it"), e.g. non-GEMM prefill kernels or launch overhead.
3. The other models' prefill rows must not get slower.

## Non-goals

- Decode (GEMV, decode attention, graph capture), except to keep it working.
- AMD paths (`gemm_kernel` / rocWMMA): must still compile, not optimized.
- Ampere `cp.async` path: must keep passing tests; not a target, though a shared improvement
  that also helps it is welcome.

## Anti-goals (a failure even if it is faster)

- **No vendor GEMM.** No cuBLAS, cuBLASLt, CUTLASS, hipBLASLt or similar inside the engine.
  The project is from scratch. cuBLAS stays only as the benchmark reference.
- **No precision loss.** fp32 accumulation stays. No fp16 accumulation, no lower precision
  operands, nothing that would move WikiText-2 perplexity or greedy tokens off their current
  match with HF transformers. The user considers this cheating.

## Constraints

- Hardware: `ssh edmonds` (edmonds.utdallas.edu) has 2× RTX 2080 Ti, the same GPU as the
  headline numbers, and an RTX A6000 that is busy for roughly the next 3 hours.
  The 2080 Tis may be used freely (repeated benchmarks, full test runs, all-model checks).
- Git: no commits during the work. Commit once, only when the work is done.
- Time: about 3 hours of work in this session.
- Build: `make` (nvcc, `NV_ARCH` covers sm_75/80/86/89). Must keep building with `make hip`.
- Correctness gates already in the repo, which must keep passing:
  `tests/test_kernels.py` (every tile config × split-K × epilogue), `tests/gate.py`
  (greedy tokens vs HF), `tests/ppl.py` (WikiText-2 perplexity vs HF).
- Measurement: `scripts/bench.py` (GEMM vs cuBLAS rows, prefill latency), with
  `GEMM_SWEEP=1` for the tile sweep; `scripts/run_all.sh` for the full per-model run.

## Decided by the user

- Target is the llama.cpp comparison row, not a cuBLAS ratio for its own sake.
- From scratch only; fp32 accumulation only.
- Scope extends to anything that improves prefill time.

## Calls made in the design (not the user's; flagged for review)

- **"No precision loss" means fp32 accumulation and the existing gates pass, not bit-identical
  output.** Reordering fp32 sums (new tiles, deeper pipeline, split-K) is not a precision cut;
  the autotuner already picks different configs per shape and GPU. Pass bar: `test_kernels.py`
  at its current tolerances, `gate.py` greedy tokens identical except where HF's own top-2
  logits tie (the README's existing standard), `ppl.py` within the current 0.003 % of HF.
- **Target number:** the stored llama.cpp result, 261.82 ms
  (`results/baselines/2080ti/llama32-3b/llamacpp.log`, best of flash-attn on/off and ubatch
  512/2048), since that is what the README compares against. "Faster" requires beating it by
  more than run-to-run noise (≥ 1 %, measured as the median of the existing bench repeats);
  within ±1 % is reported as a tie. Re-running llama.cpp is optional, only if time allows.
- **Other GPUs:** the 2080 Ti is the target. Ampere configs must not regress, which the
  per-GPU autotuner should guarantee; this gets checked on the A6000 when it is free. The
  one-time autotune on first call is not counted in prefill latency (the bench warms it up
  already).

## Left to the builder

How to make the Turing GEMM faster. Candidate directions, none decided:

- Deeper software pipeline on sm_75: more than two shared stages, with the global loads for
  tile k+2 spread across the k16 steps of tile k so the load latency hides and the per-tile
  `__syncthreads` bubble shrinks.
- Bigger warp tiles / better register reuse (more MMAs per ldmatrix), within the register
  budget, and checking occupancy and spills with `-Xptxas -v`.
- XOR-swizzled shared layout instead of +8 padding, to fit more stages in shared memory.
- New tile configs added to `kCfg` and picked by the existing autotuner.
- Non-GEMM prefill costs: the ≈ 13 ms that is neither GEMM nor attention (norms, RoPE,
  cache writes, launches).

Any new kernel variant must go through the existing autotuner and tests rather than a
hard-coded switch, and the README/SUMMARY numbers are updated only from real runs.
