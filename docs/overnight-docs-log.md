# Overnight docs log (2026-10-16)

Scope: three documents under `docs/`, written only from data already in the repo (no GPU access, no
benchmarks run). Nothing outside `docs/` is touched.

## Sources read

- `README.md`, `results/SUMMARY.md`, `results/a6000/THERMAL.md`, `docs/turing-prefill-gemm.md`,
  `docs/superpowers/plans/2026-10-15-prefill-3b-gap.md`
- `scripts/export.py`, `src/api.cu` (`load_shard`, `decode_step`, `prefill`, TP code),
  `src/kernels.cuh` (bias / RoPE epilogues), `scripts/hipinfer.py`, `scripts/bench.py`,
  `scripts/bench_tp.py`, `scripts/bench_baselines.py`, `scripts/prof_prefill.py`, `scripts/run_all.sh`
- `tests/test_kernels.py`, `tests/gate.py`, `tests/ppl.py`, `tests/test_tp.py`
- Every `results/2080ti/*/{bench,gate,ppl,tp2,tp2_test}.json`, `results/a6000/*/{bench,gate,ppl}.json`,
  `results/baselines/**`, `results/2080ti/gemm_sweep_*.json`, both `test_kernels.log` files
- Git history (8 commits; too short to count per-feature diffs, so line counts come from the current code)

## Findings that differ from, or sharpen, the README

1. RMSNorm is bit-exact against HF's fp16 order at 3 of the 5 tested shapes; at H=4096 (M=300) and
   H=3072 (M=33) the max error is 1.95e-3 and 9.8e-4 (within the test's 2e-3 tolerance). RoPE is
   bit-exact (tolerance 0) at every tested shape. Source: `results/2080ti/test_kernels.log`.
2. The 5–16 % batch-1 decode lead over llama.cpp is at a 128-token context. At 1,024 tokens it is
   3.0–14.4 % (Qwen2.5-0.5B: 43 %).
3. Llama-3.2-3B prefill is behind llama.cpp at 512 tokens as well (76.7 vs 65.4 ms, 1.17×), not only
   at 2,000.
4. One-shot all-reduce vs NCCL-only decode: 1.5–15.2 % lower latency (README says 2–15 %).
5. Perplexity within 0.003 % of HF holds for the single-GPU 2080 Ti runs (max 0.0027 %). A6000 runs reach
   0.0062 % (Llama-3.2-3B) and sharded Mistral-7B-v0.1 0.0098 %.
6. One A6000 gate divergence (Llama-3.2-1B) is at an HF top-2 gap of 0.03125, i.e. two fp16 steps at
   that logit magnitude; the README's "0 or 0.0156" holds for the 2080 Ti files.
7. `results/2080ti/gemm_sweep_*.json` contain configs 0–25 only; cfg 26 (the 256×128 tile) was added in
   the Turing GEMM change, so the sweeps predate the current build (they read 0.79–0.91× cuBLAS at
   M=2048, matching the write-up's "before" numbers).
8. No committed result file backs "batched projections ~1.15–1.2× slower than cuBLAS at 32–64
   sequences": `bench.json` times GEMMs only at M=128/512/2048, and the only M≤64 data is the stale sweep.
9. The `prof_prefill.py` output (the "~2.5 ms idle per prefill" figure) is not committed.
10. In the kernel microbenchmarks, matrices that fit in the 2080 Ti's L2 read faster than DRAM peak
    (Qwen2.5-1.5B o-proj: 738 GB/s against a 616 GB/s peak), because the same weight is replayed.
11. `export.py` does not check the architecture: a model whose tensor names match Llama's but whose math
    differs (other norm, activation, embedding scale, an o-proj or MLP bias) exports without an error.

## Progress

- [x] Survey repo and results; verify numbers (this file)
- [x] docs/porting-a-model.md
- [x] docs/profiling.md
- [ ] docs/claims.md
