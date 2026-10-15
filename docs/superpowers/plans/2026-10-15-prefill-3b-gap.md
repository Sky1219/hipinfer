# Close the Llama-3.2-3B Prefill Gap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make hipinfer's Llama-3.2-3B prefill at T=2000 on the RTX 2080 Ti beat llama.cpp's 261.82 ms (today: 318.45 ms), from-scratch kernels and fp32 accumulation only.

**Architecture:** About 88 % of the prefill is the four per-layer GEMMs, which run at 0.79–0.80× cuBLAS on Turing. The work rewrites the Turing (sm_75, `STAGES == 2`) main loop of `gemm_mma_kernel` so global loads, shared-memory stores, barriers and fragment loads overlap the MMAs instead of serializing with them, then lets the existing autotuner pick tiles. A profiling script decides whether any non-GEMM prefill kernel is also worth fixing.

**Tech Stack:** CUDA C++17 (nvcc 12.4 on edmonds), raw `mma.sync` / `ldmatrix` PTX, Python 3 + torch 2.11 for tests and benchmarks via ctypes (`scripts/hipinfer.py`).

**Spec:** `docs/superpowers/specs/2026-10-15-prefill-3b-gap-design.md`

## Global Constraints

- No cuBLAS, cuBLASLt, CUTLASS, hipBLASLt or similar inside the engine; cuBLAS only as the benchmark reference.
- fp32 accumulation stays. No fp16 accumulation or lower-precision operands.
- Pass bar: `tests/test_kernels.py` at its current tolerances; `tests/gate.py` greedy tokens identical except where HF's own top-2 logits tie; `tests/ppl.py` within 0.003 % of HF.
- Target: llama.cpp 261.82 ms (`results/baselines/2080ti/llama32-3b/llamacpp.log`). "Faster" = beats it by ≥ 1 %; within ±1 % = tie.
- Other models' prefill rows must not get slower (> 1 % is a regression).
- `make hip` must still compile. Ampere (`STAGES >= 3`, cp.async) path must keep passing tests.
- New kernel variants go through `kCfg` + the existing autotuner, never a hard-coded switch.
- README / `results/SUMMARY.md` numbers only from real runs.
- **No git commits until the work is done; one commit at the end (Task 6).**

## Environment (edmonds)

- `ssh edmonds` → work dir `/data/axd200066/hipinfer_pf` (new; the local tree currently matches `/data/axd200066/hipinfer_tpdev`).
- GPUs: index 0 and 2 = RTX 2080 Ti (use freely, but another user may share them; performance numbers only on a 2080 Ti with no other compute app). Index 1 = RTX A6000: **never run anything on it unless the user explicitly says so.** The user is using it for other work.
  Always set `CUDA_VISIBLE_DEVICES=0` for single-GPU work, `0,2` for TP.
- `PY=/data/axd200066/envs/baselines/bin/python` (torch 2.11, transformers 5.18). `NVCC=/usr/local/cuda/bin/nvcc` (12.4). No nsys/ncu.
- Weights: `/data/axd200066/models/<tag>.bin` (all 7 models + `mistral-7b-v03.bin`). `hipcc`: `/data/axd200066/rocm_venv/bin/hipcc`.
- Sync local → remote (excluding `.git`, `build`, `results`, `__pycache__`):
  `rsync -az --delete --exclude .git --exclude build --exclude results --exclude __pycache__ ./ edmonds:/data/axd200066/hipinfer_pf/`
- Work-in-progress results go to `results/2080ti-wip/` on edmonds only, never over `results/2080ti/`.

## Review Focus

1. **Prefill length that is not a tile multiple (T=2000, and T=1–127 tails):** every config must give correct rows at M=2000 on the real 3B shapes. Test added in Task 3.
2. **Very short K per split (KT = 1 or 2 K tiles in a block):** the new pipeline's prologue/epilogue must handle a block whose split has one or two tiles. Test added in Task 3.
3. **Tensor-parallel prefill (sharded N, two 2080 Tis):** the TP path calls the same GEMMs with halved N; `tests/test_tp.py` must still match 1-GPU output. Run in Task 5.
4. **Graph capture of GEMMs after the autotuner picks a new config:** `test_gemm_graph_after_workspace_growth` must still pass with the new loop. Run in Task 3 (existing test).
5. **Autotuning disabled (`HIPINFER_GEMM_AUTOTUNE=0`):** the `gemm_pick` fallback must still return a config that is valid and not slower than today on Turing. Checked in Task 4.

---

### Task 1: Remote workspace and baseline

**Files:**
- Create: `scripts/edmonds.sh` (sync + run helper)
- Output (edmonds only): `results/2080ti-wip/baseline/`

**Interfaces:**
- Produces: `scripts/edmonds.sh <cmd...>` — rsyncs the local tree to `/data/axd200066/hipinfer_pf`, then runs `<cmd...>` there via `ssh edmonds` with `cd /data/axd200066/hipinfer_pf`, `PY`, `NVCC` and `CUDA_VISIBLE_DEVICES=${GPUS:-0}` exported. Exit code is the remote command's.

- [ ] **Step 1: Write `scripts/edmonds.sh`** with the rsync line from Environment and the exports above; `set -euo pipefail`.

- [ ] **Step 2: Build and run the kernel tests on the unchanged tree**

Run: `scripts/edmonds.sh 'make NVCC=$NVCC && $PY tests/test_kernels.py > /tmp/hipinfer_pf_tk.log; tail -1 /tmp/hipinfer_pf_tk.log; grep -c FAIL /tmp/hipinfer_pf_tk.log'`
Expected: last line reports all OK, FAIL count 0.

- [ ] **Step 3: Record the baseline 3B bench**

Run: `scripts/edmonds.sh 'mkdir -p results/2080ti-wip/baseline && $PY scripts/bench.py --model unsloth/Llama-3.2-3B --weights /data/axd200066/models/llama32-3b.bin --no-hf --out results/2080ti-wip/baseline/bench.json > results/2080ti-wip/baseline/bench.log 2>&1; grep -E "T= 2000|M=2048" results/2080ti-wip/baseline/bench.log'`
Expected: prefill T=2000 within ±3 % of 318.45 ms, and the four M=2048 GEMM rows near 0.79–0.80× cuBLAS. If off by more, stop and report (machine state differs from the stored run).

---

### Task 2: Prefill kernel-time breakdown

**Files:**
- Create: `scripts/prof_prefill.py`

**Interfaces:**
- Consumes: `hipinfer.Engine(path, max_batch, max_seq, max_prefill)`, `Engine.prefill(slot, tokens)` (`scripts/hipinfer.py`).
- Produces: `python scripts/prof_prefill.py --weights <bin> [--T 2000] [--reps 5]` printing one row per kernel name: total ms per prefill, % of prefill, call count; then a family summary: `gemm`, `flash_prefill`, `other`. Families are matched by substring of the demangled name: `gemm` → contains `gemm`; `flash_prefill` → contains `flash_prefill`; anything else → `other`.

- [ ] **Step 1: Implement with `torch.profiler.profile(activities=[ProfilerActivity.CUDA])`** around `reps` prefills after 2 warm-up prefills (warm-up triggers the autotuner); kernels launched from the ctypes library are captured by CUPTI. Divide totals by `reps`.

- [ ] **Step 2: Run on the baseline**

Run: `scripts/edmonds.sh '$PY scripts/prof_prefill.py --weights /data/axd200066/models/llama32-3b.bin --T 2000 > results/2080ti-wip/baseline/prof.log; tail -5 results/2080ti-wip/baseline/prof.log'`
Expected: `gemm` family ≈ 280 ms (85–90 %), `flash_prefill` ≈ 23 ms, `other` ≈ 10–15 ms, and the family sum within 3 % of Task 1's T=2000 time.

- [ ] **Step 3: Decide Task 7.** If any single non-GEMM, non-attention kernel totals > 3 ms per prefill, Task 7 runs on it; otherwise Task 7 is skipped. Write the decision and the numbers into the task log.

---

### Task 3: Overlapped Turing main loop in `gemm_mma_kernel`

**Files:**
- Modify: `src/kernels.cuh:621-664` (the `else` branch of `if constexpr (STAGES >= 3)`, the Turing register-staged path) and the `compute` lambda `src/kernels.cuh:556-585` as needed.
- Test: `tests/test_kernels.py`

**Interfaces:**
- Consumes: kernel template `gemm_mma_kernel<BM, BN, BK, WGM, WGN, STAGES, EPI>` — signature unchanged; launch math (`gemm_mma_launch_st`, smem = `STAGES * (BM + BN) * (BK + 8) * 2`) unchanged so `cfg_smem` stays correct.
- Produces: same kernel, same configs; only the `STAGES == 2` path changes. Accumulator layout and epilogue untouched.

What is wrong today, per K tile: `gload(kt+1)` → `compute(kt)` (whose first `ldfrag(kk=0)` is exposed every tile, and with BK=32 that is half of all fragment loads) → `sstore` → `__syncthreads`. The store and barrier sit fully exposed after the MMAs.

- [ ] **Step 1: Add the Review Focus tests** to `test_gemm()` in `tests/test_kernels.py`, inside the per-config loop:

```python
        # Review Focus 1: real 3B prefill shapes at a non-tile-multiple M
        for N, K in [(5120, 3072), (3072, 8192)]:
            A = torch.randn(2000, K, device=dev, dtype=torch.float16)
            W = torch.randn(N, K, device=dev, dtype=torch.float16) * 0.02
            check(f"gemm cfg={cfg} M=2000 N={N} K={K}", hi.gemm(A, W), A.float() @ W.float().T, 3e-3, 3e-3)
```

and to `test_gemm_splitk()`, Review Focus 2 (one and two K tiles per split; `K % 32 == 0` is the API's floor):

```python
    for sp in [1, 2]:
        hi.lib.hi_gemm_force_splits(sp)
        for M, K in [(300, 32 * sp), (300, 64 * sp), (129, 96)]:
            x = torch.randn(M, K, device=dev, dtype=torch.float16)
            W = torch.randn(1024, K, device=dev, dtype=torch.float16) * 0.02
            check(f"gemm short-K split={sp} M={M} K={K}", hi.gemm(x, W), x.float() @ W.float().T, 3e-3, 3e-3)
    hi.lib.hi_gemm_force_splits(-1)
```

Skip a config/K pair the kernel rejects by contract (K not a multiple of `BK * splits`): check how `gemm_choose` / the split logic guards that before adding, and mirror the guard rather than asserting on it.

- [ ] **Step 2: Run on the unchanged kernel**

Run: `scripts/edmonds.sh 'make NVCC=$NVCC && $PY tests/test_kernels.py | grep -E "M=2000|short-K|FAIL" | tail -20'`
Expected: all new lines OK (they pin behaviour; the rewrite must keep them OK). Note how long the full test takes.

- [ ] **Step 3: Rewrite the `STAGES == 2` main loop.** Keep two shared buffers (Turing's 64 KB opt-in smem cannot fit a third buffer for the 256×128×32 tile: 3·384·40·2 = 92 KB). Required properties, in this order of expected payoff:
  1. **Cross-tile fragment prefetch:** the `ldfrag(kk=0)` of tile kt+1 is issued before the last k16 step's MMAs of tile kt, not after the barrier. That needs tile kt+1 in shared memory before the last k16 step, so the `sstore` of kt+1 and its `__syncthreads` move to just before the last k16 step of tile kt.
  2. **Spread global loads:** split `gload` across the k16 steps of a tile (A chunks in the first half, W chunks in the second) so the load instructions interleave with MMAs instead of issuing in one burst.
  3. One `__syncthreads` per K tile, as today.
  Restructure `compute` into a per-k16-step helper so the loop can place `sstore` / barrier / prefetch between steps; the `STAGES >= 3` branch must call it so it keeps its current behaviour exactly.

- [ ] **Step 4: Check registers and spills**

Run: `scripts/edmonds.sh '$NVCC -O3 -std=c++17 -gencode arch=compute_75,code=sm_75 -Xptxas -v -Xcompiler -fPIC -shared src/api.cu -o /tmp/pf_ptxas.so 2>&1 | grep -A1 gemm_mma_kernel | grep -E "spill|registers" | sort | uniq -c | head'`
Expected: `0 bytes spill stores` for every `gemm_mma_kernel` instantiation.

- [ ] **Step 5: Run the full kernel tests**

Run: `scripts/edmonds.sh 'make NVCC=$NVCC && $PY tests/test_kernels.py > /tmp/pf_tk.log; tail -1 /tmp/pf_tk.log; grep FAIL /tmp/pf_tk.log | head'`
Expected: no FAIL lines (includes every config, split-K, epilogues, and `test_gemm_graph_after_workspace_growth`, Review Focus 4).

- [ ] **Step 6: Measure the four 3B shapes**

Run: `scripts/edmonds.sh '$PY scripts/bench.py --model unsloth/Llama-3.2-3B --weights /data/axd200066/models/llama32-3b.bin --skip-e2e --gemm-sweep --out results/2080ti-wip/t3_sweep.json > results/2080ti-wip/t3_sweep.log 2>&1; grep "M= 2048" results/2080ti-wip/t3_sweep.log'`
Expected: best config per shape at M=2048 improves over the baseline sweep (`results/2080ti/gemm_sweep_llama32-3b.log`: qkv 37.0, o 37.2, gate_up 36.4, down 35.2 TFLOPs). Target ≥ 0.95× cuBLAS on gate_up. If the best config did not move, iterate on Step 3 before Task 4.

---

### Task 4: Re-tune configs and the fallback pick for Turing

**Files:**
- Modify: `src/api.cu` — `kCfg` (`src/api.cu:210-222`), `kNumCfg`, `gemm_cfg` switch (`src/api.cu:279-317`), `gemm_pick` (`src/api.cu:260-277`).

**Interfaces:**
- Consumes: Task 3's kernel (unchanged template parameters).
- Produces: zero or more new `mma.sync` entries appended **after** config 25 (existing indices keep their meaning, so stored sweeps stay comparable), each with its `case` in `gemm_cfg` and `if constexpr (EPI != 2)` where the warp tile is not 32 wide (`static_assert(EPI != 2 || WTN == 32)`).

- [ ] **Step 1: Pick candidates from the Task 3 sweep.** Add a tile only if the overlapped loop makes a shape that does not exist today plausible (candidates: 128×256×32 4×2 warps, 256×128×32 2×4 warps, 128×128×64 2×2 warps for Turing smem ≤ 64 KB with 2 stages). Each must satisfy `cfg_smem(c) + 1024 <= g_smem_optin` on sm_75.

- [ ] **Step 2: Rebuild, re-run Task 3 Step 5 (tests cover every config automatically) and Step 6 (sweep).**
Expected: no FAIL; keep only configs that are best for at least one 3B (or other model) M=2048 shape; drop the rest.

- [ ] **Step 3: Update `gemm_pick` for sm_75** so `HIPINFER_GEMM_AUTOTUNE=0` lands on the best Turing config from the sweep at M ≥ 512 (Review Focus 5). Keep the `#else` (AMD) branch and Ampere choices unchanged.

Run: `scripts/edmonds.sh 'HIPINFER_GEMM_AUTOTUNE=0 $PY scripts/bench.py --model unsloth/Llama-3.2-3B --weights /data/axd200066/models/llama32-3b.bin --no-hf --out results/2080ti-wip/t4_noauto.json 2>&1 | grep "T= 2000"'`
Expected: T=2000 prefill no slower than Task 1's baseline.

- [ ] **Step 4: 3B prefill check**

Run: `scripts/edmonds.sh '$PY scripts/bench.py --model unsloth/Llama-3.2-3B --weights /data/axd200066/models/llama32-3b.bin --no-hf --out results/2080ti-wip/t4.json > results/2080ti-wip/t4.log 2>&1; grep "T= 2000" results/2080ti-wip/t4.log'`
Expected: T=2000 ≤ 259.2 ms (beats 261.82 by ≥ 1 %). If it is above that, report the number and the per-family profile (`scripts/prof_prefill.py`) and go back to Task 3 Step 3 or Task 7 before Task 5.

---

### Task 5: Full verification on the 2080 Tis (and the A6000 when free)

**Files:**
- Output (edmonds): `results/2080ti-wip/<model>/…` via `scripts/run_all.sh`

- [ ] **Step 1: All models, single GPU + TP2**

Run: `GPUS=0,2 scripts/edmonds.sh 'WEIGHTS_DIR=/data/axd200066/models PY=$PY scripts/run_all.sh 2080ti-wip > results/2080ti-wip/run_all.log 2>&1; tail -15 results/2080ti-wip/run_all.log'`
Expected: no failures listed (kernel tests, `gate`, `ppl`, `bench`, `tp2 test`, TP ppl for the 7B/8B models — Review Focus 3).
Check `WEIGHTS_DIR` naming against `run_all.sh` (it expects `<model_tag>.bin`; `mistral-7b` maps to `mistral-7b-v03.bin` on edmonds — symlink it into a scratch dir if the names differ).

- [ ] **Step 2: Compare prefill rows**

Run: `scripts/edmonds.sh 'for m in tinyllama llama32-1b qwen25-0.5b qwen25-1.5b llama32-3b; do echo "== $m"; grep -E "^T=" results/2080ti/$m/bench.log | head -5; grep -E "^T=" results/2080ti-wip/$m/bench.log | head -5; done'`
Expected: every T row in `2080ti-wip` ≤ 1.01× the stored `2080ti` row; Llama-3.2-3B T=2000 ≤ 259.2 ms. Perplexity in every `ppl.log` within 0.003 % of HF.

- [ ] **Step 3: AMD compile check**

Run: `scripts/edmonds.sh 'make hip HIPCC=/data/axd200066/rocm_venv/bin/hipcc 2>&1 | tail -3'`
Expected: builds `build/libhipinfer_amd.so`. If the toolchain itself is broken on the unchanged tree too, record that and move on (unchanged AMD code).

- [ ] **Step 4: A6000 (only when the user explicitly says to run it; idle in `nvidia-smi` is not enough)**

Run: `GPUS=1 scripts/edmonds.sh '$PY tests/test_kernels.py | tail -1; $PY scripts/bench.py --model unsloth/Llama-3.2-3B --weights /data/axd200066/models/llama32-3b.bin --no-hf --out results/2080ti-wip/a6000_3b.json 2>&1 | grep -E "T= *(128|512|2000)"'`
Expected: tests all OK; prefill not slower than `results/a6000` / README A6000 numbers by > 1 %. If the A6000 is still busy at the end of the session, say so in the hand-off; do not block the commit on it.

---

### Task 6: Results, docs and the one commit

**Files:**
- Modify: `README.md` (status block, headline table 3B row, "What this says" bullets on prefill), `results/SUMMARY.md` (via `scripts/summarize.py`)
- Move: `results/2080ti-wip/*` → `results/2080ti/` (edmonds → local via rsync)
- Delete: `results/2080ti-wip/`

- [ ] **Step 1: Pull results** — `rsync -az edmonds:/data/axd200066/hipinfer_pf/results/2080ti-wip/ results/2080ti/` (overwrites only the files the new run produced), then regenerate `results/SUMMARY.md` with `scripts/summarize.py` as `run_all.sh` / the README describe.

- [ ] **Step 2: Update README** from those files only: the Llama-3.2-3B prefill column (vs llama.cpp 261.82 ms), the 2080 Ti prefill table, any other prefill numbers that changed, and the GEMM-vs-cuBLAS statements (e.g. "~0.8× cuBLAS"). Wording rule from the spec: "faster" only if ≥ 1 % ahead, otherwise "tie".

- [ ] **Step 3: Verify every changed README number** has a source line in `results/2080ti/**`: `grep` each new number in the results tree.
Expected: every number found.

- [ ] **Step 4: Ask the user to confirm, then make the single commit** (repo has no commits; this is the first). Include `scripts/edmonds.sh`, `scripts/prof_prefill.py`, the docs under `docs/superpowers/`, source, tests, results, README.

```bash
git add -A
git commit -m "Faster Turing prefill GEMM: overlapped main loop; Llama-3.2-3B prefill beats llama.cpp"
```
(Adjust the subject if the result is a tie or still behind; append the attribution trailer required by the session.)

---

### Task 8 (added 2026-10-15, runs after Task 4 and before Task 5): GEMM wave quantization on Turing

Why: after Tasks 3–4, T=2000 is 273.6–284.8 ms against the 259.2 ms target. The autotuned M≈2048 configs (cfg 9 / cfg 26, 256×128 tiles) run one block per SM on the 2080 Ti's 68 SMs. qkv launches 320 tiles (4.71 waves), o and down 192 (2.82 waves), gate_up 1024 (15.06 waves), so each GEMM's last wave leaves SMs idle; the estimate is ≈ 6 % of GEMM time, ≈ 15 ms per prefill. Measured GPU idle between kernels is only ≈ 2.5 ms, so the remaining gap is inside the GEMMs.

**Files:**
- Modify: `src/kernels.cuh` (`gemm_mma_kernel` scheduling and epilogue), `src/api.cu` (`kCfg`, `gemm_cfg`, `split_ws_bytes`, autotuner if needed)
- Test: `tests/test_kernels.py`

**Interfaces:**
- Consumes: Task 3's Turing loop and `mma_step`; the existing split-K fp32 workspace (`g_split.ws`, `g_split.cnt`) and its last-block-reduces protocol.
- Produces: any new scheduling mode is a new `kCfg` entry, or a split mode the autotuner times. It needs no new public API and no change to `hi_gemm`'s signature.

- [ ] **Step 1: Add the failing-or-pinning test** to `test_gemm()` for each new config. It covers the 3B M=2000 shapes at epi 0/1/2 (qkv 5120×3072, o 3072×3072, gate_up 16384×3072 interleaved SwiGLU, down 3072×8192), plus M ∈ {1, 77, 300} tails, at the existing tolerances (3e-3 / 4e-3). It also covers graph replay as in `test_gemm_graph_after_workspace_growth` if the mode uses the workspace.
- [ ] **Step 2: Implement the scheduling.** Builder's choice, subject to fp32 accumulation (fp32 partials, any order) and determinism within a run. The candidates are:
  1. **Stream-K / split tail.** Full waves run whole tiles, and the tail tiles are split along K across the idle SMs with fp32 partials, reduced by the existing last-block protocol.
  2. **A persistent kernel** with an atomic or static tile scheduler.
  3. **Per-shape tile shapes or split factors** chosen by the autotuner, so the wave count lands close to an integer.
  Option 3 is acceptable only if it reaches the target.
- [ ] **Step 3: Check registers and spills** for the new instantiations (Task 3 Step 4 command). Expect no new spills beyond cfg 26's accepted 16 B.
- [ ] **Step 4: Run the full `tests/test_kernels.py`** on a 2080 Ti. Expected: no FAIL. The A6000 run is deferred to Task 5 Step 4, which only happens on the user's explicit go-ahead.
- [ ] **Step 5: Measure** on an uncontended 2080 Ti: check `nvidia-smi --query-compute-apps` first, and run nothing while another user's process is on that GPU. Record the SM clock alongside each measurement.
  - Run `scripts/prof_prefill.py --T 2000` and the 3B bench. Expected: T=2000 ≤ 259.2 ms (median), the GEMM family time down by about the wave-tail loss, and every other T row no slower than Task 4's.
  - If T=2000 is still above 259.2 ms, report the number, the per-GEMM before/after, and what is left, rather than iterating without bound. The time cap is about 90 minutes.

### Task 7 (conditional on Task 2 Step 3): Non-GEMM prefill kernel

**Files:** decided by Task 2's profile — the kernel in `src/kernels.cuh` and its launch in `src/api.cu`.

- [ ] **Step 1: Add a test** in `tests/test_kernels.py` for that kernel at T=2000, 3B dims (hidden 3072, heads 24, kv_heads 8, head_dim 128), against the same reference its existing test uses, at the existing tolerance.
- [ ] **Step 2: Run it on the unchanged kernel** — expected OK (pins behaviour).
- [ ] **Step 3: Speed it up or fuse it** into a neighbouring GEMM epilogue; no change to math order where the existing test checks HF's exact fp16 op order.
- [ ] **Step 4: Re-run `scripts/prof_prefill.py`** — expected: that kernel's ms per prefill drops; total prefill drops by about the same amount. Then full `tests/test_kernels.py`, no FAIL.
