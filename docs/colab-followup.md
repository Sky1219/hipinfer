# The A100 follow-up run (Colab)

`hipinfer_colab_followup.ipynb` (built by `scripts/build_colab_followup.py`) is a second, about
one-hour Colab A100 run that fills the gaps in the first one (`results/a100`,
`results/baselines/a100`). Its steps are ordered by value, so a run stopped early still leaves the
most useful results. Every step is timed in `timeline.txt`. A failing step is logged and skipped,
and the results are zipped after every step.

Results go to new directories, so nothing committed is overwritten:

- `results/a100_followup/`: the MoE kernel test, the profiles, Granite, `gpu_log.csv` and `timeline.txt`
- `results/baselines/a100_followup/<model>/`: the two engine comparisons, each with hipinfer
  re-run in the same session

Before committing any number from this run, check `gpu_log.csv` for throttling, the same way as
`results/a100/THERMAL.md`. The log has the throttle-reason flags (power cap, HW slowdown, SW/HW
thermal, power brake) every 5 s.

## Step 1: build, and the MoE kernel test

`test_moe()` from `tests/test_kernels.py` checks the MoE router and both expert paths (GEMV and
grouped GEMM) against `scripts/ref_engine.py`, on Granite-, Mixtral- and OLMoE-shaped layers. Last
time it failed to import `ref_engine`; every other kernel test had already passed on this GPU.

- Read: `test_kernels_moe.log` ends in `ALL PASS`.
- Then: the MoE kernels are verified on Ampere as well as Turing.

## Step 2: profile batch-1 decode (Nsight Systems)

**The claim under test.** The README says A100 decode reaches only 61–74 % of the bandwidth floor
on the 1–3B models, against 81–87 % on a 2080 Ti. It says this is because each kernel launch costs
about the same on both GPUs, while the weights stream 2.5× faster on the A100. Until now that was
an inference from timings. Nothing measured the gaps.

**What runs.** `scripts/profile_decode.sh` on TinyLlama-1.1B and Llama-3.2-3B:

- prefill 128 random tokens, then decode 128 tokens from the CUDA graph and 64 without it;
- each decode run sits inside an NVTX range, under `nsys profile --trace=cuda,nvtx
  --cuda-graph-trace=node` (node tracing, because decode is one graph launch per token);
- the same runs again without the profiler, for reference ms/token, plus the measured DRAM read
  bandwidth (a `sum` over 2 GiB) and copy bandwidth.

`scripts/profile_decode.py summarize` reads the SQLite export and gives, per token
(`summary.json`):

| field | meaning |
|---|---|
| `kernels_per_step` | kernels in one decode step (expected 5 per layer + 3: 113 for TinyLlama, 143 for the 3B) |
| `wall_ms`, `kernel_sum_ms`, `gap_ms` | step period (first kernel start to the next step's first kernel start), the sum of kernel durations, and the difference: GPU idle between kernels |
| `gaps` | gap time inside a step and between steps; the median, p90 and max gap; the mean gap before each kernel type |
| `gemv.<role>` | per GEMV role (qkv, o, gate_up, down, lm_head): median time and achieved bandwidth, where bytes is the weight matrix |
| `gemv_fit` | least-squares fit of GEMV time = t0 + bytes / BW over every GEMV in the step. t0 is the fixed cost a GEMV pays whatever its size (block fill and tail, the fused RMSNorm prologue); BW is the streaming rate. |
| `shortfall` | wall − floor, split into gaps, fixed cost inside kernels (GEMV t0 × count + attention time above reading its KV cache at peak + embed/argmax), and the rest: GEMVs streaming below peak |
| `profiler_inflation` | engine ms/token under nsys ÷ without it |

The `nsys stats` tables (`stats_graph_*.csv`, limited to the `decode_graph` range when the nsys
version supports `--filter-nvtx`, otherwise `stats_all_*.csv`) and the `.nsys-rep` are kept for
inspection in the Nsight Systems GUI.

**How to read it.**

1. **Check the profiler first.** If `profiler_inflation.graph` is above about 1.05, node tracing
   is adding time. Its gaps then overstate the real ones. Use the reference `wall` minus the traced
   `kernel_sum_ms` as the gap estimate instead: kernel durations are less affected than gaps.
2. **Gaps** (`shortfall.gaps_share` = gaps ÷ (wall − floor)):
   - ≥ 0.5: **confirmed**. Most of the missing time is the GPU idle between kernels, which is
     launch/dispatch overhead. The fix is fewer kernels (more fusion, a persistent decode kernel).
   - ≤ 0.2: **refuted as stated**. The GPU is busy almost the whole token. The missing time is
     inside the kernels, and the README sentence has to change.
   - In between: gaps are a real but partial cause.
3. **Inside the kernels**, when gaps don't explain it:
   - A large `fixed_in_kernel_ms` (t0 × count) means every kernel pays a fixed cost whatever its
     size. That is still per-kernel overhead, and the fusion remedy is the same, but it isn't launch
     latency. The README would have to say "per-kernel fixed cost (fill, tail, norm prologue)", not
     "launch overhead". `verdict` calls this "partly".
   - A large `streaming_below_peak_ms`, with the big GEMVs (gate_up, down, lm_head) well below
     the measured read bandwidth, means the GEMV's memory pipeline doesn't keep enough bytes in
     flight for the A100's 1.5 TB/s. The fix is in the kernel (more loads in flight per SM), not in
     launches.
4. **Peak vs measured.** The floor uses the 1,555 GB/s datasheet figure. Compare each GEMV's GB/s
   with `read_bw_GBs` as well: a GEMV at 90 % of the measured read rate has little left to gain,
   whatever its share of the datasheet figure. The GEMV numbers in `bench.json`'s `kernels` section
   are microbenchmarks that loop over one matrix, which fits in the A100's 40 MB L2. That is why
   some read above 1,555 GB/s. The trace's numbers come from real decode, where every matrix comes
   from DRAM.
5. **Graph vs no graph.** `decode_nograph` shows the same split with host-launched kernels. The
   engine already reports only 0.1–0.2 ms/token between the two, so host launch cost is small.
   GPU-side gaps can still be large in both modes.

**What the committed numbers predict.** This was written before the run. The A100 shortfall per
kernel is 7.4 µs (TinyLlama), 8.9 (Llama-3.2-1B), 8.5 (Qwen2.5-1.5B), 10.2 (Llama-3.2-3B) and 11.7
(Mistral-7B). In absolute terms it is close to the 2080 Ti's (0.83 against 0.77 ms on TinyLlama,
1.46 against 1.63 ms on the 3B). The similar absolute shortfall fits a fixed per-kernel cost.
But it grows with model size, and graph-node launch gaps on an A100 are usually a few µs, so I
expect gaps to explain only part of it, with a size-dependent part inside the kernels. If
`gaps_share` comes out ≥ 0.5 on both models, that expectation was wrong and the README stands.

## Step 3: Granite-3.1-1B-A400M (MoE) end to end

This step runs the export, `tests/gate.py --router-tie 0.01`, `tests/ppl.py --windows 16` and
`scripts/bench.py`, as on the 2080 Ti (`results/2080ti/granite-moe-1b/`). A token divergence is
accepted only at an fp16 token near-tie, or when the router near-tie check explains it (HF re-run
adopting the engine's expert set where HF's own top-k margin is ≤ 0.01; `docs/moe-design.md`).

- Read: `gate.log` ends in `GATE PASS`. Lines saying "router near-tie, OK" are expected and count
  as passes. `ppl.json` shows the hipinfer and HF perplexities, top-1 agreement and KL; on the
  2080 Ti they were 7.7292 vs 7.7294, with 99.4 % top-1 agreement.
- `bench.json`: MoE decode ms/token and % of floor, where the floor counts only the experts one
  token reads. Compare it with the 2080 Ti's 63 % (README).

## Step 4: torch.compile on Qwen2.5-1.5B, QKV bias included

The first A100 run left Qwen2's QKV bias out of the gpt-fast-style baseline. Its tokens matched HF
on 0/32 and the row was marked "not valid". `scripts/bench_baselines.py` now adds the bias.

- Read: the notebook logs `torch.compile greedy tokens vs HF: 32/32`. Anything less means the
  numbers are still not valid, and the step is listed under FAILED.
- `hipinfer.json` next to it is the same-session hipinfer run. Compare decode and prefill against
  it, not against `results/baselines/a100/`.

## Step 5: vLLM on Llama-3.2-3B

The first run's disk filled before vLLM could load the 3B. This time models are deleted after each
step (the 3B is kept from step 2 to step 5 only), and `HF_HOME` is pinned so vLLM loads the export's
download instead of a second copy. vLLM is pinned to 0.31.0, the version in the committed table.
If that won't install, the latest is used; `vllm.json` records which.

- Read: `vllm.json` decode ms/token at ctx 128/1024 and prefill, against `hipinfer.json` from the
  same session.

## Afterwards

- Copy the summaries, logs, JSONs and `stats_*.csv` into the repo. The `.nsys-rep` files can stay
  out of git if they are large.
- Update the README's A100 tables. Fill the "not valid" and "not run" cells only from steps 4 and
  5's same-session pairs, and rewrite the launch-overhead paragraph to match step 2's verdict.
