# The Turing prefill GEMM (2026-10-15)

Llama-3.2-3B prefill on the 2080 Ti was the one headline row where llama.cpp won by a lot. About
90 % of that prefill is the four per-layer GEMMs, and those ran at 0.79–0.81× cuBLAS. This change
got them to 0.92–0.95× and prefill from 318.5 ms to 273.6 ms. llama.cpp is still ahead, at 261.8 ms.

| Llama-3.2-3B, RTX 2080 Ti | before | after |
|---|---|---|
| prefill, 2,000 tokens | 316–318 ms | **273.6 ms** (llama.cpp: 261.8 ms) |
| GEMMs at 2,048 rows vs cuBLAS (qkv, o, gate/up, down, lm_head) | 0.79–0.81× | **0.92–0.95×** |

## What changed

Turing has no `cp.async`, so the old main loop staged each K tile through registers. It paid for
the shared-memory store, the barrier and the next tile's first fragment loads one after another,
while the MMAs sat idle. The new loop overlaps them:

- The global loads for the next tile go out in two halves while the current tile's MMAs run.
- There is one `__syncthreads` per K tile.
- The next tile's first `ldmatrix` fragments are loaded before the last k16 step's MMAs, so those
  MMAs hide the load latency.

Three smaller changes came with it:

- A new 256×128 tile with 2×4 warps. Its 128×32 warp tiles can run the fused SwiGLU epilogue.
- The autotuner now times each candidate with the epilogue it will actually run (plain, residual
  add or SwiGLU), then re-times the three fastest back to back. When it timed everything with a
  plain epilogue, it picked a slower config for gate/up.
- The fallback choice on Turing (used before tuning, or with `HIPINFER_GEMM_AUTOTUNE=0`) no longer
  picks big tiles for short prompts. The old one took 42.9 ms at 128 tokens; the new one takes 24.0 ms.

Accumulation is still fp32, and there is still no vendor GEMM.

## How I measured it

The 273.6 ms is the median of 5 prefills on an otherwise idle 2080 Ti. These cards power-cap
under sustained load: I logged SM clocks of 1.41–1.62 GHz during long runs, against about 1.77 GHz
in short bursts. Add autotuner picks that differ slightly between processes, and repeated runs
landed anywhere between 273 and 285 ms.

A profiler trace ([scripts/prof_prefill.py](../scripts/prof_prefill.py)) showed only about 2.5 ms
of GPU idle time between kernels per prefill, so the rest is inside the GEMMs. That trace isn't
committed, so treat the 2.5 ms as unverified.

## What didn't work

At 2,048 rows the 256×128 tiles run one block per SM, so each GEMM's last wave leaves some of the
68 SMs idle. That is where most of the remaining gap is.

I tried a stream-K variant that splits the tail tiles along K across the idle SMs. It was correct
and deterministic, but 1.3–2.4× slower per GEMM on the 2080 Ti. My guess is that blocks starting
mid-K lose L2/DRAM locality, but I couldn't confirm it without profiler counters. It isn't in the code.

## Cost

The new 256×128 config spills 16 bytes of registers per thread on sm_75. I kept it because it is
still the fastest SwiGLU config I measured.
