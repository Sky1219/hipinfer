# AMD: what compiles, and what to check first on real hardware

Status: **compile-checked only.** `make hip` builds `build/libhipinfer_amd.so` for gfx90a (MI200),
gfx942 (MI300) and gfx1100 (RDNA3) with ROCm 7.2.4, rocWMMA and (optionally) RCCL, with no errors
and no warnings, MoE kernels included. Nothing has run on an AMD GPU. The logs, the toolchain
versions and the script that produced them are in
[`results/amd_build/`](../results/amd_build/).

## What AMD runs

The NVIDIA fast paths (`mma.sync` / `ldmatrix` / `cp.async` GEMMs and attention) are behind
`#if !defined(HI_AMD)`. On AMD the engine always uses the portable kernels:

| job | AMD kernel | NVIDIA equivalent that exercises the same code |
|---|---|---|
| decode projections | `gemv_kernel` | the same kernel (NVIDIA uses it too) |
| prefill / batched GEMM | `gemm_kernel` on rocWMMA 16×16×16, tile configs 0–5 | configs 0–5 (`hi_gemm_force_config`), on `nvcuda::wmma` |
| prefill attention | `flash_prefill_kernel` (WMMA) | `HIPINFER_FLASH_WMMA=1` |
| decode attention | `decode_attn_kernel` (scalar fp32) | `HIPINFER_DECODE_SCALAR=1` |
| RMSNorm, RoPE, argmax, embedding | the same kernels | the same kernels |
| MoE router, sort, gather, expert GEMVs, combine | the same kernels | the same kernels |
| tensor-parallel all-reduce | one-shot kernel, or RCCL with `RCCL=1` | one-shot kernel, or NCCL |

NVIDIA runs with those switches test the algorithms, but **not the wave64 lane mappings**, because
on NVIDIA (and on gfx1100) a warp is 32 lanes.

## How the warp width is handled

- `HI_WARP` (`src/hip_compat.h`) is a compile-time constant per device target: 64 when
  `__GFX9__` is defined (CDNA: gfx90a, gfx942), else 32 (RDNA, NVIDIA). The build check confirmed
  the macros: the gfx90a and gfx942 device passes define `__GFX9__`, and gfx1100 defines `__GFX11__`.
  ROCm 7's clang no longer defines `__AMDGCN_WAVEFRONT_SIZE`, so there is nothing to cross-check
  `HI_WARP` against at compile time.
- The host reads the warp size at run time (`g_warp = hipDeviceProp_t::warpSize`) and sizes the
  launches from it: `gemm_kernel` gets `8 * g_warp` threads (512 on CDNA, matching its
  `__launch_bounds__(8 * HI_WARP)`), `flash_prefill_kernel` `4 * g_warp`, and the GEMV grids use
  `256 / g_warp` warps per block.
- **The two must agree.** If gfx1100 were built with `-mwavefrontsize64`, the device code would
  still use `HI_WARP = 32` while the hardware runs 64-lane waves, and every reduction below would
  silently cover half a wave. The Makefile doesn't pass that flag; don't add it.
- Warp shuffles are `__shfl_xor(v, m)` on AMD, with the default width (`warpSize`, the whole
  wave) and no lane mask. Every call site is reached by the whole wave (no shuffle sits inside
  lane-divergent control flow), which is what HIP's maskless shuffles need. `hi_syncwarp()` is a
  wavefront fence plus `__builtin_amdgcn_wave_barrier`.

## Code paths that behave differently on wave64 (gfx90a, gfx942): check these first

In order of risk, for the first run on a CDNA GPU (`tests/test_kernels.py` covers every one of them):

1. **`flash_prefill_kernel`: different lane→row mapping.** `LPR = HI_WARP / 16` lanes share
   one query row: 4 on wave64 against 2 on wave32. Each lane then owns `CPL = BKV / 4` score
   columns and `OCC = 16` output columns per 64-wide chunk (wave32: `BKV / 2` and 32), and the
   row max and sum are reduced with `xor 1, 2` instead of `xor 1`. This layout has never executed
   anywhere. Test: `test_flash_prefill`, both head dims (D = 64 with BKV = 64, D = 128 with BKV = 32).
2. **`moe_router_kernel`: 64-lane top-k.** Warp 0 holds `S = kMoeMaxE / HI_WARP = 4` logits per
   lane (8 on wave32), tracks taken experts in a 4-bit mask, and runs k rounds of a 64-lane argmax
   with the NaN and tie rules of `argmax_better`. On wave64 one block also computes 4 expert logits
   at a time instead of 8. Test: `test_moe` (router and top-k vs PyTorch, including ties).
3. **`gemm_kernel` on rocWMMA.** 8 waves of 64 = 512 threads per block. The epilogue stages
   each 16×16 fragment through 512 floats of per-wave scratch, and its loop
   `for (ch = lane; ch < 32; ch += HI_WARP)` leaves lanes 32–63 idle (correct, just idle). Split-K
   partial sums meet through `__threadfence()`, an atomic counter and `hi_ldcg` (an agent-scope
   relaxed atomic load on AMD). Test: `test_gemm`, `test_gemm_splitk` for all six configs.
4. **Block reductions: `block_sum`, `block_argmax`.** A block has half as many warps (a
   1024-thread argmax has 16 waves, not 32). The second reduction level reads `sh[l]` for `l < nw`,
   so its shared arrays (32 entries) are big enough either way. Tests: `test_rmsnorm`, `test_argmax`.
5. **`decode_attn_kernel`.** 256 threads = 4 waves, so each wave runs the online softmax for 2 of
   the G query heads where wave32 runs 1 (`for (g = warp; g < G; g += nw)`). The split merge
   uses the same loop. Tests: `test_decode_attn`, `test_decode_work_reuse`.
6. **`gemv_kernel` and `warp_dot` (dense decode and MoE expert GEMVs).** One wave streams one
   weight row with `HI_WARP * 4` 16-byte loads per step, so a step covers 2048 halves of K on
   wave64. That is correct for any K, but when K/8 is small most of the unrolled loads are masked:
   Granite-MoE's down projection (I = 512, K/8 = 64) gives each lane one useful load out of four.
   This is a performance gap, not a correctness one. Tests: `test_gemv`, `test_gemv_qkv_rope`, `test_moe`.

gfx1100 (RDNA3) is wave32 and uses the same lane mappings as NVIDIA. What's new there is only
rocWMMA's RDNA3 WMMA backend under `gemm_kernel` and `flash_prefill_kernel`.

## AMD-specific, but not about wave width

- **MI300 (gfx942) has 8 XCDs, each with its own L2.** The split-K GEMM and the split decode-attention
  merge hand partial results between blocks through `__threadfence()` (agent scope), an
  `atomicAdd` counter and agent-scope atomic loads. That is correct under HIP's memory model, but
  it is the first place a cross-XCD coherence problem would show up. Test with forced split-K
  (`test_gemm_splitk`) and long-context decode (`test_decode_attn`).
- **One-shot all-reduce.** It uses system-scope release/acquire flags (`__hip_atomic_*`),
  64-bit atomic loads for remote data, and `hipHostMallocCoherent` pinned memory when the GPUs
  have no peer path. It is untested on xGMI or PCIe AMD systems. Test: `tests/test_tp.py`.
- **Shared memory.** AMD gives a workgroup 64 KB of LDS. The largest portable GEMM tile (config 3,
  256×128×32) uses 60 KB of dynamic shared memory, and `flash_prefill_kernel` uses 44 KB (D = 64)
  or 39 KB (D = 128) of static shared memory, so everything fits.

## Register and spill report

From `results/amd_build/resource_usage_<target>.log` (hipcc `-Rpass-analysis=kernel-resource-usage`,
113 kernel instantiations per target). These numbers predict performance, not correctness, and
nothing here has been timed.

- **gfx90a, gfx942: no register spills and no scratch in any kernel.**
- **gfx1100: three kernels spill.** GEMM config 3 (`gemm_kernel<256, 128, 32, 4, 2>`, store and
  residual epilogues) uses 256 VGPRs and spills 156 (244 bytes of scratch per lane). Each wave holds
  a 64×64 output tile, 16 fp32 accumulator fragments, and RDNA3's wave32 WMMA layout needs twice
  the per-lane registers of CDNA's wave64 MFMA layout. `flash_prefill_kernel<128, 32>` spills 9
  VGPRs. The autotuner times config 3 against the others, so a spilling config should simply lose,
  but the untuned fallback (`gemm_pick`, used before tuning or with
  `HIPINFER_GEMM_AUTOTUNE=0`) picks config 3 for large prefills. On gfx1100 that choice is the first
  thing to re-measure.
- **Low occupancy on CDNA**, in waves per SIMD (8 is the most these block sizes allow):

  | kernel | gfx90a | gfx942 | gfx1100 | limited by |
  |---|---|---|---|---|
  | `flash_prefill_kernel` (both head dims) | 1 | 1 | 2–3 | 40–44 KB of LDS per 4-wave block |
  | `decode_attn_kernel<G, 128>`, G = 6–8 | 1 | 1 | 10 | 254–256 VGPRs (the G-wide score array, fully unrolled over D) |
  | `decode_attn_kernel<4, 128>` (Llama-3-8B shape) | 2 | 2 | 12 | 176 VGPRs |
  | `moe_gemv_down_kernel` | 2 | 2 | 9 | 176 VGPRs (4 dot products of 4 unrolled loads) |
  | `gemv_kernel<7–8, *, 2–3>` (batch 7–8, SwiGLU / QKV epilogues) | 3 | 2 | 16 | up to 168 (gfx90a) / 196 (gfx942) VGPRs |

  None of these spill. They are the first places to look if CDNA decode or prefill turns out slow.

## Reproducing the build

```bash
docker run --rm -v "$PWD":/src -w /src -e SRC_REV="$(git describe --always --dirty)" \
  rocm/dev-ubuntu-24.04:7.2.4 bash results/amd_build/build.sh
```

No GPU is needed. The script installs `rocwmma-dev` and `rccl-dev` into the container. It builds
each target on its own, then all three together (plain, with `RCCL=1`, and with `-Wall -Wextra`),
then writes the per-target resource reports, one log per build, and `toolchain.txt`.
