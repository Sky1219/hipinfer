# AMD build logs (compile only, no AMD GPU)

Built on 2026-10-18 on a Windows laptop with no AMD GPU, inside AMD's `rocm/dev-ubuntu-24.04:7.2.4`
Docker image: ROCm 7.2.4, HIP 7.2.53211, AMD clang 22.0.0git (roc-7.2.4), rocWMMA 2.2.0, RCCL
2.27.7. Source: commit `662ab55` (src/ and Makefile). Versions are in [`toolchain.txt`](toolchain.txt),
and the script is [`build.sh`](build.sh) (its header has the `docker run` line).

| log | command | result |
|---|---|---|
| [`gfx90a.log`](gfx90a.log) | `make hip AMD_ARCH=--offload-arch=gfx90a` | builds, 0 warnings |
| [`gfx942.log`](gfx942.log) | `make hip AMD_ARCH=--offload-arch=gfx942` | builds, 0 warnings |
| [`gfx1100.log`](gfx1100.log) | `make hip AMD_ARCH=--offload-arch=gfx1100` | builds, 0 warnings |
| [`all_targets.log`](all_targets.log) | `make hip` (all three targets in one library) | builds, 0 warnings |
| [`all_targets_rccl.log`](all_targets_rccl.log) | `make hip RCCL=1` | builds, 0 warnings |
| [`all_targets_wall.log`](all_targets_wall.log) | `make hip` with `-Wall -Wextra` | builds, 3 harmless warnings ×4 passes (below) |
| `resource_usage_<target>.log` | device-only compile with `-Rpass-analysis=kernel-resource-usage` | per-kernel VGPRs, spills, LDS, occupancy |
| [`nvidia_sm75.log`](nvidia_sm75.log) | `make NVCC=nvcc NV_ARCH="-gencode arch=compute_75,code=sm_75"`, nvcc 12.0 in WSL | builds (NVIDIA check on the same commit) |

Every library holds 113 kernels per target, including all six MoE kernels (`moe_router`, `moe_sort`,
`moe_gather`, `moe_gemv_gu`, `moe_gemv_down`, `moe_combine`). The log lists them after the build,
read from each target's device code object.

The `-Wall -Wextra` warnings: `kernels.cuh:1892` compares `threadIdx.x` (unsigned) with `world`
(an int from 1 to 8); `api.cu:262` doesn't use `gemm_pick`'s `epi` parameter; `api.cu:798` leaves
`ar_nccl` unused in builds without NCCL/RCCL. None affects behaviour.

What would behave differently on 64-lane wavefronts, and what to test first on real hardware:
[docs/amd-wave64.md](../../docs/amd-wave64.md).
