// All device code. fp16 storage, fp32 accumulation everywhere. Where HF/PyTorch rounds an
// intermediate to fp16 (RMSNorm output, RoPE products, projection outputs before the
// residual add, SiLU), the kernels round at the same point so greedy tokens match HF.
// head_dim (D) is a template parameter of the attention kernels: 64 and 128 are instantiated.
#pragma once
#include "hip_compat.h"
#include <cstdint>

typedef __half half;

union H8 { uint4 u; __half2 h2[4]; __half h[8]; };

#define HI_NEG_INF (-__int_as_float(0x7f800000))

__device__ __forceinline__ void hi_syncwarp() {
#if defined(HI_AMD)
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
#else
  __syncwarp();
#endif
}

// Block-wide sum; every thread gets the result. `sh` needs >= 32 floats.
__device__ __forceinline__ float block_sum(float v, float* sh) {
  const int w = threadIdx.x / HI_WARP, l = threadIdx.x % HI_WARP, nw = blockDim.x / HI_WARP;
  v = warp_sum(v);
  __syncthreads();
  if (l == 0) sh[w] = v;
  __syncthreads();
  if (w == 0) {
    v = (l < nw) ? sh[l] : 0.f;
    v = warp_sum(v);
    if (l == 0) sh[0] = v;
  }
  __syncthreads();
  return sh[0];
}

__device__ __forceinline__ float dot8(const uint4 a, const uint4 b) {
  H8 x, y; x.u = a; y.u = b;
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    float2 fa = __half22float2(x.h2[i]), fb = __half22float2(y.h2[i]);
    s = fmaf(fa.x, fb.x, s);
    s = fmaf(fa.y, fb.y, s);
  }
  return s;
}

__device__ __forceinline__ half silu_mul(float g, float u) {
  // HF: act_fn(gate_proj(x)) * up_proj(x), every step rounded to fp16
  float gh = __half2float(__float2half(g));
  half s = __float2half(gh / (1.f + __expf(-gh)));
  return __hmul(s, __float2half(u));
}

// ---------------------------------------------------------------------------------
// RMSNorm: y = w * fp16(x * rsqrt(mean(x^2) + eps)). One block per row.
// ---------------------------------------------------------------------------------
__global__ void rmsnorm_kernel(const half* __restrict__ x, const half* __restrict__ w,
                               half* __restrict__ y, int H, float eps) {
  __shared__ float sh[32];
  const uint4* xr = reinterpret_cast<const uint4*>(x + (size_t)blockIdx.x * H);
  const uint4* wr = reinterpret_cast<const uint4*>(w);
  uint4* yr = reinterpret_cast<uint4*>(y + (size_t)blockIdx.x * H);
  float ss = 0.f;
  for (int i = threadIdx.x; i < H / 8; i += blockDim.x) ss += dot8(xr[i], xr[i]);
  const float r = rsqrtf(block_sum(ss, sh) / H + eps);
  for (int i = threadIdx.x; i < H / 8; i += blockDim.x) {
    H8 xv, wv, o; xv.u = xr[i]; wv.u = wr[i];
#pragma unroll
    for (int j = 0; j < 8; j++) o.h[j] = __hmul(wv.h[j], __float2half(__half2float(xv.h[j]) * r));
    yr[i] = o.u;
  }
}

// ---------------------------------------------------------------------------------
// RoPE on one (x[d], x[d+D/2]) pair with PyTorch's per-op fp16 rounding. Products of two fp16
// values are exact in fp32; the explicit roundings stop the compiler contracting mul+add
// into an FMA (which __hmul/__hsub allow and which drifts by 1 ulp).
// ---------------------------------------------------------------------------------
__device__ __forceinline__ void rope_pair(half x1, half x2, __half2 t, half& y1, half& y2) {
  const float c = __low2float(t), sn = __high2float(t), f1 = __half2float(x1), f2 = __half2float(x2);
  y1 = __float2half(__half2float(__float2half(f1 * c)) - __half2float(__float2half(f2 * sn)));
  y2 = __float2half(__half2float(__float2half(f2 * c)) + __half2float(__float2half(f1 * sn)));
}

// Arguments of the fused QKV epilogue (EPI 3); unused by the other epilogues.
struct RopeArgs {
  const __half2* cs;  // [max_seq][D/2] (cos, sin)
  const int* pos;     // [M] position of each row; row m goes to cache slot m (or kv_slot)
  half *kc, *vc;      // this layer's cache [slots][Hkv][max_seq][D]
  const half* bias;   // optional QKV bias [(Hq + 2 Hkv) * D] (Qwen2), or null
  int max_seq, Hq, Hkv, D;
  int kv_slot = -1;   // >= 0: every row is a position of this one slot (speculative verification)
};

// ---------------------------------------------------------------------------------
// GEMV for decode (M <= 8 activation rows): out[m, n] = sum_k x[m, k] * W[n, k].
// Weight-bandwidth bound: each warp streams one weight row (two for the paired epilogues)
// with 16-byte loads, 4 loads in flight per lane, converts each weight vector to fp32 once
// and reuses it for all MR activation rows.
// NORM: RMSNorm of x is fused in (each block normalizes the MR rows into shared memory).
// EPI 0: store; 1: out += y (residual, fp16 rounding like HF); 2: SwiGLU over weights whose
// rows are interleaved in groups of 16 (16 gate rows, then the matching 16 up rows);
// 3: QKV projection with bias, RoPE and the KV-cache write fused in: a warp computes rows d
// and d+D/2 of one head, rotates the pair, and writes q to `out` and k/v straight into the cache.
// ---------------------------------------------------------------------------------
template <int MR, bool NORM, int EPI>
__global__ void __launch_bounds__(256)
gemv_kernel(const half* __restrict__ x, int ldx, const half* __restrict__ normw, float eps,
            const half* __restrict__ W, int N, int K, half* out, int ldo, RopeArgs ra) {
  extern __shared__ __align__(16) unsigned char gemv_smem[];
  constexpr int NR = (EPI >= 2) ? 2 : 1;
#ifndef GEMV_U
#define GEMV_U 4
#endif
  constexpr int U = (MR <= 2) ? GEMV_U : 4;  // 16-byte weight loads in flight per lane
  const int lane = threadIdx.x % HI_WARP, warp = threadIdx.x / HI_WARP, nw = blockDim.x / HI_WARP;
  const int K8 = K / 8;

  const uint4* xp[MR];
  if constexpr (NORM) {
    __shared__ float sh[32];
    uint4* xs = reinterpret_cast<uint4*>(gemv_smem);
    const uint4* wn = reinterpret_cast<const uint4*>(normw);
#pragma unroll
    for (int m = 0; m < MR; m++) {
      const uint4* xr = reinterpret_cast<const uint4*>(x + (size_t)m * ldx);
      float ss = 0.f;
      for (int i = threadIdx.x; i < K8; i += blockDim.x) ss += dot8(xr[i], xr[i]);
      const float r = rsqrtf(block_sum(ss, sh) / K + eps);
      for (int i = threadIdx.x; i < K8; i += blockDim.x) {
        H8 xv, wv, o; xv.u = xr[i]; wv.u = wn[i];
#pragma unroll
        for (int j = 0; j < 8; j++) o.h[j] = __hmul(wv.h[j], __float2half(__half2float(xv.h[j]) * r));
        xs[m * K8 + i] = o.u;
      }
      xp[m] = xs + m * K8;
    }
    __syncthreads();
  } else {
#pragma unroll
    for (int m = 0; m < MR; m++) xp[m] = reinterpret_cast<const uint4*>(x + (size_t)m * ldx);
  }

  const int n_out = (EPI >= 2) ? N / 2 : N;
  for (int o = blockIdx.x * nw + warp; o < n_out; o += gridDim.x * nw) {
    int r0 = o, r1 = o;
    if constexpr (EPI == 2) { r0 = (o / 16) * 32 + (o % 16); r1 = r0 + 16; }
    if constexpr (EPI == 3) { const int HD = ra.D >> 1; r0 = (o / HD) * ra.D + (o % HD); r1 = r0 + HD; }
    const uint4* wr[NR];
    wr[0] = reinterpret_cast<const uint4*>(W + (size_t)r0 * K);
    if constexpr (NR == 2) wr[NR - 1] = reinterpret_cast<const uint4*>(W + (size_t)r1 * K);
    float acc[NR][MR];
#pragma unroll
    for (int r = 0; r < NR; r++)
#pragma unroll
      for (int m = 0; m < MR; m++) acc[r][m] = 0.f;

    for (int base = 0; base < K8; base += HI_WARP * U) {
      uint4 wv[NR][U];
#pragma unroll
      for (int u = 0; u < U; u++) {
        const int idx = base + u * HI_WARP + lane;
#pragma unroll
        for (int r = 0; r < NR; r++) wv[r][u] = idx < K8 ? __ldg(wr[r] + idx) : make_uint4(0, 0, 0, 0);
      }
#pragma unroll
      for (int u = 0; u < U; u++) {
        const int idx = base + u * HI_WARP + lane;
        if (idx < K8) {
          float wf[NR][8];
#pragma unroll
          for (int r = 0; r < NR; r++) {
            H8 t; t.u = wv[r][u];
#pragma unroll
            for (int i = 0; i < 4; i++) { const float2 f = __half22float2(t.h2[i]); wf[r][2 * i] = f.x; wf[r][2 * i + 1] = f.y; }
          }
#pragma unroll
          for (int m = 0; m < MR; m++) {
            H8 xv; xv.u = xp[m][idx];
#pragma unroll
            for (int i = 0; i < 4; i++) {
              const float2 xf = __half22float2(xv.h2[i]);
#pragma unroll
              for (int r = 0; r < NR; r++) {
                acc[r][m] = fmaf(wf[r][2 * i], xf.x, acc[r][m]);
                acc[r][m] = fmaf(wf[r][2 * i + 1], xf.y, acc[r][m]);
              }
            }
          }
        }
      }
    }
#pragma unroll
    for (int r = 0; r < NR; r++)
#pragma unroll
      for (int m = 0; m < MR; m++) acc[r][m] = warp_sum(acc[r][m]);

    if (lane < MR) {
      // select acc[.][lane] without dynamic register indexing
      float a0 = 0.f, a1 = 0.f;
#pragma unroll
      for (int m = 0; m < MR; m++) if (m == lane) { a0 = acc[0][m]; a1 = acc[NR - 1][m]; }
      if constexpr (EPI == 0) out[(size_t)lane * ldo + o] = __float2half(a0);
      else if constexpr (EPI == 1) { half* dst = out + (size_t)lane * ldo + o; *dst = __hadd(*dst, __float2half(a0)); }
      else if constexpr (EPI == 2) out[(size_t)lane * ldo + o] = silu_mul(a0, a1);
      else {
        const int D = ra.D, HD = D >> 1;
        const int head = o / HD, d = o % HD, p = ra.pos[lane], slot = ra.kv_slot >= 0 ? ra.kv_slot : lane;
        if (ra.bias) { a0 += __half2float(ra.bias[r0]); a1 += __half2float(ra.bias[r1]); }
        half y1 = __float2half(a0), y2 = __float2half(a1);
        if (head < ra.Hq + ra.Hkv) rope_pair(y1, y2, ra.cs[(size_t)p * HD + d], y1, y2);
        half* dst;
        if (head < ra.Hq) dst = out + (size_t)lane * ldo + head * D;
        else if (head < ra.Hq + ra.Hkv) dst = ra.kc + (((size_t)slot * ra.Hkv + head - ra.Hq) * ra.max_seq + p) * D;
        else dst = ra.vc + (((size_t)slot * ra.Hkv + head - ra.Hq - ra.Hkv) * ra.max_seq + p) * D;
        dst[d] = y1;
        dst[d + HD] = y2;
      }
    }
  }
}

// Output-tile rasterization for the GEMMs: the 1-D tile index walks down GROUP_M tile rows
// before stepping right, so the ~100 blocks resident at any time cover a compact block of A
// rows and W rows and both stay in L2 (plain row-major order re-streams all of W from DRAM for
// every tile row, which on a 5.5 MB L2 with 10 MB weights is most of the kernel time).
__device__ __forceinline__ void tile_swizzle(int pid, int tiles_m, int tiles_n, int& tm, int& tn) {
  constexpr int GROUP_M = 8;
  const int per_group = GROUP_M * tiles_n;
  const int first_m = (pid / per_group) * GROUP_M;
  const int gsize = min(tiles_m - first_m, GROUP_M);
  const int r = pid % per_group;
  tm = first_m + r % gsize;
  tn = r / gsize;
}

// ---------------------------------------------------------------------------------
// Tensor-core GEMM for prefill: C[M, N'] = A[M, K] @ W[N, K]^T (+ epilogue as in GEMV; EPI 0
// adds `bias` when non-null). WMMA 16x16x16 fragments (rocWMMA on AMD, same API). Block tile
// BM x BN x BK, 8 warps in a WGM x WGN grid, double-buffered (dynamic) shared memory filled
// through registers (sm_75 has no cp.async), +8 half padding per row against bank conflicts.
// ---------------------------------------------------------------------------------
template <int BM, int BN, int BK, int WGM, int WGN, int EPI>
__global__ void __launch_bounds__(8 * HI_WARP)
gemm_kernel(const half* __restrict__ A, int lda, const half* __restrict__ W, const half* __restrict__ bias,
            half* C, int ldc, int M, int N, int K, float* ws, int* counters) {
  constexpr int LDS = BK + 8;
  constexpr int WTM = BM / WGM, WTN = BN / WGN, FM = WTM / 16, FN = WTN / 16;
  constexpr int NT = 8 * HI_WARP;
  constexpr int A_CH = BM * BK / 8, B_CH = BN * BK / 8;
  constexpr int A_PT = (A_CH + NT - 1) / NT, B_PT = (B_CH + NT - 1) / NT;
  constexpr int SMEM_HALVES = 2 * (BM + BN) * LDS;
  static_assert(WGM * WGN == 8, "8 warps");
  static_assert(EPI != 2 || WTN == 32, "SwiGLU epilogue pairs one gate and one up fragment");
  static_assert(SMEM_HALVES * 2 >= 8 * 2 * 256 * 4, "epilogue scratch fits");

  extern __shared__ __align__(32) unsigned char gemm_smem_raw[];
  half* smem = reinterpret_cast<half*>(gemm_smem_raw);
  half* As = smem;                    // [2][BM][LDS]
  half* Bs = smem + 2 * BM * LDS;     // [2][BN][LDS]

  const int tid = threadIdx.x, warp = tid / HI_WARP, lane = tid % HI_WARP;
  const int wm = warp / WGN, wn = warp % WGN;
  const int tiles_m = (M + BM - 1) / BM, tiles_n = (N + BN - 1) / BN;  // gridDim.x == tiles_m * tiles_n
  int tm, tn;
  tile_swizzle(blockIdx.x, tiles_m, tiles_n, tm, tn);
  const int bm = tm * BM, bn = tn * BN;

  uint4 ra[A_PT], rb[B_PT];
  auto gload = [&](int k0) {
#pragma unroll
    for (int i = 0; i < A_PT; i++) {
      const int c = tid + i * NT;
      if (c < A_CH) {
        const int r = c / (BK / 8), kc = (c % (BK / 8)) * 8, g = bm + r;
        ra[i] = g < M ? *reinterpret_cast<const uint4*>(A + (size_t)g * lda + k0 + kc) : make_uint4(0, 0, 0, 0);
      }
    }
#pragma unroll
    for (int i = 0; i < B_PT; i++) {
      const int c = tid + i * NT;
      if (c < B_CH) {
        const int r = c / (BK / 8), kc = (c % (BK / 8)) * 8, g = bn + r;
        rb[i] = g < N ? __ldg(reinterpret_cast<const uint4*>(W + (size_t)g * K + k0 + kc)) : make_uint4(0, 0, 0, 0);
      }
    }
  };
  auto sstore = [&](int st) {
#pragma unroll
    for (int i = 0; i < A_PT; i++) {
      const int c = tid + i * NT;
      if (c < A_CH) *reinterpret_cast<uint4*>(As + st * BM * LDS + (c / (BK / 8)) * LDS + (c % (BK / 8)) * 8) = ra[i];
    }
#pragma unroll
    for (int i = 0; i < B_PT; i++) {
      const int c = tid + i * NT;
      if (c < B_CH) *reinterpret_cast<uint4*>(Bs + st * BN * LDS + (c / (BK / 8)) * LDS + (c % (BK / 8)) * 8) = rb[i];
    }
  };

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[FM][FN];
#pragma unroll
  for (int i = 0; i < FM; i++)
#pragma unroll
    for (int j = 0; j < FN; j++) wmma::fill_fragment(acc[i][j], 0.f);

  // split-K: gridDim.z blocks share an output tile, each taking K / gridDim.z
  const int KT = K / BK / gridDim.z, kz = blockIdx.z * KT * BK;
  gload(kz);
  sstore(0);
  __syncthreads();
  for (int kt = 0; kt < KT; kt++) {
    const int st = kt & 1;
    if (kt + 1 < KT) gload(kz + (kt + 1) * BK);
#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[FM];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b[FN];
#pragma unroll
      for (int i = 0; i < FM; i++)
        wmma::load_matrix_sync(a[i], As + st * BM * LDS + (wm * WTM + i * 16) * LDS + kk, LDS);
#pragma unroll
      for (int j = 0; j < FN; j++)
        wmma::load_matrix_sync(b[j], Bs + st * BN * LDS + (wn * WTN + j * 16) * LDS + kk, LDS);
#pragma unroll
      for (int i = 0; i < FM; i++)
#pragma unroll
        for (int j = 0; j < FN; j++) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
    }
    if (kt + 1 < KT) sstore(st ^ 1);
    __syncthreads();
  }

  if (gridDim.z > 1) {
    // Split-K: fp32 partial tiles go to ws; the last of the gridDim.z blocks of this output
    // tile (atomic counter, reset for the next launch / graph replay) sums them and applies
    // the epilogue, so no separate reduction launch is needed.
    const size_t plane = (size_t)tiles_m * BM * N;
    float* part = ws + blockIdx.z * plane;
#pragma unroll
    for (int i = 0; i < FM; i++)
#pragma unroll
      for (int j = 0; j < FN; j++) {
        const int col0 = bn + wn * WTN + j * 16;
        if (col0 < N)
          wmma::store_matrix_sync(part + (size_t)(bm + wm * WTM + i * 16) * N + col0, acc[i][j], N, wmma::mem_row_major);
      }
    __threadfence();
    __syncthreads();
    __shared__ int last_s;
    const int t = blockIdx.x;
    if (tid == 0) {
      last_s = atomicAdd(&counters[t], 1) == (int)gridDim.z - 1;
      if (last_s) counters[t] = 0;
    }
    __syncthreads();
    if (!last_s) return;
    __threadfence();
    auto total = [&](int r, int c) {
      float v = 0.f;
      for (int z = 0; z < (int)gridDim.z; z++) v += hi_ldcg(ws + z * plane + (size_t)r * N + c);
      return v;
    };
    if constexpr (EPI == 2) {
      for (int e = tid; e < BM * BN / 2; e += NT) {
        const int r = bm + e / (BN / 2), o = e % (BN / 2), c = bn + (o / 16) * 32 + o % 16;
        if (r < M && c < N) C[(size_t)r * ldc + bn / 2 + o] = silu_mul(total(r, c), total(r, c + 16));
      }
    } else {
      for (int e = tid; e < BM * BN; e += NT) {
        const int r = bm + e / BN, c = bn + e % BN;
        if (r < M && c < N) {
          half* dst = C + (size_t)r * ldc + c;
          if constexpr (EPI == 1) *dst = __hadd(*dst, __float2half(total(r, c)));
          else *dst = __float2half(total(r, c) + (bias ? __half2float(bias[c]) : 0.f));
        }
      }
    }
    return;
  }

  // Epilogue: stage each 16x16 fragment (two for SwiGLU) through per-warp scratch, then
  // write 8 halves per lane with 16-byte stores.
  float* scr = reinterpret_cast<float*>(smem) + warp * 512;
#pragma unroll
  for (int i = 0; i < FM; i++) {
    const int row0 = bm + wm * WTM + i * 16;
    if constexpr (EPI == 2) {
      wmma::store_matrix_sync(scr, acc[i][0], 16, wmma::mem_row_major);
      wmma::store_matrix_sync(scr + 256, acc[i][1], 16, wmma::mem_row_major);
      hi_syncwarp();
      const int col0 = (bn + wn * WTN) / 2;
      const bool col_ok = bn + wn * WTN < N;  // last tile: warps past N hold zero-padded W rows
      for (int ch = lane; ch < 32; ch += HI_WARP) {
        const int r = ch / 2, c8 = (ch % 2) * 8;
        if (row0 + r < M && col_ok) {
          H8 o;
#pragma unroll
          for (int e = 0; e < 8; e++) o.h[e] = silu_mul(scr[r * 16 + c8 + e], scr[256 + r * 16 + c8 + e]);
          *reinterpret_cast<uint4*>(C + (size_t)(row0 + r) * ldc + col0 + c8) = o.u;
        }
      }
      hi_syncwarp();
    } else {
#pragma unroll
      for (int j = 0; j < FN; j++) {
        const int col0 = bn + wn * WTN + j * 16;
        wmma::store_matrix_sync(scr, acc[i][j], 16, wmma::mem_row_major);
        hi_syncwarp();
        for (int ch = lane; ch < 32; ch += HI_WARP) {
          const int r = ch / 2, c8 = (ch % 2) * 8;
          if (row0 + r < M && col0 < N) {
            uint4* dst = reinterpret_cast<uint4*>(C + (size_t)(row0 + r) * ldc + col0 + c8);
            H8 o;
            if constexpr (EPI == 1) {
              H8 old; old.u = *dst;
#pragma unroll
              for (int e = 0; e < 8; e++) o.h[e] = __hadd(old.h[e], __float2half(scr[r * 16 + c8 + e]));
            } else if (bias) {
              H8 bv; bv.u = *reinterpret_cast<const uint4*>(bias + col0 + c8);
#pragma unroll
              for (int e = 0; e < 8; e++) o.h[e] = __float2half(scr[r * 16 + c8 + e] + __half2float(bv.h[e]));
            } else {
#pragma unroll
              for (int e = 0; e < 8; e++) o.h[e] = __float2half(scr[r * 16 + c8 + e]);
            }
            *dst = o.u;
          }
        }
        hi_syncwarp();
      }
    }
  }
}

#if !defined(HI_AMD)
// ---------------------------------------------------------------------------------
// NVIDIA tensor-core primitives shared by the mma.sync GEMM and attention kernels: m16n8k16
// fp16 MMA with fp32 accumulate (Turing runs it as two m16n8k8), ldmatrix loads of 8x8 b16
// tiles from shared memory, and cp.async 16-byte global->shared copies (Ampere+; the sm_75
// build compiles them to nothing and never launches the pipelined variants).
// ---------------------------------------------------------------------------------
// Not volatile: the asm has register outputs, so it is never dropped, and the compiler may
// schedule independent MMAs around each other (Turing needs this: the two k8 halves of one
// tile are dependent and would otherwise issue back to back).
__device__ __forceinline__ void mma1688(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
  asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(b0));
}
__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if __CUDA_ARCH__ >= 800
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
  mma1688(d, a[0], a[1], b0);
  mma1688(d, a[2], a[3], b1);
#endif
}

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], uint32_t a) {  // a: shared-window address
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const half* p) {
  ldsm_x4(r, (uint32_t)__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void ldsm_x4_trans(uint32_t (&r)[4], const half* p) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ uint32_t pack_half2(float a, float b) {
  __half2 h = __floats2half2_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool pred) {
#if __CUDA_ARCH__ >= 800
  const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
  const int n = pred ? 16 : 0;  // src-size 0 -> zero fill
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s), "l"(gmem), "r"(n));
#endif
}
__device__ __forceinline__ void cp_async_commit() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n");
#endif
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}
#endif  // !HI_AMD

#if !defined(HI_AMD)
// ---------------------------------------------------------------------------------
// NVIDIA tensor-core GEMM on raw mma.sync: C[M, N'] = A[M, K] @ W[N, K]^T (+ the same epilogues
// as gemm_kernel). Both operands are K-contiguous, which is exactly the m16n8k16 "row.col"
// operand layout, so A and W tiles go shared -> registers with plain ldmatrix.x4 (no
// transposes, no layout shuffles) and the accumulator layout is known, so the epilogue writes
// straight from registers. Shared rows are padded by 8 halves, which makes every ldmatrix
// bank-conflict free. STAGES >= 3: cp.async multi-stage pipeline (Ampere+; global->shared
// copies are asynchronous and 2 tiles are in flight while one is computed). STAGES == 2:
// register-staged double buffering for Turing, which has no cp.async.
// ---------------------------------------------------------------------------------
template <int BM, int BN, int BK, int WGM, int WGN, int STAGES, int EPI>
__global__ void __launch_bounds__(WGM * WGN * 32)
gemm_mma_kernel(const half* __restrict__ A, int lda, const half* __restrict__ W, const half* __restrict__ bias,
                half* C, int ldc, int M, int N, int K, float* ws, int* counters) {
  constexpr int LDS = BK + 8, NT = WGM * WGN * 32;
  constexpr int WTM = BM / WGM, WTN = BN / WGN, FM = WTM / 16, FN = WTN / 8;
  constexpr int A_CH = BM * BK / 8, B_CH = BN * BK / 8;  // 16-byte chunks per tile
  constexpr int A_PT = (A_CH + NT - 1) / NT, B_PT = (B_CH + NT - 1) / NT;
  static_assert(WGM * WGN == 4 || WGM * WGN == 8, "4 or 8 warps");
  static_assert(FN % 2 == 0, "ldmatrix.x4 loads two n8 tiles");
  static_assert(EPI != 2 || WTN == 32, "SwiGLU epilogue pairs 16 gate with 16 up columns per warp");
  static_assert(STAGES >= 2, "pipeline depth");

  extern __shared__ __align__(128) unsigned char gemm_mma_smem_raw[];
  half* As = reinterpret_cast<half*>(gemm_mma_smem_raw);  // [STAGES][BM][LDS]
  half* Bs = As + STAGES * BM * LDS;                  // [STAGES][BN][LDS]

  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, g = lane / 4, t = lane % 4;
  const int wm = warp / WGN, wn = warp % WGN;
  const int tiles_m = (M + BM - 1) / BM, tiles_n = (N + BN - 1) / BN;  // gridDim.x == tiles_m * tiles_n
  int tm, tn;
  tile_swizzle(blockIdx.x, tiles_m, tiles_n, tm, tn);
  const int bm = tm * BM, bn = tn * BN;
  const int KT = K / BK / gridDim.z, kz = blockIdx.z * KT * BK;

  float acc[FM][FN][4];
#pragma unroll
  for (int i = 0; i < FM; i++)
#pragma unroll
    for (int j = 0; j < FN; j++) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

  // One k16 step of fragments; two buffers so the ldmatrix for step kk+16 is in flight while
  // step kk's MMAs run.
  uint32_t af[2][FM][4], bf[2][FN][2];
  // The MMAs of one k16 step on fragment buffer cur.
  auto mma_step = [&](int cur) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    // Turing: issue every tile's first k8 half before any second half, so no MMA waits on the
    // one just before it
#pragma unroll
    for (int i = 0; i < FM; i++)
#pragma unroll
      for (int j = 0; j < FN; j++) mma1688(acc[i][j], af[cur][i][0], af[cur][i][1], bf[cur][j][0]);
#pragma unroll
    for (int i = 0; i < FM; i++)
#pragma unroll
      for (int j = 0; j < FN; j++) mma1688(acc[i][j], af[cur][i][2], af[cur][i][3], bf[cur][j][1]);
#else
#pragma unroll
    for (int i = 0; i < FM; i++)
#pragma unroll
      for (int j = 0; j < FN; j++) mma16816(acc[i][j], af[cur][i], bf[cur][j][0], bf[cur][j][1]);
#endif
  };

  if constexpr (STAGES >= 3) {
    // One k16 step of fragments from this warp's rows at as / bs into fragment buffer buf.
    auto ldfrag = [&](const half* as, const half* bs, int kk, int buf) {
#pragma unroll
      for (int i = 0; i < FM; i++) ldsm_x4(af[buf][i], as + (i * 16 + (lane & 15)) * LDS + kk + (lane >> 4) * 8);
#pragma unroll
      for (int j = 0; j < FN / 2; j++) {
        uint32_t r[4];
        ldsm_x4(r, bs + (j * 16 + (lane & 7) + ((lane >> 4) & 1) * 8) * LDS + kk + ((lane >> 3) & 1) * 8);
        bf[buf][2 * j][0] = r[0]; bf[buf][2 * j][1] = r[1]; bf[buf][2 * j + 1][0] = r[2]; bf[buf][2 * j + 1][1] = r[3];
      }
    };
    // This warp's rows of shared stage st.
    auto a_st = [&](int st) { return As + st * BM * LDS + (wm * WTM) * LDS; };
    auto b_st = [&](int st) { return Bs + st * BN * LDS + (wn * WTN) * LDS; };
    // One whole K tile from shared stage st.
    auto compute = [&](int st) {
      const half* as = a_st(st);
      const half* bs = b_st(st);
      ldfrag(as, bs, 0, 0);
#pragma unroll
      for (int kk = 0; kk < BK; kk += 16) {
        const int cur = (kk / 16) & 1;
        if (kk + 16 < BK) ldfrag(as, bs, kk + 16, cur ^ 1);
        mma_step(cur);
      }
    };
    // cp.async pipeline: STAGES-1 tiles in flight ahead of the one being computed
    auto issue = [&](int st, int k0) {
#pragma unroll
      for (int i = 0; i < A_PT; i++) {
        const int c = tid + i * NT;
        if (c < A_CH) {
          const int r = c / (BK / 8), kc = (c % (BK / 8)) * 8, gr = bm + r;
          const bool ok = gr < M;
          cp_async16(As + st * BM * LDS + r * LDS + kc, ok ? A + (size_t)gr * lda + k0 + kc : A, ok);
        }
      }
#pragma unroll
      for (int i = 0; i < B_PT; i++) {
        const int c = tid + i * NT;
        if (c < B_CH) {
          const int r = c / (BK / 8), kc = (c % (BK / 8)) * 8, gr = bn + r;
          const bool ok = gr < N;
          cp_async16(Bs + st * BN * LDS + r * LDS + kc, ok ? W + (size_t)gr * K + k0 + kc : W, ok);
        }
      }
    };
#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
      if (s < KT) issue(s, kz + s * BK);
      cp_async_commit();
    }
    for (int kt = 0; kt < KT; kt++) {
      cp_async_wait<STAGES - 2>();  // tile kt has landed (this thread's part)
      __syncthreads();              // ...everyone's part; and everyone is done with the buffer we refill next
      const int nk = kt + STAGES - 1;
      if (nk < KT) issue(nk % STAGES, kz + nk * BK);
      cp_async_commit();
      compute(kt % STAGES);
    }
  } else {
    // Turing (no cp.async): register-staged double buffering, one __syncthreads per K tile.
    // Tile kt+1 goes registers -> shared just before the LAST k16 step of tile kt, so the store
    // and barrier overlap that step's MMAs. Global loads are split so they interleave with MMAs
    // instead of issuing in one burst: A of tile kt+2 right after tile kt's barrier, W of tile
    // kt+1 at k16 step 0 of tile kt.
    uint4 ra[A_PT], rb[B_PT];
    // One 64-bit base pointer per operand plus 32-bit row offsets (chunk i is NT / (BK / 8)
    // rows below chunk 0, and i * that < BM <= 256 rows), not one 64-bit pointer per chunk.
    constexpr int RPP = NT / (BK / 8);  // rows per pass of the block over a tile
    static_assert(NT % (BK / 8) == 0, "whole rows per pass");
    const int r0 = tid / (BK / 8), kc0 = (tid % (BK / 8)) * 8;
    const half* a0 = A + (size_t)(bm + r0) * lda + kc0;
    const half* w0 = W + (size_t)(bn + r0) * K + kc0;
    const int sa = RPP * lda, sw = RPP * K;
    auto gload_a = [&](int k0) {
#pragma unroll
      for (int i = 0; i < A_PT; i++) {
        if (tid + i * NT < A_CH)
          ra[i] = bm + r0 + i * RPP < M ? *reinterpret_cast<const uint4*>(a0 + i * sa + k0) : make_uint4(0, 0, 0, 0);
      }
    };
    auto gload_b = [&](int k0) {
#pragma unroll
      for (int i = 0; i < B_PT; i++) {
        if (tid + i * NT < B_CH)
          rb[i] = bn + r0 + i * RPP < N ? __ldg(reinterpret_cast<const uint4*>(w0 + i * sw + k0)) : make_uint4(0, 0, 0, 0);
      }
    };
    // ldmatrix through 32-bit shared-window addresses (one register each, where generic pointers
    // take two): this lane's rows in stage 0.
    const uint32_t sa0 = (uint32_t)__cvta_generic_to_shared(As), sb0 = (uint32_t)__cvta_generic_to_shared(Bs);
    const uint32_t fa = sa0 + ((wm * WTM + (lane & 15)) * LDS + (lane >> 4) * 8) * 2;
    const uint32_t fb = sb0 + ((wn * WTN + (lane & 7) + ((lane >> 4) & 1) * 8) * LDS + ((lane >> 3) & 1) * 8) * 2;
    auto sstore = [&](int st) {
#pragma unroll
      for (int i = 0; i < A_PT; i++)
        if (tid + i * NT < A_CH) {
          *reinterpret_cast<uint4*>(As + st * BM * LDS + (r0 + i * RPP) * LDS + kc0) = ra[i];
        }
#pragma unroll
      for (int i = 0; i < B_PT; i++)
        if (tid + i * NT < B_CH) {
          *reinterpret_cast<uint4*>(Bs + st * BN * LDS + (r0 + i * RPP) * LDS + kc0) = rb[i];
        }
    };
    // ldfrag on shared-window addresses: the k16 step at kk of stage st into buffer buf
    auto ldfrag_s = [&](int st, int kk, int buf) {
#pragma unroll
      for (int i = 0; i < FM; i++) ldsm_x4(af[buf][i], fa + st * (BM * LDS * 2) + (i * 16 * LDS + kk) * 2);
#pragma unroll
      for (int j = 0; j < FN / 2; j++) {
        uint32_t r[4];
        ldsm_x4(r, fb + st * (BN * LDS * 2) + (j * 16 * LDS + kk) * 2);
        bf[buf][2 * j][0] = r[0]; bf[buf][2 * j][1] = r[1]; bf[buf][2 * j + 1][0] = r[2]; bf[buf][2 * j + 1][1] = r[3];
      }
    };
    constexpr int NS = BK / 16;  // k16 steps per tile (even: every tile starts on fragment buffer 0)
    static_assert(NS % 2 == 0, "BK multiple of 32");
    // Loads past the last tile re-read tile KT-1 (in bounds, never stored where it matters), so
    // the loop body has no branches around its loads, stores or barrier.
    auto ktile = [&](int t) { return kz + min(t, KT - 1) * BK; };
    gload_a(kz);
    gload_b(kz);
    sstore(0);
    __syncthreads();
    gload_a(ktile(1));
    ldfrag_s(0, 0, 0);
    for (int kt = 0; kt < KT; kt++) {
      const int st = kt & 1;
#pragma unroll
      for (int s = 0; s < NS; s++) {
        const int cur = s & 1;
        if (s + 1 < NS) {
          ldfrag_s(st, (s + 1) * 16, cur ^ 1);
          if (s == 0) gload_b(ktile(kt + 1));  // W of tile kt+1: lands during this tile's MMAs
        } else {
          sstore(st ^ 1);  // tile kt+1 (a redundant copy of tile KT-1 on the last tile)
          __syncthreads();
          ldfrag_s(st ^ 1, 0, cur ^ 1);
          gload_a(ktile(kt + 2));  // A of tile kt+2: a whole tile of MMAs to land
        }
        mma_step(cur);
      }
    }
  }

  // ---- epilogue. Accumulator layout: acc[i][j] = rows (g, g+8) of m-tile i, cols (2t, 2t+1) of n-tile j.
  const int row_base = bm + wm * WTM, col_base = bn + wn * WTN;
  if (gridDim.z > 1) {
    // split-K: fp32 partials to ws, the last block of this tile reduces and applies the epilogue
    const size_t plane = (size_t)tiles_m * BM * N;
    float* part = ws + blockIdx.z * plane;
#pragma unroll
    for (int i = 0; i < FM; i++)
#pragma unroll
      for (int j = 0; j < FN; j++) {
        const int col = col_base + j * 8 + 2 * t;
        if (col < N) {
          const int r0 = row_base + i * 16 + g;
          *reinterpret_cast<float2*>(part + (size_t)r0 * N + col) = make_float2(acc[i][j][0], acc[i][j][1]);
          *reinterpret_cast<float2*>(part + (size_t)(r0 + 8) * N + col) = make_float2(acc[i][j][2], acc[i][j][3]);
        }
      }
    __threadfence();
    __syncthreads();
    __shared__ int last_s;
    const int tile = blockIdx.x;
    if (tid == 0) {
      last_s = atomicAdd(&counters[tile], 1) == (int)gridDim.z - 1;
      if (last_s) counters[tile] = 0;
    }
    __syncthreads();
    if (!last_s) return;
    __threadfence();
    auto total = [&](int r, int c) {
      float v = 0.f;
      for (int z = 0; z < (int)gridDim.z; z++) v += hi_ldcg(ws + z * plane + (size_t)r * N + c);
      return v;
    };
    if constexpr (EPI == 2) {
      for (int e = tid; e < BM * BN / 2; e += NT) {
        const int r = bm + e / (BN / 2), o = e % (BN / 2), c = bn + (o / 16) * 32 + o % 16;
        if (r < M && c < N) C[(size_t)r * ldc + bn / 2 + o] = silu_mul(total(r, c), total(r, c + 16));
      }
    } else {
      for (int e = tid; e < BM * BN; e += NT) {
        const int r = bm + e / BN, c = bn + e % BN;
        if (r < M && c < N) {
          half* dst = C + (size_t)r * ldc + c;
          if constexpr (EPI == 1) *dst = __hadd(*dst, __float2half(total(r, c)));
          else *dst = __float2half(total(r, c) + (bias ? __half2float(bias[c]) : 0.f));
        }
      }
    }
    return;
  }

  if constexpr (EPI == 2) {
    // n-tiles 0,1 are the 16 gate columns of this warp's group, 2,3 the matching up columns
    const int ocol = col_base / 2 + 2 * t;
#pragma unroll
    for (int i = 0; i < FM; i++)
#pragma unroll
      for (int j = 0; j < 2; j++) {
        const int r0 = row_base + i * 16 + g;
        __half2 lo = __halves2half2(silu_mul(acc[i][j][0], acc[i][j + 2][0]), silu_mul(acc[i][j][1], acc[i][j + 2][1]));
        __half2 hi = __halves2half2(silu_mul(acc[i][j][2], acc[i][j + 2][2]), silu_mul(acc[i][j][3], acc[i][j + 2][3]));
        // last tile: warps past N hold zero-padded W rows
        if (r0 < M && col_base < N) *reinterpret_cast<__half2*>(C + (size_t)r0 * ldc + ocol + j * 8) = lo;
        if (r0 + 8 < M && col_base < N) *reinterpret_cast<__half2*>(C + (size_t)(r0 + 8) * ldc + ocol + j * 8) = hi;
      }
  } else {
#pragma unroll
    for (int i = 0; i < FM; i++)
#pragma unroll
      for (int j = 0; j < FN; j++) {
        const int col = col_base + j * 8 + 2 * t;
        if (col >= N) continue;
        const int r0 = row_base + i * 16 + g;
        float v0 = acc[i][j][0], v1 = acc[i][j][1], v2 = acc[i][j][2], v3 = acc[i][j][3];
        if constexpr (EPI == 0) {
          if (bias) {
            const __half2 b = *reinterpret_cast<const __half2*>(bias + col);
            const float bx = __low2float(b), by = __high2float(b);
            v0 += bx; v1 += by; v2 += bx; v3 += by;
          }
          if (r0 < M) *reinterpret_cast<__half2*>(C + (size_t)r0 * ldc + col) = __floats2half2_rn(v0, v1);
          if (r0 + 8 < M) *reinterpret_cast<__half2*>(C + (size_t)(r0 + 8) * ldc + col) = __floats2half2_rn(v2, v3);
        } else {  // residual: out = fp16(out + fp16(acc)), like HF's fp16 residual add
          if (r0 < M) {
            __half2* d = reinterpret_cast<__half2*>(C + (size_t)r0 * ldc + col);
            *d = __hadd2(*d, __floats2half2_rn(v0, v1));
          }
          if (r0 + 8 < M) {
            __half2* d = reinterpret_cast<__half2*>(C + (size_t)(r0 + 8) * ldc + col);
            *d = __hadd2(*d, __floats2half2_rn(v2, v3));
          }
        }
      }
  }
}
#endif  // !HI_AMD

#if !defined(HI_AMD)
// ---------------------------------------------------------------------------------
// Weight-streaming tensor-core GEMM for small M (batched decode, M <= 64). The projection is
// bandwidth-bound, so the kernel is organized like the GEMV: each warp owns NW weight rows and
// streams them from global memory once, straight into mma.sync B fragments (register
// double-buffered, a 128-wide k chunk in flight), with no shared-memory staging of the weights.
// A lane's 16-byte load holds 8 consecutive k of one row; the B fragment wants two k-pairs per
// lane, so the k index inside every 32-wide window is permuted (lane t's halves 0-3 are the
// virtual k16 window 0, halves 4-7 window 1). The A operand gets the same permutation, which
// makes its fragments plain 16-byte loads too: the block stages one 128-wide k chunk of A
// (M x 128) in shared memory, double-buffered, shared by the 4 warps of a k-slice. KSPLIT
// k-slices per block (4*KSPLIT warps) put more loads in flight per SM when N alone gives too
// few warps; they are summed through shared memory at the end. Split-K across blocks
// (gridDim.z) with the usual last-block reduction is also supported.
// ---------------------------------------------------------------------------------
template <int MT, int NW, int KSPLIT, int EPI>
__global__ void __launch_bounds__(128 * KSPLIT)
gemm_skinny_kernel(const half* __restrict__ A, int lda, const half* __restrict__ W, const half* __restrict__ bias,
                   half* C, int ldc, int M, int N, int K, float* ws, int* counters) {
  constexpr int NTL = NW / 8;                  // n8 tiles per warp
  constexpr int CK = 128, NWIN = CK / 32;      // k chunk per pipeline step, 32-wide windows in it
  constexpr int LDA = CK + 32;                 // row stride: 64 B mod 128 -> conflict-free 16 B row reads
  constexpr int BM = 16 * MT, BN = 4 * NW;
  constexpr int WB = NTL <= 2 ? 2 : 1;         // W register buffers (NW=32 has 4 tiles: single)
  constexpr int A_PT = BM * CK / 8 / 128;      // 16-byte A chunks per thread per stage (per k-slice)
  constexpr int ACC = MT * NTL * 4;
  static_assert(EPI != 2 || NW == 32, "SwiGLU: 16 gate + 16 up rows per warp");
  static_assert(KSPLIT == 1 || (KSPLIT - 1) * 4 * 32 * ACC * 4 <= KSPLIT * 2 * BM * LDA * 2, "k-slice reduction fits in As");
  __shared__ __align__(16) half As[KSPLIT][2][BM * LDA];

  const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32, g = lane / 4, t = lane % 4;
  const int ks = warp / 4, wg = warp % 4, stid = tid % 128;  // k-slice, row group, thread within the slice
  const int grp = blockIdx.x * 4 + wg, r0 = grp * NW;
  const bool active = r0 < N;
  const int KS = K / gridDim.z / KSPLIT, k_begin = (blockIdx.z * KSPLIT + ks) * KS, nchunks = KS / CK;
  half* myA = &As[ks][0][0];

  const half* wrow[NTL];
#pragma unroll
  for (int nt = 0; nt < NTL; nt++) wrow[nt] = W + (size_t)(active ? r0 + nt * 8 + g : 0) * K + 8 * t;
  uint4 ra[A_PT], wv[WB][NWIN][NTL];
  auto aload = [&](int c) {
#pragma unroll
    for (int i = 0; i < A_PT; i++) {
      const int idx = stid + i * 128, row = idx / (CK / 8), kc = (idx % (CK / 8)) * 8;
      ra[i] = row < M ? *reinterpret_cast<const uint4*>(A + (size_t)row * lda + k_begin + c * CK + kc) : make_uint4(0, 0, 0, 0);
    }
  };
  auto astore = [&](int buf) {
#pragma unroll
    for (int i = 0; i < A_PT; i++) {
      const int idx = stid + i * 128, row = idx / (CK / 8), kc = (idx % (CK / 8)) * 8;
      *reinterpret_cast<uint4*>(myA + buf * BM * LDA + row * LDA + kc) = ra[i];
    }
  };
  auto wload = [&](int buf, int c) {
#pragma unroll
    for (int w = 0; w < NWIN; w++)
#pragma unroll
      for (int nt = 0; nt < NTL; nt++)
        wv[buf][w][nt] = __ldg(reinterpret_cast<const uint4*>(wrow[nt] + k_begin + c * CK + 32 * w));
  };

  float acc[MT][NTL][4];
#pragma unroll
  for (int m = 0; m < MT; m++)
#pragma unroll
    for (int nt = 0; nt < NTL; nt++) acc[m][nt][0] = acc[m][nt][1] = acc[m][nt][2] = acc[m][nt][3] = 0.f;

  aload(0);
  astore(0);
  if (active) wload(0, 0);
  __syncthreads();
  for (int c = 0; c < nchunks; c++) {
    const int cur = c & 1;
    const bool more = c + 1 < nchunks;
    if (more) {
      aload(c + 1);
      if (WB == 2 && active) wload(cur ^ 1, c + 1);
    }
    if (active) {
      const int wb = WB == 2 ? cur : 0;
      const half* a_cur = myA + cur * BM * LDA;
#pragma unroll
      for (int w = 0; w < NWIN; w++) {
#pragma unroll
        for (int m = 0; m < MT; m++) {
          const uint4 lo = *reinterpret_cast<const uint4*>(a_cur + (16 * m + g) * LDA + 32 * w + 8 * t);
          const uint4 hi = *reinterpret_cast<const uint4*>(a_cur + (16 * m + 8 + g) * LDA + 32 * w + 8 * t);
          const uint32_t a0[4] = {lo.x, hi.x, lo.y, hi.y};
          const uint32_t a1[4] = {lo.z, hi.z, lo.w, hi.w};
#pragma unroll
          for (int nt = 0; nt < NTL; nt++) {
            mma16816(acc[m][nt], a0, wv[wb][w][nt].x, wv[wb][w][nt].y);
            mma16816(acc[m][nt], a1, wv[wb][w][nt].z, wv[wb][w][nt].w);
          }
        }
      }
    }
    if (more) astore(cur ^ 1);
    __syncthreads();
    if (WB == 1 && more && active) wload(0, c + 1);
  }

  if constexpr (KSPLIT > 1) {
    // sum the k-slices: slices 1.. park their accumulators in As, slice 0 adds them
    float* red = reinterpret_cast<float*>(&As[0][0][0]);
    if (ks > 0) {
      float* dst = red + (((ks - 1) * 4 + wg) * 32 + lane) * ACC;
#pragma unroll
      for (int m = 0; m < MT; m++)
#pragma unroll
        for (int nt = 0; nt < NTL; nt++)
#pragma unroll
          for (int e = 0; e < 4; e++) dst[(m * NTL + nt) * 4 + e] = acc[m][nt][e];
    }
    __syncthreads();
    if (ks > 0) return;
#pragma unroll
    for (int s = 1; s < KSPLIT; s++) {
      const float* src = red + (((s - 1) * 4 + wg) * 32 + lane) * ACC;
#pragma unroll
      for (int m = 0; m < MT; m++)
#pragma unroll
        for (int nt = 0; nt < NTL; nt++)
#pragma unroll
          for (int e = 0; e < 4; e++) acc[m][nt][e] += src[(m * NTL + nt) * 4 + e];
    }
  }
  // from here on only the 128 threads of k-slice 0 are alive

  if (gridDim.z > 1) {
    const size_t plane = (size_t)BM * N;
    float* part = ws + blockIdx.z * plane;
    if (active) {
#pragma unroll
      for (int m = 0; m < MT; m++)
#pragma unroll
        for (int nt = 0; nt < NTL; nt++) {
          const int col = r0 + nt * 8 + 2 * t, row = 16 * m + g;
          *reinterpret_cast<float2*>(part + (size_t)row * N + col) = make_float2(acc[m][nt][0], acc[m][nt][1]);
          *reinterpret_cast<float2*>(part + (size_t)(row + 8) * N + col) = make_float2(acc[m][nt][2], acc[m][nt][3]);
        }
    }
    __threadfence();
    __shared__ int last_s;
    asm volatile("bar.sync 1, 128;");  // the 128 surviving threads
    if (tid == 0) {
      last_s = atomicAdd(&counters[blockIdx.x], 1) == (int)gridDim.z - 1;
      if (last_s) counters[blockIdx.x] = 0;
    }
    asm volatile("bar.sync 1, 128;");
    if (!last_s) return;
    __threadfence();
    const int cb = blockIdx.x * BN;
    auto total = [&](int r, int c) {
      float v = 0.f;
      for (int z = 0; z < (int)gridDim.z; z++) v += hi_ldcg(ws + z * plane + (size_t)r * N + c);
      return v;
    };
    if constexpr (EPI == 2) {
      for (int e = tid; e < M * (BN / 2); e += 128) {
        const int r = e / (BN / 2), o = e % (BN / 2), c = cb + (o / 16) * 32 + o % 16;
        if (c < N) C[(size_t)r * ldc + cb / 2 + o] = silu_mul(total(r, c), total(r, c + 16));
      }
    } else {
      for (int e = tid; e < M * BN; e += 128) {
        const int r = e / BN, c = cb + e % BN;
        if (c < N) {
          half* dst = C + (size_t)r * ldc + c;
          if constexpr (EPI == 1) *dst = __hadd(*dst, __float2half(total(r, c)));
          else *dst = __float2half(total(r, c) + (bias ? __half2float(bias[c]) : 0.f));
        }
      }
    }
    return;
  }

  if (!active) return;
  if constexpr (EPI == 2) {
    const int ocol = (r0 / 32) * 16 + 2 * t;
#pragma unroll
    for (int m = 0; m < MT; m++)
#pragma unroll
      for (int j = 0; j < 2; j++) {
        const int row = 16 * m + g;
        if (row < M)
          *reinterpret_cast<__half2*>(C + (size_t)row * ldc + ocol + j * 8) =
              __halves2half2(silu_mul(acc[m][j][0], acc[m][j + 2][0]), silu_mul(acc[m][j][1], acc[m][j + 2][1]));
        if (row + 8 < M)
          *reinterpret_cast<__half2*>(C + (size_t)(row + 8) * ldc + ocol + j * 8) =
              __halves2half2(silu_mul(acc[m][j][2], acc[m][j + 2][2]), silu_mul(acc[m][j][3], acc[m][j + 2][3]));
      }
  } else {
#pragma unroll
    for (int m = 0; m < MT; m++)
#pragma unroll
      for (int nt = 0; nt < NTL; nt++) {
        const int col = r0 + nt * 8 + 2 * t, row = 16 * m + g;
        float v0 = acc[m][nt][0], v1 = acc[m][nt][1], v2 = acc[m][nt][2], v3 = acc[m][nt][3];
        if constexpr (EPI == 0) {
          if (bias) {
            const __half2 b = *reinterpret_cast<const __half2*>(bias + col);
            v0 += __low2float(b); v1 += __high2float(b); v2 += __low2float(b); v3 += __high2float(b);
          }
          if (row < M) *reinterpret_cast<__half2*>(C + (size_t)row * ldc + col) = __floats2half2_rn(v0, v1);
          if (row + 8 < M) *reinterpret_cast<__half2*>(C + (size_t)(row + 8) * ldc + col) = __floats2half2_rn(v2, v3);
        } else {
          if (row < M) {
            __half2* d = reinterpret_cast<__half2*>(C + (size_t)row * ldc + col);
            *d = __hadd2(*d, __floats2half2_rn(v0, v1));
          }
          if (row + 8 < M) {
            __half2* d = reinterpret_cast<__half2*>(C + (size_t)(row + 8) * ldc + col);
            *d = __hadd2(*d, __floats2half2_rn(v2, v3));
          }
        }
      }
  }
}
#endif  // !HI_AMD

// ---------------------------------------------------------------------------------
// RoPE (HF rotate_half convention, fp16 arithmetic like HF) fused with the KV-cache write.
// qkv row layout: [q (Hq*D) | k (Hkv*D) | v (Hkv*D)]. q is rotated in place; rotated k
// and raw v go to cache[slot][kv_head][pos][D]. cs = per-position (cos, sin) in fp16.
// Positions come from device memory (decode, so a captured graph can advance them) or
// from pos_base + row (prefill).
// ---------------------------------------------------------------------------------
__global__ void rope_cache_kernel(half* qkv, int ldq, int Hq, int Hkv, int D, const __half2* __restrict__ cs,
                                  const int* pos, int pos_base, int slot_base,
                                  half* kc, half* vc, int max_seq) {
  const int HD = D / 2;
  const int row = blockIdx.x;
  const int p = pos ? pos[row] : pos_base + row;
  const int s = pos ? slot_base + row : slot_base;
  half* base = qkv + (size_t)row * ldq;
  for (int i = threadIdx.x; i < (Hq + Hkv) * HD; i += blockDim.x) {
    const int hh = i / HD, d = i % HD;
    half* ptr = base + hh * D;
    const half x1 = ptr[d], x2 = ptr[d + HD];
    half y1, y2;
    rope_pair(x1, x2, cs[(size_t)p * HD + d], y1, y2);
    if (hh < Hq) {
      ptr[d] = y1; ptr[d + HD] = y2;
    } else {
      half* dst = kc + (((size_t)s * Hkv + (hh - Hq)) * max_seq + p) * D;
      dst[d] = y1; dst[d + HD] = y2;
    }
  }
  const half* v = base + (Hq + Hkv) * D;
  for (int i = threadIdx.x; i < Hkv * D; i += blockDim.x)
    vc[(((size_t)s * Hkv + i / D) * max_seq + p) * D + i % D] = v[i];
}

// ---------------------------------------------------------------------------------
// FlashAttention-2 style causal prefill attention on tensor cores (portable WMMA version).
// Block = (64 query rows, one query head), 4 warps x 16 rows. K/V tiles of BKV keys are
// read from the KV cache (so prefill can continue an existing prefix at p0). S = QK^T and
// O += PV run on WMMA; the online softmax runs on the fp32 S tile in shared memory, and
// the output accumulator lives in registers with a fixed lane->(row, cols) map.
// ---------------------------------------------------------------------------------
template <int D, int BKV>
__global__ void __launch_bounds__(4 * HI_WARP)
flash_prefill_kernel(const half* __restrict__ q, int ldq, const half* __restrict__ kc,
                     const half* __restrict__ vc, int max_seq, half* out, int ldo,
                     int T, int p0, int Hq, int Hkv, float scale_log2) {
  constexpr int BQ = 64, LDK = D + 8, SC = BKV > 64 ? BKV : 64, LDSF = SC + 4, LDP = BKV + 8, LDQ = D + 8;
  constexpr int LPR = HI_WARP / 16;          // lanes per row
  constexpr int CPL = BKV / LPR;             // score columns per lane
  constexpr int OCC = 64 / LPR, OC = D / LPR;  // output columns per lane per 64-wide chunk / total
  static_assert(16 * LDQ * 2 <= 16 * LDSF * 4, "Q staging fits in the per-warp S tile");
  static_assert(OCC % 8 == 0 && CPL >= 1, "lane mapping");
  __shared__ __align__(32) half Ks[BKV * LDK];
  __shared__ __align__(32) half Vs[BKV * LDK];
  __shared__ __align__(32) float Sb[4][16 * LDSF];
  __shared__ __align__(32) half Pb[4][16 * LDP];

  const int tid = threadIdx.x, w = tid / HI_WARP, lane = tid % HI_WARP;
  const int q0 = blockIdx.x * BQ, h = blockIdx.y, hk = h / (Hq / Hkv);
  const int qrow0 = q0 + w * 16;
  const int my_row = lane / LPR, cseg = (lane % LPR) * CPL, oseg = (lane % LPR) * OCC;
  const int qpos = p0 + qrow0 + my_row;
  const int kv_end = p0 + min(q0 + BQ, T);
  const half* Kh = kc + (size_t)hk * max_seq * D;
  const half* Vh = vc + (size_t)hk * max_seq * D;

  // stage this warp's 16 query rows (zero past T) through its S tile and keep them as A fragments
  half* Qst = reinterpret_cast<half*>(Sb[w]);
  for (int c = lane; c < 16 * D / 8; c += HI_WARP) {
    const int r = c / (D / 8), d8 = (c % (D / 8)) * 8;
    *reinterpret_cast<uint4*>(Qst + r * LDQ + d8) = (qrow0 + r < T)
        ? *reinterpret_cast<const uint4*>(q + (size_t)(qrow0 + r) * ldq + h * D + d8)
        : make_uint4(0, 0, 0, 0);
  }
  hi_syncwarp();
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> qa[D / 16];
#pragma unroll
  for (int k = 0; k < D / 16; k++) wmma::load_matrix_sync(qa[k], Qst + k * 16, LDQ);
  hi_syncwarp();

  float m_i = HI_NEG_INF, l_i = 0.f, o[OC];
#pragma unroll
  for (int c = 0; c < OC; c++) o[c] = 0.f;

  // K/V tiles are software-pipelined: tile j+1 is loaded into registers while tile j is
  // being computed from shared memory, so global-load latency is off the critical path.
  constexpr int NT = 4 * HI_WARP, CH = BKV * D / 8, PT = (CH + NT - 1) / NT;
  uint4 kreg[PT], vreg[PT];
  auto gload = [&](int j) {
#pragma unroll
    for (int i = 0; i < PT; i++) {
      const int c = tid + i * NT;
      if (c < CH) {
        const int r = c / (D / 8), d8 = (c % (D / 8)) * 8;
        const bool ok = j + r < kv_end;
        kreg[i] = ok ? *reinterpret_cast<const uint4*>(Kh + (size_t)(j + r) * D + d8) : make_uint4(0, 0, 0, 0);
        vreg[i] = ok ? *reinterpret_cast<const uint4*>(Vh + (size_t)(j + r) * D + d8) : make_uint4(0, 0, 0, 0);
      }
    }
  };
  gload(0);
  for (int j0 = 0; j0 < kv_end; j0 += BKV) {
    __syncthreads();  // everyone is done reading the previous tile
#pragma unroll
    for (int i = 0; i < PT; i++) {
      const int c = tid + i * NT;
      if (c < CH) {
        const int r = c / (D / 8), d8 = (c % (D / 8)) * 8;
        *reinterpret_cast<uint4*>(Ks + r * LDK + d8) = kreg[i];
        *reinterpret_cast<uint4*>(Vs + r * LDK + d8) = vreg[i];
      }
    }
    __syncthreads();
    if (j0 + BKV < kv_end) gload(j0 + BKV);
    if (p0 + qrow0 + 15 < j0) continue;  // whole warp is above the causal diagonal for this tile

    // S = Q K^T
#pragma unroll
    for (int n = 0; n < BKV / 16; n++) {
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> s;
      wmma::fill_fragment(s, 0.f);
#pragma unroll
      for (int k = 0; k < D / 16; k++) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> kb;
        wmma::load_matrix_sync(kb, Ks + n * 16 * LDK + k * 16, LDK);
        wmma::mma_sync(s, qa[k], kb, s);
      }
      wmma::store_matrix_sync(Sb[w] + n * 16, s, LDSF, wmma::mem_row_major);
    }
    hi_syncwarp();

    // online softmax on this lane's row segment
    float* srow = Sb[w] + my_row * LDSF + cseg;
    float mx = HI_NEG_INF;
#pragma unroll
    for (int c = 0; c < CPL; c++) {
      const int kp = j0 + cseg + c;
      const float sv = (kp <= qpos && kp < kv_end) ? srow[c] * scale_log2 : HI_NEG_INF;
      srow[c] = sv;
      mx = fmaxf(mx, sv);
    }
#pragma unroll
    for (int off = 1; off < LPR; off <<= 1) mx = fmaxf(mx, hi_shfl_xor(mx, off));
    const float m_new = fmaxf(m_i, mx);
    const float alpha = (m_new == HI_NEG_INF) ? 1.f : exp2f(m_i - m_new);
    float sum = 0.f;
    half* prow = Pb[w] + my_row * LDP + cseg;
#pragma unroll
    for (int c = 0; c < CPL; c++) {
      const float p = (srow[c] == HI_NEG_INF) ? 0.f : exp2f(srow[c] - m_new);
      sum += p;
      prow[c] = __float2half(p);
    }
#pragma unroll
    for (int off = 1; off < LPR; off <<= 1) sum += hi_shfl_xor(sum, off);
    l_i = l_i * alpha + sum;
    m_i = m_new;
#pragma unroll
    for (int c = 0; c < OC; c++) o[c] *= alpha;
    hi_syncwarp();

    // O += P V, 64 output columns at a time (staged through Sb)
#pragma unroll
    for (int n0 = 0; n0 < D; n0 += 64) {
#pragma unroll
      for (int n = 0; n < 4; n++) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> pv;
        wmma::fill_fragment(pv, 0.f);
#pragma unroll
        for (int k = 0; k < BKV / 16; k++) {
          wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> pa;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> vb;
          wmma::load_matrix_sync(pa, Pb[w] + k * 16, LDP);
          wmma::load_matrix_sync(vb, Vs + k * 16 * LDK + n0 + n * 16, LDK);
          wmma::mma_sync(pv, pa, vb, pv);
        }
        wmma::store_matrix_sync(Sb[w] + n * 16, pv, LDSF, wmma::mem_row_major);
      }
      hi_syncwarp();
      const float* orow = Sb[w] + my_row * LDSF + oseg;
#pragma unroll
      for (int c = 0; c < OCC; c++) o[(n0 / 64) * OCC + c] += orow[c];
      hi_syncwarp();
    }
  }

  if (qrow0 + my_row < T) {
    const float inv = 1.f / l_i;
#pragma unroll
    for (int ch = 0; ch < D / 64; ch++) {
      half* dst = out + (size_t)(qrow0 + my_row) * ldo + h * D + ch * 64 + oseg;
#pragma unroll
      for (int c = 0; c < OCC; c += 8) {
        H8 v;
#pragma unroll
        for (int e = 0; e < 8; e++) v.h[e] = __float2half(o[ch * OCC + c + e] * inv);
        *reinterpret_cast<uint4*>(dst + c) = v.u;
      }
    }
  }
}

#if !defined(HI_AMD)
// ---------------------------------------------------------------------------------
// NVIDIA fast path for causal prefill attention: FlashAttention-2 with everything in
// registers. WMMA hides its accumulator layout, which forces S and the O update through
// shared memory; with raw mma.sync the layout is known, so: S = Q K^T lands in accumulator
// registers, the online softmax runs on them directly, P is repacked in registers as the A
// operand of P V (the m16n8 accumulator layout of two adjacent key tiles *is* the m16n8k16
// A layout), and O stays in registers. K/V fragments come from shared memory via ldmatrix
// (.trans for V). Same block shape and numerics as flash_prefill_kernel (P rounded to fp16
// before P V); Turing runs each m16n8k16 as two m16n8k8.
// ---------------------------------------------------------------------------------
template <int D, int BKV>
__global__ void __launch_bounds__(128)
flash_prefill_mma_kernel(const half* __restrict__ q, int ldq, const half* __restrict__ kc,
                         const half* __restrict__ vc, int max_seq, half* out, int ldo,
                         int T, int p0, int Hq, int Hkv, float scale_log2) {
  constexpr int BQ = 64, LD = D + 8;
  __shared__ __align__(16) half Qs[BQ * LD];
  __shared__ __align__(16) half Ks[BKV * LD];
  __shared__ __align__(16) half Vs[BKV * LD];

  const int tid = threadIdx.x, w = tid / 32, lane = tid % 32;
  const int g = lane / 4, t = lane % 4;  // mma fragment coordinates
  const int q0 = blockIdx.x * BQ, h = blockIdx.y, hk = h / (Hq / Hkv);
  const int qrow0 = q0 + w * 16;
  const int qpos0 = p0 + qrow0 + g, qpos1 = qpos0 + 8;  // this lane's two query rows
  const int kv_end = p0 + min(q0 + BQ, T);
  const half* Kh = kc + (size_t)hk * max_seq * D;
  const half* Vh = vc + (size_t)hk * max_seq * D;

  for (int c = tid; c < BQ * D / 8; c += 128) {
    const int r = c / (D / 8), d8 = (c % (D / 8)) * 8;
    *reinterpret_cast<uint4*>(Qs + r * LD + d8) = (q0 + r < T)
        ? *reinterpret_cast<const uint4*>(q + (size_t)(q0 + r) * ldq + h * D + d8) : make_uint4(0, 0, 0, 0);
  }
  __syncthreads();
  uint32_t qf[D / 16][4];
#pragma unroll
  for (int kc2 = 0; kc2 < D / 16; kc2++)
    ldsm_x4(qf[kc2], Qs + (w * 16 + (lane % 16)) * LD + kc2 * 16 + (lane / 16) * 8);

  float o[D / 8][4];
#pragma unroll
  for (int i = 0; i < D / 8; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  float m0 = HI_NEG_INF, m1 = HI_NEG_INF, l0 = 0.f, l1 = 0.f;

  constexpr int CH = BKV * D / 8, PT = CH / 128;
  static_assert(CH % 128 == 0, "tile chunks per thread");
  uint4 kreg[PT], vreg[PT];
  auto gload = [&](int j) {
#pragma unroll
    for (int i = 0; i < PT; i++) {
      const int c = tid + i * 128, r = c / (D / 8), d8 = (c % (D / 8)) * 8;
      const bool ok = j + r < kv_end;
      kreg[i] = ok ? *reinterpret_cast<const uint4*>(Kh + (size_t)(j + r) * D + d8) : make_uint4(0, 0, 0, 0);
      vreg[i] = ok ? *reinterpret_cast<const uint4*>(Vh + (size_t)(j + r) * D + d8) : make_uint4(0, 0, 0, 0);
    }
  };
  gload(0);
  for (int j0 = 0; j0 < kv_end; j0 += BKV) {
    __syncthreads();
#pragma unroll
    for (int i = 0; i < PT; i++) {
      const int c = tid + i * 128, r = c / (D / 8), d8 = (c % (D / 8)) * 8;
      *reinterpret_cast<uint4*>(Ks + r * LD + d8) = kreg[i];
      *reinterpret_cast<uint4*>(Vs + r * LD + d8) = vreg[i];
    }
    __syncthreads();
    if (j0 + BKV < kv_end) gload(j0 + BKV);
    if (p0 + qrow0 + 15 < j0) continue;  // warp entirely above the causal diagonal

    // S = Q K^T: BKV/8 key tiles of 8, each lane holds rows (g, g+8) x keys (2t, 2t+1) per tile
    float s[BKV / 8][4];
#pragma unroll
    for (int i = 0; i < BKV / 8; i++) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f;
#pragma unroll
    for (int np = 0; np < BKV / 16; np++) {
#pragma unroll
      for (int kc2 = 0; kc2 < D / 16; kc2++) {
        uint32_t kb[4];
        ldsm_x4(kb, Ks + (np * 16 + (lane & 7) + ((lane >> 4) & 1) * 8) * LD + kc2 * 16 + ((lane >> 3) & 1) * 8);
        mma16816(s[2 * np], qf[kc2], kb[0], kb[1]);
        mma16816(s[2 * np + 1], qf[kc2], kb[2], kb[3]);
      }
    }

    // online softmax on registers (rows g and g+8; a row's BKV keys span the 4 lanes of a quad)
    float mx0 = HI_NEG_INF, mx1 = HI_NEG_INF;
#pragma unroll
    for (int nt = 0; nt < BKV / 8; nt++) {
#pragma unroll
      for (int c = 0; c < 4; c++) {
        const int kp = j0 + nt * 8 + 2 * t + (c & 1);
        const int qp = c < 2 ? qpos0 : qpos1;
        const float v = (kp <= qp && kp < kv_end) ? s[nt][c] * scale_log2 : HI_NEG_INF;
        s[nt][c] = v;
        if (c < 2) mx0 = fmaxf(mx0, v); else mx1 = fmaxf(mx1, v);
      }
    }
#pragma unroll
    for (int off = 1; off < 4; off <<= 1) {
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, off));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, off));
    }
    const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
    const float a0 = mn0 == HI_NEG_INF ? 1.f : exp2f(m0 - mn0);
    const float a1 = mn1 == HI_NEG_INF ? 1.f : exp2f(m1 - mn1);
    float sum0 = 0.f, sum1 = 0.f;
#pragma unroll
    for (int nt = 0; nt < BKV / 8; nt++) {
#pragma unroll
      for (int c = 0; c < 4; c++) {
        const float mref = c < 2 ? mn0 : mn1;
        const float p = s[nt][c] == HI_NEG_INF ? 0.f : exp2f(s[nt][c] - mref);
        s[nt][c] = p;
        if (c < 2) sum0 += p; else sum1 += p;
      }
    }
#pragma unroll
    for (int off = 1; off < 4; off <<= 1) {
      sum0 += __shfl_xor_sync(0xffffffffu, sum0, off);
      sum1 += __shfl_xor_sync(0xffffffffu, sum1, off);
    }
    l0 = l0 * a0 + sum0; l1 = l1 * a1 + sum1;
    m0 = mn0; m1 = mn1;
#pragma unroll
    for (int dt = 0; dt < D / 8; dt++) { o[dt][0] *= a0; o[dt][1] *= a0; o[dt][2] *= a1; o[dt][3] *= a1; }

    // O += P V, P straight from the S registers (two adjacent key tiles = one k16 A fragment)
#pragma unroll
    for (int kc2 = 0; kc2 < BKV / 16; kc2++) {
      uint32_t pa[4];
      pa[0] = pack_half2(s[2 * kc2][0], s[2 * kc2][1]);
      pa[1] = pack_half2(s[2 * kc2][2], s[2 * kc2][3]);
      pa[2] = pack_half2(s[2 * kc2 + 1][0], s[2 * kc2 + 1][1]);
      pa[3] = pack_half2(s[2 * kc2 + 1][2], s[2 * kc2 + 1][3]);
#pragma unroll
      for (int dp = 0; dp < D / 16; dp++) {
        uint32_t vb[4];
        ldsm_x4_trans(vb, Vs + (kc2 * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * LD + dp * 16 + (lane >> 4) * 8);
        mma16816(o[2 * dp], pa, vb[0], vb[1]);
        mma16816(o[2 * dp + 1], pa, vb[2], vb[3]);
      }
    }
  }

  const float i0 = 1.f / l0, i1 = 1.f / l1;
  const int r0 = qrow0 + g, r1 = r0 + 8;
#pragma unroll
  for (int dt = 0; dt < D / 8; dt++) {
    const int col = h * D + dt * 8 + 2 * t;
    if (r0 < T) *reinterpret_cast<__half2*>(out + (size_t)r0 * ldo + col) = __floats2half2_rn(o[dt][0] * i0, o[dt][1] * i0);
    if (r1 < T) *reinterpret_cast<__half2*>(out + (size_t)r1 * ldo + col) = __floats2half2_rn(o[dt][2] * i1, o[dt][3] * i1);
  }
}
#endif  // !HI_AMD

// ---------------------------------------------------------------------------------
// Split-K ("flash-decoding") attention for one new token per sequence (portable version).
// Block = (split, kv head, sequence): all G query heads of a GQA group share one pass
// over K/V, so the cache is read once per group. The grid has S splits but only
// S_eff = min(S, ceil(len / 64)) do work (len is read on device, so the launch stays
// graph-capturable while the context grows). Each split runs an online softmax over its
// keys in tiles of 256; the last split block to finish (atomic counter) merges the partials,
// so no separate combine launch is needed. With one split the block normalizes directly.
// Query b reads cache slot b, or slot kv_slot for every b when kv_slot >= 0 (speculative
// verification: the queries are consecutive positions of one sequence). A query's result
// depends only on its position, its slot's keys and S, never on the number of queries.
// ---------------------------------------------------------------------------------
template <int G, int D>
__global__ void __launch_bounds__(256)
decode_attn_kernel(const half* __restrict__ q, int ldq, const half* __restrict__ kc,
                   const half* __restrict__ vc, int max_seq, int Hkv, const int* __restrict__ pos,
                   float* part_o, float* part_ml, int* counters, int S, half* out, int ldo,
                   float scale_log2, int kv_slot) {
  constexpr int TILE = 256, NT = 256, HALF = D / 2, NP = (G * HALF + NT - 1) / NT;  // PV passes
  __shared__ float qs[G][D];
  __shared__ float sc[G][TILE];
  __shared__ float alpha_s[G], m_s[G], l_s[G];
  const int split = blockIdx.x, hk = blockIdx.y, b = blockIdx.z;
  const int tid = threadIdx.x, lane = tid % HI_WARP, warp = tid / HI_WARP, nw = blockDim.x / HI_WARP;
  const int Hq = Hkv * G;
  const int len = pos[b] + 1;
  const int S_eff = min(S, (len + 63) / 64);
  if (split >= S_eff) return;
  const int chunk = (len + S_eff - 1) / S_eff;
  const int start = split * chunk, end = min(len, start + chunk);
  const size_t cb = (size_t)(kv_slot >= 0 ? kv_slot : b);  // the cache slot this query reads
  const half* Kh = kc + (cb * Hkv + hk) * max_seq * D;
  const half* Vh = vc + (cb * Hkv + hk) * max_seq * D;

  for (int i = tid; i < G * D; i += blockDim.x)
    qs[i / D][i % D] = __half2float(q[(size_t)b * ldq + (hk * G) * D + i]) * scale_log2;
  if (tid < G) { m_s[tid] = HI_NEG_INF; l_s[tid] = 0.f; }
  // PV mapping: pair index tid + k*NT -> (head, 2 dims)
  float acc0[NP], acc1[NP];
#pragma unroll
  for (int k = 0; k < NP; k++) acc0[k] = acc1[k] = 0.f;
  __syncthreads();

  for (int t0 = start; t0 < end; t0 += TILE) {
    const int j = t0 + tid;
    if (tid < TILE) {
      float s[G];
#pragma unroll
      for (int g = 0; g < G; g++) s[g] = HI_NEG_INF;
      if (j < end) {
#pragma unroll
        for (int g = 0; g < G; g++) s[g] = 0.f;
        const uint4* kr = reinterpret_cast<const uint4*>(Kh + (size_t)j * D);
#pragma unroll
        for (int c = 0; c < D / 8; c++) {
          H8 kv; kv.u = kr[c];
#pragma unroll
          for (int e = 0; e < 8; e++) {
            const float kf = __half2float(kv.h[e]);
#pragma unroll
            for (int g = 0; g < G; g++) s[g] = fmaf(qs[g][c * 8 + e], kf, s[g]);
          }
        }
      }
#pragma unroll
      for (int g = 0; g < G; g++) sc[g][tid] = s[g];
    }
    __syncthreads();
    for (int g = warp; g < G; g += nw) {
      const float m_old = m_s[g];
      float mx = HI_NEG_INF;
      for (int i = lane; i < TILE; i += HI_WARP) mx = fmaxf(mx, sc[g][i]);
      mx = warp_max(mx);
      const float m_new = fmaxf(m_old, mx);
      float sum = 0.f;
      for (int i = lane; i < TILE; i += HI_WARP) {
        const float sv = sc[g][i];
        const float p = (sv == HI_NEG_INF) ? 0.f : exp2f(sv - m_new);
        sc[g][i] = p;
        sum += p;
      }
      sum = warp_sum(sum);
      if (lane == 0) {
        const float a = (m_new == HI_NEG_INF) ? 1.f : exp2f(m_old - m_new);
        alpha_s[g] = a;
        l_s[g] = l_s[g] * a + sum;
        m_s[g] = m_new;
      }
    }
    __syncthreads();
    const int n = min(TILE, end - t0);
#pragma unroll
    for (int k = 0; k < NP; k++) {
      const int idx = tid + k * NT, g3 = idx / HALF, dp = idx % HALF;
      if (g3 < G) {
        const float a = alpha_s[g3];
        acc0[k] *= a; acc1[k] *= a;
        const __half2* vp = reinterpret_cast<const __half2*>(Vh + (size_t)t0 * D) + dp;
        float x0 = 0.f, x1 = 0.f;
#pragma unroll 8
        for (int jj = 0; jj < n; jj++) {
          const float p = sc[g3][jj];
          const float2 v = __half22float2(vp[(size_t)jj * HALF]);
          x0 = fmaf(p, v.x, x0);
          x1 = fmaf(p, v.y, x1);
        }
        acc0[k] += x0; acc1[k] += x1;
      }
    }
    __syncthreads();
  }
  if (S_eff == 1) {
#pragma unroll
    for (int k = 0; k < NP; k++) {
      const int idx = tid + k * NT, g3 = idx / HALF, dp = idx % HALF;
      if (g3 < G) {
        const float inv = 1.f / l_s[g3];
        *reinterpret_cast<__half2*>(out + (size_t)b * ldo + (hk * G + g3) * D + 2 * dp) = __floats2half2_rn(acc0[k] * inv, acc1[k] * inv);
      }
    }
    return;
  }
#pragma unroll
  for (int k = 0; k < NP; k++) {
    const int idx = tid + k * NT, g3 = idx / HALF, dp = idx % HALF;
    if (g3 < G) {
      const size_t pi = ((size_t)b * Hq + hk * G + g3) * S + split;
      part_o[pi * D + 2 * dp] = acc0[k];
      part_o[pi * D + 2 * dp + 1] = acc1[k];
      if (dp == 0) { part_ml[pi * 2] = m_s[g3]; part_ml[pi * 2 + 1] = l_s[g3]; }
    }
  }
  __threadfence();
  __syncthreads();
  __shared__ int is_last;
  if (tid == 0) {
    const int done = atomicAdd(&counters[b * Hkv + hk], 1);
    is_last = done == S_eff - 1;
    if (is_last) counters[b * Hkv + hk] = 0;  // ready for the next launch / graph replay
  }
  __syncthreads();
  if (!is_last) return;
  __threadfence();
  // merge: per-split weights exp2(m_s - M) / L via shared memory, then independent loads of o
  __shared__ float wgt[G][64];
  for (int i = tid; i < G * S_eff; i += blockDim.x) {
    const int g = i / S_eff, s2 = i % S_eff;
    wgt[g][s2] = hi_ldcg(part_ml + (((size_t)b * Hq + hk * G + g) * S + s2) * 2);
    sc[g][s2] = hi_ldcg(part_ml + (((size_t)b * Hq + hk * G + g) * S + s2) * 2 + 1);  // reuse sc for l
  }
  __syncthreads();
  for (int g = warp; g < G; g += nw) {
    float mx = HI_NEG_INF;
    for (int s2 = lane; s2 < S_eff; s2 += HI_WARP) mx = fmaxf(mx, wgt[g][s2]);
    mx = warp_max(mx);
    float L = 0.f;
    for (int s2 = lane; s2 < S_eff; s2 += HI_WARP) L += exp2f(wgt[g][s2] - mx) * sc[g][s2];
    L = warp_sum(L);
    for (int s2 = lane; s2 < S_eff; s2 += HI_WARP) wgt[g][s2] = exp2f(wgt[g][s2] - mx) / L;
  }
  __syncthreads();
  for (int i = tid; i < G * D; i += blockDim.x) {
    const int g = i / D, d = i % D;
    const float* po = part_o + ((size_t)b * Hq + hk * G + g) * S * D + d;
    float o0 = 0.f, o1 = 0.f;
    int s2 = 0;
    for (; s2 + 1 < S_eff; s2 += 2) {
      o0 = fmaf(hi_ldcg(po + (size_t)s2 * D), wgt[g][s2], o0);
      o1 = fmaf(hi_ldcg(po + (size_t)(s2 + 1) * D), wgt[g][s2 + 1], o1);
    }
    if (s2 < S_eff) o0 = fmaf(hi_ldcg(po + (size_t)s2 * D), wgt[g][s2], o0);
    out[(size_t)b * ldo + (hk * G + g) * D + d] = __float2half(o0 + o1);
  }
}

#if !defined(HI_AMD)
// ---------------------------------------------------------------------------------
// NVIDIA fast path for decode attention on tensor cores. The G query heads of a GQA group
// are the rows of an m16 tile (rows >= G are zero padding), so one block's pass over K/V
// serves the whole group with mma.sync instead of scalar dot products. Each of the 4 warps
// owns 16 keys of every 64-key tile and keeps its own online-softmax state in registers
// (S, P and O never touch shared memory); the warps are merged through shared memory at the
// end (reusing the K tile), then splits are merged by the last block exactly as in
// decode_attn_kernel.
// ---------------------------------------------------------------------------------
template <int G, int D>
__global__ void __launch_bounds__(128)
decode_attn_mma_kernel(const half* __restrict__ q, int ldq, const half* __restrict__ kc,
                       const half* __restrict__ vc, int max_seq, int Hkv, const int* __restrict__ pos,
                       float* part_o, float* part_ml, int* counters, int S, half* out, int ldo,
                       float scale_log2, int kv_slot) {
  static_assert(G <= 8, "one quad row per head");
  constexpr int TILE = 64, LD = D + 8;
  static_assert(4 * G * D * 4 <= TILE * LD * 2, "warp-merge scratch aliases the K tile");
  __shared__ __align__(16) half Qs[16 * LD];
  __shared__ __align__(16) half Ks[TILE * LD];
  __shared__ __align__(16) half Vs[TILE * LD];
  __shared__ float red_m[4][G], red_l[4][G];
  __shared__ float wgt[G][64];
  __shared__ float lsum[G][64];
  __shared__ int is_last;
  float* red_o = reinterpret_cast<float*>(Ks);  // [4][G][D], valid after the K/V loop

  const int split = blockIdx.x, hk = blockIdx.y, b = blockIdx.z;
  const int tid = threadIdx.x, w = tid / 32, lane = tid % 32, g = lane / 4, t = lane % 4;
  const int Hq = Hkv * G;
  const int len = pos[b] + 1;
  const int S_eff = min(S, (len + TILE - 1) / TILE);
  if (split >= S_eff) return;
  const int chunk = (len + S_eff - 1) / S_eff;
  const int start = split * chunk, end = min(len, start + chunk);
  const size_t cb = (size_t)(kv_slot >= 0 ? kv_slot : b);  // the cache slot this query reads
  const half* Kh = kc + (cb * Hkv + hk) * max_seq * D;
  const half* Vh = vc + (cb * Hkv + hk) * max_seq * D;

  for (int c = tid; c < 16 * D / 8; c += 128) {
    const int r = c / (D / 8), d8 = (c % (D / 8)) * 8;
    *reinterpret_cast<uint4*>(Qs + r * LD + d8) = r < G
        ? *reinterpret_cast<const uint4*>(q + (size_t)b * ldq + (hk * G + r) * D + d8) : make_uint4(0, 0, 0, 0);
  }
  __syncthreads();
  uint32_t qf[D / 16][4];
#pragma unroll
  for (int k = 0; k < D / 16; k++) ldsm_x4(qf[k], Qs + (lane % 16) * LD + k * 16 + (lane / 16) * 8);

  float o[D / 8][4];
#pragma unroll
  for (int i = 0; i < D / 8; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  float m0 = HI_NEG_INF, l0 = 0.f;  // row g (rows g + 8 are padding and ignored)

  for (int t0 = start; t0 < end; t0 += TILE) {
    __syncthreads();
    for (int c = tid; c < TILE * D / 8; c += 128) {
      const int r = c / (D / 8), d8 = (c % (D / 8)) * 8;
      const bool ok = t0 + r < end;
      *reinterpret_cast<uint4*>(Ks + r * LD + d8) = ok ? *reinterpret_cast<const uint4*>(Kh + (size_t)(t0 + r) * D + d8) : make_uint4(0, 0, 0, 0);
      *reinterpret_cast<uint4*>(Vs + r * LD + d8) = ok ? *reinterpret_cast<const uint4*>(Vh + (size_t)(t0 + r) * D + d8) : make_uint4(0, 0, 0, 0);
    }
    __syncthreads();
    const int kw = w * 16;
    if (t0 + kw >= end) continue;

    float s[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
    for (int k = 0; k < D / 16; k++) {
      uint32_t kb[4];
      ldsm_x4(kb, Ks + (kw + (lane & 7) + ((lane >> 4) & 1) * 8) * LD + k * 16 + ((lane >> 3) & 1) * 8);
      mma16816(s[0], qf[k], kb[0], kb[1]);
      mma16816(s[1], qf[k], kb[2], kb[3]);
    }
    float mx = HI_NEG_INF;
#pragma unroll
    for (int nt = 0; nt < 2; nt++)
#pragma unroll
      for (int c = 0; c < 2; c++) {
        const int kp = t0 + kw + nt * 8 + 2 * t + c;
        const float v = kp < end ? s[nt][c] * scale_log2 : HI_NEG_INF;
        s[nt][c] = v;
        mx = fmaxf(mx, v);
      }
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
    const float mn = fmaxf(m0, mx);  // finite: this warp has >= 1 valid key
    const float a = exp2f(m0 - mn);
    float sum = 0.f;
#pragma unroll
    for (int nt = 0; nt < 2; nt++) {
#pragma unroll
      for (int c = 0; c < 2; c++) {
        const float p = s[nt][c] == HI_NEG_INF ? 0.f : exp2f(s[nt][c] - mn);
        s[nt][c] = p;
        sum += p;
      }
      s[nt][2] = s[nt][3] = 0.f;  // padding rows contribute nothing
    }
    sum += __shfl_xor_sync(0xffffffffu, sum, 1);
    sum += __shfl_xor_sync(0xffffffffu, sum, 2);
    l0 = l0 * a + sum;
    m0 = mn;
#pragma unroll
    for (int dt = 0; dt < D / 8; dt++) { o[dt][0] *= a; o[dt][1] *= a; }
    uint32_t pa[4];
    pa[0] = pack_half2(s[0][0], s[0][1]);
    pa[1] = pack_half2(s[0][2], s[0][3]);
    pa[2] = pack_half2(s[1][0], s[1][1]);
    pa[3] = pack_half2(s[1][2], s[1][3]);
#pragma unroll
    for (int dp = 0; dp < D / 16; dp++) {
      uint32_t vb[4];
      ldsm_x4_trans(vb, Vs + (kw + (lane & 7) + ((lane >> 3) & 1) * 8) * LD + dp * 16 + (lane >> 4) * 8);
      mma16816(o[2 * dp], pa, vb[0], vb[1]);
      mma16816(o[2 * dp + 1], pa, vb[2], vb[3]);
    }
  }

  // merge the 4 warps (each covered a different 16-key slice of every tile)
  __syncthreads();  // all warps are done with the K tile; reuse it as merge scratch
  if (g < G) {
    if (t == 0) { red_m[w][g] = m0; red_l[w][g] = l0; }
#pragma unroll
    for (int dt = 0; dt < D / 8; dt++) {
      red_o[(w * G + g) * D + dt * 8 + 2 * t] = o[dt][0];
      red_o[(w * G + g) * D + dt * 8 + 2 * t + 1] = o[dt][1];
    }
  }
  __syncthreads();
  for (int i = tid; i < G * D; i += 128) {
    const int r = i / D, d = i % D;
    float M = HI_NEG_INF;
#pragma unroll
    for (int ww = 0; ww < 4; ww++) M = fmaxf(M, red_m[ww][r]);
    float L = 0.f, O = 0.f;
#pragma unroll
    for (int ww = 0; ww < 4; ww++) {
      if (red_m[ww][r] == HI_NEG_INF) continue;
      const float e = exp2f(red_m[ww][r] - M);
      L += red_l[ww][r] * e;
      O += red_o[(ww * G + r) * D + d] * e;
    }
    if (S_eff == 1) {
      out[(size_t)b * ldo + (hk * G + r) * D + d] = __float2half(O / L);
    } else {
      const size_t idx = ((size_t)b * Hq + hk * G + r) * S + split;
      part_o[idx * D + d] = O;
      if (d == 0) { part_ml[idx * 2] = M; part_ml[idx * 2 + 1] = L; }
    }
  }
  if (S_eff == 1) return;

  // split merge: the last split block of this (sequence, kv head) combines all partials
  __threadfence();
  __syncthreads();
  if (tid == 0) {
    const int done = atomicAdd(&counters[b * Hkv + hk], 1);
    is_last = done == S_eff - 1;
    if (is_last) counters[b * Hkv + hk] = 0;
  }
  __syncthreads();
  if (!is_last) return;
  __threadfence();
  for (int i = tid; i < G * S_eff; i += 128) {
    const int r = i / S_eff, s2 = i % S_eff;
    const float* ml = part_ml + (((size_t)b * Hq + hk * G + r) * S + s2) * 2;
    wgt[r][s2] = hi_ldcg(ml);
    lsum[r][s2] = hi_ldcg(ml + 1);
  }
  __syncthreads();
  for (int r = w; r < G; r += 4) {
    float mx = HI_NEG_INF;
    for (int s2 = lane; s2 < S_eff; s2 += 32) mx = fmaxf(mx, wgt[r][s2]);
    mx = warp_max(mx);
    float L = 0.f;
    for (int s2 = lane; s2 < S_eff; s2 += 32) L += exp2f(wgt[r][s2] - mx) * lsum[r][s2];
    L = warp_sum(L);
    for (int s2 = lane; s2 < S_eff; s2 += 32) wgt[r][s2] = exp2f(wgt[r][s2] - mx) / L;
  }
  __syncthreads();
  for (int i = tid; i < G * D; i += 128) {
    const int r = i / D, d = i % D;
    const float* po = part_o + ((size_t)b * Hq + hk * G + r) * S * D + d;
    float o0 = 0.f, o1 = 0.f;
    int s2 = 0;
    for (; s2 + 1 < S_eff; s2 += 2) {
      o0 = fmaf(hi_ldcg(po + (size_t)s2 * D), wgt[r][s2], o0);
      o1 = fmaf(hi_ldcg(po + (size_t)(s2 + 1) * D), wgt[r][s2 + 1], o1);
    }
    if (s2 < S_eff) o0 = fmaf(hi_ldcg(po + (size_t)s2 * D), wgt[r][s2], o0);
    out[(size_t)b * ldo + (hk * G + r) * D + d] = __float2half(o0 + o1);
  }
}
#endif  // !HI_AMD

// ---------------------------------------------------------------------------------
// Embedding gather and greedy argmax (first index on ties, like torch.argmax). argmax
// also advances the device-side position and appends to the token history, so a decode
// step needs no host round trip.
// ---------------------------------------------------------------------------------
__global__ void embed_kernel(const int* __restrict__ tok, const half* __restrict__ E, half* x, int H) {
  const uint4* src = reinterpret_cast<const uint4*>(E + (size_t)tok[blockIdx.x] * H);
  uint4* dst = reinterpret_cast<uint4*>(x + (size_t)blockIdx.x * H);
  for (int i = threadIdx.x; i < H / 8; i += blockDim.x) dst[i] = src[i];
}

// Block-wide argmax of l[0, V): the largest value, lowest index on ties. Valid in thread 0.
// Does (v, i) beat (best, idx)? NaN ranks above every number (as in torch.argmax), ties go to
// the lower index.
__device__ __forceinline__ bool argmax_better(float v, int i, float best, int idx) {
  const bool vn = v != v, bn = best != best;
  if (vn != bn) return vn;
  return (vn || v == best) ? i < idx : v > best;
}

// Starts from (-inf, 0), so a row that is all -inf yields index 0, never an out-of-range token.
__device__ __forceinline__ void block_argmax(const half* __restrict__ l, int V, float& best, int& idx) {
  __shared__ float bv[32];
  __shared__ int bi[32];
  const int lane = threadIdx.x % HI_WARP, w = threadIdx.x / HI_WARP, nw = blockDim.x / HI_WARP;
  best = HI_NEG_INF;
  idx = 0;
  for (int i = threadIdx.x; i < V; i += blockDim.x) {
    const float v = __half2float(l[i]);
    if (argmax_better(v, i, best, idx)) { best = v; idx = i; }
  }
  for (int o = HI_WARP / 2; o > 0; o >>= 1) {
    const float ov = hi_shfl_xor(best, o);
    const int oi = hi_shfl_xor_i(idx, o);
    if (argmax_better(ov, oi, best, idx)) { best = ov; idx = oi; }
  }
  if (lane == 0) { bv[w] = best; bi[w] = idx; }
  __syncthreads();
  if (threadIdx.x == 0)
    for (int i = 1; i < nw; i++)
      if (argmax_better(bv[i], bi[i], best, idx)) { best = bv[i]; idx = bi[i]; }
}

// Record sequence b's next token: tok_out[b], advance pos[b] (if pos), append to its history.
__device__ __forceinline__ void emit_token(int b, int idx, int* tok_out, int* pos, int* hist, int max_seq) {
  tok_out[b] = idx;
  if (pos) {
    const int p = pos[b] + 1;
    pos[b] = p;
    if (p < max_seq) hist[(size_t)b * max_seq + p] = idx;
  }
}

__global__ void argmax_kernel(const half* __restrict__ logits, int V, int* tok_out, int* pos,
                              int* hist, int max_seq) {
  float best;
  int idx;
  block_argmax(logits + (size_t)blockIdx.x * V, V, best, idx);
  if (threadIdx.x == 0) emit_token(blockIdx.x, idx, tok_out, pos, hist, max_seq);
}

// Tensor-parallel argmax: each rank reduces its vocabulary slice to one 64-bit key per row,
// (order-preserving float bits << 32) | ~index, so the cross-rank max of the keys is the largest
// logit with the lowest index on ties, exactly what argmax_kernel picks on one GPU.
__device__ __forceinline__ unsigned long long argmax_key(float v, unsigned idx) {
  unsigned u = __float_as_uint(v);
  u = v != v ? 0xffffffffu : (u & 0x80000000u) ? ~u : (u | 0x80000000u);  // NaN (either sign) above +inf
  return ((unsigned long long)u << 32) | (0xffffffffu - idx);
}

__global__ void argmax_key_kernel(const half* __restrict__ logits, int ld, int V, int v0, unsigned long long* keys) {
  float best;
  int idx;
  block_argmax(logits + (size_t)blockIdx.x * ld, V, best, idx);
  if (threadIdx.x == 0) keys[blockIdx.x] = argmax_key(best, (unsigned)(v0 + idx));
}

__global__ void argmax_finish_kernel(const unsigned long long* keys, int B, int* tok_out, int* pos, int* hist, int max_seq) {
  const int b = threadIdx.x;
  if (b < B) emit_token(b, (int)(0xffffffffu - (unsigned)(keys[b] & 0xffffffffull)), tok_out, pos, hist, max_seq);
}

// ---------------------------------------------------------------------------------
// One-shot all-reduce for tensor parallelism, for the small messages of decode (one hidden
// row per sequence, twice per layer), where NCCL's ring protocol is latency-bound.
//
// Each block is an independent channel over a contiguous range of the message, processed in
// pieces of up to kARPiece bytes: copy the piece into this rank's staging buffer (readable by
// every rank: this GPU's memory over P2P, or pinned host memory when the GPUs have no peer
// path), raise this rank's flag in every rank's flag array, wait for every rank's flag, then sum
// all ranks' copies of the piece in rank order, so all ranks end bitwise identical (fp32
// accumulation). Pieces let blocks drift apart, overlapping one block's PCIe reads with
// another's writes on large (prefill) messages.
//
// Block b owns sub-buffer b of each of two staging slots and its own call counter (`epoch`),
// which advances once per piece identically on all ranks (same message, same grid). The
// slot alternates per piece; block b overwrites a sub-buffer two pieces later, only after
// passing the barrier of the piece in between, which a peer's block b reaches only after it
// finished reading the earlier piece. Nothing else ever touches sub-buffer b, so the grid
// size may change from call to call (sized to the message).
// ---------------------------------------------------------------------------------
constexpr int kARMaxRanks = 8, kARBlocks = 16, kARThreads = 256, kARPiece = 32 << 10;
constexpr size_t kARSlotBytes = (size_t)kARBlocks * kARPiece;  // one staging slot: a piece per block
constexpr size_t kARRankBytes = 2 * kARSlotBytes + kARBlocks * kARMaxRanks * sizeof(unsigned);  // 2 slots + flags
struct ARArgs {
  char* stage[kARMaxRanks];      // rank r's buffer: 2 staging slots, then its flags [kARBlocks][kARMaxRanks]
  unsigned* epoch;               // this rank's per-block counters (local device memory)
  int rank, world;
};
__device__ __forceinline__ unsigned* ar_flags(const ARArgs& a, int r) {
  return reinterpret_cast<unsigned*>(a.stage[r] + 2 * kARSlotBytes);
}

template <int OP>  // 0: fp16 sum, 1: uint64 max (argmax keys)
__global__ void __launch_bounds__(kARThreads) allreduce_oneshot_kernel(ARArgs a, uint4* data, long long nvec) {
  const int b = blockIdx.x;
  const long long per = (nvec + gridDim.x - 1) / gridDim.x;
  const long long v0 = min(nvec, b * per), v1 = min(nvec, v0 + per);
  constexpr int cap = kARPiece / 16;
  unsigned e = a.epoch[b];
  long long p0 = v0;
  do {
    const int m = (int)min((long long)cap, v1 - p0);
    e++;
    const size_t so = (size_t)(e & 1) * kARSlotBytes + (size_t)b * kARPiece;
    uint4* mine = reinterpret_cast<uint4*>(a.stage[a.rank] + so);
    const uint4* src = data + p0;
    for (int i = threadIdx.x; i < m; i += blockDim.x) mine[i] = src[i];
    __syncthreads();
    if (threadIdx.x < a.world) {
      __threadfence_system();
      hi_st_release_sys(ar_flags(a, threadIdx.x) + b * kARMaxRanks + a.rank, e);
      // wrap-safe "flag < e" (the epoch counter wraps after 2^32 pieces)
      while ((int)(hi_ld_acquire_sys(ar_flags(a, a.rank) + b * kARMaxRanks + threadIdx.x) - e) < 0) hi_backoff();
    }
    __syncthreads();
    for (int i = threadIdx.x; i < m; i += blockDim.x) {
      // rank r's copy of vector i (this rank's own from its source, the others' from their staging)
      auto load = [&](int r) { return r == a.rank ? src[i] : hi_ld_remote(reinterpret_cast<const uint4*>(a.stage[r] + so) + i); };
      uint4 res;
      if constexpr (OP == 0) {
        float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        for (int r = 0; r < a.world; r++) {
          const uint4 v = load(r);
          const __half2* h = reinterpret_cast<const __half2*>(&v);
#pragma unroll
          for (int j = 0; j < 4; j++) {
            const float2 f = __half22float2(h[j]);
            acc[2 * j] += f.x; acc[2 * j + 1] += f.y;
          }
        }
        __half2* o = reinterpret_cast<__half2*>(&res);
#pragma unroll
        for (int j = 0; j < 4; j++) o[j] = __floats2half2_rn(acc[2 * j], acc[2 * j + 1]);
      } else {
        unsigned long long m0 = 0, m1 = 0;
        for (int r = 0; r < a.world; r++) {
          const uint4 v = load(r);
          const unsigned long long* u = reinterpret_cast<const unsigned long long*>(&v);
          m0 = u[0] > m0 ? u[0] : m0;
          m1 = u[1] > m1 ? u[1] : m1;
        }
        unsigned long long* o = reinterpret_cast<unsigned long long*>(&res);
        o[0] = m0; o[1] = m1;
      }
      data[p0 + i] = res;
    }
    p0 += m;
  } while (p0 < v1);
  if (threadIdx.x == 0) a.epoch[b] = e;
}

// =================================================================================
// Mixture of Experts (docs/moe-design.md).
// Everything from here to the end of the file is checked by tests/test_kernels.py test_moe().
// Rounding follows HF's default (grouped_mm) experts path: router logits rounded to fp16, each
// expert's down projection rounded to fp16, times its routing weight (an fp16-rounded product
// when the model rounds its weights to fp16), summed over the k experts in fp32, rounded once,
// optionally scaled (Granite's residual multiplier, rounded again), added to the residual.
// =================================================================================
constexpr int kMoeMaxE = 256, kMoeMaxK = 16;
constexpr int kMoeNormTopk = 1, kMoeFp16Weights = 2;  // moe_flags bits (weight-file header)

// NR dot products x[r] . w[r] over K8 uint4 (8 halves each), one warp: 16-byte loads, 4 per lane
// in flight, fp32 accumulation. Every lane gets the warp-reduced sums.
template <int NR>
__device__ __forceinline__ void warp_dot(const uint4* const (&x)[NR], const uint4* const (&w)[NR], int K8, float (&acc)[NR]) {
  constexpr int U = 4;
  const int lane = threadIdx.x % HI_WARP;
#pragma unroll
  for (int r = 0; r < NR; r++) acc[r] = 0.f;
  for (int base = 0; base < K8; base += HI_WARP * U) {
    uint4 wv[NR][U], xv[NR][U];
#pragma unroll
    for (int u = 0; u < U; u++) {
      const int i = base + u * HI_WARP + lane;
#pragma unroll
      for (int r = 0; r < NR; r++) {
        wv[r][u] = i < K8 ? __ldg(w[r] + i) : make_uint4(0, 0, 0, 0);
        xv[r][u] = i < K8 ? x[r][i] : make_uint4(0, 0, 0, 0);
      }
    }
#pragma unroll
    for (int u = 0; u < U; u++)
#pragma unroll
      for (int r = 0; r < NR; r++) acc[r] += dot8(wv[r][u], xv[r][u]);
  }
#pragma unroll
  for (int r = 0; r < NR; r++) acc[r] = warp_sum(acc[r]);
}

// ---------------------------------------------------------------------------------
// Router + top-k + routing weights, one block per token. Logits l = fp16(xn . Wr[e])
// (one warp per expert), then warp 0 picks the k largest in k rounds of a warp argmax (NaN above
// numbers, ties to the lower index: argmax_better), in descending order. Weights: softmax over the
// k (kMoeNormTopk; Mixtral's renormalised top-k and Granite's top-k softmax are the same thing) or
// over all E (OLMoE), rounded to fp16 with kMoeFp16Weights. Writes idx[t*k + j], w[t*k + j].
// E <= kMoeMaxE, k <= kMoeMaxK, H % 8 == 0.
// ---------------------------------------------------------------------------------
__global__ void __launch_bounds__(256)
moe_router_kernel(const half* __restrict__ xn, int ldx, const half* __restrict__ Wr, int E, int H, int k, int flags,
                  int* __restrict__ idx_out, float* __restrict__ w_out) {
  __shared__ float logit[kMoeMaxE];
  const int t = blockIdx.x, lane = threadIdx.x % HI_WARP, warp = threadIdx.x / HI_WARP, nw = blockDim.x / HI_WARP;
  const uint4* xr = reinterpret_cast<const uint4*>(xn + (size_t)t * ldx);
  for (int ex = warp; ex < E; ex += nw) {
    const uint4* const x[1] = {xr};
    const uint4* const w[1] = {reinterpret_cast<const uint4*>(Wr + (size_t)ex * H)};
    float acc[1];
    warp_dot<1>(x, w, H / 8, acc);
    if (lane == 0) logit[ex] = __half2float(__float2half(acc[0]));  // HF rounds the router linear to fp16
  }
  __syncthreads();
  if (warp != 0) return;
  constexpr int S = kMoeMaxE / HI_WARP;  // logits per lane
  float v[S];
#pragma unroll
  for (int i = 0; i < S; i++) { const int ex = i * HI_WARP + lane; v[i] = ex < E ? logit[ex] : HI_NEG_INF; }
  unsigned taken = 0;  // bit i: this lane's expert i * HI_WARP + lane is already selected
  int sel_i[kMoeMaxK];
  float sel_v[kMoeMaxK];
  for (int j = 0; j < k; j++) {
    float best = HI_NEG_INF;
    int bi = 0x7fffffff;  // above every expert index, so any untaken expert beats it
#pragma unroll
    for (int i = 0; i < S; i++) {
      const int ex = i * HI_WARP + lane;
      if (ex < E && !(taken >> i & 1) && argmax_better(v[i], ex, best, bi)) { best = v[i]; bi = ex; }
    }
    for (int o = HI_WARP / 2; o > 0; o >>= 1) {
      const float ov = hi_shfl_xor(best, o);
      const int oi = hi_shfl_xor_i(bi, o);
      if (argmax_better(ov, oi, best, bi)) { best = ov; bi = oi; }
    }
    // every lane now holds the same winner; its owner marks it taken
    if (bi % HI_WARP == lane) taken |= 1u << (bi / HI_WARP);
    sel_i[j] = bi;
    sel_v[j] = best;
  }
  const float m = sel_v[0];
  float denom = 0.f;
  if (flags & kMoeNormTopk) {
    for (int j = 0; j < k; j++) denom += expf(sel_v[j] - m);
  } else {
    float s = 0.f;
    for (int ex = lane; ex < E; ex += HI_WARP) s += expf(logit[ex] - m);
    denom = warp_sum(s);
  }
  if (lane == 0)
    for (int j = 0; j < k; j++) {
      float wj = expf(sel_v[j] - m) / denom;
      if (flags & kMoeFp16Weights) wj = __half2float(__float2half(wj));
      idx_out[(size_t)t * k + j] = sel_i[j];
      w_out[(size_t)t * k + j] = wj;
    }
}

// ---------------------------------------------------------------------------------
// Small-batch (decode) experts as GEMVs over the selected experts only. Pair
// p = token * k + j runs expert idx[p] on token p / k; grid.y = pairs, so the launch shape is
// fixed and the expert choice stays on the device (CUDA-graph capturable).
// Gate/up + SwiGLU: one warp per (pair, unit o): weight rows r0 = (o/16)*32 + o%16 and r0 + 16
// of the expert's interleaved wgu [2I, H]; act[p, o] = silu(gate) * up, rounded like silu_mul.
// ---------------------------------------------------------------------------------
__global__ void __launch_bounds__(256)
moe_gemv_gu_kernel(const half* __restrict__ xn, int ldx, const half* __restrict__ wgu, const int* __restrict__ idx, int k,
                   int H, int I, half* __restrict__ act) {
  const int p = blockIdx.y, lane = threadIdx.x % HI_WARP, warp = threadIdx.x / HI_WARP, nw = blockDim.x / HI_WARP;
  const half* W = wgu + (size_t)idx[p] * 2 * I * H;
  const uint4* xr = reinterpret_cast<const uint4*>(xn + (size_t)(p / k) * ldx);
  for (int o = blockIdx.x * nw + warp; o < I; o += gridDim.x * nw) {
    const int r0 = (o / 16) * 32 + (o % 16);
    const uint4* const x[2] = {xr, xr};
    const uint4* const w[2] = {reinterpret_cast<const uint4*>(W + (size_t)r0 * H),
                               reinterpret_cast<const uint4*>(W + (size_t)(r0 + 16) * H)};
    float acc[2];
    warp_dot<2>(x, w, H / 8, acc);
    if (lane == 0) act[(size_t)p * I + o] = silu_mul(acc[0], acc[1]);
  }
}

// Down projection + combine + residual, one warp per (token b, hidden column h): the
// k dot products wdown[idx[b,j]][h] . act[b*k + j] (4 at a time), each rounded to fp16, weighted,
// summed over j in fp32, rounded, scaled, and added to x[b, h]. No atomics; deterministic.
__global__ void __launch_bounds__(256)
moe_gemv_down_kernel(const half* __restrict__ act, const half* __restrict__ wdown, const int* __restrict__ idx,
                     const float* __restrict__ wts, int k, int H, int I, int flags, float scale, half* x, int ldx) {
  const int b = blockIdx.y, lane = threadIdx.x % HI_WARP, warp = threadIdx.x / HI_WARP, nw = blockDim.x / HI_WARP;
  for (int h = blockIdx.x * nw + warp; h < H; h += gridDim.x * nw) {
    float sum = 0.f;
    for (int j0 = 0; j0 < k; j0 += 4) {
      const uint4* xs[4];
      const uint4* ws[4];
#pragma unroll
      for (int r = 0; r < 4; r++) {
        const int p = b * k + (j0 + r < k ? j0 + r : j0);  // past k: repeat a valid row, result unused
        xs[r] = reinterpret_cast<const uint4*>(act + (size_t)p * I);
        ws[r] = reinterpret_cast<const uint4*>(wdown + ((size_t)idx[p] * H + h) * I);
      }
      const uint4* const x4[4] = {xs[0], xs[1], xs[2], xs[3]};
      const uint4* const w4[4] = {ws[0], ws[1], ws[2], ws[3]};
      float acc[4];
      warp_dot<4>(x4, w4, I / 8, acc);
#pragma unroll
      for (int r = 0; r < 4; r++)
        if (j0 + r < k) {
          const float c = __half2float(__float2half(acc[r])) * wts[(size_t)b * k + j0 + r];
          sum += (flags & kMoeFp16Weights) ? __half2float(__float2half(c)) : c;
        }
    }
    if (lane == 0) {
      half o = __float2half(sum);
      if (scale != 1.f) o = __float2half(__half2float(o) * scale);
      half* dst = x + (size_t)b * ldx + h;
      *dst = __hadd(*dst, o);
    }
  }
}

// ---------------------------------------------------------------------------------
// Grouped (prefill / batched) path, part 1: counting sort of the n = M*k (token, slot) pairs by
// expert, one block. off[E+1] = exclusive prefix of the counts; perm[q] = the pair at
// sorted position q; inv[pair] = q. The order within one expert follows the shared-memory atomics,
// which changes no result (every row of a GEMM is computed independently of the others).
// ---------------------------------------------------------------------------------
__global__ void __launch_bounds__(1024)
moe_sort_kernel(const int* __restrict__ idx, int n, int E, int* __restrict__ off, int* __restrict__ perm,
                int* __restrict__ inv) {
  __shared__ int cnt[kMoeMaxE], cur[kMoeMaxE];
  for (int e = threadIdx.x; e < E; e += blockDim.x) cnt[e] = 0;
  __syncthreads();
  for (int p = threadIdx.x; p < n; p += blockDim.x) atomicAdd(&cnt[idx[p]], 1);
  __syncthreads();
  if (threadIdx.x == 0) {
    int s = 0;
    for (int e = 0; e < E; e++) { off[e] = s; cur[e] = s; s += cnt[e]; }
    off[E] = s;
  }
  __syncthreads();
  for (int p = threadIdx.x; p < n; p += blockDim.x) {
    const int q = atomicAdd(&cur[idx[p]], 1);
    perm[q] = p;
    inv[p] = q;
  }
}

// Part 2: rows[q] = xn[perm[q] / k], so each expert's tokens are contiguous.
__global__ void moe_gather_kernel(const half* __restrict__ xn, int ldx, const int* __restrict__ perm, int k, int H,
                                  half* __restrict__ rows) {
  const int q = blockIdx.x;
  const uint4* src = reinterpret_cast<const uint4*>(xn + (size_t)(perm[q] / k) * ldx);
  uint4* dst = reinterpret_cast<uint4*>(rows + (size_t)q * H);
  for (int i = threadIdx.x; i < H / 8; i += blockDim.x) dst[i] = src[i];
}

// Part 3 (after the per-expert GEMMs wrote y[q] = expert output of sorted row q, fp16): token t's
// weighted sum over its k rows y[inv[t*k + j]], rounded as in moe_gemv_down_kernel, added to x[t]
//. One block per token, 8 columns per thread.
__global__ void moe_combine_kernel(const half* __restrict__ y, const int* __restrict__ inv, const float* __restrict__ wts,
                                   int k, int H, int flags, float scale, half* x, int ldx) {
  const int t = blockIdx.x;
  uint4* xr = reinterpret_cast<uint4*>(x + (size_t)t * ldx);
  for (int i = threadIdx.x; i < H / 8; i += blockDim.x) {
    float s[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int j = 0; j < k; j++) {
      const float w = wts[(size_t)t * k + j];
      H8 v;
      v.u = reinterpret_cast<const uint4*>(y + (size_t)inv[(size_t)t * k + j] * H)[i];
#pragma unroll
      for (int u = 0; u < 8; u++) {
        const float c = __half2float(v.h[u]) * w;
        s[u] += (flags & kMoeFp16Weights) ? __half2float(__float2half(c)) : c;
      }
    }
    H8 xv;
    xv.u = xr[i];
#pragma unroll
    for (int u = 0; u < 8; u++) {
      half o = __float2half(s[u]);
      if (scale != 1.f) o = __float2half(__half2float(o) * scale);
      xv.h[u] = __hadd(xv.h[u], o);
    }
    xr[i] = xv.u;
  }
}
