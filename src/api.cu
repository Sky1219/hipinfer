// Host side: kernel launchers (exported with a C ABI so tests can drive each kernel from
// Python on torch tensors) and the Llama-family engine (weights, KV cache, prefill,
// graph-captured decode).
#include "kernels.cuh"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <vector>

#define EXPORT extern "C" __attribute__((visibility("default")))

// Device properties and the per-device state below are thread_local and indexed by device: a
// tensor-parallel engine drives each GPU from its own worker thread, and a single-GPU caller
// may switch between GPUs on one thread. hi_init() selects the current device's entry.
static const int kMaxDev = 64;
struct DevProps { int warp, sms, smem_optin, cc; bool init = false; };
static thread_local DevProps g_props[kMaxDev];
static thread_local int g_dev = -1;
static thread_local int g_warp = 32, g_sms = 80, g_smem_optin = 48 * 1024, g_cc = 75;
static void hi_init() {
  int dev; HI_CHECK(hipGetDevice(&dev));
  if (dev == g_dev) return;
  if (dev >= kMaxDev) { fprintf(stderr, "hipinfer: device %d >= %d not supported\n", dev, kMaxDev); exit(1); }
  DevProps& d = g_props[dev];
  if (!d.init) {
    hipDeviceProp_t p; HI_CHECK(hipGetDeviceProperties(&p, dev));
    d.warp = p.warpSize; d.sms = p.multiProcessorCount; d.cc = p.major * 10 + p.minor;
    HI_CHECK(hipDeviceGetAttribute(&d.smem_optin, HI_ATTR_SMEM_OPTIN, dev));
    d.init = true;
  }
  g_dev = dev; g_warp = d.warp; g_sms = d.sms; g_smem_optin = d.smem_optin; g_cc = d.cc;
}

// Make device d current for a scope (restoring the caller's device afterwards).
struct DevGuard {
  int prev, d;
  explicit DevGuard(int dev) : d(dev) { HI_CHECK(hipGetDevice(&prev)); if (prev != d) HI_CHECK(hipSetDevice(d)); }
  ~DevGuard() { if (prev != d) (void)hipSetDevice(prev); }
};
// Largest dynamic shared-memory allocation a block may opt into (64 KB on Turing, ~99 KB on Ampere).
EXPORT int hi_max_smem() { hi_init(); return g_smem_optin; }
// Can the fused-RMSNorm GEMV hold M rows of K halves?
static bool norm_fits(int M, int K) { return (size_t)M * K * 2 + 1024 <= (size_t)g_smem_optin; }
#define ST(s) ((hipStream_t)(s))

// Split-K workspace: fp32 partial planes (grown only outside graph capture), plus a separate
// fixed buffer of per-output-tile counters that start at zero and are left zeroed by the
// kernel (kept apart so partial planes of one shape never land on another's counters).
// Captured CUDA graphs hold the workspace pointer they were captured with, so a buffer that is
// outgrown is retired, not freed, until the thread exits (growth at least doubles, so retired
// buffers add up to less than the live one).
struct SplitWs {
  float* ws = nullptr;
  size_t ws_bytes = 0;
  int* cnt = nullptr;
  std::vector<float*> retired;
  ~SplitWs() {  // a tensor-parallel worker thread exits with its engine; release its GPU's buffers
    if (ws) (void)hipFree(ws);
    if (cnt) (void)hipFree(cnt);
    for (float* p : retired) (void)hipFree(p);
  }
};
static thread_local SplitWs g_split_dev[kMaxDev];
#define g_split (g_split_dev[g_dev])
static const int kMaxTiles = 1 << 16;

// ------------------------------------------------------------------ launchers
EXPORT void hi_rmsnorm(const void* x, const void* w, void* y, int M, int H, float eps, void* s) {
  hi_init();
  rmsnorm_kernel<<<M, 256, 0, ST(s)>>>((const half*)x, (const half*)w, (half*)y, H, eps);
}

template <int MR, bool NORM, int EPI>
static void gemv_launch(const half* x, int ldx, const half* nw, float eps, const half* W,
                        half* out, int ldo, int N, int K, hipStream_t s, RopeArgs ra = {}) {
  const int warps = 256 / g_warp, n_out = EPI >= 2 ? N / 2 : N;
  const size_t sm = NORM ? (size_t)MR * K * sizeof(half) : 0;
  // One wave, balanced: every warp gets the same number of output rows, and the grid fits in
  // the resident block slots (the fused-norm smem can limit those), so the norm prologue is
  // paid once per block and no SM is left with an extra row.
  static thread_local int per_sm_dev[kMaxDev];
  static thread_local size_t per_sm_key[kMaxDev];  // sm + 1 the entry was computed for (0: none)
  int& per_sm = per_sm_dev[g_dev];
  if (per_sm_key[g_dev] != sm + 1) {
    if (sm + 1024 > 48 * 1024) {  // the kernel also has a little static smem; opt in above 48 KB total
      if (sm + 1024 > (size_t)g_smem_optin) {
        fprintf(stderr, "gemv: fused RMSNorm needs %zu B of shared memory (device max %d); use the unfused path\n",
                sm + 1024, g_smem_optin);
        exit(1);
      }
      HI_CHECK(hipFuncSetAttribute((const void*)gemv_kernel<MR, NORM, EPI>,
                                   hipFuncAttributeMaxDynamicSharedMemorySize, (int)sm));
    }
    HI_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, gemv_kernel<MR, NORM, EPI>, 256, sm));
    if (per_sm < 1) per_sm = 1;
    per_sm_key[g_dev] = sm + 1;
  }
  const int slots = per_sm * g_sms * warps;
  const int rows_per_warp = (n_out + slots - 1) / slots;
  const int grid = (n_out + warps * rows_per_warp - 1) / (warps * rows_per_warp);
  gemv_kernel<MR, NORM, EPI><<<grid, 256, sm, s>>>(x, ldx, nw, eps, W, N, K, out, ldo, ra);
}

template <int MR, bool NORM>
static void gemv_epi(const half* x, int ldx, const half* nw, float eps, const half* W, half* out,
                     int ldo, int N, int K, int epi, hipStream_t s) {
  if (epi == 0) gemv_launch<MR, NORM, 0>(x, ldx, nw, eps, W, out, ldo, N, K, s);
  else if (epi == 1) gemv_launch<MR, NORM, 1>(x, ldx, nw, eps, W, out, ldo, N, K, s);
  else gemv_launch<MR, NORM, 2>(x, ldx, nw, eps, W, out, ldo, N, K, s);
}

template <int MR>
static void gemv_mr(const half* x, int ldx, const half* nw, float eps, const half* W, half* out,
                    int ldo, int N, int K, int epi, hipStream_t s) {
  if (nw) gemv_epi<MR, true>(x, ldx, nw, eps, W, out, ldo, N, K, epi, s);
  else gemv_epi<MR, false>(x, ldx, nw, eps, W, out, ldo, N, K, epi, s);
}

// Widest GEMV launch: 8 rows with the RMSNorm fused (its shared memory holds the normalized
// rows), kGemvMaxRows without (speculative verification of up to 15 draft tokens in one weight
// pass). A row's arithmetic does not depend on the row count: row m of an M-row GEMV is
// bit-identical to a 1-row GEMV of that row.
static const int kGemvMaxRows = 16;

// out[M, *] (epi) x[M, K] @ W[N, K]^T for small M, optionally RMSNorm-ing x first (normw != null)
EXPORT void hi_gemv(const void* x, int ldx, const void* normw, float eps, const void* W, void* out,
                    int ldo, int M, int N, int K, int epi, void* s) {
  hi_init();
  const int cap = normw ? 8 : kGemvMaxRows;
  for (int m0 = 0; m0 < M; m0 += cap) {
    const int mr = M - m0 < cap ? M - m0 : cap;
    const half* xp = (const half*)x + (size_t)m0 * ldx;
    half* op = (half*)out + (size_t)m0 * ldo;
    const half* nw = (const half*)normw;
    const half* w = (const half*)W;
#define HI_GW(m) case m: gemv_epi<m, false>(xp, ldx, nw, eps, w, op, ldo, N, K, epi, ST(s)); break
    switch (mr) {
      case 1: gemv_mr<1>(xp, ldx, nw, eps, w, op, ldo, N, K, epi, ST(s)); break;
      case 2: gemv_mr<2>(xp, ldx, nw, eps, w, op, ldo, N, K, epi, ST(s)); break;
      case 3: gemv_mr<3>(xp, ldx, nw, eps, w, op, ldo, N, K, epi, ST(s)); break;
      case 4: gemv_mr<4>(xp, ldx, nw, eps, w, op, ldo, N, K, epi, ST(s)); break;
      case 5: gemv_mr<5>(xp, ldx, nw, eps, w, op, ldo, N, K, epi, ST(s)); break;
      case 6: gemv_mr<6>(xp, ldx, nw, eps, w, op, ldo, N, K, epi, ST(s)); break;
      case 7: gemv_mr<7>(xp, ldx, nw, eps, w, op, ldo, N, K, epi, ST(s)); break;
      case 8: gemv_mr<8>(xp, ldx, nw, eps, w, op, ldo, N, K, epi, ST(s)); break;
      HI_GW(9); HI_GW(10); HI_GW(11); HI_GW(12); HI_GW(13); HI_GW(14); HI_GW(15); HI_GW(16);
    }
#undef HI_GW
  }
}

// Fused QKV GEMV (hi_gemv_qkv_rope); kv_slot >= 0 sends every row's k/v to that one slot.
static void gemv_qkv_rope(const half* x, int ldx, const half* nw, float eps, const half* w, const half* bias, half* o,
                          int ldo, int M, int K, int Hq, int Hkv, int D, const __half2* cs, const int* pos, half* kc,
                          half* vc, int max_seq, int kv_slot, hipStream_t s) {
  const int N = (Hq + 2 * Hkv) * D;
  RopeArgs ra{cs, pos, kc, vc, bias, max_seq, Hq, Hkv, D, kv_slot};
#define HI_QKV(mr) case mr: if (nw) gemv_launch<mr, true, 3>(x, ldx, nw, eps, w, o, ldo, N, K, s, ra); \
                            else gemv_launch<mr, false, 3>(x, ldx, nw, eps, w, o, ldo, N, K, s, ra); break
#define HI_QKV_NN(mr) case mr: gemv_launch<mr, false, 3>(x, ldx, nw, eps, w, o, ldo, N, K, s, ra); break
  if (nw && M > 8) { fprintf(stderr, "hi_gemv_qkv_rope: M must be 1..8 with the fused RMSNorm\n"); exit(1); }
  switch (M) {
    HI_QKV(1); HI_QKV(2); HI_QKV(3); HI_QKV(4); HI_QKV(5); HI_QKV(6); HI_QKV(7); HI_QKV(8);
    HI_QKV_NN(9); HI_QKV_NN(10); HI_QKV_NN(11); HI_QKV_NN(12); HI_QKV_NN(13); HI_QKV_NN(14); HI_QKV_NN(15);
    HI_QKV_NN(16);
    default: fprintf(stderr, "hi_gemv_qkv_rope: M must be 1..%d\n", kGemvMaxRows); exit(1);
  }
#undef HI_QKV
#undef HI_QKV_NN
}

// QKV projection (RMSNorm fused) with bias, RoPE and the KV-cache write fused into the epilogue.
// q goes to out[m, 0:Hq*D]; k/v go to cache slot m at position pos[m]. M <= 8 with normw, <= 16 without.
EXPORT void hi_gemv_qkv_rope(const void* x, int ldx, const void* normw, float eps, const void* W, const void* bias,
                             void* out, int ldo, int M, int K, int Hq, int Hkv, int D, const void* cs, const int* pos,
                             void* kc, void* vc, int max_seq, void* s) {
  hi_init();
  gemv_qkv_rope((const half*)x, ldx, (const half*)normw, eps, (const half*)W, (const half*)bias, (half*)out, ldo, M, K,
                Hq, Hkv, D, (const __half2*)cs, pos, (half*)kc, (half*)vc, max_seq, -1, ST(s));
}

template <int BM, int BN, int BK, int WGM, int WGN, int EPI>
static void gemm_launch(const half* A, int lda, const half* W, const half* bias, half* C, int ldc, int M, int N, int K,
                        int splits, float* ws, hipStream_t s) {
  constexpr int smem = 2 * (BM + BN) * (BK + 8) * 2;
  static thread_local unsigned long long attr = 0;  // bit d: set on device d (never inside graph capture: warmed up eagerly)
  if (!(attr >> g_dev & 1) && smem > 48 * 1024) {
    HI_CHECK(hipFuncSetAttribute((const void*)gemm_kernel<BM, BN, BK, WGM, WGN, EPI>,
                                 hipFuncAttributeMaxDynamicSharedMemorySize, smem));
  }
  attr |= 1ull << g_dev;
  dim3 grid(((N + BN - 1) / BN) * ((M + BM - 1) / BM), 1, splits);
  gemm_kernel<BM, BN, BK, WGM, WGN, EPI><<<grid, 8 * g_warp, smem, s>>>(A, lda, W, bias, C, ldc, M, N, K, ws, g_split.cnt);
}

#if !defined(HI_AMD)
// mma.sync GEMM: 3-stage cp.async pipeline on sm_80+, register double buffering on Turing.
static int gemm_mma_stages() { return g_cc >= 80 ? 3 : 2; }

template <int BM, int BN, int BK, int WGM, int WGN, int STAGES, int EPI>
static void gemm_mma_launch_st(const half* A, int lda, const half* W, const half* bias, half* C, int ldc, int M, int N,
                               int K, int splits, float* ws, hipStream_t s) {
  constexpr int smem = STAGES * (BM + BN) * (BK + 8) * 2;
  static thread_local unsigned long long attr = 0;  // bit d: set on device d
  if (!(attr >> g_dev & 1) && smem > 48 * 1024) {
    HI_CHECK(hipFuncSetAttribute((const void*)gemm_mma_kernel<BM, BN, BK, WGM, WGN, STAGES, EPI>,
                                 hipFuncAttributeMaxDynamicSharedMemorySize, smem));
  }
  attr |= 1ull << g_dev;
  dim3 grid(((N + BN - 1) / BN) * ((M + BM - 1) / BM), 1, splits);
  gemm_mma_kernel<BM, BN, BK, WGM, WGN, STAGES, EPI><<<grid, WGM * WGN * 32, smem, s>>>(A, lda, W, bias, C, ldc, M, N, K, ws, g_split.cnt);
}
template <int BM, int BN, int BK, int WGM, int WGN, int EPI>
static void gemm_mma_launch(const half* A, int lda, const half* W, const half* bias, half* C, int ldc, int M, int N, int K,
                            int splits, float* ws, hipStream_t s) {
  if (gemm_mma_stages() >= 3) gemm_mma_launch_st<BM, BN, BK, WGM, WGN, 3, EPI>(A, lda, W, bias, C, ldc, M, N, K, splits, ws, s);
  else gemm_mma_launch_st<BM, BN, BK, WGM, WGN, 2, EPI>(A, lda, W, bias, C, ldc, M, N, K, splits, ws, s);
}
#endif

// Tile configs {BM, BN, BK, warp grid}: 0-5 portable WMMA (double-buffered through registers),
// 6-17 NVIDIA mma.sync (ldmatrix + cp.async pipeline; 6-12 eight warps, 13-17 four warps),
// 18-25 NVIDIA weight-streaming "skinny" kernels for M <= 64 (bn = columns per block), 26 NVIDIA
// mma.sync 256x128 with a 2x4 warp grid (9's tile; its 32-column warp tiles also run the SwiGLU
// epilogue, which 9 cannot). The autotuner picks per shape.
struct GemmCfg { int bm, bn, bk, wgm, wgn; bool mma; int skinny, ksplit; };
static const GemmCfg kCfg[] = {
    {128, 128, 32, 2, 4, false, 0, 0}, {64, 128, 32, 2, 4, false, 0, 0}, {64, 64, 32, 4, 2, false, 0, 0}, {256, 128, 32, 4, 2, false, 0, 0},
    {128, 64, 64, 4, 2, false, 0, 0},  {64, 128, 64, 2, 4, false, 0, 0},
    {128, 128, 32, 2, 4, true, 0, 0},  {64, 128, 32, 2, 4, true, 0, 0},  {64, 64, 32, 4, 2, true, 0, 0},  {256, 128, 32, 4, 2, true, 0, 0},
    {128, 64, 64, 4, 2, true, 0, 0},   {64, 128, 64, 2, 4, true, 0, 0},  {128, 256, 32, 2, 4, true, 0, 0},
    {128, 128, 32, 2, 2, true, 0, 0},  {64, 128, 32, 2, 2, true, 0, 0},  {64, 64, 32, 2, 2, true, 0, 0},  {64, 128, 32, 1, 4, true, 0, 0},
    {128, 64, 32, 2, 2, true, 0, 0},
    // skinny: bn = columns per block, bk = k granularity (128 * k-slices per block)
    {64, 32, 128, 4, 1, true, 8, 1},   {64, 64, 128, 4, 1, true, 16, 1}, {64, 128, 128, 4, 1, true, 32, 1},
    {64, 32, 256, 8, 1, true, 8, 2},   {64, 64, 256, 8, 1, true, 16, 2}, {64, 128, 256, 8, 1, true, 32, 2},
    {64, 32, 512, 16, 1, true, 8, 4},  {64, 64, 512, 16, 1, true, 16, 4},
    {256, 128, 32, 2, 4, true, 0, 0}};
#if defined(HI_AMD)
static const int kNumCfg = 6;
#else
static const int kNumCfg = 27;
#endif
static int g_gemm_force = -1;  // benchmarking hook: force a tile config
EXPORT void hi_gemm_force_config(int c) { g_gemm_force = c; }
EXPORT int hi_gemm_num_configs() { return kNumCfg; }
EXPORT void hi_gemm_config_desc(int c, char* out, int n) {
  const GemmCfg& k = kCfg[c];
  if (k.skinny) snprintf(out, n, "skinny mma.sync, %d weight rows per warp, %d k-slices", k.skinny, k.ksplit);
  else snprintf(out, n, "%s %dx%dx%d warps %dx%d", k.mma ? "mma.sync" : "wmma", k.bm, k.bn, k.bk, k.wgm, k.wgn);
}

static size_t cfg_smem(int c) {
  const GemmCfg& k = kCfg[c];
  if (k.skinny) return 0;
#if defined(HI_AMD)
  const int stages = 2;
#else
  const int stages = k.mma ? gemm_mma_stages() : 2;
#endif
  return (size_t)stages * (k.bm + k.bn) * (k.bk + 8) * 2;
}

#if !defined(HI_AMD)
template <int NW, int KSPLIT, int EPI>
static void gemm_skinny_launch(const half* A, int lda, const half* W, const half* bias, half* C, int ldc, int M, int N, int K,
                               int splits, float* ws, hipStream_t s) {
  dim3 grid((N + 4 * NW - 1) / (4 * NW), 1, splits);
  const int MT = (M + 15) / 16;
  // shared memory holds KSPLIT k-slices of the A chunk: MT * KSPLIT <= 4 (40 KB)
#define HI_SK(mt) case mt: if constexpr (mt * KSPLIT <= 4) { gemm_skinny_kernel<mt, NW, KSPLIT, EPI><<<grid, 128 * KSPLIT, 0, s>>>(A, lda, W, bias, C, ldc, M, N, K, ws, g_split.cnt); break; } [[fallthrough]];
  switch (MT) { HI_SK(1) HI_SK(2) HI_SK(3) HI_SK(4) default: fprintf(stderr, "skinny gemm: unsupported M %d for %d k-slices\n", M, KSPLIT); exit(1); }
#undef HI_SK
}
#endif

// Fallback choice (graph capture before tuning, or autotuning disabled).
static int gemm_pick(int M, int N, int K, int epi) {
#if !defined(HI_AMD)
  const int b256 = ((M + 255) / 256) * ((N + 127) / 128);
  if (g_cc < 80) {
    // Turing, from tile sweeps of its overlapped 2-stage loop on a 2080 Ti (3B shapes, M = 128 to
    // 2048, the engine's epilogues): 256x128 from M = 512 once that is >= 2 waves (26, its
    // 2x4-warp twin, under SwiGLU, which 9 cannot run); else the BK=64 64x128 tile, which beats
    // 6 by 1.3-2.6x at M = 128 and runs every epilogue
    if (M >= 512 && b256 >= 2 * g_sms && cfg_smem(9) + 1024 <= (size_t)g_smem_optin) return epi == 2 ? 26 : 9;
    if (M > 64) return 11;
  }
  if (b256 >= 2 * g_sms && cfg_smem(9) + 1024 <= (size_t)g_smem_optin) return 9;
  if (M > 64) return 6;
  return 8;
#else
  // from the WMMA tile sweep on a 2080 Ti: BK=64 tiles win almost everywhere; 256x128 once it
  // still fills >= 4 blocks per SM; narrow 64x64 for decode-sized M unless N or K is large
  const int b256 = ((M + 255) / 256) * ((N + 127) / 128);
  if (b256 >= 4 * g_sms) return 3;
  if (M > 64) return 4;
  if (N >= 8192) return 5;
  if (K >= 4096) return 4;
  return 2;
#endif
}

template <int EPI>
static void gemm_cfg(int cfg, int sp, float* ws, const half* A, int lda, const half* W, const half* bias, half* C, int ldc,
                     int M, int N, int K, hipStream_t s) {
#define HI_G(BM, BN, BK, GM, GN) gemm_launch<BM, BN, BK, GM, GN, EPI>(A, lda, W, bias, C, ldc, M, N, K, sp, ws, s)
#define HI_M(BM, BN, BK, GM, GN) gemm_mma_launch<BM, BN, BK, GM, GN, EPI>(A, lda, W, bias, C, ldc, M, N, K, sp, ws, s)
  switch (cfg) {
    case 0: HI_G(128, 128, 32, 2, 4); break;
    case 1: HI_G(64, 128, 32, 2, 4); break;
    case 2: HI_G(64, 64, 32, 4, 2); break;
    case 3: if constexpr (EPI != 2) HI_G(256, 128, 32, 4, 2); break;
    case 4: HI_G(128, 64, 64, 4, 2); break;
    case 5: HI_G(64, 128, 64, 2, 4); break;
#if !defined(HI_AMD)
    case 6: HI_M(128, 128, 32, 2, 4); break;
    case 7: HI_M(64, 128, 32, 2, 4); break;
    case 8: HI_M(64, 64, 32, 4, 2); break;
    case 9: if constexpr (EPI != 2) HI_M(256, 128, 32, 4, 2); break;
    case 10: HI_M(128, 64, 64, 4, 2); break;
    case 11: HI_M(64, 128, 64, 2, 4); break;
    case 12: if constexpr (EPI != 2) HI_M(128, 256, 32, 2, 4); break;
    case 13: if constexpr (EPI != 2) HI_M(128, 128, 32, 2, 2); break;
    case 14: if constexpr (EPI != 2) HI_M(64, 128, 32, 2, 2); break;
    case 15: HI_M(64, 64, 32, 2, 2); break;
    case 16: HI_M(64, 128, 32, 1, 4); break;
    case 17: HI_M(128, 64, 32, 2, 2); break;
#define HI_S(NW, KS) gemm_skinny_launch<NW, KS, EPI>(A, lda, W, bias, C, ldc, M, N, K, sp, ws, s)
    case 18: if constexpr (EPI != 2) HI_S(8, 1); break;
    case 19: if constexpr (EPI != 2) HI_S(16, 1); break;
    case 20: HI_S(32, 1); break;
    case 21: if constexpr (EPI != 2) HI_S(8, 2); break;
    case 22: if constexpr (EPI != 2) HI_S(16, 2); break;
    case 23: HI_S(32, 2); break;
    case 24: if constexpr (EPI != 2) HI_S(8, 4); break;
    case 25: if constexpr (EPI != 2) HI_S(16, 4); break;
#undef HI_S
    case 26: HI_M(256, 128, 32, 2, 4); break;
#endif
    default: fprintf(stderr, "bad gemm cfg %d\n", cfg); exit(1);
  }
#undef HI_G
#undef HI_M
}

static size_t split_ws_bytes(int cfg, int sp, int M, int N) {
  if (sp <= 1) return 0;
  if (kCfg[cfg].skinny) {
    const size_t tn = (N + kCfg[cfg].bn - 1) / kCfg[cfg].bn;
    if (tn > (size_t)kMaxTiles) return ~(size_t)0;
    return (size_t)sp * ((M + 15) / 16 * 16) * N * 4;
  }
  const size_t tm = (M + kCfg[cfg].bm - 1) / kCfg[cfg].bm, tn = (N + kCfg[cfg].bn - 1) / kCfg[cfg].bn;
  if (tm * tn > (size_t)kMaxTiles) return ~(size_t)0;
  return (size_t)sp * tm * kCfg[cfg].bm * N * 4;
}
static bool ensure_ws(size_t bytes, hipStream_t s) {
  if (bytes == ~(size_t)0) return false;
  SplitWs& w = g_split;
  if (bytes <= w.ws_bytes && w.cnt) return true;
  hipStreamCaptureStatus cap;
  HI_CHECK(hipStreamIsCapturing(s, &cap));
  if (cap != hipStreamCaptureStatusNone) return false;
  if (!w.cnt) {
    HI_CHECK(hipMalloc(&w.cnt, kMaxTiles * sizeof(int)));
    HI_CHECK(hipMemset(w.cnt, 0, kMaxTiles * sizeof(int)));
  }
  if (bytes <= w.ws_bytes) return true;
  bytes = std::max(bytes, 2 * w.ws_bytes);
  if (w.ws) w.retired.push_back(w.ws);  // graphs captured earlier may still use it
  HI_CHECK(hipMalloc(&w.ws, bytes));
  HI_CHECK(hipMemset(w.ws, 0, bytes));
  w.ws_bytes = bytes;
  return true;
}
static bool split_valid(int cfg, int sp, int K) { return (K / kCfg[cfg].bk) % sp == 0; }

static bool cfg_valid(int cfg, int M, int N, int K, int epi) {
  if (cfg < 0 || cfg >= kNumCfg) return false;
  const GemmCfg& k = kCfg[cfg];
  if (K % k.bk) return false;                      // whole K tiles only
  if (k.skinny) return M <= 64 && ((M + 15) / 16) * k.ksplit <= 4 && N % k.skinny == 0 && (epi != 2 || k.skinny == 32);
  if (epi == 2 && k.bn / k.wgn != 32) return false; // SwiGLU pairing needs 32-column warp tiles
  if (cfg_smem(cfg) + 1024 > (size_t)g_smem_optin) return false;
  return true;
}
// configs appended after the original 18 are never the last-resort fallback, so stored sweeps keep their meaning
static constexpr int kFallbackMaxCfg = 17;
static int cfg_fallback(int M, int N, int K, int epi) {
  for (int c = kFallbackMaxCfg; c >= 0; c--) if (cfg_valid(c, M, N, K, epi) && kCfg[c].bk == 32 && !kCfg[c].skinny) return c;
  return 2;
}

static int g_split_force = -1;  // benchmarking hook
EXPORT void hi_gemm_force_splits(int sp) { g_split_force = sp; }

// Per-GPU autotuning: the first call for a (M bucket, N, K, swiglu) key times every valid
// (tile config, split-K factor) on the real operands (into a scratch buffer, with the
// calling epilogue; epi 0 and 1 share a key) and caches the fastest. Small-M shapes (batched
// decode) need split-K to put enough weight-streaming blocks on the GPU. Calls during graph
// capture, or with HIPINFER_GEMM_AUTOTUNE=0, use gemm_pick with no split.
struct GemmChoice { int cfg, splits; };
static thread_local std::map<std::tuple<int, int, int, bool>, GemmChoice> g_tuned_dev[kMaxDev];
static GemmChoice gemm_choose(const half* A, int lda, const half* W, int M, int N, int K, int epi, hipStream_t s) {
  int mb = 16;
  while (mb < M && mb < 4096) mb *= 2;
  const auto key = std::make_tuple(mb, N, K, epi == 2);
  GemmChoice ch{gemm_pick(M, N, K, epi), 1};
  if (!cfg_valid(ch.cfg, M, N, K, epi)) ch.cfg = cfg_fallback(M, N, K, epi);
  if (g_gemm_force >= 0 || g_split_force >= 1) {
    if (g_gemm_force >= 0) ch.cfg = g_gemm_force;
    if (g_split_force >= 1) ch.splits = g_split_force;
    return ch;
  }
  auto& g_tuned = g_tuned_dev[g_dev];
  const auto it = g_tuned.find(key);
  if (it != g_tuned.end()) return it->second;
  static const bool enabled = !getenv("HIPINFER_GEMM_AUTOTUNE") || atoi(getenv("HIPINFER_GEMM_AUTOTUNE")) != 0;
  static const bool verbose = getenv("HIPINFER_GEMM_VERBOSE") && atoi(getenv("HIPINFER_GEMM_VERBOSE")) != 0;
  hipStreamCaptureStatus cap;
  HI_CHECK(hipStreamIsCapturing(s, &cap));
  if (!enabled || cap != hipStreamCaptureStatusNone) return ch;

  half* scratch;
  HI_CHECK(hipMalloc(&scratch, (size_t)M * N * sizeof(half)));
  if (epi == 1) HI_CHECK(hipMemsetAsync(scratch, 0, (size_t)M * N * sizeof(half), s));  // only the residual epilogue reads it
  hipEvent_t e0, e1;
  HI_CHECK(hipEventCreate(&e0)); HI_CHECK(hipEventCreate(&e1));
  // timed with the shape's own epilogue (into scratch; SwiGLU writes N / 2 columns), since the
  // epilogue's cost differs between tile configs
  auto run = [&](int c, int sp) {
    if (epi == 0) gemm_cfg<0>(c, sp, g_split.ws, A, lda, W, nullptr, scratch, N, M, N, K, s);
    else if (epi == 1) gemm_cfg<1>(c, sp, g_split.ws, A, lda, W, nullptr, scratch, N, M, N, K, s);
    else gemm_cfg<2>(c, sp, g_split.ws, A, lda, W, nullptr, scratch, N / 2, M, N, K, s);
  };
  auto time8 = [&](int c, int sp) {
    for (int r = 0; r < 2; r++) run(c, sp);  // warm-up
    HI_CHECK(hipEventRecord(e0, s));
    for (int r = 0; r < 8; r++) run(c, sp);
    HI_CHECK(hipEventRecord(e1, s));
    HI_CHECK(hipEventSynchronize(e1));
    float ms; HI_CHECK(hipEventElapsedTime(&ms, e0, e1));
    return ms;
  };
  // Pass 1 times every candidate once, in config order, and keeps the 3 fastest. Configs timed
  // late in a long sweep run on a hotter, lower-clocked GPU (a 2080 Ti loses several % within
  // a large shape's sweep), so pass 2 re-times the finalists back to back for 3 rounds and
  // keeps each one's best; the winner is decided on that like-for-like comparison.
  std::vector<std::pair<float, GemmChoice>> timed;  // every candidate: {ms, choice}
  for (int c = 0; c < hi_gemm_num_configs(); c++) {
    if (!cfg_valid(c, M, N, K, epi)) continue;
    const int tiles = kCfg[c].skinny ? (N + kCfg[c].bn - 1) / kCfg[c].bn
                                      : ((M + kCfg[c].bm - 1) / kCfg[c].bm) * ((N + kCfg[c].bn - 1) / kCfg[c].bn);
    for (int sp = 1; sp <= 16; sp *= 2) {
      if (!split_valid(c, sp, K) || (sp > 1 && tiles * sp > 8 * g_sms)) continue;
      if (!ensure_ws(split_ws_bytes(c, sp, M, N), s)) continue;
      const float ms = time8(c, sp);
      if (verbose) fprintf(stderr, "  cfg %2d x%-2d %7.1f us\n", c, sp, ms / 8 * 1e3f);
      timed.push_back({ms, {c, sp}});
    }
  }
  float best_ms = 1e30f;
  const size_t nfin = std::min<size_t>(3, timed.size());
  std::partial_sort(timed.begin(), timed.begin() + nfin, timed.end(),
                    [](const auto& a, const auto& b) { return a.first < b.first; });
  for (size_t f = 0; f < nfin; f++) timed[f].first = 1e30f;
  for (int r = 0; r < 3; r++)
    for (size_t f = 0; f < nfin; f++) timed[f].first = std::min(timed[f].first, time8(timed[f].second.cfg, timed[f].second.splits));
  for (size_t f = 0; f < nfin; f++)
    if (timed[f].first < best_ms) { best_ms = timed[f].first; ch = timed[f].second; }
  if (verbose)
    for (size_t f = 0; f < nfin; f++)
      fprintf(stderr, "  finalist cfg %2d x%-2d %7.1f us\n", timed[f].second.cfg, timed[f].second.splits, timed[f].first / 8 * 1e3f);
  (void)hipEventDestroy(e0); (void)hipEventDestroy(e1);
  HI_CHECK(hipFree(scratch));
  g_tuned[key] = ch;
  if (verbose) {
    char d[64]; hi_gemm_config_desc(ch.cfg, d, 64);
    fprintf(stderr, "gemm autotune M=%d N=%d K=%d epi=%d -> cfg %d (%s) splits %d: %.1f us, %.1f TFLOPs, %.0f GB/s of weights\n",
            M, N, K, epi, ch.cfg, d, ch.splits, best_ms / 8 * 1e3f, 2.0 * M * N * K / (best_ms / 8 * 1e-3) / 1e12,
            2.0 * N * K / (best_ms / 8 * 1e-3) / 1e9);
  }
  return ch;
}

// C (epi) A[M, K] @ W[N, K]^T (+ bias[N] for epi 0 when non-null). K % 32 == 0, N % 16 == 0 (N % 32 for epi 2).
EXPORT void hi_gemm(const void* A, int lda, const void* W, const void* bias, void* C, int ldc, int M, int N, int K, int epi,
                    void* s) {
  hi_init();
  if (K % 32 || N % (epi == 2 ? 32 : 16)) { fprintf(stderr, "hi_gemm: unsupported shape N=%d K=%d epi=%d\n", N, K, epi); exit(1); }
  GemmChoice ch = gemm_choose((const half*)A, lda, (const half*)W, M, N, K, epi, ST(s));
  if (!cfg_valid(ch.cfg, M, N, K, epi)) ch.cfg = cfg_fallback(M, N, K, epi);
  if (ch.splits > 1 && (!split_valid(ch.cfg, ch.splits, K) || !ensure_ws(split_ws_bytes(ch.cfg, ch.splits, M, N), ST(s))))
    ch.splits = 1;
  const half* a = (const half*)A; const half* w = (const half*)W; const half* b = (const half*)bias; half* c = (half*)C;
  float* ws = g_split.ws;
  if (epi == 0) gemm_cfg<0>(ch.cfg, ch.splits, ws, a, lda, w, b, c, ldc, M, N, K, ST(s));
  else if (epi == 1) gemm_cfg<1>(ch.cfg, ch.splits, ws, a, lda, w, nullptr, c, ldc, M, N, K, ST(s));
  else gemm_cfg<2>(ch.cfg, ch.splits, ws, a, lda, w, nullptr, c, ldc, M, N, K, ST(s));
}

EXPORT void hi_rope_cache(void* qkv, int ldq, int M, int Hq, int Hkv, int D, const void* cs, const int* pos,
                          int pos_base, int slot_base, void* kc, void* vc, int max_seq, void* s) {
  rope_cache_kernel<<<M, 256, 0, ST(s)>>>((half*)qkv, ldq, Hq, Hkv, D, (const __half2*)cs, pos, pos_base,
                                          slot_base, (half*)kc, (half*)vc, max_seq);
}

static bool head_dim_ok(int D, const char* who) {
  if (D == 64 || D == 128) return true;
  fprintf(stderr, "%s: head_dim %d not supported (64 or 128)\n", who, D);
  return false;
}
static void check_head_dim(int D, const char* who) { if (!head_dim_ok(D, who)) exit(1); }

EXPORT void hi_flash_prefill(const void* q, int ldq, const void* kc, const void* vc, int max_seq,
                             void* out, int ldo, int T, int p0, int Hq, int Hkv, int D, void* s) {
  hi_init();
  check_head_dim(D, "hi_flash_prefill");
  const float scale_log2 = 1.4426950408889634f / sqrtf((float)D);
  dim3 grid((T + 63) / 64, Hq);
  const half* Q = (const half*)q; const half* K = (const half*)kc; const half* V = (const half*)vc; half* O = (half*)out;
#if !defined(HI_AMD)
  static const bool use_mma = !getenv("HIPINFER_FLASH_WMMA");  // env var forces the portable kernel
  if (use_mma) {
    if (D == 64) flash_prefill_mma_kernel<64, 64><<<grid, 128, 0, ST(s)>>>(Q, ldq, K, V, max_seq, O, ldo, T, p0, Hq, Hkv, scale_log2);
    else flash_prefill_mma_kernel<128, 32><<<grid, 128, 0, ST(s)>>>(Q, ldq, K, V, max_seq, O, ldo, T, p0, Hq, Hkv, scale_log2);
    return;
  }
#endif
  if (D == 64) flash_prefill_kernel<64, 64><<<grid, 4 * g_warp, 0, ST(s)>>>(Q, ldq, K, V, max_seq, O, ldo, T, p0, Hq, Hkv, scale_log2);
  else flash_prefill_kernel<128, 32><<<grid, 4 * g_warp, 0, ST(s)>>>(Q, ldq, K, V, max_seq, O, ldo, T, p0, Hq, Hkv, scale_log2);
}

static int decode_splits(int B, int Hkv) {
  int S = (2 * g_sms + B * Hkv - 1) / (B * Hkv);
  return S < 1 ? 1 : S > 64 ? 64 : S;
}

// work: kDecodeCounters zero-initialized int counters (one per (sequence, kv head), left zeroed by the
// kernel), then B*Hq*64*(D+2) floats of partials (64 = max splits). The counters sit at a fixed
// offset so that a launch with a smaller B never finds a larger launch's partials in its counters.
static const int kDecodeCounters = 4096;
static size_t decode_work_bytes(int B, int Hq, int Hkv, int D) {
  (void)Hkv;
  return (size_t)kDecodeCounters * 4 + (size_t)B * Hq * 64 * (D + 2) * 4;
}
EXPORT size_t hi_decode_work_bytes(int B, int Hq, int Hkv, int D) { return decode_work_bytes(B, Hq, Hkv, D); }

template <int G, int D>
static void decode_attn_g(dim3 grid, bool mma, const half* q, int ldq, const half* kc, const half* vc, int max_seq, int Hkv,
                          const int* pos, float* po, float* pml, int* counters, int S, half* out, int ldo, float sl2,
                          int kv_slot, hipStream_t s) {
#if !defined(HI_AMD)
  if (mma) {
    decode_attn_mma_kernel<G, D><<<grid, 128, 0, s>>>(q, ldq, kc, vc, max_seq, Hkv, pos, po, pml, counters, S, out, ldo, sl2,
                                                      kv_slot);
    return;
  }
#endif
  (void)mma;
  decode_attn_kernel<G, D><<<grid, 256, 0, s>>>(q, ldq, kc, vc, max_seq, Hkv, pos, po, pml, counters, S, out, ldo, sl2,
                                                kv_slot);
}

template <int D>
static void decode_attn_d(int G, dim3 grid, bool mma, const half* q, int ldq, const half* kc, const half* vc, int max_seq,
                          int Hkv, const int* pos, float* po, float* pml, int* counters, int S, half* out, int ldo,
                          float sl2, int kv_slot, hipStream_t s) {
#define HI_DA(g) case g: decode_attn_g<g, D>(grid, mma, q, ldq, kc, vc, max_seq, Hkv, pos, po, pml, counters, S, out, ldo, sl2, kv_slot, s); break
  switch (G) {
    HI_DA(1); HI_DA(2); HI_DA(3); HI_DA(4); HI_DA(5); HI_DA(6); HI_DA(7); HI_DA(8);
    default: fprintf(stderr, "unsupported GQA group %d (1..8)\n", G); exit(1);
  }
#undef HI_DA
}

// Decode attention of B queries with S splits; kv_slot >= 0: every query reads that cache slot.
static void decode_attn(const half* q, int ldq, const half* kc, const half* vc, int max_seq, const int* pos, half* out,
                        int ldo, int B, int Hq, int Hkv, int D, void* work, int S, int kv_slot, hipStream_t s) {
  check_head_dim(D, "hi_decode_attn");
  const int G = Hq / Hkv;
  if (B * Hkv > kDecodeCounters) { fprintf(stderr, "hi_decode_attn: B*Hkv %d > %d\n", B * Hkv, kDecodeCounters); exit(1); }
  int* counters = (int*)work;
  float* po = (float*)work + kDecodeCounters;
  float* pml = po + (size_t)B * Hq * S * D;
  const float scale_log2 = 1.4426950408889634f / sqrtf((float)D);
  dim3 grid(S, Hkv, B);
  static const bool use_mma = !getenv("HIPINFER_DECODE_SCALAR");  // env var forces the portable kernel
  if (D == 64)
    decode_attn_d<64>(G, grid, use_mma, q, ldq, kc, vc, max_seq, Hkv, pos, po, pml, counters, S, out, ldo, scale_log2,
                      kv_slot, s);
  else
    decode_attn_d<128>(G, grid, use_mma, q, ldq, kc, vc, max_seq, Hkv, pos, po, pml, counters, S, out, ldo, scale_log2,
                       kv_slot, s);
}

EXPORT void hi_decode_attn(const void* q, int ldq, const void* kc, const void* vc, int max_seq, const int* pos,
                           void* out, int ldo, int B, int Hq, int Hkv, int D, void* work, void* s) {
  hi_init();
  decode_attn((const half*)q, ldq, (const half*)kc, (const half*)vc, max_seq, pos, (half*)out, ldo, B, Hq, Hkv, D, work,
              decode_splits(B, Hkv), -1, ST(s));
}

EXPORT void hi_argmax(const void* logits, int B, int V, int* tok_out, void* s) {
  argmax_kernel<<<B, 1024, 0, ST(s)>>>((const half*)logits, V, tok_out, nullptr, nullptr, 0);
}

// ------------------------------------------------------------------ mixture of experts
// docs/moe-design.md has the design. Checked by tests/test_kernels.py test_moe() (kernels vs PyTorch)
// and end to end against HF by tests/gate.py and tests/ppl.py (Granite MoE 1B).
static int gemv_max();

// Router + top-k for M normalised rows xn: idx[M, k] (int) and routing weights w[M, k] (fp32),
// in descending order. flags: kMoeNormTopk, kMoeFp16Weights (weight-file moe_flags).
EXPORT void hi_moe_router(const void* xn, int ldx, const void* Wr, int M, int E, int H, int k, int flags, int* idx, float* w,
                          void* s) {
  hi_init();
  if (E > kMoeMaxE || k > kMoeMaxK || k > E || H % 8) {
    fprintf(stderr, "hi_moe_router: unsupported E %d / k %d / H %d (E <= %d, k <= %d)\n", E, k, H, kMoeMaxE, kMoeMaxK);
    exit(1);
  }
  moe_router_kernel<<<M, 256, 0, ST(s)>>>((const half*)xn, ldx, (const half*)Wr, E, H, k, flags, idx, w);
}

// Expert workspace, each piece on a 256-byte boundary: perm, inv [M*k] int, off [E+1] int,
// rows [M*k, H], y [M*k, H] and act [M*k, I] fp16. The GEMV path uses only act.
static size_t moe_al(size_t b) { return (b + 255) / 256 * 256; }
EXPORT size_t hi_moe_work_bytes(int M, int k, int E, int H, int I) {
  const size_t n = (size_t)M * k;
  return 2 * moe_al(n * 4) + moe_al((size_t)(E + 1) * 4) + 2 * moe_al(n * H * 2) + moe_al(n * I * 2);
}

// x[m] += MoE(xn[m]) for M rows, given the routing (idx, w from hi_moe_router): wgu [E][2I, H]
// (gate/up interleaved per expert), wdown [E][H, I]. path 0 picks the GEMV path for M <= the
// GEMV limit (HIPINFER_GEMV_MAX), else the grouped GEMM path; 1 / 2 force GEMV / grouped. The
// grouped path reads the per-expert row counts back to the host (a stream sync), so it cannot
// run inside graph capture.
EXPORT void hi_moe_experts(const void* xn, int ldxn, const void* wgu, const void* wdown, const int* idx, const float* w,
                           int M, int E, int k, int H, int I, int flags, float scale, void* x, int ldx, int path, void* work,
                           void* s) {
  hi_init();
  if (H % 32 || I % 32) { fprintf(stderr, "hi_moe_experts: H %d and I %d must be multiples of 32\n", H, I); exit(1); }
  const hipStream_t st = ST(s);
  const size_t n = (size_t)M * k;
  char* p = (char*)work;
  int* perm = (int*)p;            p += moe_al(n * 4);
  int* inv = (int*)p;             p += moe_al(n * 4);
  int* off = (int*)p;             p += moe_al((size_t)(E + 1) * 4);
  half* rows = (half*)p;          p += moe_al(n * H * 2);
  half* y = (half*)p;             p += moe_al(n * H * 2);
  half* act = (half*)p;
  const half* Wgu = (const half*)wgu; const half* Wd = (const half*)wdown;
  if (path == 1 || (path == 0 && M <= gemv_max())) {
    const int warps = 256 / g_warp;
    moe_gemv_gu_kernel<<<dim3((I + warps - 1) / warps, (unsigned)n), 256, 0, st>>>((const half*)xn, ldxn, Wgu, idx, k, H, I, act);
    moe_gemv_down_kernel<<<dim3((H + warps - 1) / warps, M), 256, 0, st>>>(act, Wd, idx, w, k, H, I, flags, scale, (half*)x, ldx);
    return;
  }
  hipStreamCaptureStatus cap;
  HI_CHECK(hipStreamIsCapturing(st, &cap));
  if (cap != hipStreamCaptureStatusNone) {
    fprintf(stderr, "hi_moe_experts: the grouped path (M %d) cannot run inside graph capture\n", M);
    exit(1);
  }
  moe_sort_kernel<<<1, 1024, 0, st>>>(idx, (int)n, E, off, perm, inv);
  moe_gather_kernel<<<(unsigned)n, 128, 0, st>>>((const half*)xn, ldxn, perm, k, H, rows);
  std::vector<int> ho(E + 1);
  HI_CHECK(hipMemcpyAsync(ho.data(), off, (size_t)(E + 1) * 4, hipMemcpyDeviceToHost, st));
  HI_CHECK(hipStreamSynchronize(st));
  for (int e = 0; e < E; e++) {
    const int r0 = ho[e], m = ho[e + 1] - ho[e];
    if (!m) continue;
    hi_gemm(rows + (size_t)r0 * H, H, Wgu + (size_t)e * 2 * I * H, nullptr, act + (size_t)r0 * I, I, m, 2 * I, H, 2, s);
    hi_gemm(act + (size_t)r0 * I, I, Wd + (size_t)e * H * I, nullptr, y + (size_t)r0 * H, H, m, H, I, 0, s);
  }
  moe_combine_kernel<<<M, 128, 0, st>>>(y, inv, w, k, H, flags, scale, (half*)x, ldx);
}

// ------------------------------------------------------------------ tensor parallelism
// Megatron-style sharding over G GPUs in one process, one worker thread per GPU:
//   QKV  column-parallel by heads (each rank owns heads/G query and kv heads, and their KV cache)
//   O    row-parallel, then all-reduce of the hidden rows
//   gate/up column-parallel by intermediate units, down row-parallel, then all-reduce
//   lm_head vocabulary-parallel; greedy argmax reduces one 64-bit key per row across ranks
// Embedding and norms are replicated, so after each all-reduce every rank holds the same
// residual stream. All-reduces run either on NCCL/RCCL (built with HI_NCCL) or on the one-shot
// kernel in kernels.cuh, which wins on latency for decode-sized messages.
#if defined(HI_NCCL)
  #if defined(__HIPCC__)
    #include <rccl/rccl.h>
  #else
    #include <nccl.h>
  #endif
  #define NC_CHECK(x) do { ncclResult_t r_ = (x); if (r_ != ncclSuccess) { \
    fprintf(stderr, "NCCL error %s at %s:%d\n", ncclGetErrorString(r_), __FILE__, __LINE__); exit(1); } } while (0)
#endif

// Persistent worker threads, one per GPU (device set once); run(f) calls f(rank) on every worker
// and returns when all are done.
struct Pool {
  int n;
  std::vector<std::thread> th;
  std::mutex m;
  std::condition_variable cv, done;
  const std::function<void(int)>* job = nullptr;  // valid while run() waits
  int gen = 0, pending = 0;
  bool stop = false;
  explicit Pool(const std::vector<int>& devs) : n((int)devs.size()) {
    for (int i = 0; i < n; i++)
      th.emplace_back([this, i, d = devs[i]] {
        HI_CHECK(hipSetDevice(d));
        int seen = 0;
        for (;;) {
          const std::function<void(int)>* f;
          {
            std::unique_lock<std::mutex> l(m);
            cv.wait(l, [&] { return stop || gen != seen; });
            if (stop) return;
            seen = gen;
            f = job;
          }
          (*f)(i);
          std::lock_guard<std::mutex> l(m);
          if (--pending == 0) done.notify_all();
        }
      });
  }
  void run(const std::function<void(int)>& f) {
    std::unique_lock<std::mutex> l(m);
    job = &f; pending = n; gen++;
    cv.notify_all();
    done.wait(l, [&] { return pending == 0; });
  }
  ~Pool() {
    { std::lock_guard<std::mutex> l(m); stop = true; }
    cv.notify_all();
    for (auto& t : th) t.join();
  }
};

struct Engine;
struct TPGroup {
  Pool pool;
  bool p2p = false, has_nccl = false;
  int mode = 0;                     // all-reduce backend: 0 auto, 1 NCCL, 2 one-shot
  size_t oneshot_max = 512u << 10;  // auto: messages up to this many bytes go one-shot
  int ar_blocks = kARBlocks;        // one-shot: most blocks (channels) per all-reduce
  bool overlap = true;              // prefill: overlap each half's all-reduce with the other half's compute
  char* host_buf = nullptr;         // staging + flags in pinned host memory (GPUs without P2P)
  std::vector<char*> dev_buf;       // per-rank staging + flags in device memory (P2P)
  std::vector<ARArgs> args;         // per rank (args[r].epoch: its device-side counters)
#if defined(HI_NCCL)
  std::vector<ncclComm_t> nccl;
#endif
  std::vector<Engine*> ranks;
  explicit TPGroup(const std::vector<int>& d) : pool(d) {}
};

static TPGroup* tp_group(int world, const int* devs) {
  std::vector<int> d(world);
  for (int r = 0; r < world; r++) d[r] = devs ? devs[r] : r;
  TPGroup* g = new TPGroup(d);
  if (const char* v = getenv("HIPINFER_ALLREDUCE")) g->mode = !strcmp(v, "nccl") ? 1 : !strcmp(v, "oneshot") ? 2 : 0;
  if (const char* v = getenv("HIPINFER_AR_ONESHOT_MAX")) g->oneshot_max = strtoull(v, nullptr, 10);
  if (const char* v = getenv("HIPINFER_TP_OVERLAP")) g->overlap = atoi(v) != 0;
  if (const char* v = getenv("HIPINFER_AR_BLOCKS")) g->ar_blocks = std::max(1, std::min(kARBlocks, atoi(v)));
  const char* no_p2p = getenv("HIPINFER_NO_P2P");
  bool p2p = !(no_p2p && atoi(no_p2p));
  for (int i = 0; i < world; i++)
    for (int j = 0; j < world; j++)
      if (i != j) { int ok = 0; HI_CHECK(hipDeviceCanAccessPeer(&ok, d[i], d[j])); p2p = p2p && ok; }
  g->p2p = p2p;
  const size_t per_rank = kARRankBytes;
  g->dev_buf.assign(world, nullptr);
  g->args.assign(world, ARArgs{});
  if (!p2p) {
    HI_CHECK(hipHostMalloc((void**)&g->host_buf, world * per_rank, HI_HOST_SHARED_FLAGS));
    memset(g->host_buf, 0, world * per_rank);
  }
  g->pool.run([&](int r) {
    if (p2p) {
      for (int j = 0; j < world; j++) {
        if (j == r) continue;
        const hipError_t err = hipDeviceEnablePeerAccess(d[j], 0);
        if (err != hipSuccess && err != hipErrorPeerAccessAlreadyEnabled) HI_CHECK(err);
        (void)hipGetLastError();
      }
      HI_CHECK(hipMalloc((void**)&g->dev_buf[r], per_rank));
      HI_CHECK(hipMemset(g->dev_buf[r], 0, per_rank));
    }
    HI_CHECK(hipMalloc((void**)&g->args[r].epoch, kARBlocks * sizeof(unsigned)));
    HI_CHECK(hipMemset(g->args[r].epoch, 0, kARBlocks * sizeof(unsigned)));
  });
  for (int q = 0; q < world; q++) {
    for (int r = 0; r < world; r++) g->args[q].stage[r] = p2p ? g->dev_buf[r] : g->host_buf + r * per_rank;
    g->args[q].rank = q;
    g->args[q].world = world;
  }
#if defined(HI_NCCL)
  g->nccl.resize(world);
  int cur;
  HI_CHECK(hipGetDevice(&cur));
  NC_CHECK(ncclCommInitAll(g->nccl.data(), world, d.data()));
  HI_CHECK(hipSetDevice(cur));
  g->has_nccl = true;
#endif
  return g;
}

static void tp_group_free(TPGroup* g) {
  g->pool.run([&](int r) {
#if defined(HI_NCCL)
    (void)ncclCommDestroy(g->nccl[r]);
#endif
    if (g->dev_buf[r]) (void)hipFree(g->dev_buf[r]);
    (void)hipFree(g->args[r].epoch);
  });
  if (g->host_buf) (void)hipHostFree(g->host_buf);
  delete g;
}

static bool ar_nccl(const TPGroup* g, size_t bytes) {
  if (!g->has_nccl || g->mode == 2) return false;
  return g->mode == 1 || bytes > g->oneshot_max;
}

// ------------------------------------------------------------------ engine
struct Config {
  int vocab, hidden, inter, layers, n_heads, n_kv, head_dim, max_pos;
  int window;  // sliding-window attention span (0: none); the engine needs max_seq <= window
  float eps, attn_scaling;
  bool tied, qkv_bias;
  // MoE (weight-file version 3): inter is per expert
  bool moe = false, qk_norm = false;
  int n_experts = 0, top_k = 0, moe_flags = 0;
  float moe_scale = 1.f;
};
// Dense layers use wgu / wdown; MoE layers router [E, H], ewgu [E][2I, H], ewdown [E][H, I].
struct Layer { half *attn_norm, *wqkv, *bqkv, *wo, *mlp_norm, *wgu, *wdown, *router, *ewgu, *ewdown; };

// One engine per GPU. With tensor parallelism each rank holds its shard (hq query heads, hkv kv
// heads, il intermediate units, vocabulary rows [v0, v0 + vl)); with one GPU those are the
// full model's sizes.
struct Engine {
  Config c;
  int dev = 0;  // this engine's GPU
  int rank = 0, world = 1;
  TPGroup* g = nullptr;
  int hq, hkv, il, vl, v0;
  std::vector<void*> wallocs;
  half *embed, *final_norm, *lm_head;
  std::vector<Layer> L;
  int maxB, max_seq, max_pf, Mmax, qkv_dim;
  half *kc, *vc, *x, *xn, *qkv, *att, *act, *logits;  // logits [maxB, vl]
  half* pf_logits = nullptr;            // TP: [max_pf, vl] staging for hi_prefill_logits
  half* hstage = nullptr;               // TP: pinned host [max(max_pf, maxB), vl], logits gathered on the host
  unsigned long long* keys = nullptr;   // TP: per-row argmax keys
  __half2* rope;
  int *d_tok, *d_pos, *d_hist, *d_ptok;
  float* work;
  int* moe_idx = nullptr;               // MoE: [Mmax, k] routed experts, moe_w their weights
  float* moe_w = nullptr;
  void* moe_work = nullptr;             // MoE: hi_moe_experts workspace for Mmax rows
  hipStream_t st;
  hipStream_t cs = nullptr;            // TP: all-reduce stream for the overlapped prefill
  hipEvent_t ev[8];                    // TP: compute <-> comm ordering in the overlapped prefill
  std::vector<hipGraphExec_t> graphs;  // per batch size
  size_t layer_cache;                  // elements per layer of the KV cache
  half* dbg = nullptr;                 // hi_prefill_debug: where prefill copies its intermediates (null: off)
  int dbg_T = 0;                       // rows of each dbg plane
  // hi_verify (speculative decoding): up to kGemvMaxRows rows of one slot, allocated on first use
  int *v_tok = nullptr, *v_pos = nullptr, *v_out = nullptr;
  half* v_logits = nullptr;            // [kGemvMaxRows, vocab]
  void* v_work = nullptr;              // decode-attention workspace for kGemvMaxRows queries
  std::map<std::pair<int, int>, hipGraphExec_t> vgraphs;  // (slot, rows)
};

// Vocabulary-sharded logits ([rows, vl] at src, row stride vl) into the caller's [rows, vocab]
// buffer dst, which lives on rank 0's GPU (or the host). Rank 0 copies its columns directly. The
// other ranks stage theirs in pinned host memory, and tp_gather_logits copies them on once every
// rank is idle: a direct copy into another GPU's memory (no P2P) can be ordered behind that GPU's
// all-reduce, which waits for this GPU: a deadlock.
static void tp_export_logits(Engine* e, const half* src, int rows, half* dst, hipStream_t s) {
  if (e->rank == 0) {
    HI_CHECK(hipMemcpy2DAsync(dst, (size_t)e->c.vocab * 2, src, (size_t)e->vl * 2, (size_t)e->vl * 2, rows,
                              hipMemcpyDefault, s));
    return;
  }
  if (!e->hstage)
    HI_CHECK(hipHostMalloc((void**)&e->hstage, (size_t)std::max(e->max_pf, e->maxB) * e->vl * 2, HI_HOST_SHARED_FLAGS));
  HI_CHECK(hipMemcpyAsync(e->hstage, src, (size_t)rows * e->vl * 2, hipMemcpyDeviceToHost, s));
}
static void tp_gather_logits(Engine* e, int rows, void* dst) {
  for (Engine* r : e->g->ranks)
    if (r->rank > 0)
      HI_CHECK(hipMemcpy2D((half*)dst + r->v0, (size_t)r->c.vocab * 2, r->hstage, (size_t)r->vl * 2, (size_t)r->vl * 2,
                           rows, hipMemcpyDefault));
}

// Run f on every rank (each on its GPU's worker thread), or inline for a single-GPU engine.
template <class F>
static void each_rank(Engine* e, F f) {
  if (e->world == 1) { DevGuard dg(e->dev); f(e); return; }
  e->g->pool.run([&](int r) { f(e->g->ranks[r]); });
}
// Same, returning rank 0's result (every rank computes the same value).
template <class F>
static auto rank0(Engine* e, F f) -> decltype(f(e)) {
  decltype(f(e)) v{};
  each_rank(e, [&](Engine* r) { auto x = f(r); if (r->rank == 0) v = x; });
  return v;
}

// All-reduce `bytes` at x over all ranks, in place (bytes % 16 == 0): OP 0 sums fp16, OP 1 takes
// the max of uint64 (argmax keys).
template <int OP>
static void tp_allreduce_op(Engine* e, void* x, size_t bytes, hipStream_t s) {
  TPGroup* g = e->g;
#if defined(HI_NCCL)
  if (ar_nccl(g, bytes)) {
    if (OP == 0) NC_CHECK(ncclAllReduce(x, x, bytes / 2, ncclHalf, ncclSum, g->nccl[e->rank], s));
    else NC_CHECK(ncclAllReduce(x, x, bytes / 8, ncclUint64, ncclMax, g->nccl[e->rank], s));
    return;
  }
#endif
  // one block per 8 KB up to ar_blocks: a 1-row decode message is one block (one flag round trip)
  const long long nv = (long long)(bytes / 16);
  const int nb = (int)std::min<long long>(g->ar_blocks, std::max<long long>(1, (nv + 511) / 512));
  allreduce_oneshot_kernel<OP><<<nb, kARThreads, 0, s>>>(g->args[e->rank], (uint4*)x, nv);
}

// Sum the n halves at x over all ranks, in place (n % 8 == 0).
static void tp_allreduce(Engine* e, half* x, size_t n, hipStream_t s) {
  if (e->world > 1) tp_allreduce_op<0>(e, x, n * 2, s);
}

// Greedy pick for rows [0, B) of this rank's logits slice (row stride vl) into slots [slot,
// slot + B): token, position and history. With tensor parallelism: per-rank argmax keys, a
// cross-rank max, then the update.
static void greedy(Engine* e, const half* logits, int B, int slot, hipStream_t s) {
  int* tok = e->d_tok + slot;
  int* pos = e->d_pos + slot;
  int* hist = e->d_hist + (size_t)slot * e->max_seq;
  if (e->world == 1) {
    argmax_kernel<<<B, 1024, 0, s>>>(logits, e->c.vocab, tok, pos, hist, e->max_seq);
    return;
  }
  argmax_key_kernel<<<B, 1024, 0, s>>>(logits, e->vl, e->vl, e->v0, e->keys);
  tp_allreduce_op<1>(e, e->keys, (size_t)(B + 1) / 2 * 16, s);
  argmax_finish_kernel<<<1, (B + 31) / 32 * 32, 0, s>>>(e->keys, B, tok, pos, hist, e->max_seq);
}

// Capture f's launches on s into an instantiated graph.
template <class F>
static hipGraphExec_t capture(hipStream_t s, F f) {
  hipGraph_t g;
  hipGraphExec_t ge;
  HI_CHECK(hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal));
  f();
  HI_CHECK(hipStreamEndCapture(s, &g));
  HI_CHECK(hiGraphInstantiate(&ge, g));
  HI_CHECK(hipGraphDestroy(g));
  return ge;
}

// GPU milliseconds of the work f enqueues on s.
template <class F>
static float time_ms(hipStream_t s, F f) {
  hipEvent_t a, b;
  HI_CHECK(hipEventCreate(&a)); HI_CHECK(hipEventCreate(&b));
  HI_CHECK(hipEventRecord(a, s));
  f();
  HI_CHECK(hipEventRecord(b, s));
  HI_CHECK(hipEventSynchronize(b));
  HI_CHECK(hipGetLastError());
  float ms; HI_CHECK(hipEventElapsedTime(&ms, a, b));
  (void)hipEventDestroy(a); (void)hipEventDestroy(b);
  return ms;
}

static void* dalloc(size_t bytes) { void* p; HI_CHECK(hipMalloc(&p, bytes)); HI_CHECK(hipMemset(p, 0, bytes)); return p; }
static void* walloc(Engine* e, size_t bytes) { void* p; HI_CHECK(hipMalloc(&p, bytes)); e->wallocs.push_back(p); return p; }

// Weight file (written by scripts/export.py, version 2): 64-byte header, then inv_freq (fp32)
// and fp16 tensors, each starting on a 256-byte boundary, in the order consumed below. Rank
// `rank` of `world` reads only its shard (row ranges directly, column ranges via a 2-D copy).
// Returns null (after a message on stderr) for a bad file or an unsupported model / setting.
static Engine* load_shard(const char* path, int rank, int world, int maxB, int max_seq, int max_pf) {
  hi_init();
  if (maxB < 1 || max_seq < 1 || max_pf < 1) {
    fprintf(stderr, "hi_load: max_batch (%d), max_seq (%d) and max_prefill (%d) must be >= 1\n", maxB, max_seq, max_pf);
    return nullptr;
  }
  FILE* f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", path); return nullptr; }
  int32_t hdr[32] = {0};
  if (fread(hdr, 4, 16, f) != 16 || hdr[0] != 0x48495046) { fprintf(stderr, "bad weight file\n"); fclose(f); return nullptr; }
  if (hdr[1] != 2 && hdr[1] != 3) { fprintf(stderr, "weight file version %d; re-export with scripts/export.py (version 2, or 3 for MoE)\n", hdr[1]); fclose(f); return nullptr; }
  // version 3 (MoE): a 128-byte header, MoE fields at bytes 64..79
  if (hdr[1] == 3 && fread(hdr + 16, 4, 16, f) != 16) { fprintf(stderr, "bad weight file\n"); fclose(f); return nullptr; }
  Engine* e = new Engine();
  HI_CHECK(hipGetDevice(&e->dev));
  e->rank = rank; e->world = world;
  Config& c = e->c;
  c.vocab = hdr[2]; c.hidden = hdr[3]; c.inter = hdr[4]; c.layers = hdr[5];
  c.n_heads = hdr[6]; c.n_kv = hdr[7]; c.head_dim = hdr[8]; c.max_pos = hdr[9];
  memcpy(&c.eps, &hdr[10], 4); memcpy(&c.attn_scaling, &hdr[11], 4);
  c.tied = hdr[12] & 1; c.qkv_bias = (hdr[12] & 2) != 0;
  c.window = hdr[13];  // zero padding in files exported before the field existed
  c.moe = (hdr[12] & 4) != 0; c.qk_norm = (hdr[12] & 8) != 0;
  if (c.moe) { c.n_experts = hdr[16]; c.top_k = hdr[17]; c.moe_flags = hdr[18]; memcpy(&c.moe_scale, &hdr[19], 4); }
  auto fail = [&]() -> Engine* { fclose(f); delete e; return nullptr; };
  if (c.moe != (hdr[1] == 3)) { fprintf(stderr, "hi_load: MoE flag does not match file version %d\n", hdr[1]); return fail(); }
  if (c.moe) {
    if (world > 1) {
      fprintf(stderr, "hi_load: MoE models run on one GPU for now (expert parallelism: docs/moe-design.md)\n");
      return fail();
    }
    if (c.qk_norm) { fprintf(stderr, "hi_load: QK-norm (OLMoE) is not implemented in the engine yet\n"); return fail(); }
    if (c.n_experts < 1 || c.n_experts > kMoeMaxE || c.top_k < 1 || c.top_k > kMoeMaxK || c.top_k > c.n_experts ||
        c.hidden % 32 || c.inter % 32) {
      fprintf(stderr, "hi_load: unsupported MoE shape: %d experts, top-%d, hidden %d, expert inter %d\n", c.n_experts,
              c.top_k, c.hidden, c.inter);
      return fail();
    }
  }
  if (!head_dim_ok(c.head_dim, "hi_load")) return fail();
  if (c.n_kv < 1 || c.n_heads % c.n_kv || c.n_heads / c.n_kv > 8) {
    fprintf(stderr, "unsupported GQA group %d/%d\n", c.n_heads, c.n_kv);
    return fail();
  }
  // The engine attends to the whole history; that equals sliding-window attention only while
  // every sequence fits in the window.
  if (c.window > 0 && max_seq > c.window) {
    fprintf(stderr, "hi_load: model uses sliding-window attention (%d tokens), which the engine does not implement; "
            "max_seq %d must be <= %d\n", c.window, max_seq, c.window);
    return fail();
  }
  const int H = c.hidden, I = c.inter, D = c.head_dim, Hq = c.n_heads, Hkv = c.n_kv, V = c.vocab;
  if (Hq % world || Hkv % world || I % (32 * world)) {
    fprintf(stderr, "tensor parallel x%d needs heads (%d), kv heads (%d) and intermediate/32 (%d) divisible by %d\n",
            world, Hq, Hkv, I, world);
    return fail();
  }
  e->hq = Hq / world; e->hkv = Hkv / world; e->il = I / world;
  if ((long long)maxB * e->hkv > kDecodeCounters) {
    fprintf(stderr, "hi_load: max_batch %d x %d kv heads per GPU exceeds %d\n", maxB, e->hkv, kDecodeCounters);
    return fail();
  }
  const int vr = ((V + world - 1) / world + 15) / 16 * 16;
  e->v0 = rank * vr; e->vl = V - e->v0 < vr ? V - e->v0 : vr;
  const int hq = e->hq, hkv = e->hkv, il = e->il;
  e->qkv_dim = (hq + 2 * hkv) * D;

  size_t off = hdr[1] == 3 ? 128 : 64;
  auto next = [&](size_t bytes) { const size_t o = off; off += (bytes + 255) / 256 * 256; return o; };
  std::vector<char> hb;
  auto read_at = [&](size_t o, size_t n) -> const char* {
    if (hb.size() < n) hb.resize(n);
    if (fseek(f, (long)o, SEEK_SET) || fread(hb.data(), 1, n, f) != n) throw std::runtime_error("weight file too small");
    return hb.data();
  };
  // rows [r0, r0 + nr) of a [*, cols] fp16 tensor at file offset o
  auto rows_to = [&](half* dst, size_t o, size_t cols, size_t r0, size_t nr) {
    HI_CHECK(hipMemcpy(dst, read_at(o + r0 * cols * 2, nr * cols * 2), nr * cols * 2, hipMemcpyHostToDevice));
  };
  auto rows = [&](size_t o, size_t cols, size_t r0, size_t nr) {
    half* p = (half*)walloc(e, nr * cols * 2);
    rows_to(p, o, cols, r0, nr);
    return p;
  };
  // columns [c0, c0 + nc) of a [nrows, cols] tensor
  auto columns = [&](size_t o, size_t nrows, size_t cols, size_t c0, size_t nc) {
    half* p = (half*)walloc(e, nrows * nc * 2);
    const char* h = read_at(o, nrows * cols * 2);
    HI_CHECK(hipMemcpy2D(p, nc * 2, h + c0 * 2, cols * 2, nc * 2, nrows, hipMemcpyHostToDevice));
    return p;
  };

  std::vector<float> inv(D / 2);  // fp32 RoPE frequencies
  try {
    memcpy(inv.data(), read_at(next((D / 2) * 4), (D / 2) * 4), (D / 2) * 4);
    e->embed = rows(next((size_t)V * H * 2), H, 0, V);
    e->final_norm = rows(next((size_t)H * 2), H, 0, 1);
    e->lm_head = c.tied ? e->embed + (size_t)e->v0 * H : rows(next((size_t)V * H * 2), H, e->v0, e->vl);
    e->L.resize(c.layers);
    const size_t QKV = (size_t)(Hq + 2 * Hkv) * D;
    // this rank's q, k and v rows of the [q; k; v] projection (or bias)
    auto qkv_rows = [&](half* dst, size_t o, size_t cols) {
      rows_to(dst, o, cols, (size_t)rank * hq * D, (size_t)hq * D);
      rows_to(dst + (size_t)hq * D * cols, o, cols, (size_t)Hq * D + (size_t)rank * hkv * D, (size_t)hkv * D);
      rows_to(dst + (size_t)(hq + hkv) * D * cols, o, cols, (size_t)(Hq + Hkv) * D + (size_t)rank * hkv * D, (size_t)hkv * D);
    };
    for (auto& l : e->L) {
      l.attn_norm = rows(next((size_t)H * 2), H, 0, 1);
      l.wqkv = (half*)walloc(e, (size_t)e->qkv_dim * H * 2);
      qkv_rows(l.wqkv, next(QKV * H * 2), H);
      l.bqkv = nullptr;
      if (c.qkv_bias) { l.bqkv = (half*)walloc(e, (size_t)e->qkv_dim * 2); qkv_rows(l.bqkv, next(QKV * 2), 1); }
      l.wo = columns(next((size_t)H * Hq * D * 2), H, (size_t)Hq * D, (size_t)rank * hq * D, (size_t)hq * D);
      l.mlp_norm = rows(next((size_t)H * 2), H, 0, 1);
      l.router = l.ewgu = l.ewdown = nullptr;
      if (c.moe) {  // one GPU (checked above): every expert, stacked as in the file
        l.wgu = l.wdown = nullptr;
        const size_t E = c.n_experts;
        l.router = rows(next(E * H * 2), H, 0, E);
        l.ewgu = rows(next(E * 2 * I * H * 2), H, 0, E * 2 * I);
        l.ewdown = rows(next(E * H * I * 2), I, 0, E * H);
        continue;
      }
      // gate/up rows are interleaved in groups of 16 units, so a rank's units are a contiguous row range
      l.wgu = rows(next((size_t)2 * I * H * 2), H, (size_t)rank * 2 * il, (size_t)2 * il);
      l.wdown = columns(next((size_t)H * I * 2), H, I, (size_t)rank * il, il);
    }
  } catch (const std::exception& ex) {
    fprintf(stderr, "hi_load: %s: %s\n", path, ex.what());
    for (void* p : e->wallocs) (void)hipFree(p);
    return fail();
  }
  fclose(f);

  e->maxB = maxB; e->max_seq = max_seq; e->max_pf = max_pf;
  e->Mmax = ((max_pf > maxB ? max_pf : maxB) + 127) / 128 * 128;
  e->layer_cache = (size_t)maxB * hkv * max_seq * D;
  e->kc = (half*)dalloc(e->layer_cache * c.layers * 2);
  e->vc = (half*)dalloc(e->layer_cache * c.layers * 2);
  e->x = (half*)dalloc((size_t)e->Mmax * H * 2);
  e->xn = (half*)dalloc((size_t)e->Mmax * H * 2);
  e->qkv = (half*)dalloc((size_t)e->Mmax * e->qkv_dim * 2);
  e->att = (half*)dalloc((size_t)e->Mmax * hq * D * 2);
  e->act = (half*)dalloc((size_t)e->Mmax * il * 2);
  e->logits = (half*)dalloc((size_t)maxB * e->vl * 2);
  e->keys = (unsigned long long*)dalloc((size_t)(maxB + 2) * 8);
  e->d_tok = (int*)dalloc(maxB * 4);
  e->d_pos = (int*)dalloc(maxB * 4);
  e->d_hist = (int*)dalloc((size_t)maxB * max_seq * 4);
  e->d_ptok = (int*)dalloc(e->Mmax * 4);
  e->work = (float*)dalloc(decode_work_bytes(maxB, hq, hkv, D));
  if (c.moe) {
    e->moe_idx = (int*)dalloc((size_t)e->Mmax * c.top_k * 4);
    e->moe_w = (float*)dalloc((size_t)e->Mmax * c.top_k * 4);
    e->moe_work = dalloc(hi_moe_work_bytes(e->Mmax, c.top_k, c.n_experts, H, I));
  }

  // RoPE table exactly as HF computes it: fp32 inv_freq (incl. llama3 / linear scaling, from
  // the export) and fp32 angles; cos/sin times attention_scaling, rounded to fp16.
  std::vector<__half2> rt((size_t)max_seq * (D / 2));
  for (int i = 0; i < D / 2; i++)
    for (int p = 0; p < max_seq; p++) {
      const float a = (float)p * inv[i];
      rt[(size_t)p * (D / 2) + i] = __halves2half2(__float2half((float)cos((double)a) * c.attn_scaling),
                                                   __float2half((float)sin((double)a) * c.attn_scaling));
    }
  e->rope = (__half2*)dalloc(rt.size() * 4);
  HI_CHECK(hipMemcpy(e->rope, rt.data(), rt.size() * 4, hipMemcpyHostToDevice));
  // Blocking streams: legacy-stream copies and memsets (weights, split-K workspace, state) are
  // ordered before engine kernels. That ordering is per device, so it cannot couple two ranks; the
  // engine never copies between GPUs directly (see tp_gather_logits).
  HI_CHECK(hipStreamCreate(&e->st));
  if (world > 1) {
    HI_CHECK(hipStreamCreate(&e->cs));
    for (auto& ev : e->ev) HI_CHECK(hipEventCreateWithFlags(&ev, hipEventDisableTiming));
  }
  e->graphs.assign(maxB + 1, nullptr);
  return e;
}

static void free_shard(Engine* e) {
  for (auto g : e->graphs) if (g) (void)hipGraphExecDestroy(g);
  for (auto& g : e->vgraphs) (void)hipGraphExecDestroy(g.second);
  for (void* p : e->wallocs) (void)hipFree(p);
  void* ps[] = {e->kc, e->vc, e->x, e->xn, e->qkv, e->att, e->act, e->logits, e->keys, e->rope,
                e->d_tok, e->d_pos, e->d_hist, e->d_ptok, e->work, e->pf_logits, e->moe_idx, e->moe_w, e->moe_work,
                e->v_tok, e->v_pos, e->v_out, e->v_logits, e->v_work};
  for (void* p : ps) if (p) (void)hipFree(p);
  if (e->hstage) (void)hipHostFree(e->hstage);
  (void)hipStreamDestroy(e->st);
  if (e->cs) {
    (void)hipStreamDestroy(e->cs);
    for (auto ev : e->ev) (void)hipEventDestroy(ev);
  }
  delete e;
}

// Load a model sharded over `world` GPUs (device ids `devs`, or 0..world-1 when null; with one
// GPU, devs[0] or the current device). The returned handle drives all of them; every hi_*
// engine call below runs on all ranks, each on its own GPU whatever the caller's current device.
EXPORT Engine* hi_load_tp(const char* path, int maxB, int max_seq, int max_pf, int world, const int* devs) {
  if (world <= 1) {
    int dev;
    if (devs) dev = devs[0];
    else HI_CHECK(hipGetDevice(&dev));
    DevGuard dg(dev);
    return load_shard(path, 0, 1, maxB, max_seq, max_pf);
  }
  if (world > kARMaxRanks) { fprintf(stderr, "tensor parallel: at most %d GPUs\n", kARMaxRanks); return nullptr; }
  TPGroup* g = tp_group(world, devs);
  g->ranks.assign(world, nullptr);
  g->pool.run([&](int r) {
    Engine* e = load_shard(path, r, world, maxB, max_seq, max_pf);
    if (e) e->g = g;
    g->ranks[r] = e;
  });
  if (!std::all_of(g->ranks.begin(), g->ranks.end(), [](Engine* e) { return e != nullptr; })) {
    g->pool.run([&](int r) { if (g->ranks[r]) free_shard(g->ranks[r]); });
    tp_group_free(g);
    return nullptr;
  }
  return g->ranks[0];
}

EXPORT Engine* hi_load(const char* path, int maxB, int max_seq, int max_pf) { return hi_load_tp(path, maxB, max_seq, max_pf, 1, nullptr); }

EXPORT void hi_free(Engine* e) {
  if (e->world == 1) { DevGuard dg(e->dev); free_shard(e); return; }
  TPGroup* g = e->g;
  g->pool.run([&](int r) { free_shard(g->ranks[r]); });
  tp_group_free(g);
}

// {vocab, hidden, inter, layers, heads, kv heads, head_dim, max_pos, tied, qkv_bias, sliding window (0: none)}
EXPORT void hi_config(Engine* e, int* out) {
  const Config& c = e->c;
  int v[] = {c.vocab, c.hidden, c.inter, c.layers, c.n_heads, c.n_kv, c.head_dim, c.max_pos, (int)c.tied, (int)c.qkv_bias,
             c.window};
  memcpy(out, v, sizeof(v));
}

// {n_experts, top_k, moe_flags}: zeros for a dense model
EXPORT void hi_moe_config(Engine* e, int* out) {
  const Config& c = e->c;
  out[0] = c.n_experts; out[1] = c.top_k; out[2] = c.moe_flags;
}

// Weight bytes one GPU reads per decode step (the roofline numerator; this rank's shard). For MoE:
// one token's (B = 1) reads, the router plus its top_k experts.
EXPORT size_t hi_weight_bytes(Engine* e) {
  const Config& c = e->c;
  const size_t mlp = c.moe ? (size_t)c.n_experts * c.hidden + (size_t)c.top_k * 3 * c.inter * c.hidden
                           : (size_t)3 * e->il * c.hidden;
  size_t per_layer = (size_t)e->qkv_dim * c.hidden + (size_t)c.hidden * e->hq * c.head_dim +
                     mlp + 2 * (size_t)c.hidden + (c.qkv_bias ? e->qkv_dim : 0);
  return (per_layer * c.layers + (size_t)e->vl * c.hidden + c.hidden) * 2;
}

// {world, peer access between all GPUs, NCCL built in, backend (0 auto, 1 NCCL, 2 one-shot), auto threshold
// bytes, prefill overlap}
EXPORT void hi_tp_info(Engine* e, long long* out) {
  TPGroup* g = e->g;
  out[0] = e->world;
  out[1] = g ? g->p2p : 0;
  out[2] = g ? g->has_nccl : 0;
  out[3] = g ? g->mode : 0;
  out[4] = g ? (long long)g->oneshot_max : 0;
  out[5] = g ? g->overlap : 0;
}

// Choose the all-reduce backend (drops captured decode graphs, which bake it in).
EXPORT void hi_tp_set_allreduce(Engine* e, int mode, long long oneshot_max) {
  if (e->world == 1) return;
  e->g->mode = mode;
  if (oneshot_max >= 0) e->g->oneshot_max = (size_t)oneshot_max;
  each_rank(e, [&](Engine* r) {
    HI_CHECK(hipStreamSynchronize(r->st));
    for (auto& gr : r->graphs) if (gr) { (void)hipGraphExecDestroy(gr); gr = nullptr; }
  });
}

// Overlap prefill all-reduces with compute (two row halves on two streams); on by default.
EXPORT void hi_tp_set_overlap(Engine* e, int on) { if (e->g) e->g->overlap = on != 0; }

// All-reduce self-test: rank r contributes ((7i + 3r) % 11 - 5) / 4 (exact in fp16); returns the
// largest deviation from the exact sum over all ranks.
EXPORT float hi_tp_allreduce_check(Engine* e, size_t n) {
  if (e->world == 1) return 0.f;
  auto val = [](size_t i, int r) { return (float)((int)((7 * i + 3 * r) % 11) - 5) * 0.25f; };
  std::vector<float> err(e->world, 0.f);
  each_rank(e, [&](Engine* r) {
    std::vector<half> h(n);
    for (size_t i = 0; i < n; i++) h[i] = __float2half(val(i, r->rank));
    half* d;
    HI_CHECK(hipMalloc(&d, n * 2));
    HI_CHECK(hipMemcpy(d, h.data(), n * 2, hipMemcpyHostToDevice));
    tp_allreduce(r, d, n, r->st);
    HI_CHECK(hipStreamSynchronize(r->st));
    HI_CHECK(hipMemcpy(h.data(), d, n * 2, hipMemcpyDeviceToHost));
    HI_CHECK(hipFree(d));
    float m = 0.f;
    for (size_t i = 0; i < n; i++) {
      float want = 0.f;
      for (int q = 0; q < r->world; q++) want += val(i, q);
      m = fmaxf(m, fabsf(__half2float(h[i]) - want));
    }
    err[r->rank] = m;
  });
  float m = 0.f;
  for (float v : err) m = fmaxf(m, v);
  return m;
}

// Microseconds per all-reduce of n halves with the current backend, timed as `iters`
// back-to-back all-reduces inside one CUDA graph (no launch gaps).
EXPORT float hi_tp_allreduce_bench(Engine* e, size_t n, int iters) {
  if (e->world == 1) return 0.f;
  return rank0(e, [&](Engine* r) {
    half* d = (half*)dalloc(n * 2);
    hipStream_t s = r->st;
    for (int i = 0; i < 3; i++) tp_allreduce(r, d, n, s);
    hipGraphExec_t ge = capture(s, [&] { for (int i = 0; i < iters; i++) tp_allreduce(r, d, n, s); });
    HI_CHECK(hipGraphLaunch(ge, s));
    const float ms = time_ms(s, [&] { HI_CHECK(hipGraphLaunch(ge, s)); });
    (void)hipGraphExecDestroy(ge);
    HI_CHECK(hipFree(d));
    return ms * 1e3f / iters;
  });
}

// ------------------------------------------------------------------ debug dump (hi_prefill_debug)
// While e->dbg is set (only inside hi_prefill_debug: one GPU, one prefill chunk), prefill copies
// each sub-step's output rows into it: fp16 [dbg_T, width] planes back to back, plane 0 the
// embedding, then per layer {attn_norm, qkv (before RoPE), rope (q | k after RoPE), attn,
// o_proj (residual stream after the O GEMM), mlp_norm, gate_up (SwiGLU output), down (residual
// stream after the down GEMM)}, then final_norm and logits. scripts/layer_diff.py engine_layout
// is the same layout. Copies run on the prefill stream. Decode dumps only through
// hi_decode_debug (one step, no graph), and only the planes its MoE layers fill.
enum { kDbgAttnNorm, kDbgQkv, kDbgRope, kDbgAttn, kDbgO, kDbgMlpNorm, kDbgGateUp, kDbgDown, kDbgSteps };
static int dbg_planes(const Engine* e) { return 1 + e->c.layers * kDbgSteps + 2; }
static size_t dbg_width(const Engine* e, int p) {
  const Config& c = e->c;
  const int last = dbg_planes(e) - 1;  // logits; last - 1: final_norm
  if (p == last) return c.vocab;
  if (p == 0 || p == last - 1) return c.hidden;
  switch ((p - 1) % kDbgSteps) {
    case kDbgQkv: return (size_t)(c.n_heads + 2 * c.n_kv) * c.head_dim;
    case kDbgRope: return (size_t)(c.n_heads + c.n_kv) * c.head_dim;
    case kDbgAttn: return (size_t)c.n_heads * c.head_dim;
    case kDbgGateUp: return c.inter;
    default: return c.hidden;
  }
}
static half* dbg_plane(const Engine* e, int p) {
  size_t off = 0;
  for (int i = 0; i < p; i++) off += dbg_width(e, i);
  return e->dbg + off * e->dbg_T;
}
// Copy rows [r0, r0 + m) of src (row stride ld; src points at row r0) into plane p.
static void dbg_copy(Engine* e, int p, const half* src, int ld, int r0, int m, hipStream_t s) {
  if (!e->dbg) return;
  const size_t w = dbg_width(e, p);
  HI_CHECK(hipMemcpy2DAsync(dbg_plane(e, p) + (size_t)r0 * w, w * 2, src, (size_t)ld * 2, w * 2, m,
                            hipMemcpyDeviceToDevice, s));
}
static void dbg_step(Engine* e, const Layer& w, int step, const half* src, int ld, int r0, int m, hipStream_t s) {
  if (e->dbg) dbg_copy(e, 1 + (int)(&w - e->L.data()) * kDbgSteps + step, src, ld, r0, m, s);
}
// RoPE output: q rotated in place in the QKV buffer, k rotated into the cache (slot, positions pb + r0...).
static void dbg_rope(Engine* e, const Layer& w, const half* q, int ld, const half* kc, int slot, int r0, int m, int pb,
                     hipStream_t s) {
  if (!e->dbg) return;
  const int D = e->c.head_dim, p = 1 + (int)(&w - e->L.data()) * kDbgSteps + kDbgRope;
  const size_t wd = dbg_width(e, p);
  half* dst = dbg_plane(e, p) + (size_t)r0 * wd;
  HI_CHECK(hipMemcpy2DAsync(dst, wd * 2, q, (size_t)ld * 2, (size_t)e->hq * D * 2, m, hipMemcpyDeviceToDevice, s));
  for (int h = 0; h < e->hkv; h++)
    HI_CHECK(hipMemcpy2DAsync(dst + (size_t)(e->hq + h) * D, wd * 2,
                              kc + (((size_t)slot * e->hkv + h) * e->max_seq + pb + r0) * D, (size_t)D * 2,
                              (size_t)D * 2, m, hipMemcpyDeviceToDevice, s));
}

// Attention block for rows [r0, r0 + m) of the n prompt rows: norm, QKV, RoPE + cache write,
// causal attention over the cache (rows before r0 already written), O projection into e->x.
static void attn_rows(Engine* e, const Layer& w, half* kc, half* vc, int slot, int r0, int m, int pb, hipStream_t s) {
  const Config& c = e->c;
  const int H = c.hidden, D = c.head_dim, hq = e->hq, hkv = e->hkv, HD = hq * D, ld = e->qkv_dim;
  const int ep = e->rank == 0 ? 1 : 0;  // row-parallel outputs: rank 0 adds the residual, the others store partials
  half* q = e->qkv + (size_t)r0 * ld;
  hi_rmsnorm(e->x + (size_t)r0 * H, w.attn_norm, e->xn + (size_t)r0 * H, m, H, c.eps, s);
  dbg_step(e, w, kDbgAttnNorm, e->xn + (size_t)r0 * H, H, r0, m, s);
  hi_gemm(e->xn + (size_t)r0 * H, H, w.wqkv, w.bqkv, q, ld, m, ld, H, 0, s);
  dbg_step(e, w, kDbgQkv, q, ld, r0, m, s);
  hi_rope_cache(q, ld, m, hq, hkv, D, e->rope, nullptr, pb + r0, slot, kc, vc, e->max_seq, s);
  dbg_rope(e, w, q, ld, kc, slot, r0, m, pb, s);
  hi_flash_prefill(q, ld, kc + (size_t)slot * hkv * e->max_seq * D, vc + (size_t)slot * hkv * e->max_seq * D, e->max_seq,
                   e->att + (size_t)r0 * HD, HD, m, pb + r0, hq, hkv, D, s);
  dbg_step(e, w, kDbgAttn, e->att + (size_t)r0 * HD, HD, r0, m, s);
  hi_gemm(e->att + (size_t)r0 * HD, HD, w.wo, nullptr, e->x + (size_t)r0 * H, H, m, H, HD, ep, s);
  dbg_step(e, w, kDbgO, e->x + (size_t)r0 * H, H, r0, m, s);
}

// MoE block for rows [r0, r0 + m): norm, router + top-k, experts + combine into
// e->x. Up to the GEMV limit the experts run as GEMVs (graph-capturable); above it, grouped GEMMs.
static void moe_rows(Engine* e, const Layer& w, int r0, int m, hipStream_t s) {
  const Config& c = e->c;
  const int H = c.hidden;
  half* xn = e->xn + (size_t)r0 * H;
  hi_rmsnorm(e->x + (size_t)r0 * H, w.mlp_norm, xn, m, H, c.eps, s);
  dbg_step(e, w, kDbgMlpNorm, xn, H, r0, m, s);  // the router's input (gate_up has no single value: left zero)
  hi_moe_router(xn, H, w.router, m, c.n_experts, H, c.top_k, c.moe_flags, e->moe_idx, e->moe_w, s);
  hi_moe_experts(xn, H, w.ewgu, w.ewdown, e->moe_idx, e->moe_w, m, c.n_experts, c.top_k, H, c.inter, c.moe_flags,
                 c.moe_scale, e->x + (size_t)r0 * H, H, 0, e->moe_work, s);
  dbg_step(e, w, kDbgDown, e->x + (size_t)r0 * H, H, r0, m, s);
}

// MLP block for rows [r0, r0 + m): norm, gate/up + SwiGLU, down projection into e->x.
static void mlp_rows(Engine* e, const Layer& w, int r0, int m, hipStream_t s) {
  if (e->c.moe) { moe_rows(e, w, r0, m, s); return; }
  const Config& c = e->c;
  const int H = c.hidden, il = e->il, ep = e->rank == 0 ? 1 : 0;
  hi_rmsnorm(e->x + (size_t)r0 * H, w.mlp_norm, e->xn + (size_t)r0 * H, m, H, c.eps, s);
  dbg_step(e, w, kDbgMlpNorm, e->xn + (size_t)r0 * H, H, r0, m, s);
  hi_gemm(e->xn + (size_t)r0 * H, H, w.wgu, nullptr, e->act + (size_t)r0 * il, il, m, 2 * il, H, 2, s);
  dbg_step(e, w, kDbgGateUp, e->act + (size_t)r0 * il, il, r0, m, s);
  hi_gemm(e->act + (size_t)r0 * il, il, w.wdown, nullptr, e->x + (size_t)r0 * H, H, m, H, il, ep, s);
  dbg_step(e, w, kDbgDown, e->x + (size_t)r0 * H, H, r0, m, s);
}

// Run the transformer layers over n tokens already embedded into e->x, continuing slot `slot`
// at position pb. Leaves the final hidden states (pre-norm) in e->x.
//
// With tensor parallelism and n >= 256 the rows are split into two halves A and B so that every
// all-reduce (on the comm stream) runs while the other half computes:
//   compute: attn A | attn B | mlp A | mlp B | attn A' ...
//   comm:           | AR A   | AR B  | AR A  | AR B   ...
// (B's attention reads A's KV cache entries of the same layer, written just before.)
static void prefill_layers(Engine* e, int slot, int n, int pb) {
  const Config& c = e->c;
  const int H = c.hidden;
  hipStream_t s = e->st;
  const bool split = e->world > 1 && e->g->overlap && n >= 256;
  const int n1 = split ? (n / 2 + 63) / 64 * 64 : n;
  const int rows[2] = {0, n1}, cnt[2] = {n1, n - n1};
  hipStream_t cs = e->cs;
  hipEvent_t* ev = e->ev;  // ev[0..3]: compute -> comm (attn A, attn B, mlp A, mlp B); ev[4..7]: comm -> compute
  for (int l = 0; l < c.layers; l++) {
    const Layer& w = e->L[l];
    half* kc = e->kc + e->layer_cache * l;
    half* vc = e->vc + e->layer_cache * l;
    if (!split) {
      attn_rows(e, w, kc, vc, slot, 0, n, pb, s);
      tp_allreduce(e, e->x, (size_t)n * H, s);
      mlp_rows(e, w, 0, n, s);
      tp_allreduce(e, e->x, (size_t)n * H, s);
      continue;
    }
    for (int stage = 0; stage < 2; stage++)      // 0: attention, 1: MLP
      for (int h = 0; h < 2; h++) {
        const int k = stage * 2 + h;
        // this half's previous all-reduce (MLP of the last layer, or attention of this one) must be done
        if (l > 0 || stage > 0) HI_CHECK(hipStreamWaitEvent(s, ev[4 + (k + 2) % 4], 0));
        if (stage == 0) attn_rows(e, w, kc, vc, slot, rows[h], cnt[h], pb, s);
        else mlp_rows(e, w, rows[h], cnt[h], s);
        HI_CHECK(hipEventRecord(ev[k], s));
        HI_CHECK(hipStreamWaitEvent(cs, ev[k], 0));
        tp_allreduce(e, e->x + (size_t)rows[h] * H, (size_t)cnt[h] * H, cs);
        HI_CHECK(hipEventRecord(ev[4 + k], cs));
      }
  }
  if (split) {  // the last two all-reduces before anything reads e->x
    HI_CHECK(hipStreamWaitEvent(s, ev[6], 0));
    HI_CHECK(hipStreamWaitEvent(s, ev[7], 0));
  }
}

// Prefill `T` prompt tokens into cache slot `slot`, continuing from position p0 (0 for a
// fresh sequence). Leaves the greedy next token in d_tok[slot], its position in d_pos[slot]
// and the last-position logits in logits[slot]. Returns that token. When `all_logits` is
// non-null (device pointer, [T, vocab] fp16) the logits of every position are written there too.
static int prefill(Engine* e, int slot, const int* tokens, int T, int p0, half* all_logits) {
  const Config& c = e->c;
  const int H = c.hidden, V = c.vocab, vl = e->vl;
  hipStream_t s = e->st;
  if (all_logits && e->world > 1 && !e->pf_logits) e->pf_logits = (half*)dalloc((size_t)e->max_pf * vl * 2);
  for (int t0 = 0; t0 < T; t0 += e->max_pf) {
    const int n = T - t0 < e->max_pf ? T - t0 : e->max_pf;
    const int pb = p0 + t0;
    HI_CHECK(hipMemcpyAsync(e->d_ptok, tokens + t0, n * 4, hipMemcpyHostToDevice, s));
    embed_kernel<<<n, 256, 0, s>>>(e->d_ptok, e->embed, e->x, H);
    dbg_copy(e, 0, e->x, H, 0, n, s);
    prefill_layers(e, slot, n, pb);
    if (all_logits) {
      hi_rmsnorm(e->x, e->final_norm, e->xn, n, H, c.eps, s);
      dbg_copy(e, dbg_planes(e) - 2, e->xn, H, 0, n, s);
      if (e->world == 1) {
        hi_gemm(e->xn, H, e->lm_head, nullptr, all_logits + (size_t)t0 * V, V, n, V, H, 0, s);
      } else {  // this rank's vocabulary columns (hi_prefill_logits requires T <= max_prefill: one chunk)
        hi_gemm(e->xn, H, e->lm_head, nullptr, e->pf_logits, vl, n, vl, H, 0, s);
        tp_export_logits(e, e->pf_logits, n, all_logits + e->v0, s);
      }
    }
    if (t0 + n >= T) {
      half* lg = e->logits + (size_t)slot * vl;
      if (norm_fits(1, H))
        hi_gemv(e->x + (size_t)(n - 1) * H, H, e->final_norm, c.eps, e->lm_head, lg, vl, 1, vl, H, 0, s);
      else {
        hi_rmsnorm(e->x + (size_t)(n - 1) * H, e->final_norm, e->xn, 1, H, c.eps, s);
        hi_gemv(e->xn, H, nullptr, 0.f, e->lm_head, lg, vl, 1, vl, H, 0, s);
      }
      const int pm1 = p0 + T - 1;
      HI_CHECK(hipMemcpyAsync(e->d_pos + slot, &pm1, 4, hipMemcpyHostToDevice, s));
      greedy(e, lg, 1, slot, s);
    }
  }
  int tok;
  HI_CHECK(hipMemcpyAsync(&tok, e->d_tok + slot, 4, hipMemcpyDeviceToHost, s));
  HI_CHECK(hipStreamSynchronize(s));
  return tok;
}

// Argument checks for the engine API (a failed one prints why and the call returns -1): an
// out-of-range slot, batch size, token or position would index past the KV cache, the RoPE
// table, the token history, the embedding or the per-batch-size graphs.
#define HI_REQ(cond, ...) do { if (!(cond)) { fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); return false; } } while (0)
static bool tokens_ok(const Engine* e, const int* tok, int n, const char* who) {
  for (int i = 0; i < n; i++) HI_REQ(tok[i] >= 0 && tok[i] < e->c.vocab, "%s: token %d at %d outside vocab %d", who, tok[i], i, e->c.vocab);
  return true;
}
static bool prefill_ok(const Engine* e, int slot, const int* tokens, int T, int p0, const char* who) {
  HI_REQ(slot >= 0 && slot < e->maxB, "%s: slot %d outside [0, max_batch %d)", who, slot, e->maxB);
  HI_REQ(T >= 1 && p0 >= 0, "%s: needs T >= 1 and p0 >= 0 (T %d, p0 %d)", who, T, p0);
  HI_REQ((long long)p0 + T <= e->max_seq, "%s: p0 %d + T %d exceeds max_seq %d", who, p0, T, e->max_seq);
  return tokens_ok(e, tokens, T, who);
}

EXPORT int hi_prefill(Engine* e, int slot, const int* tokens, int T, int p0) {
  if (!prefill_ok(e, slot, tokens, T, p0, "hi_prefill")) return -1;
  return rank0(e, [&](Engine* r) { return prefill(r, slot, tokens, T, p0, nullptr); });
}

// Prefill that also returns the logits of all T positions (device pointer dst on the engine's
// first GPU, [T, vocab] fp16), for perplexity evaluation. T <= max_prefill.
EXPORT int hi_prefill_logits(Engine* e, int slot, const int* tokens, int T, int p0, void* dst) {
  if (!prefill_ok(e, slot, tokens, T, p0, "hi_prefill_logits")) return -1;
  if (T > e->max_pf) { fprintf(stderr, "hi_prefill_logits: T %d > max_prefill %d\n", T, e->max_pf); return -1; }
  const int tok = rank0(e, [&](Engine* r) { return prefill(r, slot, tokens, T, p0, (half*)dst); });
  if (e->world > 1) tp_gather_logits(e, T, dst);
  return tok;
}

// Bytes hi_prefill_debug writes for T prompt tokens.
EXPORT size_t hi_debug_bytes(Engine* e, int T) {
  size_t n = 0;
  for (int p = 0; p < dbg_planes(e); p++) n += dbg_width(e, p);
  return n * (size_t)T * 2;
}

// Debug (scripts/layer_diff.py; see the dump layout above attn_rows): prefill
// like hi_prefill_logits and also copy every sub-step's activations for the T rows into dst, a
// device pointer on the engine's GPU of hi_debug_bytes(e, T) bytes (zeroed first, so a plane the
// engine never wrote compares as all zeros). One GPU, T <= max_prefill. Returns the greedy next
// token, or -1 for bad arguments.
EXPORT int hi_prefill_debug(Engine* e, int slot, const int* tokens, int T, int p0, void* dst) {
  if (!prefill_ok(e, slot, tokens, T, p0, "hi_prefill_debug")) return -1;
  if (e->world > 1) { fprintf(stderr, "hi_prefill_debug: tensor-parallel engines are not supported\n"); return -1; }
  if (T > e->max_pf) { fprintf(stderr, "hi_prefill_debug: T %d > max_prefill %d\n", T, e->max_pf); return -1; }
  DevGuard dg(e->dev);
  HI_CHECK(hipMemsetAsync(dst, 0, hi_debug_bytes(e, T), e->st));
  e->dbg = (half*)dst;
  e->dbg_T = T;
  const int tok = prefill(e, slot, tokens, T, p0, dbg_plane(e, dbg_planes(e) - 1));  // logits land in their plane
  e->dbg = nullptr;
  return tok;
}

// Batches up to HIPINFER_GEMV_MAX (default 4) rows take the GEMV path (fused norm / RoPE /
// cache-write epilogues, fp32 FMAs); larger ones the tensor-core GEMMs (skinny weight-streaming
// kernels up to 64 rows, tiled GEMM above).
static int gemv_max() {
  static const int v = getenv("HIPINFER_GEMV_MAX") ? atoi(getenv("HIPINFER_GEMV_MAX")) : 4;
  return v < 8 ? v : 8;
}

// One decode step for slots [0, B): feed d_tok at d_pos, write the next greedy token back
// into d_tok, advance d_pos, append to d_hist. Pure device-side state -> graph-capturable.
static void decode_step(Engine* e, int B) {
  const Config& c = e->c;
  const int H = c.hidden, D = c.head_dim, hq = e->hq, hkv = e->hkv, il = e->il, HD = hq * D, vl = e->vl;
  const int ep = e->rank == 0 ? 1 : 0;  // row-parallel outputs: rank 0 adds the residual, the others store partials
  hipStream_t s = e->st;
  const bool small = B <= gemv_max();
  // The fused RMSNorm keeps B normalized rows in shared memory; for very wide models on a
  // small-smem GPU (e.g. hidden 4096 x 8 rows on Turing) run the norm as its own kernel.
  const bool fuse = norm_fits(B, H);
  embed_kernel<<<B, 256, 0, s>>>(e->d_tok, e->embed, e->x, H);
  for (int l = 0; l < c.layers; l++) {
    const Layer& w = e->L[l];
    half* kc = e->kc + e->layer_cache * l;
    half* vc = e->vc + e->layer_cache * l;
    if (small && fuse) {
      hi_gemv_qkv_rope(e->x, H, w.attn_norm, c.eps, w.wqkv, w.bqkv, e->qkv, e->qkv_dim, B, H, hq, hkv, D, e->rope, e->d_pos,
                       kc, vc, e->max_seq, s);
    } else if (small) {
      hi_rmsnorm(e->x, w.attn_norm, e->xn, B, H, c.eps, s);
      hi_gemv_qkv_rope(e->xn, H, nullptr, c.eps, w.wqkv, w.bqkv, e->qkv, e->qkv_dim, B, H, hq, hkv, D, e->rope, e->d_pos,
                       kc, vc, e->max_seq, s);
    } else {
      hi_rmsnorm(e->x, w.attn_norm, e->xn, B, H, c.eps, s);
      hi_gemm(e->xn, H, w.wqkv, w.bqkv, e->qkv, e->qkv_dim, B, e->qkv_dim, H, 0, s);
      hi_rope_cache(e->qkv, e->qkv_dim, B, hq, hkv, D, e->rope, e->d_pos, 0, 0, kc, vc, e->max_seq, s);
    }
    hi_decode_attn(e->qkv, e->qkv_dim, kc, vc, e->max_seq, e->d_pos, e->att, HD, B, hq, hkv, D, e->work, s);
    if (c.moe) {  // one GPU only (no all-reduce); GEMV experts for small B, grouped GEMMs above
      if (small) hi_gemv(e->att, HD, nullptr, 0.f, w.wo, e->x, H, B, H, HD, ep, s);
      else hi_gemm(e->att, HD, w.wo, nullptr, e->x, H, B, H, HD, ep, s);
      moe_rows(e, w, 0, B, s);
      continue;
    }
    if (small) {
      hi_gemv(e->att, HD, nullptr, 0.f, w.wo, e->x, H, B, H, HD, ep, s);
      tp_allreduce(e, e->x, (size_t)B * H, s);
      if (fuse) hi_gemv(e->x, H, w.mlp_norm, c.eps, w.wgu, e->act, il, B, 2 * il, H, 2, s);
      else { hi_rmsnorm(e->x, w.mlp_norm, e->xn, B, H, c.eps, s); hi_gemv(e->xn, H, nullptr, 0.f, w.wgu, e->act, il, B, 2 * il, H, 2, s); }
      hi_gemv(e->act, il, nullptr, 0.f, w.wdown, e->x, H, B, H, il, ep, s);
    } else {
      hi_gemm(e->att, HD, w.wo, nullptr, e->x, H, B, H, HD, ep, s);
      tp_allreduce(e, e->x, (size_t)B * H, s);
      hi_rmsnorm(e->x, w.mlp_norm, e->xn, B, H, c.eps, s);
      hi_gemm(e->xn, H, w.wgu, nullptr, e->act, il, B, 2 * il, H, 2, s);
      hi_gemm(e->act, il, w.wdown, nullptr, e->x, H, B, H, il, ep, s);
    }
    tp_allreduce(e, e->x, (size_t)B * H, s);
  }
  if (small && fuse) hi_gemv(e->x, H, e->final_norm, c.eps, e->lm_head, e->logits, vl, B, vl, H, 0, s);
  else {
    hi_rmsnorm(e->x, e->final_norm, e->xn, B, H, c.eps, s);
    if (small) hi_gemv(e->xn, H, nullptr, 0.f, e->lm_head, e->logits, vl, B, vl, H, 0, s);
    else hi_gemm(e->xn, H, e->lm_head, nullptr, e->logits, vl, B, vl, H, 0, s);
  }
  greedy(e, e->logits, B, 0, s);
}

static float decode_rank(Engine* e, int B, int steps, int use_graph) {
  hipStream_t s = e->st;
  // MoE above the GEMV limit takes the grouped path, which syncs on the per-expert counts: no graph
  if (e->c.moe && B > gemv_max()) use_graph = 0;
  if (use_graph && !e->graphs[B]) {
    if (B > gemv_max()) {  // GEMM path: tune the decode shapes now, since tuning can't run inside capture
      const Config& c = e->c;
      const Layer& w = e->L[0];
      const int H = c.hidden, HD = e->hq * c.head_dim;
      gemm_choose(e->xn, H, w.wqkv, B, e->qkv_dim, H, 0, s);
      gemm_choose(e->att, HD, w.wo, B, H, HD, 1, s);
      gemm_choose(e->xn, H, w.wgu, B, 2 * e->il, H, 2, s);
      gemm_choose(e->act, e->il, w.wdown, B, H, e->il, 1, s);
      gemm_choose(e->xn, H, e->lm_head, B, e->vl, H, 0, s);
    }
    e->graphs[B] = capture(s, [&] { decode_step(e, B); });
  }
  return time_ms(s, [&] {
    for (int i = 0; i < steps; i++) {
      if (use_graph) HI_CHECK(hipGraphLaunch(e->graphs[B], s));
      else decode_step(e, B);
    }
  });
}

// Every decode step writes the KV cache at its slot's position: slots [0, B) must stay below
// max_seq for all `steps` (all ranks hold the same positions; rank 0's are read).
static bool decode_ok(Engine* e, int B, int steps) {
  HI_REQ(B >= 1 && B <= e->maxB, "hi_decode: B %d outside [1, max_batch %d]", B, e->maxB);
  HI_REQ(steps >= 0, "hi_decode: steps %d < 0", steps);
  DevGuard dg(e->dev);
  std::vector<int> pos(B);
  HI_CHECK(hipStreamSynchronize(e->st));
  HI_CHECK(hipMemcpy(pos.data(), e->d_pos, (size_t)B * 4, hipMemcpyDeviceToHost));
  for (int b = 0; b < B; b++)
    HI_REQ(pos[b] >= 0 && (long long)pos[b] + steps <= e->max_seq,
           "hi_decode: slot %d at position %d + %d steps exceeds max_seq %d", b, pos[b], steps, e->max_seq);
  return true;
}

// Run `steps` decode steps on slots [0, B). Returns elapsed GPU milliseconds (rank 0's), or -1
// for bad arguments.
EXPORT float hi_decode(Engine* e, int B, int steps, int use_graph) {
  if (!decode_ok(e, B, steps)) return -1.f;
  return rank0(e, [&](Engine* r) { return decode_rank(r, B, steps, use_graph); });
}

// One decode step on slots [0, B), run without a graph, that dumps into dst (device pointer on
// the engine's GPU, hi_debug_bytes(e, B) bytes) in the hi_prefill_debug layout with B rows. Only
// MoE layers dump: their mlp_norm (the router's input) and down planes; the rest stay zero.
// For scripts/moe_router_trace.py. Returns 0, or -1 for bad arguments.
EXPORT int hi_decode_debug(Engine* e, int B, void* dst) {
  if (!decode_ok(e, B, 1)) return -1;
  if (e->world > 1) { fprintf(stderr, "hi_decode_debug: tensor-parallel engines are not supported\n"); return -1; }
  DevGuard dg(e->dev);
  HI_CHECK(hipMemsetAsync(dst, 0, hi_debug_bytes(e, B), e->st));
  e->dbg = (half*)dst;
  e->dbg_T = B;
  decode_step(e, B);
  e->dbg = nullptr;
  HI_CHECK(hipStreamSynchronize(e->st));
  return 0;
}

// ------------------------------------------------------------------ speculative verification
// docs/speculative.md. R consecutive tokens of the sequence in slot `slot` (v_tok, at positions
// v_pos = p0 .. p0 + R - 1) go through the kernels a B=1 decode step uses, with R activation
// rows: the GEMVs (a row's arithmetic does not depend on the row count) and the decode attention
// with B=1's split count and every query reading this slot's cache. All R rows' k/v are written
// before any attention runs, and query i reads positions <= p0 + i only, so row i's logits, and
// the k/v it leaves in the cache, are bit-identical to what plain greedy decoding computes at
// position p0 + i. Accepting the draft tokens the target's argmax agrees with therefore
// reproduces plain greedy output token for token. Above 8 rows (or when the fused norm's shared
// memory does not fit) the RMSNorms run as rmsnorm_kernel, the same arithmetic as the GEMV's
// fused prologue. v_out[i] = the argmax after row i. Dense models, one GPU.
static void verify_step(Engine* e, int slot, int R) {
  const Config& c = e->c;
  const int H = c.hidden, D = c.head_dim, hq = e->hq, hkv = e->hkv, il = e->il, HD = hq * D, V = c.vocab;
  hipStream_t s = e->st;
  const bool fuse = R <= 8 && norm_fits(R, H);
  const int S = decode_splits(1, hkv);  // B=1's split count, whatever R is
  embed_kernel<<<R, 256, 0, s>>>(e->v_tok, e->embed, e->x, H);
  for (int l = 0; l < c.layers; l++) {
    const Layer& w = e->L[l];
    half* kc = e->kc + e->layer_cache * l;
    half* vc = e->vc + e->layer_cache * l;
    if (fuse) {
      gemv_qkv_rope(e->x, H, w.attn_norm, c.eps, w.wqkv, w.bqkv, e->qkv, e->qkv_dim, R, H, hq, hkv, D, e->rope, e->v_pos,
                    kc, vc, e->max_seq, slot, s);
    } else {
      hi_rmsnorm(e->x, w.attn_norm, e->xn, R, H, c.eps, s);
      gemv_qkv_rope(e->xn, H, nullptr, c.eps, w.wqkv, w.bqkv, e->qkv, e->qkv_dim, R, H, hq, hkv, D, e->rope, e->v_pos,
                    kc, vc, e->max_seq, slot, s);
    }
    decode_attn(e->qkv, e->qkv_dim, kc, vc, e->max_seq, e->v_pos, e->att, HD, R, hq, hkv, D, e->v_work, S, slot, s);
    hi_gemv(e->att, HD, nullptr, 0.f, w.wo, e->x, H, R, H, HD, 1, s);
    if (fuse) hi_gemv(e->x, H, w.mlp_norm, c.eps, w.wgu, e->act, il, R, 2 * il, H, 2, s);
    else { hi_rmsnorm(e->x, w.mlp_norm, e->xn, R, H, c.eps, s); hi_gemv(e->xn, H, nullptr, 0.f, w.wgu, e->act, il, R, 2 * il, H, 2, s); }
    hi_gemv(e->act, il, nullptr, 0.f, w.wdown, e->x, H, R, H, il, 1, s);
  }
  if (fuse) hi_gemv(e->x, H, e->final_norm, c.eps, e->lm_head, e->v_logits, V, R, V, H, 0, s);
  else {
    hi_rmsnorm(e->x, e->final_norm, e->xn, R, H, c.eps, s);
    hi_gemv(e->xn, H, nullptr, 0.f, e->lm_head, e->v_logits, V, R, V, H, 0, s);
  }
  argmax_kernel<<<R, 1024, 0, s>>>(e->v_logits, V, e->v_out, nullptr, nullptr, 0);
}

static bool verify_ok(const Engine* e, int slot, const int* tokens, int R, int p0) {
  HI_REQ(e->world == 1 && !e->c.moe, "hi_verify: dense models on one GPU only");
  HI_REQ(gemv_max() >= 1, "hi_verify: B=1 decode must take the GEMV path (HIPINFER_GEMV_MAX >= 1)");
  HI_REQ(R >= 1 && R <= kGemvMaxRows, "hi_verify: R %d outside [1, %d]", R, kGemvMaxRows);
  return prefill_ok(e, slot, tokens, R, p0, "hi_verify");
}

EXPORT int hi_verify_max_rows() { return kGemvMaxRows; }

// Speculative verification (verify_step): the greedy next token after each of the R tokens at
// positions p0 .. p0 + R - 1 of slot `slot`, into out[0, R) (host). With logits_dst (device
// pointer on the engine's GPU, [R, vocab] fp16) non-null the logits are copied there too. The
// rows' k/v are written to the cache; the slot's next token, position and history are left
// alone (hi_rewind sets them). Returns 0, or -1 for bad arguments.
EXPORT int hi_verify(Engine* e, int slot, const int* tokens, int R, int p0, int* out, void* logits_dst, int use_graph) {
  if (!verify_ok(e, slot, tokens, R, p0)) return -1;
  DevGuard dg(e->dev);
  hi_init();  // norm_fits and decode_splits read this device's properties
  hipStream_t s = e->st;
  if (!e->v_tok) {
    e->v_tok = (int*)dalloc(kGemvMaxRows * 4);
    e->v_pos = (int*)dalloc(kGemvMaxRows * 4);
    e->v_out = (int*)dalloc(kGemvMaxRows * 4);
    e->v_logits = (half*)dalloc((size_t)kGemvMaxRows * e->c.vocab * 2);
    e->v_work = dalloc(decode_work_bytes(kGemvMaxRows, e->hq, e->hkv, e->c.head_dim));
  }
  int pos[kGemvMaxRows];
  for (int i = 0; i < R; i++) pos[i] = p0 + i;
  HI_CHECK(hipMemcpyAsync(e->v_tok, tokens, (size_t)R * 4, hipMemcpyHostToDevice, s));
  HI_CHECK(hipMemcpyAsync(e->v_pos, pos, (size_t)R * 4, hipMemcpyHostToDevice, s));
  if (use_graph) {
    hipGraphExec_t& g = e->vgraphs[{slot, R}];
    if (!g) g = capture(s, [&] { verify_step(e, slot, R); });
    HI_CHECK(hipGraphLaunch(g, s));
  } else {
    verify_step(e, slot, R);
  }
  if (logits_dst)
    HI_CHECK(hipMemcpyAsync(logits_dst, e->v_logits, (size_t)R * e->c.vocab * 2, hipMemcpyDeviceToDevice, s));
  HI_CHECK(hipMemcpyAsync(out, e->v_out, (size_t)R * 4, hipMemcpyDeviceToHost, s));
  HI_CHECK(hipStreamSynchronize(s));
  return 0;
}

// Set slot `slot`'s cache length to `len` (positions [0, len) are kept) and its next input token
// to `tok`, so the next decode step feeds tok at position len. The cache is contiguous per slot,
// so this is the whole rollback of rejected speculative tokens. Returns 0, or -1 for bad arguments.
EXPORT int hi_rewind(Engine* e, int slot, int len, int tok) {
  auto ok = [&] {
    HI_REQ(slot >= 0 && slot < e->maxB, "hi_rewind: slot %d outside [0, max_batch %d)", slot, e->maxB);
    HI_REQ(len >= 0 && len < e->max_seq, "hi_rewind: length %d outside [0, max_seq %d)", len, e->max_seq);
    return tokens_ok(e, &tok, 1, "hi_rewind");
  };
  if (!ok()) return -1;
  each_rank(e, [&](Engine* r) {
    HI_CHECK(hipMemcpy(r->d_tok + slot, &tok, 4, hipMemcpyHostToDevice));
    HI_CHECK(hipMemcpy(r->d_pos + slot, &len, 4, hipMemcpyHostToDevice));
  });
  return 0;
}

EXPORT void hi_sync(Engine* e) { each_rank(e, [](Engine* r) { HI_CHECK(hipStreamSynchronize(r->st)); }); }

// Copy tokens hist[slot][from, from + n) (positions) to host. Rank 0's history is every rank's
// (and its last argmax waited for all of them). Returns 0, or -1 for bad arguments.
EXPORT int hi_history(Engine* e, int slot, int from, int n, int* out) {
  auto ok = [&] {
    HI_REQ(slot >= 0 && slot < e->maxB, "hi_history: slot %d outside [0, max_batch %d)", slot, e->maxB);
    HI_REQ(from >= 0 && n >= 0 && (long long)from + n <= e->max_seq,
           "hi_history: [%d, %d + %d) outside max_seq %d", from, from, n, e->max_seq);
    return true;
  };
  if (!ok()) return -1;
  DevGuard dg(e->dev);
  HI_CHECK(hipStreamSynchronize(e->st));
  HI_CHECK(hipMemcpy(out, e->d_hist + (size_t)slot * e->max_seq + from, (size_t)n * 4, hipMemcpyDeviceToHost));
  return 0;
}

// Copy the latest logits of slots [0, B) (fp16, [B, vocab]) to dst (host or device pointer).
// Returns 0, or -1 for bad arguments.
EXPORT int hi_logits(Engine* e, int B, void* dst) {
  if (B < 1 || B > e->maxB) { fprintf(stderr, "hi_logits: B %d outside [1, max_batch %d]\n", B, e->maxB); return -1; }
  if (e->world == 1) {
    DevGuard dg(e->dev);
    HI_CHECK(hipStreamSynchronize(e->st));
    HI_CHECK(hipMemcpy(dst, e->logits, (size_t)B * e->c.vocab * 2, hipMemcpyDefault));
    return 0;
  }
  each_rank(e, [&](Engine* r) {
    tp_export_logits(r, r->logits, B, (half*)dst + r->v0, r->st);
    HI_CHECK(hipStreamSynchronize(r->st));
  });
  tp_gather_logits(e, B, dst);
  return 0;
}

// Set slot b's next input token and its position (e.g. to restart a batch). Returns 0, or -1
// for bad arguments.
EXPORT int hi_set_state(Engine* e, int B, const int* tok, const int* pos) {
  auto ok = [&] {
    HI_REQ(B >= 1 && B <= e->maxB, "hi_set_state: B %d outside [1, max_batch %d]", B, e->maxB);
    for (int b = 0; b < B; b++)
      HI_REQ(pos[b] >= 0 && pos[b] < e->max_seq, "hi_set_state: slot %d position %d outside max_seq %d", b, pos[b], e->max_seq);
    return tokens_ok(e, tok, B, "hi_set_state");
  };
  if (!ok()) return -1;
  each_rank(e, [&](Engine* r) {
    HI_CHECK(hipMemcpy(r->d_tok, tok, B * 4, hipMemcpyHostToDevice));
    HI_CHECK(hipMemcpy(r->d_pos, pos, B * 4, hipMemcpyHostToDevice));
  });
  return 0;
}

EXPORT void* hi_stream(Engine* e) { return e->st; }
