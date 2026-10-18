// Eviction scoring and selection (docs/eviction.md). Device code only; included by api.cu after
// kernels.cuh.
//
// Scores are attention weights summed over queries and over the query heads of a kv head:
//   score[kv head][key] += sum_{query, head in group} softmax(q k^T / sqrt(D))[query, key]
// (H2O's accumulated attention, whole history). The attention kernels never hold the softmax
// matrix, so the weights are rebuilt afterwards from what they leave behind:
//   decode   decode_attn_*_kernel<TRACK> stores each key's logit (one query per head) and the
//            log-sum-exp; decode_score_kernel turns them into weights: p = exp2(logit - lse).
//   prefill  flash_prefill_*_kernel stores each row's log-sum-exp; prefill_score_kernel then
//            runs over keys instead of queries, recomputes q k^T, and column-sums
//            exp2(logit - lse) for its own keys, so no two blocks touch one score (no atomics).
// Logits and log-sum-exps are in the log2 domain (scaled by log2(e) / sqrt(D)), as the
// attention kernels' online softmax is.
//
// Selection, per problem (one (layer, kv head) of one sequence), over a ground set of n slots:
//   evict_select_kernel  the k smallest keys among eligible slots, ties to the lower index
//                        (radix select over order-preserving uint32 keys); H2O keeps the
//                        top-m scores with it, WFL removes the cheapest slots with it.
//   wfl_prep_kernel      centres keys and values per problem and L2-normalizes them (fp32), as
//                        SubCache's slot_similarity, and normalizes the weights w = score / sum.
//   wfl_top2_kernel      for each listed row i: best and second-best kept neighbour under
//                        sim(i, j) = max(cos_k, 0) * max(cos_v, 0) (squared for power 2),
//                        sim(i, i) = 1, computed on the fly (no n x n matrix). Ties go to the
//                        lower index.
//   wfl_round_kernel     the removal loss of every kept candidate, removes the k cheapest, and
//                        lists the rows whose best or second-best was removed: only those can
//                        change (the max over a subset that still holds both is the same pair),
//                        so the next top-2 pass rescans only them.
#pragma once
#include "kernels.cuh"

// ------------------------------------------------------------------ block helpers
__device__ __forceinline__ unsigned long long hi_ballot(bool p) {
#if defined(HI_AMD)
  return __ballot(p);
#else
  return __ballot_sync(0xffffffffu, p);
#endif
}

// Exclusive prefix count of `pred` over the block in thread order, and the block total.
// Every thread must call it. `sh` needs >= 33 ints (one per warp + total).
__device__ __forceinline__ int block_scan_pred(bool pred, int* sh, int* total) {
  const int lane = threadIdx.x % HI_WARP, w = threadIdx.x / HI_WARP, nw = blockDim.x / HI_WARP;
  const unsigned long long bal = hi_ballot(pred);
  const unsigned long long lt = lane == 0 ? 0ull : (~0ull >> (64 - lane));
  const int in_warp = __popcll(bal & lt);
  __syncthreads();  // sh may still be read by a previous call
  if (lane == 0) sh[w] = __popcll(bal);
  __syncthreads();
  if (threadIdx.x == 0) {
    int run = 0;
    for (int i = 0; i < nw; i++) { const int c = sh[i]; sh[i] = run; run += c; }
    sh[32] = run;
  }
  __syncthreads();
  *total = sh[32];
  return sh[w] + in_warp;
}

// Order-preserving map of a float to uint32 (larger float -> larger uint).
__device__ __forceinline__ unsigned f2key(float f) {
  const unsigned u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// The k-th smallest of key(j), j < n (k >= 1, keys of ineligible slots 0xffffffff, at least k
// eligible). Returns it in *T, and in *take how many slots with key == T belong to the k
// smallest (the rest of the k have key < T). 4-bit radix select, MSB first: 8 passes over n.
template <class KeyF>
__device__ void block_kth(int n, int k, KeyF key, unsigned* T, int* take) {
  __shared__ int hist[16];
  __shared__ unsigned s_prefix;
  __shared__ int s_k;
  unsigned prefix = 0, mask = 0;
  int kk = k;
  for (int shift = 28; shift >= 0; shift -= 4) {
    if (threadIdx.x < 16) hist[threadIdx.x] = 0;
    __syncthreads();
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
      const unsigned u = key(j);
      if ((u & mask) == prefix) atomicAdd(&hist[(u >> shift) & 15], 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      int cum = 0, b = 0;
      for (; b < 15; b++) {
        if (cum + hist[b] >= kk) break;
        cum += hist[b];
      }
      s_prefix = prefix | ((unsigned)b << shift);
      s_k = kk - cum;
    }
    __syncthreads();
    prefix = s_prefix;
    kk = s_k;
    mask |= 15u << shift;
    __syncthreads();  // s_prefix / s_k / hist are rewritten next pass
  }
  *T = prefix;
  *take = kk;
}

// Mark the k smallest keys (ties to the lower index): mark(j) is called once for each.
template <class KeyF, class MarkF>
__device__ void block_select_k(int n, int k, KeyF key, MarkF mark) {
  __shared__ int scan_sh[33];
  if (k <= 0) return;
  unsigned T;
  int take;
  block_kth(n, k, key, &T, &take);
  int base = 0;  // slots with key == T taken so far (in index order)
  for (int j0 = 0; j0 < n; j0 += blockDim.x) {
    const int j = j0 + threadIdx.x;
    const unsigned u = j < n ? key(j) : 0xffffffffu;
    int tot;
    const int r = block_scan_pred(j < n && u == T, scan_sh, &tot);
    if (j < n && (u < T || (u == T && base + r < take))) mark(j);
    base += tot;
  }
}

// ------------------------------------------------------------------ decode finish
// grid (blocks, Hkv, B): adds every key's attention weight, summed over the G query heads of its
// kv head, to score [B][Hkv][max_seq]. lg / lse come from decode_attn_*_kernel<TRACK>. Optional
// window (win_row[b] >= 0: this step's query is window row win_row[b]): its softmax rows go to
// win [B][Hq][W][max_seq] (fp16, if non-null) and its per-key mass is added to wscore.
__global__ void __launch_bounds__(256)
decode_score_kernel(const float* __restrict__ lg, const float* __restrict__ lse, const int* __restrict__ pos,
                    const int* __restrict__ shift, int max_seq, int Hkv, int G, float* score,
                    half* win, const int* __restrict__ win_row, int W, float* wscore) {
  const int hk = blockIdx.y, b = blockIdx.z, Hq = Hkv * G;
  const int len = pos[b] + 1 - (shift ? shift[b] : 0);
  __shared__ float ls[8];
  if (threadIdx.x < G) ls[threadIdx.x] = lse[(size_t)b * Hq + hk * G + threadIdx.x];
  __syncthreads();
  const int wr = win_row ? win_row[b] : -1;
  const float* l0 = lg + ((size_t)b * Hq + hk * G) * max_seq;
  float* sc = score + ((size_t)b * Hkv + hk) * max_seq;
  for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < len; j += gridDim.x * blockDim.x) {
    float sum = 0.f;
    for (int g = 0; g < G; g++) {
      const float p = exp2f(l0[(size_t)g * max_seq + j] - ls[g]);
      sum += p;
      if (wr >= 0 && win) win[(((size_t)b * Hq + hk * G + g) * W + wr) * max_seq + j] = __float2half(p);
    }
    sc[j] += sum;
    if (wr >= 0 && wscore) wscore[((size_t)b * Hkv + hk) * max_seq + j] += sum;
  }
}

// ------------------------------------------------------------------ prefill score pass
// Second prefill pass, parallel over keys. Block = (64 keys, one kv head), 4 warps. The K tile
// stays in shared memory; the block walks every query row that can see one of its keys, for
// each query head of the group, 16 rows per warp: S = Q K^T on WMMA (as flash_prefill_kernel),
// p = exp2(S * scale_log2 - lse[row]) masked causally, summed down each column in registers.
// q / lse / T / p0 as for the prefill attention (lse [Hq][T] from its first pass); kc is one
// slot's [Hkv][max_seq][D] cache with keys 0 .. p0 + T - 1 written. Adds to score
// [Hkv][max_seq]. Window: call rows >= win_r0 are window rows (row win_off + row - win_r0 of
// win [Hq][W][max_seq], fp16, if non-null); their per-key mass is added to wscore (if non-null).
template <int D>
__global__ void __launch_bounds__(4 * HI_WARP)
prefill_score_kernel(const half* __restrict__ q, int ldq, const half* __restrict__ kc, int max_seq,
                     const float* __restrict__ lse, int T, int p0, int Hq, int Hkv, float scale_log2,
                     float* score, half* win, int win_r0, int win_off, int W, float* wscore) {
  constexpr int BK = 64, LDK = D + 8, LDS = BK + 4, LDQ = D + 8, CPL = BK / HI_WARP;
  static_assert(16 * LDQ * 2 <= 16 * LDS * 4, "Q staging fits in the per-warp S tile");
  __shared__ __align__(32) half Ks[BK * LDK];
  __shared__ __align__(32) float Sb[4][16 * LDS];
  __shared__ float lse_s[4][16];
  __shared__ float red[2][4][BK];

  const int tid = threadIdx.x, w = tid / HI_WARP, lane = tid % HI_WARP;
  const int k0 = blockIdx.x * BK, hk = blockIdx.y, G = Hq / Hkv;
  const int kv_end = p0 + T;
  const half* Kh = kc + (size_t)hk * max_seq * D;
  for (int c = tid; c < BK * D / 8; c += 4 * HI_WARP) {
    const int r = c / (D / 8), d8 = (c % (D / 8)) * 8;
    *reinterpret_cast<uint4*>(Ks + r * LDK + d8) = (k0 + r < kv_end)
        ? *reinterpret_cast<const uint4*>(Kh + (size_t)(k0 + r) * D + d8) : make_uint4(0, 0, 0, 0);
  }
  __syncthreads();

  float col[CPL], wcol[CPL];
#pragma unroll
  for (int c = 0; c < CPL; c++) col[c] = wcol[c] = 0.f;
  // row i (at position p0 + i) sees key k iff k <= p0 + i: rows before k0 - p0 see none of these keys
  const int i_lo = max(0, k0 - p0) / 64 * 64;
  half* Qst = reinterpret_cast<half*>(Sb[w]);
  for (int g = 0; g < G; g++) {
    const int h = hk * G + g;
    for (int i0 = i_lo; i0 < T; i0 += 64) {
      const int r0 = i0 + w * 16;  // this warp's 16 rows; warps work independently from here on
      if (r0 >= T || p0 + r0 + 15 < k0) continue;
      for (int c = lane; c < 16 * D / 8; c += HI_WARP) {
        const int r = c / (D / 8), d8 = (c % (D / 8)) * 8;
        *reinterpret_cast<uint4*>(Qst + r * LDQ + d8) = (r0 + r < T)
            ? *reinterpret_cast<const uint4*>(q + (size_t)(r0 + r) * ldq + h * D + d8) : make_uint4(0, 0, 0, 0);
      }
      if (lane < 16) lse_s[w][lane] = (r0 + lane < T) ? lse[(size_t)h * T + r0 + lane] : 0.f;
      hi_syncwarp();
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> qa[D / 16];
#pragma unroll
      for (int k = 0; k < D / 16; k++) wmma::load_matrix_sync(qa[k], Qst + k * 16, LDQ);
      hi_syncwarp();
#pragma unroll
      for (int n = 0; n < BK / 16; n++) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> s;
        wmma::fill_fragment(s, 0.f);
#pragma unroll
        for (int k = 0; k < D / 16; k++) {
          wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> kb;
          wmma::load_matrix_sync(kb, Ks + n * 16 * LDK + k * 16, LDK);
          wmma::mma_sync(s, qa[k], kb, s);
        }
        wmma::store_matrix_sync(Sb[w] + n * 16, s, LDS, wmma::mem_row_major);
      }
      hi_syncwarp();
#pragma unroll
      for (int c = 0; c < CPL; c++) {
        const int cc = lane + c * HI_WARP, key = k0 + cc;
        if (key >= kv_end) continue;
        float sum = 0.f, wsum = 0.f;
        for (int r = 0; r < 16; r++) {
          const int row = r0 + r;
          if (row >= T) break;
          const float p = key <= p0 + row ? exp2f(Sb[w][r * LDS + cc] * scale_log2 - lse_s[w][r]) : 0.f;
          sum += p;
          if (row >= win_r0) {
            wsum += p;
            if (win) win[((size_t)h * W + win_off + row - win_r0) * max_seq + key] = __float2half(p);
          }
        }
        col[c] += sum;
        wcol[c] += wsum;
      }
      hi_syncwarp();  // Sb[w] / lse_s[w] are rewritten next iteration
    }
  }
#pragma unroll
  for (int c = 0; c < CPL; c++) { red[0][w][lane + c * HI_WARP] = col[c]; red[1][w][lane + c * HI_WARP] = wcol[c]; }
  __syncthreads();
  if (tid < BK && k0 + tid < kv_end) {
    const size_t at = (size_t)hk * max_seq + k0 + tid;
    score[at] += red[0][0][tid] + red[0][1][tid] + red[0][2][tid] + red[0][3][tid];
    if (wscore) wscore[at] += red[1][0][tid] + red[1][1][tid] + red[1][2][tid] + red[1][3][tid];
  }
}

// ------------------------------------------------------------------ selection
// Problem p of P is (layer p / hkv, kv head p % hkv); its slot j of a per-slot array lives at
// base + (p / hkv) * ls + (p % hkv) * hs + j (times D for the K / V cache).
struct ProbStride {
  long long ls, hs;
  int hkv;
  __host__ __device__ long long at(int p) const { return (p / hkv) * ls + (long long)(p % hkv) * hs; }
};

// grid P, block 1024: flags[p][j] = 1 for the k slots of [lo, hi) with the largest score
// (largest = 1) or smallest (0); ties to the lower index. Other flags are left as they are.
__global__ void __launch_bounds__(1024)
evict_select_kernel(const float* __restrict__ score, ProbStride st, int lo, int hi, int k, int largest,
                    unsigned char* flags, int ld) {
  const int p = blockIdx.x;
  const float* s = score + st.at(p);
  unsigned char* f = flags + (size_t)p * ld;
  block_select_k(hi - lo, k,
                 [&](int j) { const unsigned u = f2key(s[lo + j]); return largest ? ~u : u; },
                 [&](int j) { f[lo + j] = 1; });
}

// grid P, block 1024: the ground set of WFL in slot order, gidx[p][0..ng): slots < sink, slots
// >= n - recent (status 1: held), and the candidates in between that are in the pool
// (pool[p][j] != 0, or all of them when pool is null; status 0).
__global__ void __launch_bounds__(1024)
evict_ground_kernel(int n, int sink, int recent, const unsigned char* __restrict__ pool, int* gidx,
                    unsigned char* status, int ld) {
  __shared__ int sh[33];
  const int p = blockIdx.x;
  int base = 0;
  for (int j0 = 0; j0 < n; j0 += blockDim.x) {
    const int j = j0 + threadIdx.x;
    const bool held = j < sink || j >= n - recent;
    const bool in = j < n && (held || !pool || pool[(size_t)p * n + j]);
    int tot;
    const int r = block_scan_pred(in, sh, &tot);
    if (in) { gidx[(size_t)p * ld + base + r] = j; status[(size_t)p * ld + base + r] = held ? 1 : 0; }
    base += tot;
  }
}

// grid P, block 256: gather the ground set of every problem: keys and values centred over the
// ground set and L2-normalized (fp32, kn / vn [P][n][D]; torch normalize's 1e-12 floor), the
// weights w = score / max(sum score, 1e-9), and reset the WFL state: kept = 1, every row listed
// for the first top-2 pass, hit = -1.
__global__ void __launch_bounds__(256)
wfl_prep_kernel(const half* __restrict__ kc, const half* __restrict__ vc, ProbStride kst,
                const float* __restrict__ score, ProbStride sst, const int* __restrict__ gidx, int n, int D,
                float* kn, float* vn, float* w, unsigned char* kept, int* hit, int* rl, int* rcount) {
  __shared__ float red[32];
  __shared__ float part[2][256];
  __shared__ float mean_k[128], mean_v[128];
  const int p = blockIdx.x, tid = threadIdx.x, lane = tid % HI_WARP, wp = tid / HI_WARP, nw = blockDim.x / HI_WARP;
  const half* K = kc + kst.at(p) * D;
  const half* V = vc + kst.at(p) * D;
  const float* S = score + sst.at(p);
  const int* gi = gidx + (size_t)p * n;
  // per-dimension means: thread (grp, d) sums rows grp, grp + ngrp, ... (D divides 256, D <= 128)
  {
    const int ngrp = blockDim.x / D, grp = tid / D, d = tid % D;
    float sk = 0.f, sv = 0.f;
    for (int i = grp; i < n; i += ngrp) {
      sk += __half2float(K[(size_t)gi[i] * D + d]);
      sv += __half2float(V[(size_t)gi[i] * D + d]);
    }
    part[0][tid] = sk;
    part[1][tid] = sv;
    __syncthreads();
    if (tid < D) {
      float a = 0.f, b = 0.f;
      for (int g = 0; g < ngrp; g++) { a += part[0][g * D + tid]; b += part[1][g * D + tid]; }
      mean_k[tid] = a / n;
      mean_v[tid] = b / n;
    }
  }
  float ws = 0.f;
  for (int i = tid; i < n; i += blockDim.x) ws += S[gi[i]];
  ws = block_sum(ws, red);
  const float winv = 1.f / fmaxf(ws, 1e-9f);
  // one warp per row: centre, normalize
  for (int i = wp; i < n; i += nw) {
    const size_t src = (size_t)gi[i] * D, dst = ((size_t)p * n + i) * D;
    float nk = 0.f, nv = 0.f;
    for (int d = lane; d < D; d += HI_WARP) {
      const float a = __half2float(K[src + d]) - mean_k[d], b = __half2float(V[src + d]) - mean_v[d];
      nk += a * a;
      nv += b * b;
    }
    nk = 1.f / fmaxf(sqrtf(warp_sum(nk)), 1e-12f);
    nv = 1.f / fmaxf(sqrtf(warp_sum(nv)), 1e-12f);
    for (int d = lane; d < D; d += HI_WARP) {
      kn[dst + d] = (__half2float(K[src + d]) - mean_k[d]) * nk;
      vn[dst + d] = (__half2float(V[src + d]) - mean_v[d]) * nv;
    }
  }
  for (int i = tid; i < n; i += blockDim.x) {
    const size_t at = (size_t)p * n + i;
    w[at] = S[gi[i]] * winv;
    kept[at] = 1;
    hit[at] = -1;
    rl[at] = i;
  }
  if (tid == 0) rcount[p] = n;
}

// grid (ceil(n / 64), P), block 256. Rows: the first rcount[p] entries of rl[p]; for each, the
// best and second-best kept column (value, index; ties to the lower index; second is -3 / -1
// when fewer than two columns are kept). Thread (ty, tx) owns rows ty*4 .. +3 and columns
// tx*4 .. +3 of each 64 x 64 tile: k and v dot products in fp32 over D in chunks of 32 through
// shared memory, then a running top-2 per row over its columns (visited in increasing order),
// merged across the 16 threads of a row at the end.
struct Top2 { float b1, b2; int a1, a2; };
__device__ __forceinline__ bool t2_better(float v, int i, float u, int j) { return v > u || (v == u && i < j); }
__device__ __forceinline__ Top2 t2_merge(Top2 x, Top2 y) {  // x, y over disjoint columns
  Top2 r;
  if (t2_better(y.b1, y.a1, x.b1, x.a1)) {
    r.b1 = y.b1; r.a1 = y.a1;
    if (t2_better(x.b1, x.a1, y.b2, y.a2)) { r.b2 = x.b1; r.a2 = x.a1; } else { r.b2 = y.b2; r.a2 = y.a2; }
  } else {
    r.b1 = x.b1; r.a1 = x.a1;
    if (t2_better(y.b1, y.a1, x.b2, x.a2)) { r.b2 = y.b1; r.a2 = y.a1; } else { r.b2 = x.b2; r.a2 = x.a2; }
  }
  return r;
}

__global__ void __launch_bounds__(256)
wfl_top2_kernel(const float* __restrict__ kn, const float* __restrict__ vn, int n, int D,
                const unsigned char* __restrict__ kept, const int* __restrict__ rl, const int* __restrict__ rcount,
                int power, float* best, float* sec, int* arg, int* arg2) {
  constexpr int BT = 64, DC = 32;
  __shared__ __align__(16) float As[2][DC][BT];  // [k / v][d][row]
  __shared__ __align__(16) float Bs[2][DC][BT];  // [k / v][d][col]
  __shared__ int rows_s[BT];
  __shared__ unsigned char kept_s[BT];
  const int p = blockIdx.y, r0 = blockIdx.x * BT, cnt = rcount[p];
  if (r0 >= cnt) return;
  const int tid = threadIdx.x, ty = tid / 16, tx = tid % 16;
  if (tid < BT) rows_s[tid] = r0 + tid < cnt ? rl[(size_t)p * n + r0 + tid] : -1;
  const float* K = kn + (size_t)p * n * D;
  const float* V = vn + (size_t)p * n * D;
  Top2 t[4];
#pragma unroll
  for (int a = 0; a < 4; a++) t[a] = Top2{-3.f, -3.f, 0x7fffffff, -1};
  // tile loads: thread -> (row tid % 64, dims (tid / 64) * 4 .. +3 and +16 .. +19)
  const int lr = tid % BT, ld0 = (tid / BT) * 4;
  for (int c0 = 0; c0 < n; c0 += BT) {
    float ck[4][4], cv[4][4];
#pragma unroll
    for (int a = 0; a < 4; a++)
#pragma unroll
      for (int c = 0; c < 4; c++) ck[a][c] = cv[a][c] = 0.f;
    for (int d0 = 0; d0 < D; d0 += DC) {
      __syncthreads();  // previous tile fully consumed (and rows_s visible on the first pass)
      if (d0 == 0 && tid < BT) kept_s[tid] = c0 + tid < n ? kept[(size_t)p * n + c0 + tid] : 0;
      const int ra = rows_s[lr], cb = c0 + lr;
#pragma unroll
      for (int h = 0; h < 2; h++) {
        const int dd = ld0 + h * 16;
        float4 ak = make_float4(0.f, 0.f, 0.f, 0.f), av = ak, bk = ak, bv = ak;
        if (ra >= 0) {
          ak = *reinterpret_cast<const float4*>(K + (size_t)ra * D + d0 + dd);
          av = *reinterpret_cast<const float4*>(V + (size_t)ra * D + d0 + dd);
        }
        if (cb < n) {
          bk = *reinterpret_cast<const float4*>(K + (size_t)cb * D + d0 + dd);
          bv = *reinterpret_cast<const float4*>(V + (size_t)cb * D + d0 + dd);
        }
        As[0][dd][lr] = ak.x; As[0][dd + 1][lr] = ak.y; As[0][dd + 2][lr] = ak.z; As[0][dd + 3][lr] = ak.w;
        As[1][dd][lr] = av.x; As[1][dd + 1][lr] = av.y; As[1][dd + 2][lr] = av.z; As[1][dd + 3][lr] = av.w;
        Bs[0][dd][lr] = bk.x; Bs[0][dd + 1][lr] = bk.y; Bs[0][dd + 2][lr] = bk.z; Bs[0][dd + 3][lr] = bk.w;
        Bs[1][dd][lr] = bv.x; Bs[1][dd + 1][lr] = bv.y; Bs[1][dd + 2][lr] = bv.z; Bs[1][dd + 3][lr] = bv.w;
      }
      __syncthreads();
#pragma unroll 8
      for (int d = 0; d < DC; d++) {
        const float4 xk = *reinterpret_cast<const float4*>(&As[0][d][ty * 4]);
        const float4 xv = *reinterpret_cast<const float4*>(&As[1][d][ty * 4]);
        const float4 yk = *reinterpret_cast<const float4*>(&Bs[0][d][tx * 4]);
        const float4 yv = *reinterpret_cast<const float4*>(&Bs[1][d][tx * 4]);
        const float ak[4] = {xk.x, xk.y, xk.z, xk.w}, av[4] = {xv.x, xv.y, xv.z, xv.w};
        const float bk[4] = {yk.x, yk.y, yk.z, yk.w}, bv[4] = {yv.x, yv.y, yv.z, yv.w};
#pragma unroll
        for (int a = 0; a < 4; a++)
#pragma unroll
          for (int c = 0; c < 4; c++) {
            ck[a][c] = fmaf(ak[a], bk[c], ck[a][c]);
            cv[a][c] = fmaf(av[a], bv[c], cv[a][c]);
          }
      }
    }
#pragma unroll
    for (int c = 0; c < 4; c++) {
      const int col = c0 + tx * 4 + c;
      if (col >= n || !kept_s[tx * 4 + c]) continue;
#pragma unroll
      for (int a = 0; a < 4; a++) {
        const int row = rows_s[ty * 4 + a];
        float s = fmaxf(ck[a][c], 0.f) * fmaxf(cv[a][c], 0.f);
        if (power == 2) s *= s;
        if (row == col) s = 1.f;
        if (s > t[a].b1) { t[a].b2 = t[a].b1; t[a].a2 = t[a].a1; t[a].b1 = s; t[a].a1 = col; }
        else if (s > t[a].b2) { t[a].b2 = s; t[a].a2 = col; }
      }
    }
  }
  // merge the 16 threads (tx) of each row group; they are 16 consecutive lanes of one warp
#pragma unroll
  for (int a = 0; a < 4; a++) {
#pragma unroll
    for (int off = 1; off < 16; off <<= 1) {
      Top2 o;
      o.b1 = hi_shfl_xor(t[a].b1, off); o.b2 = hi_shfl_xor(t[a].b2, off);
      o.a1 = hi_shfl_xor_i(t[a].a1, off); o.a2 = hi_shfl_xor_i(t[a].a2, off);
      t[a] = t2_merge(t[a], o);
    }
    const int row = rows_s[ty * 4 + a];
    if (tx == 0 && row >= 0) {
      const size_t at = (size_t)p * n + row;
      best[at] = t[a].b1; sec[at] = t[a].b2; arg[at] = t[a].a1; arg2[at] = t[a].a2;
    }
  }
}

// grid P, block 1024. One removal round of policies.wfl_prune:
//   loss[j] = mix * sum_{i : arg_i = j} w_i * max(best_i - max(sec_i, 0), 0) + (1 - mix) * w_j
// over kept candidates j (status 0); the k with the smallest loss (ties to the lower index) are
// removed (kept = 0, hit = round). Then rl / rcount list the rows whose best or second-best
// column was removed, for the next top-2 pass (all rows when incremental is 0).
__global__ void __launch_bounds__(1024)
wfl_round_kernel(int n, int k, int round, float mix, int incremental, const float* __restrict__ w,
                 const float* __restrict__ best, const float* __restrict__ sec, const int* __restrict__ arg,
                 const int* __restrict__ arg2, const unsigned char* __restrict__ status, float* acc,
                 unsigned char* kept, int* hit, int* rl, int* rcount) {
  __shared__ int count;
  const int p = blockIdx.x;
  const size_t o = (size_t)p * n;
  for (int j = threadIdx.x; j < n; j += blockDim.x) acc[o + j] = 0.f;
  if (threadIdx.x == 0) count = 0;
  __syncthreads();
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float g = w[o + i] * fmaxf(best[o + i] - fmaxf(sec[o + i], 0.f), 0.f);
    if (g > 0.f) atomicAdd(&acc[o + arg[o + i]], g);
  }
  __syncthreads();
  block_select_k(n, k,
                 [&](int j) {
                   return (kept[o + j] && status[o + j] == 0) ? f2key(mix * hi_ldcg(acc + o + j) + (1.f - mix) * w[o + j])
                                                               : 0xffffffffu;
                 },
                 [&](int j) { kept[o + j] = 0; hit[o + j] = round; });
  __syncthreads();
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int b2 = arg2[o + i];
    if (!incremental || hit[o + arg[o + i]] == round || (b2 >= 0 && hit[o + b2] == round))
      rl[o + atomicAdd(&count, 1)] = i;
  }
  __syncthreads();
  if (threadIdx.x == 0) rcount[p] = count;
}

// grid P (one layer's problems), block 1024: klist[p][0..cap) = the kept slots in order, where a
// slot j < n is kept if flags[p][j] != 0, j < sink or j >= n - recent. err[0] is set when a
// problem does not keep exactly cap slots.
__global__ void __launch_bounds__(1024)
evict_list_kernel(int n, int sink, int recent, const unsigned char* __restrict__ flags, int ld, int cap,
                  int* klist, int* err) {
  __shared__ int sh[33];
  const int p = blockIdx.x;
  int base = 0;
  for (int j0 = 0; j0 < n; j0 += blockDim.x) {
    const int j = j0 + threadIdx.x;
    const bool keep = j < n && (j < sink || j >= n - recent || flags[(size_t)p * ld + j]);
    int tot;
    const int r = block_scan_pred(keep, sh, &tot);
    if (keep && base + r < cap) klist[(size_t)p * cap + base + r] = j;
    base += tot;
  }
  if (threadIdx.x == 0 && base != cap) atomicExch(err, 1);
}

// grid (cap, P): row i of problem p <- row klist[p][i] (rows of D halves and the per-slot
// floats), through tmp so that the in-place move cannot read an already overwritten row. phase 0
// gathers into tmp, phase 1 copies tmp back to rows 0 .. cap - 1.
__global__ void __launch_bounds__(128)
evict_move_kernel(int phase, const int* __restrict__ klist, int cap, half* kc, half* vc, ProbStride kst, int D,
                  float* s0, float* s1, ProbStride sst, half* tmp_kv, float* tmp_s) {
  const int i = blockIdx.x, p = blockIdx.y;
  const int src = klist[(size_t)p * cap + i];
  half* K = kc + kst.at(p) * D;
  half* V = vc + kst.at(p) * D;
  half* tk = tmp_kv + ((size_t)p * cap + i) * 2 * D;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    if (phase == 0) { tk[d] = K[(size_t)src * D + d]; tk[D + d] = V[(size_t)src * D + d]; }
    else { K[(size_t)i * D + d] = tk[d]; V[(size_t)i * D + d] = tk[D + d]; }
  }
  if (threadIdx.x == 0) {
    float* ts = tmp_s + ((size_t)p * cap + i) * 2;
    float* a = s0 + sst.at(p);
    float* b = s1 ? s1 + sst.at(p) : nullptr;
    if (phase == 0) { ts[0] = a[src]; ts[1] = b ? b[src] : 0.f; }
    else { a[i] = ts[0]; if (b) b[i] = ts[1]; }
  }
}

// grid P, block 256: out[p][gidx[p][i]] = in[p][i] for i < ng (WFL's kept candidates back to
// slot indices); out must be zeroed first.
__global__ void scatter_flags_kernel(const unsigned char* __restrict__ in, const int* __restrict__ gidx, int ng,
                                     unsigned char* out, int ld) {
  const int p = blockIdx.x;
  for (int i = threadIdx.x; i < ng; i += blockDim.x)
    if (in[(size_t)p * ng + i]) out[(size_t)p * ld + gidx[(size_t)p * ng + i]] = 1;
}

// grid P, block 256: out[p][i] = kept[p][i] && status[p][i] == 0 (WFL's kept candidates).
__global__ void wfl_output_kernel(const unsigned char* __restrict__ kept, const unsigned char* __restrict__ status, int n,
                                  unsigned char* out) {
  const size_t o = (size_t)blockIdx.x * n;
  for (int i = threadIdx.x; i < n; i += blockDim.x) out[o + i] = kept[o + i] && status[o + i] == 0;
}
