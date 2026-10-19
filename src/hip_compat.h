// Portability layer: the code is written against the HIP API. Under hipcc (AMD or
// NVIDIA platform) the real HIP headers are used; under plain nvcc the hip* names are
// mapped onto CUDA so the same source builds without a ROCm install.
#pragma once

#if defined(__HIPCC__)
  #include <hip/hip_runtime.h>
  #include <hip/hip_fp16.h>
  #if defined(__HIP_PLATFORM_AMD__)
    #define HI_AMD 1
    #include <rocwmma/rocwmma.hpp>
    namespace wmma = rocwmma;
  #else
    #include <mma.h>
    namespace wmma = nvcuda::wmma;
  #endif
#else
  #include <cuda_runtime.h>
  #include <cuda_fp16.h>
  #include <mma.h>
  namespace wmma = nvcuda::wmma;

  #define hipError_t                cudaError_t
  #define hipSuccess                cudaSuccess
  #define hipGetErrorString         cudaGetErrorString
  #define hipGetLastError           cudaGetLastError
  #define hipMalloc                 cudaMalloc
  #define hipFree                   cudaFree
  #define hipMemcpy                 cudaMemcpy
  #define hipMemcpyAsync            cudaMemcpyAsync
  #define hipMemset                 cudaMemset
  #define hipMemsetAsync            cudaMemsetAsync
  #define hipMemset2DAsync          cudaMemset2DAsync
  #define hipMemcpyHostToDevice     cudaMemcpyHostToDevice
  #define hipMemcpyDeviceToHost     cudaMemcpyDeviceToHost
  #define hipMemcpyDeviceToDevice   cudaMemcpyDeviceToDevice
  #define hipMemcpyDefault          cudaMemcpyDefault
  #define hipStream_t               cudaStream_t
  #define hipStreamCreate           cudaStreamCreate
  #define hipStreamDestroy          cudaStreamDestroy
  #define hipStreamSynchronize      cudaStreamSynchronize
  #define hipEvent_t                cudaEvent_t
  #define hipEventCreate            cudaEventCreate
  #define hipEventDestroy           cudaEventDestroy
  #define hipEventRecord            cudaEventRecord
  #define hipEventSynchronize       cudaEventSynchronize
  #define hipEventElapsedTime       cudaEventElapsedTime
  #define hipDeviceSynchronize      cudaDeviceSynchronize
  #define hipSetDevice              cudaSetDevice
  #define hipGetDevice              cudaGetDevice
  #define hipDeviceProp_t           cudaDeviceProp
  #define hipGetDeviceProperties    cudaGetDeviceProperties
  #define hipFuncSetAttribute       cudaFuncSetAttribute
  #define hipOccupancyMaxActiveBlocksPerMultiprocessor cudaOccupancyMaxActiveBlocksPerMultiprocessor
  #define hipFuncAttributeMaxDynamicSharedMemorySize cudaFuncAttributeMaxDynamicSharedMemorySize
  #define hipDeviceGetAttribute     cudaDeviceGetAttribute
  #define HI_ATTR_SMEM_OPTIN        cudaDevAttrMaxSharedMemoryPerBlockOptin
  #define hipStreamIsCapturing      cudaStreamIsCapturing
  #define hipStreamCaptureStatus    cudaStreamCaptureStatus
  #define hipStreamCaptureStatusNone cudaStreamCaptureStatusNone
  #define hipGraph_t                cudaGraph_t
  #define hipGraphExec_t            cudaGraphExec_t
  #define hipStreamBeginCapture     cudaStreamBeginCapture
  #define hipStreamEndCapture       cudaStreamEndCapture
  #define hipStreamCaptureModeThreadLocal cudaStreamCaptureModeThreadLocal
  #define hipGraphLaunch            cudaGraphLaunch
  #define hipGraphDestroy           cudaGraphDestroy
  #define hipGraphExecDestroy       cudaGraphExecDestroy
  #define hipDeviceCanAccessPeer    cudaDeviceCanAccessPeer
  #define hipDeviceEnablePeerAccess cudaDeviceEnablePeerAccess
  #define hipErrorPeerAccessAlreadyEnabled cudaErrorPeerAccessAlreadyEnabled
  #define hipHostMalloc             cudaHostAlloc
  #define hipHostFree               cudaFreeHost
  #define hipMemcpy2D               cudaMemcpy2D
  #define hipMemcpy2DAsync          cudaMemcpy2DAsync
  #define hipStreamWaitEvent        cudaStreamWaitEvent
  #define hipEventCreateWithFlags   cudaEventCreateWithFlags
  #define hipEventDisableTiming     cudaEventDisableTiming
  #define HI_HOST_SHARED_FLAGS      (cudaHostAllocMapped | cudaHostAllocPortable)
#endif

#if !defined(HI_HOST_SHARED_FLAGS)
  // pinned host memory every device maps, coherent for system-scope atomics
  #define HI_HOST_SHARED_FLAGS (hipHostMallocMapped | hipHostMallocPortable | hipHostMallocCoherent)
#endif

#if !defined(HI_ATTR_SMEM_OPTIN)
  // HIP has no opt-in attribute: the per-block LDS limit is the limit
  #define HI_ATTR_SMEM_OPTIN hipDeviceAttributeMaxSharedMemoryPerBlock
#endif

#include <cstdio>
#include <cstdlib>

#define HI_CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
  fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); \
  exit(1); } } while (0)

// Graph instantiation differs in signature between CUDA 12 and HIP.
static inline hipError_t hiGraphInstantiate(hipGraphExec_t* exec, hipGraph_t g) {
#if defined(__HIPCC__)
  return hipGraphInstantiate(exec, g, nullptr, nullptr, 0);
#else
  return cudaGraphInstantiate(exec, g, 0);
#endif
}

// Wavefront width is a compile-time property of the device target: 64 on CDNA (gfx9),
// 32 on RDNA and NVIDIA. Kernels derive warps-per-block from blockDim / HI_WARP, so the
// host only has to launch a multiple of 64 threads to be valid on every target.
#if defined(HI_AMD) && defined(__GFX9__)
  #define HI_WARP 64
#else
  #define HI_WARP 32
#endif

__device__ __forceinline__ float hi_shfl_xor(float v, int m) {
#if defined(HI_AMD)
  return __shfl_xor(v, m);
#else
  return __shfl_xor_sync(0xffffffffu, v, m);
#endif
}

__device__ __forceinline__ int hi_shfl_xor_i(int v, int m) {
#if defined(HI_AMD)
  return __shfl_xor(v, m);
#else
  return __shfl_xor_sync(0xffffffffu, v, m);
#endif
}

// Load that bypasses the (non-coherent) L1, for data written by other blocks in this launch.
__device__ __forceinline__ float hi_ldcg(const float* p) {
#if defined(HI_AMD)
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
  return __ldcg(p);
#endif
}

// System-scope flag store / load for the cross-GPU all-reduce barrier: the flags live in another
// GPU's memory (P2P) or in pinned host memory, written and polled by different devices.
__device__ __forceinline__ void hi_st_release_sys(unsigned* p, unsigned v) {
#if defined(HI_AMD)
  __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
  asm volatile("st.release.sys.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
#endif
}
__device__ __forceinline__ unsigned hi_ld_acquire_sys(const unsigned* p) {
#if defined(HI_AMD)
  return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
  unsigned v;
  asm volatile("ld.acquire.sys.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
#endif
}
// Short pause in a spin-wait, so pollers of remote flags leave the link to data traffic.
__device__ __forceinline__ void hi_backoff() {
#if defined(HI_AMD)
  __builtin_amdgcn_s_sleep(1);
#else
  __nanosleep(64);
#endif
}
// 16-byte load that is never served from a cache line filled before the barrier (remote data).
__device__ __forceinline__ uint4 hi_ld_remote(const uint4* p) {
#if defined(HI_AMD)
  const unsigned long long* q = reinterpret_cast<const unsigned long long*>(p);
  const unsigned long long a = __hip_atomic_load(q, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  const unsigned long long b = __hip_atomic_load(q + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  uint4 r;
  r.x = (unsigned)a; r.y = (unsigned)(a >> 32); r.z = (unsigned)b; r.w = (unsigned)(b >> 32);
  return r;
#else
  return __ldcv(p);
#endif
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = HI_WARP / 2; o > 0; o >>= 1) v += hi_shfl_xor(v, o);
  return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = HI_WARP / 2; o > 0; o >>= 1) v = fmaxf(v, hi_shfl_xor(v, o));
  return v;
}
