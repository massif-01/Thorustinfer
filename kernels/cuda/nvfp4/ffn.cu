#include "abi.h"
#include "fht.h"
#include "gemm.cuh"
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <exception>

using Ptr = uint64_t;
constexpr size_t capacity = 64ull << 20;
__global__ void rotate_gate(const half *x, const signed char *signs, half *rot,
                            float *maxima) {
  const int row = blockIdx.y, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int group = blockIdx.x * 4 + warp, base = group * 128 + lane;
  float v[4];
#pragma unroll
  for (int i = 0; i < 4; i++)
    v[i] = __half2float(x[row * 2560 + base + 32 * i]) * signs[base + 32 * i];
  thorustinfer_fht128(v, 4);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    half a = __float2half_rn(v[i] * 0.08838834764831845f);
    rot[row * 2560 + base + 32 * i] = a;
    v[i] = __half2float(a);
  }
  float mx = thorustinfer_rotation_max(v, 4);
  if (!lane)
    maxima[row * 20 + group] = mx;
}
// Scale each token independently. The CUTLASS epilogue broadcasts its alpha
// across that token's output channels, so other owners cannot change its codes.
__global__ void set_scales(const float *maxima, int groups, const float *w1,
                           const float *w2, float *inverse, float *alpha1,
                           float *alpha2) {
  const int row = blockIdx.x, lane = threadIdx.x;
  float maximum = 0;
  for (int i = lane; i < groups; i += 32)
    maximum = fmaxf(maximum, maxima[row * groups + i]);
  for (int d = 16; d; d /= 2)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, d));
  if (lane == 0) {
    const float scale = maximum ? maximum / 2688.f : 1.f;
    inverse[row] = 1.f / scale;
    alpha1[row] = scale * w1[0];
    alpha2[row] = scale * w2[0];
  }
}
template <int K>
__global__ void pack(const half *x, const float *g, unsigned char *q,
                     unsigned char *scales) {
  int row = blockIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= K / 16)
    return;
  const half *src = x + (row * K + j * 16);
  float v[16], mx = 0;
#pragma unroll
  for (int i = 0; i < 16; i++) {
    v[i] = __half2float(src[i]);
    mx = fmaxf(mx, fabsf(v[i]));
  }
  unsigned char code = __nv_cvt_float_to_fp8(mx * g[row] * (1.f / 6.f),
                                             __NV_SATFINITE, __NV_E4M3);
  float sf =
      __half2float(static_cast<half>(__nv_cvt_fp8_to_halfraw(code, __NV_E4M3)));
  int idx = ((((row / 128) * (K / 64) + j / 4) * 32 + (row % 32)) * 4 +
             (row % 128) / 32) *
                4 +
            j % 4;
  scales[idx] = code;
  float inv = sf ? g[row] / sf : 0.f;
  unsigned long long bits = 0;
#pragma unroll
  for (int i = 0; i < 8; i++)
    bits |= (unsigned long long)__nv_cvt_float2_to_fp4x2(
                make_float2(v[2 * i] * inv, v[2 * i + 1] * inv), __NV_E2M1,
                cudaRoundNearest)
            << (8 * i);
  reinterpret_cast<unsigned long long *>(q)[row * (K / 16) + j] = bits;
}
struct Arena {
  char *start;
  size_t used = 0;
  void *get(size_t n) {
    void *p = start + used;
    used += (n + 255) & ~size_t(255);
    return p;
  }
};
static void gemm(int m, int n, int k, void *y, const void *q, const void *w,
                 const void *as, const void *ws, const float *alpha, Arena a,
                 cudaStream_t st) {
  nvfp4_gemm(m, n, k, y, q, w, as, ws, alpha, a.start + a.used,
             capacity - a.used, st);
}
// Gate/up use one prepared activation and distinct global weight scales.
// Outputs have separate model-defined views.
extern "C" __attribute__((visibility("default"))) int
orinfer_nvfp4_ffn_v1(int kind, int m, const Ptr *b, cudaStream_t st) {
  if ((kind != 0 && kind != 1) || m <= 0 || m > 4096)
    return -1;
  bool gate = kind == 1;
  int k = gate ? 2560 : 9216, n = gate ? 9216 : 2560;
  Arena a{(char *)b[5]};
  auto *g = (float *)a.get(size_t(m) * sizeof(float));
  auto *alpha1 = (float *)a.get(size_t(m) * sizeof(float));
  auto *alpha2 = (float *)a.get(size_t(m) * sizeof(float));
  auto *q = (unsigned char *)a.get(size_t(m) * k / 2);
  auto *as = (unsigned char *)a.get(size_t((m + 127) / 128) * 128 * (k / 16));
  const half *rot = (const half *)b[0];
  const float *mx = (const float *)b[6];
  if (gate) {
    auto *r = (half *)a.get(size_t(m) * k * 2);
    auto *v = (float *)a.get(size_t(m) * 20 * 4);
    rotate_gate<<<dim3(5, m), 128, 0, st>>>((const half *)b[0],
                                            (const signed char *)b[6], r, v);
    if (auto error = cudaGetLastError(); error != cudaSuccess)
      return int(error);
    rot = r;
    mx = v;
  }
  set_scales<<<m, 32, 0, st>>>(mx, gate ? 20 : 36, (const float *)b[3],
                               (const float *)(gate ? b[9] : b[3]), g, alpha1,
                               alpha2);
  if (auto error = cudaGetLastError(); error != cudaSuccess)
    return int(error);
  if (gate)
    pack<2560><<<dim3(2, m), 128, 0, st>>>(rot, g, q, as);
  else
    pack<9216><<<dim3(5, m), 128, 0, st>>>(rot, g, q, as);
  if (auto error = cudaGetLastError(); error != cudaSuccess)
    return int(error);
  try {
    gemm(m, n, k, (void *)b[4], q, (void *)b[1], as, (void *)b[2], alpha1, a,
         st);
    if (gate)
      gemm(m, n, k, (void *)b[10], q, (void *)b[7], as, (void *)b[8], alpha2, a,
           st);
  } catch (const std::exception &e) {
    fprintf(stderr, "NVFP4 FFN: %s\n", e.what());
    return -2;
  }
  return int(cudaGetLastError());
}
