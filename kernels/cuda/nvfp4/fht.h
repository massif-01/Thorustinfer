
__device__ __forceinline__ void thorustinfer_fht128(float *values, int size) {
#pragma unroll
  for (int stage = 0; stage < 5; ++stage) {
#pragma unroll
    for (int i = 0; i < size; ++i) {
      values[i] = __shfl_xor_sync(0xffffffff, values[i], 1 << stage) +
                  values[i] * (1 - 2 * ((int(threadIdx.x) >> stage) & 1));
    }
  }
#pragma unroll
  for (int i = 0; i < size; i += 4) {
    float x0 = values[i], x1 = values[i + 1], x2 = values[i + 2],
          x3 = values[i + 3];
    float a = x1 + x0, b = x0 - x1, c = x3 + x2, d = x2 - x3;
    values[i] = c + a;
    values[i + 1] = d + b;
    values[i + 2] = a - c;
    values[i + 3] = b - d;
  }
}
__device__ __forceinline__ float thorustinfer_rotation_max(const float *values,
                                                      int size) {
  float result = 0;
#pragma unroll
  for (int i = 0; i < size; ++i)
    result = fmaxf(result, fabsf(values[i]));
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    result = fmaxf(result, __shfl_xor_sync(0xffffffff, result, mask));
  return result;
}
