// Thor FFN ABI v1. Device addresses and scratch remain owned by the caller.
// Work is enqueued on the supplied stream. rows is 1..4096. Scratch is
// 64 MiB with 256-byte alignment. W/S/G mean E2M1 bank, CUTLASS-layout E4M3
// block scales and FP32 tensor scale. Both outputs use the model's FP16 layout.
// kind 1: normalized input, gate W/S/G, gate output, scratch, rotation signs,
//         up W/S/G, up output (11 pointers).
// kind 0: rotated SwiGLU input, down W/S/G, output, scratch, tile maxima
//         (7 pointers). Zero denotes success.
#pragma once
#include <cuda_runtime_api.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
__attribute__((visibility("default"))) int
orinfer_nvfp4_ffn_v1(int kind, int rows, const uint64_t *pointers, cudaStream_t stream);
#ifdef __cplusplus
}
#endif
