#ifndef TENSOR_H
#define TENSOR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "status.h"

typedef enum {
    TENSOR_DTYPE_FP32 = 0,
    TENSOR_DTYPE_INT8 = 1,
} tensor_dtype_t;

typedef struct {
    float* restrict data;
    int8_t* restrict qdata;
    float* scales;
    int num_scales;
    tensor_dtype_t dtype;
    int dims[4]; // N, C, H, W
    int stride[4];
    bool is_owner;
} tensor_t;

status_t tensor_allocate(tensor_t* tensor, int n, int c, int h, int w);
status_t tensor_free(tensor_t* tensor);
status_t tensor_copy(tensor_t* dest, const tensor_t* src);
status_t tensor_fill(tensor_t* tensor, float value);

/* Symmetric quantize float plane to int8: q = round(x / scale), clamped. */
void tensor_quantize_symmetric(const float* src, int8_t* dst, int count, float scale);

// Optimized GEMM: C = alpha * A * B + beta * C
// A: M x K, B: K x N, C: M x N
status_t tensor_gemm(float* restrict C, const float* restrict A, const float* restrict B, int M, int N, int K,
                     float alpha, float beta);

/* INT8 GEMM: C_fp32[i,j] = input_scale * weight_scales[i] * sum_k A[i,k]*B[k,j]. weight_scales has M entries. */
status_t tensor_gemm_int8(float* restrict C, const int8_t* restrict A, const int8_t* restrict B,
                          const float* weight_scales, int M, int N, int K, float input_scale);

/* Weight-only INT8: quantize activations per GEMM, then SIMD s8×s8 matmul. */
status_t tensor_gemm_weight_int8(float* restrict C, const int8_t* restrict W, const float* restrict X,
                                 const float* weight_scales, int M, int N, int K);

/* Human-readable INT8 GEMM kernel (avx512-vnni, avx-vnni, avx2, neon, scalar). */
const char* tensor_gemm_int8_backend(void);

// Memory alignment for SIMD (NEON)
void* malloc_aligned(size_t size, size_t alignment);
void free_aligned(void* ptr);

#endif
