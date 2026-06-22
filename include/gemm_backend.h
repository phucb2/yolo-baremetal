#ifndef GEMM_BACKEND_H
#define GEMM_BACKEND_H

#include <stdint.h>
#include "status.h"

/* Active GEMM backend label for logs/benches (compile-time selected). */
const char* gemm_backend_name(void);

/* Sub-backend for INT8 (e.g. avx2, onednn). */
const char* gemm_backend_int8_name(void);

/* C = alpha * A * B + beta * C; A[M×K], B[K×N], C[M×N] row-major. */
status_t gemm_fp32(float* restrict C, const float* restrict A, const float* restrict B, int M, int N, int K,
                   float alpha, float beta);

status_t gemm_int8(float* restrict C, const int8_t* restrict A, const int8_t* restrict B,
                   const float* weight_scales, int M, int N, int K, float input_scale);

status_t gemm_weight_int8(float* restrict C, const int8_t* restrict W, const float* restrict X,
                          const float* weight_scales, int M, int N, int K);

void gemm_backend_shutdown(void);

#endif
