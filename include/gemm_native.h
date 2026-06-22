#ifndef GEMM_NATIVE_H
#define GEMM_NATIVE_H

#include <stdint.h>
#include "status.h"

status_t gemm_native_fp32(float* restrict C, const float* restrict A, const float* restrict B, int M, int N, int K,
                          float alpha, float beta);

status_t gemm_native_int8(float* restrict C, const int8_t* restrict A, const int8_t* restrict B,
                           const float* weight_scales, int M, int N, int K, float input_scale);

status_t gemm_native_weight_int8(float* restrict C, const int8_t* restrict W, const float* restrict X,
                                 const float* weight_scales, int M, int N, int K);

const char* gemm_native_int8_backend_name(void);

#endif
