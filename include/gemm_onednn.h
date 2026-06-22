#ifndef GEMM_ONEDNN_H
#define GEMM_ONEDNN_H

#include <stdint.h>
#include "status.h"

status_t gemm_onednn_fp32(float* restrict C, const float* restrict A, const float* restrict B, int M, int N, int K);

status_t gemm_onednn_int8(float* restrict C, const int8_t* restrict A, const int8_t* restrict B,
                          const float* weight_scales, int M, int N, int K, float input_scale);

void gemm_onednn_shutdown(void);

#endif
