#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#include "gemm_backend.h"
#include "gemm_native.h"
#include "platform.h"
#include "tensor.h"

#ifdef USE_ONEDNN
#include "gemm_onednn.h"
#endif

#define GEMM_TINY_THRESHOLD 4096

static int gemm_is_tiny(int M, int N, int K) {
    return (int64_t)M * (int64_t)N * (int64_t)K < GEMM_TINY_THRESHOLD;
}

const char* gemm_backend_name(void) {
#ifdef USE_ONEDNN
    return "oneDNN";
#elif defined(USE_OPENBLAS)
    return "OpenBLAS";
#else
    return "native/AVX";
#endif
}

const char* gemm_backend_int8_name(void) {
#ifdef USE_ONEDNN
    return "onednn";
#else
    return gemm_native_int8_backend_name();
#endif
}

status_t gemm_fp32(float* restrict C, const float* restrict A, const float* restrict B, int M, int N, int K,
                   float alpha, float beta) {
#ifdef USE_ONEDNN
    if (!gemm_is_tiny(M, N, K) && alpha == 1.0f && beta == 0.0f) {
        return gemm_onednn_fp32(C, A, B, M, N, K);
    }
#endif
    return gemm_native_fp32(C, A, B, M, N, K, alpha, beta);
}

status_t gemm_int8(float* restrict C, const int8_t* restrict A, const int8_t* restrict B,
                   const float* weight_scales, int M, int N, int K, float input_scale) {
#ifdef USE_ONEDNN
    if (!gemm_is_tiny(M, N, K)) {
        return gemm_onednn_int8(C, A, B, weight_scales, M, N, K, input_scale);
    }
#endif
    return gemm_native_int8(C, A, B, weight_scales, M, N, K, input_scale);
}

static float gemm_weight_input_scale(const float* X, int count) {
    float act_max = 0.0f;
    for (int i = 0; i < count; i++) {
        float v = fabsf(X[i]);
        if (v > act_max) act_max = v;
    }
    if (act_max < 1e-8f) act_max = 1e-8f;
    return act_max / 127.0f;
}

status_t gemm_weight_int8(float* restrict C, const int8_t* restrict W, const float* restrict X,
                          const float* weight_scales, int M, int N, int K) {
    if (!C || !W || !X || !weight_scales) return ERROR_NULL_POINTER;

    const int count = K * N;
    const float input_scale = gemm_weight_input_scale(X, count);

    int8_t* xq = (int8_t*)malloc_aligned((size_t)count, 64);
    if (!xq) return ERROR_OUT_OF_MEMORY;
    tensor_quantize_symmetric(X, xq, count, input_scale);

    status_t st = gemm_int8(C, W, xq, weight_scales, M, N, K, input_scale);
    free_aligned(xq);
    return st;
}

void gemm_backend_shutdown(void) {
#ifdef USE_ONEDNN
    gemm_onednn_shutdown();
#endif
}
