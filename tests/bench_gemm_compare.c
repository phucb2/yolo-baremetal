/* Side-by-side benchmark: native GEMM vs oneDNN (FP32 + INT8).
 * Build: cmake --preset win-onednn, target bench_gemm_compare */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gemm_backend.h"
#include "gemm_native.h"
#include "gemm_onednn.h"
#include "tensor.h"
#include "utils.h"

static void fill_pattern(float* p, size_t n) {
    for (size_t i = 0; i < n; i++) {
        p[i] = (float)((i * 7u) % 100u) * 0.001f - 0.05f;
    }
}

static float max_abs_diff(const float* a, const float* b, size_t n) {
    float md = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > md) md = d;
    }
    return md;
}

static float compute_input_scale(const float* X, int count) {
    float act_max = 0.0f;
    for (int i = 0; i < count; i++) {
        float v = fabsf(X[i]);
        if (v > act_max) act_max = v;
    }
    if (act_max < 1e-8f) act_max = 1e-8f;
    return act_max / 127.0f;
}

static void run_fp32_case(const char* label, int M, int N, int K, int repeats) {
    float* A = (float*)malloc((size_t)M * (size_t)K * sizeof(float));
    float* B = (float*)malloc((size_t)K * (size_t)N * sizeof(float));
    float* C_native = (float*)malloc((size_t)M * (size_t)N * sizeof(float));
    float* C_odnn = (float*)malloc((size_t)M * (size_t)N * sizeof(float));
    if (!A || !B || !C_native || !C_odnn) {
        fprintf(stderr, "bench_gemm_compare: allocation failed for %s\n", label);
        free(A);
        free(B);
        free(C_native);
        free(C_odnn);
        return;
    }
    fill_pattern(A, (size_t)M * (size_t)K);
    fill_pattern(B, (size_t)K * (size_t)N);

    for (int w = 0; w < 3; w++) {
        gemm_native_fp32(C_native, A, B, M, N, K, 1.0f, 0.0f);
        gemm_onednn_fp32(C_odnn, A, B, M, N, K);
    }

    gemm_native_fp32(C_native, A, B, M, N, K, 1.0f, 0.0f);
    gemm_onednn_fp32(C_odnn, A, B, M, N, K);
    float diff = max_abs_diff(C_native, C_odnn, (size_t)M * (size_t)N);

    timer_t t_native, t_odnn;
    timer_start(&t_native);
    for (int r = 0; r < repeats; r++) {
        gemm_native_fp32(C_native, A, B, M, N, K, 1.0f, 0.0f);
    }
    timer_stop(&t_native);

    timer_start(&t_odnn);
    for (int r = 0; r < repeats; r++) {
        gemm_onednn_fp32(C_odnn, A, B, M, N, K);
    }
    timer_stop(&t_odnn);

    double ms_native = timer_elapsed_ms(&t_native);
    double ms_odnn = timer_elapsed_ms(&t_odnn);
    double total_flops = 2.0 * (double)M * (double)N * (double)K * (double)repeats;
    double gflops_native = (total_flops / (ms_native / 1000.0)) / 1e9;
    double gflops_odnn = (total_flops / (ms_odnn / 1000.0)) / 1e9;

    printf("  %-28s M=%4d N=%6d K=%4d\n", label, M, N, K);
    printf("    native       %8.2f ms  %7.3f GFLOP/s\n", ms_native, gflops_native);
    printf("    oneDNN       %8.2f ms  %7.3f GFLOP/s  speedup=%.2fx  max|diff|=%.2e\n", ms_odnn, gflops_odnn,
           ms_native / ms_odnn, diff);

    free(A);
    free(B);
    free(C_native);
    free(C_odnn);
}

static void run_int8_case(const char* label, int M, int N, int K, int repeats) {
    int8_t* W = (int8_t*)malloc((size_t)M * (size_t)K);
    float* X = (float*)malloc((size_t)K * (size_t)N * sizeof(float));
    float* scales = (float*)malloc((size_t)M * sizeof(float));
    float* C_native = (float*)malloc((size_t)M * (size_t)N * sizeof(float));
    float* C_odnn = (float*)malloc((size_t)M * (size_t)N * sizeof(float));
    if (!W || !X || !scales || !C_native || !C_odnn) {
        fprintf(stderr, "bench_gemm_compare: int8 allocation failed for %s\n", label);
        free(W);
        free(X);
        free(scales);
        free(C_native);
        free(C_odnn);
        return;
    }
    for (int i = 0; i < M * K; i++) W[i] = (int8_t)((i % 17) - 8);
    fill_pattern(X, (size_t)K * (size_t)N);
    for (int i = 0; i < M; i++) scales[i] = 0.01f + (float)i * 0.001f;

    const int count = K * N;

    for (int w = 0; w < 3; w++) {
        gemm_native_weight_int8(C_native, W, X, scales, M, N, K);
        float is = compute_input_scale(X, count);
        int8_t* xqw = (int8_t*)malloc((size_t)count);
        if (xqw) {
            tensor_quantize_symmetric(X, xqw, count, is);
            gemm_onednn_int8(C_odnn, W, xqw, scales, M, N, K, is);
            free(xqw);
        }
    }

    gemm_native_weight_int8(C_native, W, X, scales, M, N, K);

    float input_scale = compute_input_scale(X, count);
    int8_t* xq = (int8_t*)malloc((size_t)count);
    if (!xq) {
        fprintf(stderr, "bench_gemm_compare: xq allocation failed\n");
        free(W);
        free(X);
        free(scales);
        free(C_native);
        free(C_odnn);
        return;
    }
    tensor_quantize_symmetric(X, xq, count, input_scale);
    gemm_onednn_int8(C_odnn, W, xq, scales, M, N, K, input_scale);
    float diff = max_abs_diff(C_native, C_odnn, (size_t)M * (size_t)N);

    timer_t t_native, t_odnn;
    timer_start(&t_native);
    for (int r = 0; r < repeats; r++) {
        gemm_native_weight_int8(C_native, W, X, scales, M, N, K);
    }
    timer_stop(&t_native);

    timer_start(&t_odnn);
    for (int r = 0; r < repeats; r++) {
        input_scale = compute_input_scale(X, count);
        tensor_quantize_symmetric(X, xq, count, input_scale);
        gemm_onednn_int8(C_odnn, W, xq, scales, M, N, K, input_scale);
    }
    timer_stop(&t_odnn);

    double ms_native = timer_elapsed_ms(&t_native);
    double ms_odnn = timer_elapsed_ms(&t_odnn);
    double total_ops = 2.0 * (double)M * (double)N * (double)K * (double)repeats;
    double gops_native = (total_ops / (ms_native / 1000.0)) / 1e9;
    double gops_odnn = (total_ops / (ms_odnn / 1000.0)) / 1e9;

    printf("  %-28s M=%4d N=%6d K=%4d\n", label, M, N, K);
    printf("    native weight_int8  %8.2f ms  %7.3f Gop/s\n", ms_native, gops_native);
    printf("    oneDNN              %8.2f ms  %7.3f Gop/s  speedup=%.2fx  max|diff|=%.2e\n", ms_odnn, gops_odnn,
           ms_native / ms_odnn, diff);

    free(W);
    free(X);
    free(scales);
    free(C_native);
    free(C_odnn);
    free(xq);
}

int main(void) {
    printf("GEMM compare: native vs oneDNN\n");
    printf("  active backend (tensor_gemm): %s\n", gemm_backend_name());
    printf("  INT8 sub-backend: %s\n", gemm_backend_int8_name());
    printf("  oneDNN threads: 1 (set OMP_NUM_THREADS=1 if needed)\n\n");

    printf("FP32 (C = A * B)\n");
    run_fp32_case("tiny (unit-test-like)", 5, 7, 4, 50000);
    run_fp32_case("medium", 128, 320, 128, 500);
    run_fp32_case("large-N (conv-like)", 256, 4096, 256, 20);

    printf("\nINT8 weight-only (quantize X + matmul)\n");
    run_int8_case("int8 medium", 128, 320, 128, 500);
    run_int8_case("int8 large-N (conv-like)", 256, 4096, 256, 20);

    gemm_backend_shutdown();
    return 0;
}
