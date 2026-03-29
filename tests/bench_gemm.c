/* Micro-benchmark for tensor_gemm (1x1 conv path). Build: make tests/bench_gemm */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tensor.h"
#include "utils.h"

static void fill_pattern(float* p, size_t n) {
    for (size_t i = 0; i < n; i++) {
        p[i] = (float)((i * 7u) % 100u) * 0.001f - 0.05f;
    }
}

static void run_case(const char* label, int M, int N, int K, int repeats) {
    float* A = (float*)malloc((size_t)M * (size_t)K * sizeof(float));
    float* B = (float*)malloc((size_t)K * (size_t)N * sizeof(float));
    float* C = (float*)malloc((size_t)M * (size_t)N * sizeof(float));
    if (!A || !B || !C) {
        fprintf(stderr, "bench_gemm: allocation failed for %s\n", label);
        free(A);
        free(B);
        free(C);
        return;
    }
    fill_pattern(A, (size_t)M * (size_t)K);
    fill_pattern(B, (size_t)K * (size_t)N);

    for (int w = 0; w < 3; w++) {
        tensor_gemm(C, A, B, M, N, K, 1.0f, 0.0f);
    }

    timer_t t;
    timer_start(&t);
    for (int r = 0; r < repeats; r++) {
        tensor_gemm(C, A, B, M, N, K, 1.0f, 0.0f);
    }
    timer_stop(&t);
    double ms = timer_elapsed_ms(&t);
    double total_flops = 2.0 * (double)M * (double)N * (double)K * (double)repeats;
    double gflops = (total_flops / (ms / 1000.0)) / 1e9;

    printf("  %-28s M=%4d N=%6d K=%4d reps=%5d  total=%8.2f ms  %.3f GFLOP/s\n", label, M, N, K, repeats,
           ms, gflops);

    free(A);
    free(B);
    free(C);
}

int main(void) {
#ifdef USE_OPENBLAS
    const char* backend = "OpenBLAS (cblas_sgemm)";
#else
    const char* backend = "scalar/AVX (hand-rolled)";
#endif
    printf("tensor_gemm micro-benchmark — %s\n", backend);
    printf("  (set OPENBLAS_NUM_THREADS=1 for stable OpenBLAS timings)\n\n");

    run_case("tiny (unit-test-like)", 5, 7, 4, 50000);
    run_case("medium", 128, 320, 128, 500);
    /* Conv-like 1x1: out_c x (H*W) @ in_c — large spatial N */
    run_case("large-N (conv-like)", 256, 4096, 256, 20);

    return 0;
}
