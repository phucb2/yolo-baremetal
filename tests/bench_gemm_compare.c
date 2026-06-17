/* Side-by-side benchmark: tensor_gemm vs oneDNN matmul (FP32 + INT8).
 * Build: cmake -DUSE_ONEDNN=ON with vcpkg toolchain, target bench_gemm_compare */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dnnl.h"
#include "tensor.h"
#include "utils.h"

#define DNNL_CHECK(st)                                                                                                 \
    do {                                                                                                               \
        dnnl_status_t _st = (st);                                                                                      \
        if (_st != dnnl_success) {                                                                                     \
            fprintf(stderr, "oneDNN error %d at %s:%d\n", (int)_st, __FILE__, __LINE__);                               \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

typedef struct {
    dnnl_engine_t engine;
    dnnl_stream_t stream;
} odnn_runtime_t;

typedef struct {
    dnnl_primitive_t prim;
    dnnl_memory_t src_mem;
    dnnl_memory_t weights_mem;
    dnnl_memory_t dst_mem;
    dnnl_memory_t scratchpad_mem;
    int has_scratchpad;
} odnn_matmul_t;

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

static dnnl_memory_desc_t make_ab_md(const dnnl_dims_t dims, dnnl_data_type_t dt) {
    dnnl_memory_desc_t md = NULL;
    dnnl_dims_t strides = {dims[1], 1};
    DNNL_CHECK(dnnl_memory_desc_create_with_strides(&md, 2, dims, dt, strides));
    return md;
}

static void odnn_matmul_destroy(odnn_matmul_t* ctx) {
    if (!ctx) return;
    if (ctx->prim) dnnl_primitive_destroy(ctx->prim);
    if (ctx->src_mem) dnnl_memory_destroy(ctx->src_mem);
    if (ctx->weights_mem) dnnl_memory_destroy(ctx->weights_mem);
    if (ctx->dst_mem) dnnl_memory_destroy(ctx->dst_mem);
    if (ctx->scratchpad_mem) dnnl_memory_destroy(ctx->scratchpad_mem);
    memset(ctx, 0, sizeof(*ctx));
}

static void odnn_matmul_init_fp32(odnn_matmul_t* ctx, odnn_runtime_t* rt, int M, int N, int K, float* A, float* B,
                                  float* C) {
    memset(ctx, 0, sizeof(*ctx));

    dnnl_dims_t src_dims = {M, K};
    dnnl_dims_t weights_dims = {K, N};
    dnnl_dims_t dst_dims = {M, N};

    dnnl_memory_desc_t src_md = make_ab_md(src_dims, dnnl_f32);
    dnnl_memory_desc_t weights_md = make_ab_md(weights_dims, dnnl_f32);
    dnnl_memory_desc_t dst_md = make_ab_md(dst_dims, dnnl_f32);

    dnnl_primitive_desc_t pd;
    DNNL_CHECK(dnnl_matmul_primitive_desc_create(&pd, rt->engine, src_md, weights_md, NULL, dst_md, NULL));
    DNNL_CHECK(dnnl_primitive_create(&ctx->prim, pd));

    DNNL_CHECK(dnnl_memory_create(&ctx->src_mem, src_md, rt->engine, A));
    DNNL_CHECK(dnnl_memory_create(&ctx->weights_mem, weights_md, rt->engine, B));
    DNNL_CHECK(dnnl_memory_create(&ctx->dst_mem, dst_md, rt->engine, C));

    const_dnnl_memory_desc_t scratchpad_md = dnnl_primitive_desc_query_md(pd, dnnl_query_scratchpad_md, 0);
    if (scratchpad_md && dnnl_memory_desc_get_size(scratchpad_md) > 0) {
        DNNL_CHECK(dnnl_memory_create(&ctx->scratchpad_mem, scratchpad_md, rt->engine, NULL));
        ctx->has_scratchpad = 1;
    }

    dnnl_memory_desc_destroy(src_md);
    dnnl_memory_desc_destroy(weights_md);
    dnnl_memory_desc_destroy(dst_md);
    dnnl_primitive_desc_destroy(pd);
}

static void odnn_matmul_init_int8(odnn_matmul_t* ctx, odnn_runtime_t* rt, int M, int N, int K, int8_t* W, int8_t* xq,
                                  float* C) {
    memset(ctx, 0, sizeof(*ctx));

    dnnl_dims_t src_dims = {M, K};
    dnnl_dims_t weights_dims = {K, N};
    dnnl_dims_t dst_dims = {M, N};

    dnnl_memory_desc_t src_md = NULL;
    dnnl_memory_desc_t weights_md = NULL;
    dnnl_memory_desc_t dst_md = NULL;
    DNNL_CHECK(dnnl_memory_desc_create_with_tag(&src_md, 2, src_dims, dnnl_s8, dnnl_ab));
    DNNL_CHECK(dnnl_memory_desc_create_with_tag(&weights_md, 2, weights_dims, dnnl_s8, dnnl_ab));
    DNNL_CHECK(dnnl_memory_desc_create_with_tag(&dst_md, 2, dst_dims, dnnl_f32, dnnl_ab));

    dnnl_primitive_desc_t pd;
    dnnl_status_t st = dnnl_matmul_primitive_desc_create(&pd, rt->engine, src_md, weights_md, NULL, dst_md, NULL);
    if (st == dnnl_unimplemented) {
        dnnl_memory_desc_destroy(dst_md);
        DNNL_CHECK(dnnl_memory_desc_create_with_tag(&dst_md, 2, dst_dims, dnnl_s32, dnnl_ab));
        st = dnnl_matmul_primitive_desc_create(&pd, rt->engine, src_md, weights_md, NULL, dst_md, NULL);
    }
    DNNL_CHECK(st);
    DNNL_CHECK(dnnl_primitive_create(&ctx->prim, pd));

    DNNL_CHECK(dnnl_memory_create(&ctx->src_mem, src_md, rt->engine, W));
    DNNL_CHECK(dnnl_memory_create(&ctx->weights_mem, weights_md, rt->engine, xq));
    DNNL_CHECK(dnnl_memory_create(&ctx->dst_mem, dst_md, rt->engine, C));

    const_dnnl_memory_desc_t scratchpad_md = dnnl_primitive_desc_query_md(pd, dnnl_query_scratchpad_md, 0);
    if (scratchpad_md && dnnl_memory_desc_get_size(scratchpad_md) > 0) {
        DNNL_CHECK(dnnl_memory_create(&ctx->scratchpad_mem, scratchpad_md, rt->engine, NULL));
        ctx->has_scratchpad = 1;
    }

    dnnl_memory_desc_destroy(src_md);
    dnnl_memory_desc_destroy(weights_md);
    dnnl_memory_desc_destroy(dst_md);
    dnnl_primitive_desc_destroy(pd);
}

static void odnn_matmul_execute(odnn_matmul_t* ctx, odnn_runtime_t* rt) {
    dnnl_exec_arg_t args[8];
    int n_args = 0;
    args[n_args++] = (dnnl_exec_arg_t){DNNL_ARG_SRC, ctx->src_mem};
    args[n_args++] = (dnnl_exec_arg_t){DNNL_ARG_WEIGHTS, ctx->weights_mem};
    args[n_args++] = (dnnl_exec_arg_t){DNNL_ARG_DST, ctx->dst_mem};
    if (ctx->has_scratchpad) {
        args[n_args++] = (dnnl_exec_arg_t){DNNL_ARG_SCRATCHPAD, ctx->scratchpad_mem};
    }
    DNNL_CHECK(dnnl_primitive_execute(ctx->prim, rt->stream, n_args, args));
    DNNL_CHECK(dnnl_stream_wait(rt->stream));
}

static void apply_row_scales(float* C, int M, int N, const float* row_scales) {
    for (int i = 0; i < M; i++) {
        const float s = row_scales[i];
        float* row = C + i * N;
        for (int j = 0; j < N; j++) row[j] *= s;
    }
}

static void odnn_int8_weight_gemm(float* C, const float* weight_scales, float input_scale, int M, int N,
                                  odnn_matmul_t* ctx, odnn_runtime_t* rt) {
    float row_scales[512];
    float* scales = row_scales;
    float* scales_heap = NULL;
    if (M > (int)(sizeof(row_scales) / sizeof(row_scales[0]))) {
        scales_heap = (float*)malloc((size_t)M * sizeof(float));
        if (!scales_heap) {
            fprintf(stderr, "bench_gemm_compare: row scale allocation failed\n");
            exit(1);
        }
        scales = scales_heap;
    }
    for (int i = 0; i < M; i++) scales[i] = weight_scales[i] * input_scale;

    odnn_matmul_execute(ctx, rt);
    apply_row_scales(C, M, N, scales);
    free(scales_heap);
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

static void run_fp32_case(odnn_runtime_t* rt, const char* label, int M, int N, int K, int repeats) {
    float* A = (float*)malloc((size_t)M * (size_t)K * sizeof(float));
    float* B = (float*)malloc((size_t)K * (size_t)N * sizeof(float));
    float* C_yolo = (float*)malloc((size_t)M * (size_t)N * sizeof(float));
    float* C_odnn = (float*)malloc((size_t)M * (size_t)N * sizeof(float));
    if (!A || !B || !C_yolo || !C_odnn) {
        fprintf(stderr, "bench_gemm_compare: allocation failed for %s\n", label);
        free(A);
        free(B);
        free(C_yolo);
        free(C_odnn);
        return;
    }
    fill_pattern(A, (size_t)M * (size_t)K);
    fill_pattern(B, (size_t)K * (size_t)N);

    odnn_matmul_t odnn;
    odnn_matmul_init_fp32(&odnn, rt, M, N, K, A, B, C_odnn);

    for (int w = 0; w < 3; w++) {
        tensor_gemm(C_yolo, A, B, M, N, K, 1.0f, 0.0f);
        odnn_matmul_execute(&odnn, rt);
    }

    tensor_gemm(C_yolo, A, B, M, N, K, 1.0f, 0.0f);
    odnn_matmul_execute(&odnn, rt);
    float diff = max_abs_diff(C_yolo, C_odnn, (size_t)M * (size_t)N);

    timer_t t_yolo, t_odnn;
    timer_start(&t_yolo);
    for (int r = 0; r < repeats; r++) {
        tensor_gemm(C_yolo, A, B, M, N, K, 1.0f, 0.0f);
    }
    timer_stop(&t_yolo);

    timer_start(&t_odnn);
    for (int r = 0; r < repeats; r++) {
        odnn_matmul_execute(&odnn, rt);
    }
    timer_stop(&t_odnn);

    double ms_yolo = timer_elapsed_ms(&t_yolo);
    double ms_odnn = timer_elapsed_ms(&t_odnn);
    double total_flops = 2.0 * (double)M * (double)N * (double)K * (double)repeats;
    double gflops_yolo = (total_flops / (ms_yolo / 1000.0)) / 1e9;
    double gflops_odnn = (total_flops / (ms_odnn / 1000.0)) / 1e9;
    double speedup = ms_yolo / ms_odnn;

    printf("  %-28s M=%4d N=%6d K=%4d\n", label, M, N, K);
    printf("    tensor_gemm  %8.2f ms  %7.3f GFLOP/s\n", ms_yolo, gflops_yolo);
    printf("    oneDNN       %8.2f ms  %7.3f GFLOP/s  speedup=%.2fx  max|diff|=%.2e\n", ms_odnn, gflops_odnn,
           speedup, diff);

    odnn_matmul_destroy(&odnn);
    free(A);
    free(B);
    free(C_yolo);
    free(C_odnn);
}

static void run_int8_case(odnn_runtime_t* rt, const char* label, int M, int N, int K, int repeats) {
    int8_t* W = (int8_t*)malloc((size_t)M * (size_t)K);
    float* X = (float*)malloc((size_t)K * (size_t)N * sizeof(float));
    float* scales = (float*)malloc((size_t)M * sizeof(float));
    float* C_yolo = (float*)malloc((size_t)M * (size_t)N * sizeof(float));
    float* C_odnn = (float*)malloc((size_t)M * (size_t)N * sizeof(float));
    int8_t* xq = (int8_t*)malloc((size_t)K * (size_t)N);
    if (!W || !X || !scales || !C_yolo || !C_odnn || !xq) {
        fprintf(stderr, "bench_gemm_compare: int8 allocation failed for %s\n", label);
        free(W);
        free(X);
        free(scales);
        free(C_yolo);
        free(C_odnn);
        free(xq);
        return;
    }
    for (int i = 0; i < M * K; i++) W[i] = (int8_t)((i % 17) - 8);
    fill_pattern(X, (size_t)K * (size_t)N);
    for (int i = 0; i < M; i++) scales[i] = 0.01f + (float)i * 0.001f;

    odnn_matmul_t odnn;
    memset(&odnn, 0, sizeof(odnn));

    float input_scale = compute_input_scale(X, K * N);
    tensor_quantize_symmetric(X, xq, K * N, input_scale);
    odnn_matmul_init_int8(&odnn, rt, M, N, K, W, xq, C_odnn);

    for (int w = 0; w < 3; w++) {
        tensor_gemm_weight_int8(C_yolo, W, X, scales, M, N, K);
        tensor_quantize_symmetric(X, xq, K * N, input_scale);
        odnn_int8_weight_gemm(C_odnn, scales, input_scale, M, N, &odnn, rt);
    }

    tensor_gemm_weight_int8(C_yolo, W, X, scales, M, N, K);
    tensor_quantize_symmetric(X, xq, K * N, input_scale);
    odnn_int8_weight_gemm(C_odnn, scales, input_scale, M, N, &odnn, rt);
    float diff = max_abs_diff(C_yolo, C_odnn, (size_t)M * (size_t)N);

    timer_t t_yolo, t_odnn;
    timer_start(&t_yolo);
    for (int r = 0; r < repeats; r++) {
        tensor_gemm_weight_int8(C_yolo, W, X, scales, M, N, K);
    }
    timer_stop(&t_yolo);

    timer_start(&t_odnn);
    for (int r = 0; r < repeats; r++) {
        tensor_quantize_symmetric(X, xq, K * N, input_scale);
        odnn_int8_weight_gemm(C_odnn, scales, input_scale, M, N, &odnn, rt);
    }
    timer_stop(&t_odnn);

    double ms_yolo = timer_elapsed_ms(&t_yolo);
    double ms_odnn = timer_elapsed_ms(&t_odnn);
    double total_ops = 2.0 * (double)M * (double)N * (double)K * (double)repeats;
    double gops_yolo = (total_ops / (ms_yolo / 1000.0)) / 1e9;
    double gops_odnn = (total_ops / (ms_odnn / 1000.0)) / 1e9;
    double speedup = ms_yolo / ms_odnn;

    printf("  %-28s M=%4d N=%6d K=%4d\n", label, M, N, K);
    printf("    tensor_gemm_weight_int8  %8.2f ms  %7.3f Gop/s\n", ms_yolo, gops_yolo);
    printf("    oneDNN                   %8.2f ms  %7.3f Gop/s  speedup=%.2fx  max|diff|=%.2e\n", ms_odnn, gops_odnn,
           speedup, diff);

    odnn_matmul_destroy(&odnn);
    free(W);
    free(X);
    free(scales);
    free(C_yolo);
    free(C_odnn);
    free(xq);
}

int main(void) {
#ifdef USE_OPENBLAS
    const char* fp32_backend = "OpenBLAS (cblas_sgemm)";
#else
    const char* fp32_backend = "scalar/AVX (hand-rolled)";
#endif

    odnn_runtime_t rt;
    DNNL_CHECK(dnnl_engine_create(&rt.engine, dnnl_cpu, 0));
    DNNL_CHECK(dnnl_stream_create(&rt.stream, rt.engine, dnnl_stream_default_flags));

    printf("GEMM compare: tensor_gemm vs oneDNN matmul\n");
    printf("  tensor_gemm FP32 backend: %s\n", fp32_backend);
    printf("  tensor_gemm INT8 backend: %s\n", tensor_gemm_int8_backend());
    printf("  oneDNN threads: 1 (set OMP_NUM_THREADS=1 if needed)\n\n");

    printf("FP32 (C = A * B)\n");
    run_fp32_case(&rt, "tiny (unit-test-like)", 5, 7, 4, 50000);
    run_fp32_case(&rt, "medium", 128, 320, 128, 500);
    run_fp32_case(&rt, "large-N (conv-like)", 256, 4096, 256, 20);

    printf("\nINT8 weight-only (quantize X + matmul)\n");
    run_int8_case(&rt, "int8 medium", 128, 320, 128, 500);
    run_int8_case(&rt, "int8 large-N (conv-like)", 256, 4096, 256, 20);

    dnnl_stream_destroy(rt.stream);
    dnnl_engine_destroy(rt.engine);
    return 0;
}
