#include <stdlib.h>
#include <string.h>

#include "dnnl.h"
#include "gemm_onednn.h"
#include "status.h"

#define DNNL_CHECK(st)                                                                                                 \
    do {                                                                                                               \
        dnnl_status_t _st = (st);                                                                                      \
        if (_st != dnnl_success) return ERROR_NOT_IMPLEMENTED;                                                             \
    } while (0)

#define ODNN_CACHE_MAX 64

typedef enum { ODNN_KIND_FP32 = 0, ODNN_KIND_INT8 = 1 } odnn_kind_t;

typedef struct {
    int M, N, K;
    odnn_kind_t kind;
    dnnl_primitive_t prim;
    dnnl_memory_t src_mem;
    dnnl_memory_t weights_mem;
    dnnl_memory_t dst_mem;
    dnnl_memory_t scratchpad_mem;
    int has_scratchpad;
    int valid;
} odnn_cache_entry_t;

static dnnl_engine_t g_engine = NULL;
static dnnl_stream_t g_stream = NULL;
static odnn_cache_entry_t g_cache[ODNN_CACHE_MAX];
static int g_cache_count = 0;

static status_t odnn_runtime_init(void) {
    if (g_engine) return SUCCESS;
    DNNL_CHECK(dnnl_engine_create(&g_engine, dnnl_cpu, 0));
    DNNL_CHECK(dnnl_stream_create(&g_stream, g_engine, dnnl_stream_default_flags));
    return SUCCESS;
}

static dnnl_memory_desc_t make_ab_md(const dnnl_dims_t dims, dnnl_data_type_t dt) {
    dnnl_memory_desc_t md = NULL;
    dnnl_dims_t strides = {dims[1], 1};
    dnnl_status_t st = dnnl_memory_desc_create_with_strides(&md, 2, dims, dt, strides);
    if (st != dnnl_success) return NULL;
    return md;
}

static void odnn_entry_destroy(odnn_cache_entry_t* e) {
    if (!e) return;
    if (e->prim) dnnl_primitive_destroy(e->prim);
    if (e->src_mem) dnnl_memory_destroy(e->src_mem);
    if (e->weights_mem) dnnl_memory_destroy(e->weights_mem);
    if (e->dst_mem) dnnl_memory_destroy(e->dst_mem);
    if (e->scratchpad_mem) dnnl_memory_destroy(e->scratchpad_mem);
    memset(e, 0, sizeof(*e));
}

static status_t odnn_entry_init_fp32(odnn_cache_entry_t* e, int M, int N, int K) {
    memset(e, 0, sizeof(*e));
    e->M = M;
    e->N = N;
    e->K = K;
    e->kind = ODNN_KIND_FP32;

    dnnl_dims_t src_dims = {M, K};
    dnnl_dims_t weights_dims = {K, N};
    dnnl_dims_t dst_dims = {M, N};

    dnnl_memory_desc_t src_md = make_ab_md(src_dims, dnnl_f32);
    dnnl_memory_desc_t weights_md = make_ab_md(weights_dims, dnnl_f32);
    dnnl_memory_desc_t dst_md = make_ab_md(dst_dims, dnnl_f32);
    if (!src_md || !weights_md || !dst_md) {
        if (src_md) dnnl_memory_desc_destroy(src_md);
        if (weights_md) dnnl_memory_desc_destroy(weights_md);
        if (dst_md) dnnl_memory_desc_destroy(dst_md);
        return ERROR_NOT_IMPLEMENTED;
    }

    dnnl_primitive_desc_t pd = NULL;
    DNNL_CHECK(dnnl_matmul_primitive_desc_create(&pd, g_engine, src_md, weights_md, NULL, dst_md, NULL));
    DNNL_CHECK(dnnl_primitive_create(&e->prim, pd));

    DNNL_CHECK(dnnl_memory_create(&e->src_mem, src_md, g_engine, NULL));
    DNNL_CHECK(dnnl_memory_create(&e->weights_mem, weights_md, g_engine, NULL));
    DNNL_CHECK(dnnl_memory_create(&e->dst_mem, dst_md, g_engine, NULL));

    const_dnnl_memory_desc_t scratchpad_md = dnnl_primitive_desc_query_md(pd, dnnl_query_scratchpad_md, 0);
    if (scratchpad_md && dnnl_memory_desc_get_size(scratchpad_md) > 0) {
        DNNL_CHECK(dnnl_memory_create(&e->scratchpad_mem, scratchpad_md, g_engine, NULL));
        e->has_scratchpad = 1;
    }

    dnnl_memory_desc_destroy(src_md);
    dnnl_memory_desc_destroy(weights_md);
    dnnl_memory_desc_destroy(dst_md);
    dnnl_primitive_desc_destroy(pd);
    e->valid = 1;
    return SUCCESS;
}

static status_t odnn_entry_init_int8(odnn_cache_entry_t* e, int M, int N, int K) {
    memset(e, 0, sizeof(*e));
    e->M = M;
    e->N = N;
    e->K = K;
    e->kind = ODNN_KIND_INT8;

    dnnl_dims_t src_dims = {M, K};
    dnnl_dims_t weights_dims = {K, N};
    dnnl_dims_t dst_dims = {M, N};

    dnnl_memory_desc_t src_md = NULL;
    dnnl_memory_desc_t weights_md = NULL;
    dnnl_memory_desc_t dst_md = NULL;
    DNNL_CHECK(dnnl_memory_desc_create_with_tag(&src_md, 2, src_dims, dnnl_s8, dnnl_ab));
    DNNL_CHECK(dnnl_memory_desc_create_with_tag(&weights_md, 2, weights_dims, dnnl_s8, dnnl_ab));
    DNNL_CHECK(dnnl_memory_desc_create_with_tag(&dst_md, 2, dst_dims, dnnl_f32, dnnl_ab));

    dnnl_primitive_desc_t pd = NULL;
    dnnl_status_t st = dnnl_matmul_primitive_desc_create(&pd, g_engine, src_md, weights_md, NULL, dst_md, NULL);
    if (st == dnnl_unimplemented) {
        dnnl_memory_desc_destroy(dst_md);
        DNNL_CHECK(dnnl_memory_desc_create_with_tag(&dst_md, 2, dst_dims, dnnl_s32, dnnl_ab));
        st = dnnl_matmul_primitive_desc_create(&pd, g_engine, src_md, weights_md, NULL, dst_md, NULL);
    }
    DNNL_CHECK(st);
    DNNL_CHECK(dnnl_primitive_create(&e->prim, pd));

    DNNL_CHECK(dnnl_memory_create(&e->src_mem, src_md, g_engine, NULL));
    DNNL_CHECK(dnnl_memory_create(&e->weights_mem, weights_md, g_engine, NULL));
    DNNL_CHECK(dnnl_memory_create(&e->dst_mem, dst_md, g_engine, NULL));

    const_dnnl_memory_desc_t scratchpad_md = dnnl_primitive_desc_query_md(pd, dnnl_query_scratchpad_md, 0);
    if (scratchpad_md && dnnl_memory_desc_get_size(scratchpad_md) > 0) {
        DNNL_CHECK(dnnl_memory_create(&e->scratchpad_mem, scratchpad_md, g_engine, NULL));
        e->has_scratchpad = 1;
    }

    dnnl_memory_desc_destroy(src_md);
    dnnl_memory_desc_destroy(weights_md);
    dnnl_memory_desc_destroy(dst_md);
    dnnl_primitive_desc_destroy(pd);
    e->valid = 1;
    return SUCCESS;
}

static odnn_cache_entry_t* odnn_cache_get(int M, int N, int K, odnn_kind_t kind) {
    for (int i = 0; i < g_cache_count; i++) {
        odnn_cache_entry_t* e = &g_cache[i];
        if (e->valid && e->M == M && e->N == N && e->K == K && e->kind == kind) return e;
    }

    odnn_cache_entry_t* slot = NULL;
    if (g_cache_count < ODNN_CACHE_MAX) {
        slot = &g_cache[g_cache_count++];
    } else {
        odnn_entry_destroy(&g_cache[0]);
        memmove(&g_cache[0], &g_cache[1], (size_t)(g_cache_count - 1) * sizeof(g_cache[0]));
        slot = &g_cache[g_cache_count - 1];
    }

    status_t st = (kind == ODNN_KIND_FP32) ? odnn_entry_init_fp32(slot, M, N, K) : odnn_entry_init_int8(slot, M, N, K);
    if (st != SUCCESS) {
        odnn_entry_destroy(slot);
        if (slot == &g_cache[g_cache_count - 1] && g_cache_count > 0) g_cache_count--;
        return NULL;
    }
    return slot;
}

static status_t odnn_execute(odnn_cache_entry_t* e, void* src, void* weights, void* dst) {
    DNNL_CHECK(dnnl_memory_set_data_handle(e->src_mem, src));
    DNNL_CHECK(dnnl_memory_set_data_handle(e->weights_mem, weights));
    DNNL_CHECK(dnnl_memory_set_data_handle(e->dst_mem, dst));

    dnnl_exec_arg_t args[8];
    int n_args = 0;
    args[n_args++] = (dnnl_exec_arg_t){DNNL_ARG_SRC, e->src_mem};
    args[n_args++] = (dnnl_exec_arg_t){DNNL_ARG_WEIGHTS, e->weights_mem};
    args[n_args++] = (dnnl_exec_arg_t){DNNL_ARG_DST, e->dst_mem};
    if (e->has_scratchpad) {
        args[n_args++] = (dnnl_exec_arg_t){DNNL_ARG_SCRATCHPAD, e->scratchpad_mem};
    }
    DNNL_CHECK(dnnl_primitive_execute(e->prim, g_stream, n_args, args));
    DNNL_CHECK(dnnl_stream_wait(g_stream));
    return SUCCESS;
}

static void apply_row_scales(float* C, int M, int N, const float* row_scales) {
    for (int i = 0; i < M; i++) {
        const float s = row_scales[i];
        float* row = C + i * N;
        for (int j = 0; j < N; j++) row[j] *= s;
    }
}

status_t gemm_onednn_fp32(float* restrict C, const float* restrict A, const float* restrict B, int M, int N, int K) {
    if (!C || !A || !B) return ERROR_NULL_POINTER;
    status_t st = odnn_runtime_init();
    if (st != SUCCESS) return st;

    odnn_cache_entry_t* e = odnn_cache_get(M, N, K, ODNN_KIND_FP32);
    if (!e) return ERROR_NOT_IMPLEMENTED;
    return odnn_execute(e, (void*)A, (void*)B, (void*)C);
}

status_t gemm_onednn_int8(float* restrict C, const int8_t* restrict A, const int8_t* restrict B,
                          const float* weight_scales, int M, int N, int K, float input_scale) {
    if (!C || !A || !B || !weight_scales) return ERROR_NULL_POINTER;

    status_t st = odnn_runtime_init();
    if (st != SUCCESS) return st;

    odnn_cache_entry_t* e = odnn_cache_get(M, N, K, ODNN_KIND_INT8);
    if (!e) return ERROR_NOT_IMPLEMENTED;

    st = odnn_execute(e, (void*)A, (void*)B, (void*)C);
    if (st != SUCCESS) return st;

    float row_scales[512];
    float* scales = row_scales;
    float* scales_heap = NULL;
    if (M > (int)(sizeof(row_scales) / sizeof(row_scales[0]))) {
        scales_heap = (float*)malloc((size_t)M * sizeof(float));
        if (!scales_heap) return ERROR_OUT_OF_MEMORY;
        scales = scales_heap;
    }
    for (int i = 0; i < M; i++) scales[i] = weight_scales[i] * input_scale;
    apply_row_scales(C, M, N, scales);
    free(scales_heap);
    return SUCCESS;
}

void gemm_onednn_shutdown(void) {
    for (int i = 0; i < g_cache_count; i++) {
        odnn_entry_destroy(&g_cache[i]);
    }
    g_cache_count = 0;
    if (g_stream) {
        dnnl_stream_destroy(g_stream);
        g_stream = NULL;
    }
    if (g_engine) {
        dnnl_engine_destroy(g_engine);
        g_engine = NULL;
    }
}
