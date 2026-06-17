/* Automated checks for plan.md "DONE" components (tensor, layers, loader, BN, detection). */

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "detection.h"
#include "detect.h"
#include "layers.h"
#include "model.h"
#include "tensor.h"
#include "utils.h"

static int failures;

#define CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL: %s\n", msg); \
            failures++; \
        } \
    } while (0)

static void naive_gemm(float* C, const float* A, const float* B, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++) s += A[i * K + k] * B[k * N + j];
            C[i * N + j] = s;
        }
    }
}

static void test_tensor_gemm(void) {
    const int M = 5, N = 7, K = 4;
    float A[5 * 4], B[4 * 7], C_ref[5 * 7], C_avx[5 * 7];
    for (int i = 0; i < M * K; i++) A[i] = (float)(i % 3) * 0.25f - 0.5f;
    for (int i = 0; i < K * N; i++) B[i] = (float)(i % 5) * 0.1f - 0.2f;
    naive_gemm(C_ref, A, B, M, N, K);
    memset(C_avx, 0, sizeof(C_avx));
    CHECK(tensor_gemm(C_avx, A, B, M, N, K, 1.0f, 0.0f) == SUCCESS, "tensor_gemm status");
    float max_diff = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float d = fabsf(C_avx[i] - C_ref[i]);
        if (d > max_diff) max_diff = d;
    }
    CHECK(max_diff < 1e-5f, "tensor_gemm vs reference");
}

static void test_silu(void) {
    tensor_t t;
    tensor_allocate(&t, 1, 1, 1, 1);
    t.data[0] = 0.0f;
    CHECK(silu_forward(&t) == SUCCESS, "silu_forward status");
    float expected = 0.0f / (1.0f + expf(0.0f));
    CHECK(fabsf(t.data[0] - expected) < 1e-6f, "silu value at 0");
    tensor_free(&t);
}

static void test_conv1x1_gemm_path(void) {
    tensor_t in, w, out, bias;
    tensor_allocate(&in, 1, 2, 2, 2);
    tensor_allocate(&w, 3, 2, 1, 1);
    tensor_allocate(&out, 1, 3, 2, 2);
    tensor_allocate(&bias, 3, 1, 1, 1);
    for (int i = 0; i < 8; i++) in.data[i] = (float)i;
    for (int i = 0; i < 6; i++) w.data[i] = (i == 0) ? 1.0f : 0.0f;
    bias.data[0] = 0.0f;
    bias.data[1] = 1.0f;
    bias.data[2] = 2.0f;
    conv_params_t p = {1, 0, 1};
    CHECK(conv2d_forward(&out, &in, &w, &bias, p, false) == SUCCESS, "conv2d 1x1");
    for (int i = 0; i < 12; i++) {
        float want = (i < 4) ? in.data[i] : (i < 8 ? 1.0f : 2.0f);
        CHECK(fabsf(out.data[i] - want) < 1e-5f, "conv2d 1x1 output channel");
    }
    tensor_free(&in);
    tensor_free(&w);
    tensor_free(&out);
    tensor_free(&bias);
}

/* Reference conv (same semantics as layers.c conv2d) for k>1 regression. */
static void naive_conv2d_ref(float* out, const float* in, const float* w, const float* bias_,
                             int out_c, int in_c, int kh, int kw, int in_h, int in_w, int out_h, int out_w,
                             int stride, int pad) {
    for (int oc = 0; oc < out_c; oc++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                float sum = bias_ ? bias_[oc] : 0.0f;
                for (int ic = 0; ic < in_c; ic++) {
                    for (int k_h = 0; k_h < kh; k_h++) {
                        for (int k_w = 0; k_w < kw; k_w++) {
                            int ih = oh * stride - pad + k_h;
                            int iw = ow * stride - pad + k_w;
                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                sum += in[ic * in_h * in_w + ih * in_w + iw] *
                                       w[oc * in_c * kh * kw + ic * kh * kw + k_h * kw + k_w];
                            }
                        }
                    }
                }
                out[oc * out_h * out_w + oh * out_w + ow] = sum;
            }
        }
    }
}

static void test_conv3x3_im2col_gemm(void) {
    /* 3x3, same padding, stride 1: 4x4 -> 4x4 */
    int in_h = 4, in_w = 4, kh = 3, kw = 3;
    int pad = 1;
    int out_h = in_h;
    int out_w = in_w;
    int in_c = 2, out_c = 3;
    tensor_t in, w, out, bias;
    tensor_allocate(&in, 1, in_c, in_h, in_w);
    tensor_allocate(&w, out_c, in_c, kh, kw);
    tensor_allocate(&out, 1, out_c, out_h, out_w);
    tensor_allocate(&bias, out_c, 1, 1, 1);
    for (int i = 0; i < in_c * in_h * in_w; i++) in.data[i] = (float)(i % 7) * 0.1f - 0.3f;
    for (int i = 0; i < out_c * in_c * kh * kw; i++) w.data[i] = (float)(i % 5) * 0.07f - 0.1f;
    for (int i = 0; i < out_c; i++) bias.data[i] = (float)i * 0.02f;

    float* ref = (float*)malloc((size_t)out_c * out_h * out_w * sizeof(float));
    CHECK(ref != NULL, "conv3x3 ref malloc");
    conv_params_t p = {1, pad, 1};
    naive_conv2d_ref(ref, in.data, w.data, bias.data, out_c, in_c, kh, kw, in_h, in_w, out_h, out_w,
                     p.stride, p.padding);
    CHECK(conv2d_forward(&out, &in, &w, &bias, p, false) == SUCCESS, "conv2d 3x3 status");

    float max_diff = 0.0f;
    for (int i = 0; i < out_c * out_h * out_w; i++) {
        float d = fabsf(out.data[i] - ref[i]);
        if (d > max_diff) max_diff = d;
    }
    CHECK(max_diff < 1e-4f, "conv2d 3x3 im2col+gemm vs naive ref");
    free(ref);
    tensor_free(&in);
    tensor_free(&w);
    tensor_free(&out);
    tensor_free(&bias);
}

static void test_conv_fuse_silu_parity(void) {
    tensor_t in, w, out_sep, out_fused, bias;
    tensor_allocate(&in, 1, 2, 3, 3);
    tensor_allocate(&w, 3, 2, 1, 1);
    tensor_allocate(&out_sep, 1, 3, 3, 3);
    tensor_allocate(&out_fused, 1, 3, 3, 3);
    tensor_allocate(&bias, 3, 1, 1, 1);
    for (int i = 0; i < 2 * 3 * 3; i++) in.data[i] = (float)(i % 5) * 0.1f - 0.2f;
    for (int i = 0; i < 3 * 2 * 1 * 1; i++) w.data[i] = (float)(i % 3) * 0.15f - 0.1f;
    for (int i = 0; i < 3; i++) bias.data[i] = (float)i * 0.01f;

    conv_params_t p = {1, 0, 1};
    CHECK(conv2d_forward(&out_sep, &in, &w, &bias, p, false) == SUCCESS, "conv2d sep");
    CHECK(silu_forward(&out_sep) == SUCCESS, "silu sep");
    CHECK(conv2d_forward(&out_fused, &in, &w, &bias, p, true) == SUCCESS, "conv2d fused silu");

    float max_diff = 0.0f;
    for (int i = 0; i < 3 * 3 * 3; i++) {
        float d = fabsf(out_sep.data[i] - out_fused.data[i]);
        if (d > max_diff) max_diff = d;
    }
    CHECK(max_diff < 1e-5f, "fused SiLU parity vs conv + silu");

    tensor_free(&in);
    tensor_free(&w);
    tensor_free(&out_sep);
    tensor_free(&out_fused);
    tensor_free(&bias);

    /* im2col + GEMM path */
    tensor_t in2, w2, os, of, b2;
    tensor_allocate(&in2, 1, 2, 4, 4);
    tensor_allocate(&w2, 3, 2, 3, 3);
    tensor_allocate(&os, 1, 3, 4, 4);
    tensor_allocate(&of, 1, 3, 4, 4);
    tensor_allocate(&b2, 3, 1, 1, 1);
    for (int i = 0; i < 2 * 4 * 4; i++) in2.data[i] = (float)(i % 7) * 0.11f - 0.3f;
    for (int i = 0; i < 3 * 2 * 3 * 3; i++) w2.data[i] = (float)(i % 5) * 0.07f - 0.1f;
    for (int i = 0; i < 3; i++) b2.data[i] = (float)i * 0.02f;
    conv_params_t p2 = {1, 1, 1};
    CHECK(conv2d_forward(&os, &in2, &w2, &b2, p2, false) == SUCCESS, "conv2d 3x3 sep");
    CHECK(silu_forward(&os) == SUCCESS, "silu 3x3 sep");
    CHECK(conv2d_forward(&of, &in2, &w2, &b2, p2, true) == SUCCESS, "conv2d 3x3 fused");
    max_diff = 0.0f;
    for (int i = 0; i < 3 * 4 * 4; i++) {
        float d = fabsf(os.data[i] - of.data[i]);
        if (d > max_diff) max_diff = d;
    }
    CHECK(max_diff < 1e-4f, "fused SiLU parity 3x3 im2col");

    tensor_free(&in2);
    tensor_free(&w2);
    tensor_free(&os);
    tensor_free(&of);
    tensor_free(&b2);
}

static void test_fold_bn(void) {
    tensor_t cw, cb, bn_w, bn_b, bn_m, bn_v;
    tensor_allocate(&cw, 1, 1, 1, 1);
    tensor_allocate(&cb, 1, 1, 1, 1);
    tensor_allocate(&bn_w, 1, 1, 1, 1);
    tensor_allocate(&bn_b, 1, 1, 1, 1);
    tensor_allocate(&bn_m, 1, 1, 1, 1);
    tensor_allocate(&bn_v, 1, 1, 1, 1);
    cw.data[0] = 2.0f;
    cb.data[0] = 1.0f;
    bn_w.data[0] = 1.0f;
    bn_b.data[0] = 0.0f;
    bn_m.data[0] = 0.0f;
    bn_v.data[0] = 1.0f;
    fold_bn(&cw, &cb, &bn_w, &bn_b, &bn_m, &bn_v);
    float scale = 1.0f / sqrtf(1.0f + 1e-5f);
    CHECK(fabsf(cw.data[0] - 2.0f * scale) < 1e-5f, "fold_bn weight scale");
    CHECK(fabsf(cb.data[0] - 1.0f * scale) < 1e-5f, "fold_bn bias");
    tensor_free(&cw);
    tensor_free(&cb);
    tensor_free(&bn_w);
    tensor_free(&bn_b);
    tensor_free(&bn_m);
    tensor_free(&bn_v);
}

static void write_named_tensor(FILE* f, const char* name, int n, int c, int h, int w, float fill) {
    int nl = (int)strlen(name);
    fwrite(&nl, sizeof(int), 1, f);
    fwrite(name, 1, (size_t)nl, f);
    int dims = 4;
    fwrite(&dims, sizeof(int), 1, f);
    fwrite(&n, sizeof(int), 1, f);
    fwrite(&c, sizeof(int), 1, f);
    fwrite(&h, sizeof(int), 1, f);
    fwrite(&w, sizeof(int), 1, f);
    size_t n_el = (size_t)n * c * h * w;
    for (size_t i = 0; i < n_el; i++) {
        float v = fill;
        fwrite(&v, sizeof(float), 1, f);
    }
}

static void test_fold_bn_nested_names_load(void) {
    /* Unfused nested Conv+BN; fold_all_bn should fuse and drop *.bn.* (same math as test_fold_bn). */
    char path[] = "/tmp/yolo26_nested_bn.bin";
    FILE* f = fopen(path, "wb");
    assert(f);
    int nc = 80;
    int total = 6;
    fwrite(&nc, sizeof(int), 1, f);
    fwrite(&total, sizeof(int), 1, f);
    write_named_tensor(f, "model.2.cv1.conv.weight", 1, 1, 1, 1, 2.0f);
    write_named_tensor(f, "model.2.cv1.conv.bias", 1, 1, 1, 1, 1.0f);
    write_named_tensor(f, "model.2.cv1.bn.weight", 1, 1, 1, 1, 1.0f);
    write_named_tensor(f, "model.2.cv1.bn.bias", 1, 1, 1, 1, 0.0f);
    write_named_tensor(f, "model.2.cv1.bn.running_mean", 1, 1, 1, 1, 0.0f);
    write_named_tensor(f, "model.2.cv1.bn.running_var", 1, 1, 1, 1, 1.0f);
    fclose(f);

    model_t model;
    model_create(&model, 64, 64);
    status_t st = model_load_weights(&model, path);
    CHECK(st == SUCCESS, "nested bn model_load_weights");
    CHECK(model.num_weights == 2, "nested bn: conv.w + conv.bias only after fold");
    tensor_t* cw = model_get_weight(&model, "model.2.cv1.conv.weight");
    tensor_t* cb = model_get_weight(&model, "model.2.cv1.conv.bias");
    CHECK(cw && cb, "nested fused tensors");
    CHECK(model_get_weight(&model, "model.2.cv1.bn.weight") == NULL, "bn.weight removed");
    float scale = 1.0f / sqrtf(1.0f + 1e-5f);
    CHECK(fabsf(cw->data[0] - 2.0f * scale) < 1e-5f, "nested folded w");
    CHECK(fabsf(cb->data[0] - 1.0f * scale) < 1e-5f, "nested folded b");
    model_destroy(&model);
    remove(path);
}

static void test_model_load_minimal(void) {
    char path[] = "/tmp/yolo26_test_weights.bin";
    FILE* f = fopen(path, "wb");
    assert(f);
    int nc = 80;
    int total = 1;
    fwrite(&nc, sizeof(int), 1, f);
    fwrite(&total, sizeof(int), 1, f);
    write_named_tensor(f, "model.0.conv.weight", 1, 1, 1, 1, 3.5f);
    fclose(f);

    model_t model;
    model_create(&model, 64, 64);
    status_t st = model_load_weights(&model, path);
    CHECK(st == SUCCESS, "model_load_weights minimal bin");
    tensor_t* tw = model_get_weight(&model, "model.0.conv.weight");
    CHECK(tw && tw->data[0] == 3.5f, "named tensor lookup");
    model_destroy(&model);
    remove(path);
}

#define C3K2_MAP_MAX 64
typedef struct {
    char name[128];
    tensor_t t;
} named_tensor_entry_t;

static FILE* open_test_data_bin(const char* filename) {
    char buf[256];
    snprintf(buf, sizeof(buf), "tests/data/%s", filename);
    FILE* f = fopen(buf, "rb");
    if (f) return f;
    snprintf(buf, sizeof(buf), "../tests/data/%s", filename);
    return fopen(buf, "rb");
}

static int load_tensor_map_fp(FILE* f, named_tensor_entry_t* map, int max_n) {
    int n = 0;
    while (n < max_n) {
        if (load_named_tensor(f, map[n].name, &map[n].t, 1) != SUCCESS) break;
        n++;
    }
    fclose(f);
    return n;
}

static tensor_t* map_find(named_tensor_entry_t* map, int n, const char* name) {
    for (int i = 0; i < n; i++) {
        if (strcmp(map[i].name, name) == 0) return &map[i].t;
    }
    return NULL;
}

static void free_tensor_map(named_tensor_entry_t* map, int n) {
    for (int i = 0; i < n; i++) tensor_free(&map[i].t);
}

static float max_abs_diff_tensor(const tensor_t* a, const tensor_t* b) {
    size_t na = (size_t)a->dims[0] * a->dims[1] * a->dims[2] * a->dims[3];
    size_t nb = (size_t)b->dims[0] * b->dims[1] * b->dims[2] * b->dims[3];
    if (na != nb) return INFINITY;
    float m = 0.0f;
    for (size_t i = 0; i < na; i++) {
        float d = fabsf(a->data[i] - b->data[i]);
        if (d > m) m = d;
    }
    return m;
}

static void test_c3k2_fixture(const char* bin_filename, const char* tag, int n_blocks, bool shortcut) {
    named_tensor_entry_t map[C3K2_MAP_MAX];
    FILE* fp = open_test_data_bin(bin_filename);
    if (!fp) {
        fprintf(stderr, "SKIP: open tests/data/%s\n", bin_filename);
        return;
    }
    int nmap = load_tensor_map_fp(fp, map, C3K2_MAP_MAX);
    if (nmap <= 0) {
        fprintf(stderr, "SKIP: empty %s\n", bin_filename);
        free_tensor_map(map, nmap > 0 ? nmap : 0);
        return;
    }

    char name[160];
    snprintf(name, sizeof(name), "%s_input", tag);
    tensor_t* input = map_find(map, nmap, name);
    snprintf(name, sizeof(name), "%s_output", tag);
    tensor_t* expect = map_find(map, nmap, name);
    snprintf(name, sizeof(name), "%s_cv1_weight", tag);
    tensor_t* cv1_w = map_find(map, nmap, name);
    snprintf(name, sizeof(name), "%s_cv1_bias", tag);
    tensor_t* cv1_b = map_find(map, nmap, name);
    snprintf(name, sizeof(name), "%s_cv2_weight", tag);
    tensor_t* cv2_w = map_find(map, nmap, name);
    snprintf(name, sizeof(name), "%s_cv2_bias", tag);
    tensor_t* cv2_b = map_find(map, nmap, name);
    if (!input || !expect || !cv1_w || !cv1_b || !cv2_w || !cv2_b) {
        fprintf(stderr, "SKIP: missing tensors in %s\n", bin_filename);
        free_tensor_map(map, nmap);
        return;
    }

    int h = input->dims[2], w = input->dims[3];
    int c_total = cv1_w->dims[0];
    int c_half = c_total / 2;
    snprintf(name, sizeof(name), "%s_m0_cv1_weight", tag);
    tensor_t* m0cv1 = map_find(map, nmap, name);
    CHECK(m0cv1 != NULL, "c3k2 m0_cv1_weight");
    int c_mid = m0cv1->dims[0];

    tensor_t output;
    tensor_t* buffers = (tensor_t*)calloc((size_t)(n_blocks + 3), sizeof(tensor_t));
    tensor_t* b_w = (tensor_t*)calloc((size_t)(n_blocks * 4), sizeof(tensor_t));
    CHECK(buffers != NULL && b_w != NULL, "c3k2 calloc");

    tensor_allocate(&buffers[0], 1, c_total, h, w);
    for (int i = 0; i < n_blocks; i++) tensor_allocate(&buffers[1 + i], 1, c_half, h, w);
    tensor_allocate(&buffers[n_blocks + 1], 1, c_mid, h, w);
    int concat_c = c_total + n_blocks * c_half;
    tensor_allocate(&buffers[n_blocks + 2], 1, concat_c, h, w);
    tensor_allocate(&output, 1, expect->dims[1], h, w);

    for (int i = 0; i < n_blocks; i++) {
        snprintf(name, sizeof(name), "%s_m%d_cv1_weight", tag, i);
        tensor_t* p = map_find(map, nmap, name);
        CHECK(p != NULL, "c3k2 bottleneck cv1_weight");
        b_w[i * 4 + 0] = *p;
        snprintf(name, sizeof(name), "%s_m%d_cv1_bias", tag, i);
        p = map_find(map, nmap, name);
        CHECK(p != NULL, "c3k2 bottleneck cv1_bias");
        b_w[i * 4 + 1] = *p;
        snprintf(name, sizeof(name), "%s_m%d_cv2_weight", tag, i);
        p = map_find(map, nmap, name);
        CHECK(p != NULL, "c3k2 bottleneck cv2_weight");
        b_w[i * 4 + 2] = *p;
        snprintf(name, sizeof(name), "%s_m%d_cv2_bias", tag, i);
        p = map_find(map, nmap, name);
        CHECK(p != NULL, "c3k2 bottleneck cv2_bias");
        b_w[i * 4 + 3] = *p;
    }

    status_t st = c3k2_forward(&output, input, n_blocks, shortcut, cv1_w, cv1_b, cv2_w, cv2_b, b_w, buffers);
    CHECK(st == SUCCESS, "c3k2_forward status");

    float md = max_abs_diff_tensor(&output, expect);
    CHECK(md < 5e-4f, "c3k2 max abs diff vs PyTorch golden");

    tensor_free(&output);
    for (int i = 0; i < n_blocks + 3; i++) tensor_free(&buffers[i]);
    free(buffers);
    free(b_w);
    free_tensor_map(map, nmap);
}

static void test_sppf_fixture(const char* bin_filename, int kernel_size, int n_pool, bool shortcut) {
    named_tensor_entry_t map[C3K2_MAP_MAX];
    FILE* fp = open_test_data_bin(bin_filename);
    if (!fp) {
        fprintf(stderr, "SKIP: open tests/data/%s\n", bin_filename);
        return;
    }
    int nmap = load_tensor_map_fp(fp, map, C3K2_MAP_MAX);
    if (nmap <= 0) {
        fprintf(stderr, "SKIP: empty %s\n", bin_filename);
        free_tensor_map(map, nmap > 0 ? nmap : 0);
        return;
    }

    tensor_t* input = map_find(map, nmap, "sppf_input");
    tensor_t* expect = map_find(map, nmap, "sppf_output");
    tensor_t* cv1_w = map_find(map, nmap, "sppf_cv1_weight");
    tensor_t* cv1_b = map_find(map, nmap, "sppf_cv1_bias");
    tensor_t* cv2_w = map_find(map, nmap, "sppf_cv2_weight");
    tensor_t* cv2_b = map_find(map, nmap, "sppf_cv2_bias");
    if (!input || !expect || !cv1_w || !cv1_b || !cv2_w || !cv2_b) {
        fprintf(stderr, "SKIP: missing SPPF tensors in %s\n", bin_filename);
        free_tensor_map(map, nmap);
        return;
    }

    int h = input->dims[2], w = input->dims[3];
    int c_ = cv1_w->dims[0];
    int concat_c = c_ * (n_pool + 1);

    tensor_t* buffers = (tensor_t*)calloc((size_t)(n_pool + 2), sizeof(tensor_t));
    CHECK(buffers != NULL, "sppf calloc");
    tensor_allocate(&buffers[0], 1, c_, h, w);
    for (int i = 1; i <= n_pool; i++) tensor_allocate(&buffers[i], 1, c_, h, w);
    tensor_allocate(&buffers[n_pool + 1], 1, concat_c, h, w);

    tensor_t output;
    tensor_allocate(&output, 1, expect->dims[1], h, w);

    status_t st = sppf_forward(&output, input, cv1_w, cv1_b, cv2_w, cv2_b, kernel_size, n_pool,
                               shortcut, buffers);
    CHECK(st == SUCCESS, "sppf_forward status");

    float md = max_abs_diff_tensor(&output, expect);
    CHECK(md < 5e-4f, "sppf max abs diff vs PyTorch golden");

    tensor_free(&output);
    for (int i = 0; i < n_pool + 2; i++) tensor_free(&buffers[i]);
    free(buffers);
    free_tensor_map(map, nmap);
}

static void test_c2psa_fixture(void) {
    named_tensor_entry_t map[C3K2_MAP_MAX];
    FILE* fp = open_test_data_bin("c2psa_test.bin");
    if (!fp) {
        fprintf(stderr, "SKIP: open tests/data/c2psa_test.bin\n");
        return;
    }
    int nmap = load_tensor_map_fp(fp, map, C3K2_MAP_MAX);
    if (nmap <= 0) {
        fprintf(stderr, "SKIP: empty c2psa_test.bin\n");
        free_tensor_map(map, nmap > 0 ? nmap : 0);
        return;
    }

    const int n_blocks = 2;
    const float e = 0.5f;
    const float attn_ratio = 0.5f;

    tensor_t* input = map_find(map, nmap, "c2psa_input");
    tensor_t* expect = map_find(map, nmap, "c2psa_output");
    tensor_t* cv1_w = map_find(map, nmap, "c2psa_cv1_weight");
    tensor_t* cv1_b = map_find(map, nmap, "c2psa_cv1_bias");
    tensor_t* cv2_w = map_find(map, nmap, "c2psa_cv2_weight");
    tensor_t* cv2_b = map_find(map, nmap, "c2psa_cv2_bias");
    if (!input || !expect || !cv1_w || !cv1_b || !cv2_w || !cv2_b) {
        fprintf(stderr, "SKIP: missing C2PSA tensors\n");
        free_tensor_map(map, nmap);
        return;
    }

    tensor_t psa_stack[32];
    char name[192];
    for (int bi = 0; bi < n_blocks; bi++) {
        snprintf(name, sizeof(name), "c2psa_m%d_qkv_weight", bi);
        tensor_t* p = map_find(map, nmap, name);
        CHECK(p != NULL, "c2psa qkv_weight");
        psa_stack[bi * 10 + 0] = *p;
        snprintf(name, sizeof(name), "c2psa_m%d_qkv_bias", bi);
        p = map_find(map, nmap, name);
        CHECK(p != NULL, "c2psa qkv_bias");
        psa_stack[bi * 10 + 1] = *p;
        snprintf(name, sizeof(name), "c2psa_m%d_proj_weight", bi);
        p = map_find(map, nmap, name);
        CHECK(p != NULL, "c2psa proj_weight");
        psa_stack[bi * 10 + 2] = *p;
        snprintf(name, sizeof(name), "c2psa_m%d_proj_bias", bi);
        p = map_find(map, nmap, name);
        CHECK(p != NULL, "c2psa proj_bias");
        psa_stack[bi * 10 + 3] = *p;
        snprintf(name, sizeof(name), "c2psa_m%d_pe_weight", bi);
        p = map_find(map, nmap, name);
        CHECK(p != NULL, "c2psa pe_weight");
        psa_stack[bi * 10 + 4] = *p;
        snprintf(name, sizeof(name), "c2psa_m%d_pe_bias", bi);
        p = map_find(map, nmap, name);
        CHECK(p != NULL, "c2psa pe_bias");
        psa_stack[bi * 10 + 5] = *p;
        snprintf(name, sizeof(name), "c2psa_m%d_ffn0_weight", bi);
        p = map_find(map, nmap, name);
        CHECK(p != NULL, "c2psa ffn0_weight");
        psa_stack[bi * 10 + 6] = *p;
        snprintf(name, sizeof(name), "c2psa_m%d_ffn0_bias", bi);
        p = map_find(map, nmap, name);
        CHECK(p != NULL, "c2psa ffn0_bias");
        psa_stack[bi * 10 + 7] = *p;
        snprintf(name, sizeof(name), "c2psa_m%d_ffn1_weight", bi);
        p = map_find(map, nmap, name);
        CHECK(p != NULL, "c2psa ffn1_weight");
        psa_stack[bi * 10 + 8] = *p;
        snprintf(name, sizeof(name), "c2psa_m%d_ffn1_bias", bi);
        p = map_find(map, nmap, name);
        CHECK(p != NULL, "c2psa ffn1_bias");
        psa_stack[bi * 10 + 9] = *p;
    }

    int c1 = input->dims[1];
    int c_hidden = (int)((float)c1 * e);
    int h = input->dims[2], w = input->dims[3];

    tensor_t buffers[3];
    tensor_allocate(&buffers[0], 1, 2 * c_hidden, h, w);
    tensor_allocate(&buffers[1], 1, c_hidden, h, w);
    tensor_allocate(&buffers[2], 1, 2 * c_hidden, h, w);

    tensor_t output;
    tensor_allocate(&output, 1, c1, h, w);

    status_t st = c2psa_forward(&output, input, n_blocks, e, attn_ratio, cv1_w, cv1_b, cv2_w, cv2_b, psa_stack,
                                buffers);
    CHECK(st == SUCCESS, "c2psa_forward status");

    float md = max_abs_diff_tensor(&output, expect);
    CHECK(md < 5e-4f, "c2psa max abs diff vs PyTorch golden");

    tensor_free(&output);
    for (int i = 0; i < 3; i++) tensor_free(&buffers[i]);
    free_tensor_map(map, nmap);
}

static void test_decode_detections(void) {
    tensor_t head;
    tensor_allocate(&head, 1, 2, 6, 1);
    tensor_fill(&head, 0.0f);
    /* Same box as old normalized test: cx,cy,w,h 0.5,0.5,0.2,0.2 on 640x480 -> xyxy pixels */
    head.data[0] = 256.0f;
    head.data[1] = 192.0f;
    head.data[2] = 384.0f;
    head.data[3] = 288.0f;
    head.data[4] = 0.95f;
    head.data[5] = 3.0f;

    detection_results_t res;
    detection_t dbuf[2];
    res.detections = dbuf;
    res.capacity = 2;
    CHECK(decode_detections(&res, &head, 0.5f) == SUCCESS, "decode_detections");
    CHECK(res.count == 1, "decode count");
    CHECK(res.detections[0].class_id == 3, "decode class");
    CHECK(fabsf(res.detections[0].x1 - 256.0f) < 1.0f, "decode x1");
    CHECK(fabsf(res.detections[0].y1 - 192.0f) < 1.0f, "decode y1");
    tensor_free(&head);
}

static void test_detect_postprocess(void) {
    /* N=1, nc=3: single row [xyxy] + [p0,p1,p2] */
    float pred[7];
    pred[0] = 0.0f;
    pred[1] = 0.0f;
    pred[2] = 10.0f;
    pred[3] = 10.0f;
    pred[4] = 0.2f;
    pred[5] = 0.5f;
    pred[6] = 0.3f;

    tensor_t out;
    tensor_allocate(&out, 1, 3, 6, 1);
    tensor_fill(&out, -1.0f);
    CHECK(detect_postprocess_from_pred(pred, 1, 3, 3, &out) == SUCCESS, "detect_postprocess_from_pred");
    CHECK(fabsf(out.data[0] - 0.0f) < 1e-5f, "postproc x1");
    CHECK(fabsf(out.data[4] - 0.5f) < 1e-5f, "postproc score");
    CHECK(fabsf(out.data[5] - 1.0f) < 1e-5f, "postproc class");
    tensor_free(&out);
}

static void test_gemm_int8_parity(void) {
    const int M = 5, N = 7, K = 4;
    int8_t A[M * K];
    int8_t B[K * N];
    float scales[M];
    float C_ref[M * N];
    float C_simd[M * N];

    for (int i = 0; i < M * K; i++) A[i] = (int8_t)((i % 11) - 5);
    for (int i = 0; i < K * N; i++) B[i] = (int8_t)((i % 9) - 4);
    for (int i = 0; i < M; i++) scales[i] = 0.02f + (float)i * 0.003f;
    const float input_scale = 0.04f;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++) acc += (int32_t)A[i * K + k] * (int32_t)B[k * N + j];
            C_ref[i * N + j] = scales[i] * input_scale * (float)acc;
        }
    }

    CHECK(tensor_gemm_int8(C_simd, A, B, scales, M, N, K, input_scale) == SUCCESS, "tensor_gemm_int8");
    float md = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float d = fabsf(C_simd[i] - C_ref[i]);
        if (d > md) md = d;
    }
    CHECK(md < 1e-4f, "gemm_int8 simd vs ref");
}

static void test_conv_int8_parity(void) {
    const int in_c = 4, out_c = 3, h = 5, w = 5;
    tensor_t in, out_int8, wf, wf_fp32, bias;
    tensor_allocate(&in, 1, in_c, h, w);
    tensor_allocate(&out_int8, 1, out_c, h, w);
    tensor_allocate(&wf_fp32, out_c, in_c, 1, 1);
    tensor_allocate(&wf, out_c, in_c, 1, 1);
    tensor_allocate(&bias, 1, out_c, 1, 1);

    for (int i = 0; i < in_c * h * w; i++) in.data[i] = (float)(i % 11) * 0.07f - 0.2f;
    for (int i = 0; i < out_c * in_c; i++) wf_fp32.data[i] = (float)(i % 7) * 0.05f - 0.1f;
    for (int i = 0; i < out_c; i++) bias.data[i] = (float)i * 0.01f;

    wf.dtype = TENSOR_DTYPE_INT8;
    wf.num_scales = out_c;
    wf.scales = (float*)malloc((size_t)out_c * sizeof(float));
    wf.qdata = (int8_t*)malloc((size_t)out_c * (size_t)in_c);
    CHECK(wf.scales && wf.qdata, "int8 weight alloc");
    for (int oc = 0; oc < out_c; oc++) {
        float amax = 0.0f;
        for (int ic = 0; ic < in_c; ic++) {
            float v = fabsf(wf_fp32.data[oc * in_c + ic]);
            if (v > amax) amax = v;
        }
        wf.scales[oc] = amax / 127.0f;
        if (wf.scales[oc] < 1e-8f) wf.scales[oc] = 1e-8f;
        for (int ic = 0; ic < in_c; ic++) {
            float q = roundf(wf_fp32.data[oc * in_c + ic] / wf.scales[oc]);
            if (q > 127.0f) q = 127.0f;
            if (q < -128.0f) q = -128.0f;
            wf.qdata[oc * in_c + ic] = (int8_t)q;
        }
    }

    conv_params_t p = {1, 0, 1, 0.0f};
    const int plane = h * w;
    tensor_t out_ref;
    tensor_allocate(&out_ref, 1, out_c, h, w);
    CHECK(tensor_gemm_weight_int8(out_ref.data, wf.qdata, in.data, wf.scales, out_c, plane, in_c) == SUCCESS,
          "int8 gemm ref");
    for (int oc = 0; oc < out_c; oc++) {
        float b = bias.data[oc];
        for (int pix = 0; pix < plane; pix++) {
            out_ref.data[oc * plane + pix] += b;
        }
    }
    CHECK(conv2d_forward(&out_int8, &in, &wf, &bias, p, false) == SUCCESS, "conv int8");

    float md = max_abs_diff_tensor(&out_ref, &out_int8);
    CHECK(md < 1e-4f, "int8 conv vs gemm ref");

    free(wf.scales);
    free(wf.qdata);
    wf.scales = NULL;
    wf.qdata = NULL;
    wf.dtype = TENSOR_DTYPE_FP32;
    tensor_free(&in);
    tensor_free(&out_ref);
    tensor_free(&out_int8);
    tensor_free(&wf);
    tensor_free(&wf_fp32);
    tensor_free(&bias);
}

int main(void) {
    failures = 0;
    test_tensor_gemm();
    test_silu();
    test_conv1x1_gemm_path();
    test_conv3x3_im2col_gemm();
    test_conv_fuse_silu_parity();
    test_gemm_int8_parity();
    test_conv_int8_parity();
    test_fold_bn();
    test_fold_bn_nested_names_load();
    test_model_load_minimal();
    test_c3k2_fixture("c3k2_unit.bin", "unit", 2, true);
    test_c3k2_fixture("c3k2_yaml.bin", "yaml", 2, false);
    test_sppf_fixture("sppf_test.bin", 5, 3, false);
    test_sppf_fixture("sppf_shortcut.bin", 5, 3, true);
    test_c2psa_fixture();
    test_decode_detections();
    test_detect_postprocess();

    if (failures == 0) {
        printf("test_core: all checks passed\n");
        return 0;
    }
    fprintf(stderr, "test_core: %d failures\n", failures);
    return 1;
}
