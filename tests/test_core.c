/* Automated checks for plan.md "DONE" components (tensor, layers, loader, BN, detection). */

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "detection.h"
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
    CHECK(conv2d_forward(&out, &in, &w, &bias, p) == SUCCESS, "conv2d 1x1");
    for (int i = 0; i < 12; i++) {
        float want = (i < 4) ? in.data[i] : (i < 8 ? 1.0f : 2.0f);
        CHECK(fabsf(out.data[i] - want) < 1e-5f, "conv2d 1x1 output channel");
    }
    tensor_free(&in);
    tensor_free(&w);
    tensor_free(&out);
    tensor_free(&bias);
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
        if (load_named_tensor(f, map[n].name, &map[n].t) != SUCCESS) break;
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

static void test_decode_detections(void) {
    tensor_t head;
    tensor_allocate(&head, 1, 2, 85, 1);
    tensor_fill(&head, 0.0f);
    head.data[0 * 85 + 0] = 0.5f;
    head.data[0 * 85 + 1] = 0.5f;
    head.data[0 * 85 + 2] = 0.2f;
    head.data[0 * 85 + 3] = 0.2f;
    head.data[0 * 85 + 4 + 3] = 0.95f;

    detection_results_t res;
    detection_t dbuf[2];
    res.detections = dbuf;
    res.capacity = 2;
    CHECK(decode_detections(&res, &head, 0.5f, 640.0f, 480.0f) == SUCCESS, "decode_detections");
    CHECK(res.count == 1, "decode count");
    CHECK(res.detections[0].class_id == 3, "decode class");
    CHECK(fabsf(res.detections[0].x1 - 256.0f) < 1.0f, "decode x1");
    tensor_free(&head);
}

int main(void) {
    failures = 0;
    test_tensor_gemm();
    test_silu();
    test_conv1x1_gemm_path();
    test_fold_bn();
    test_model_load_minimal();
    test_c3k2_fixture("c3k2_unit.bin", "unit", 2, true);
    test_c3k2_fixture("c3k2_yaml.bin", "yaml", 2, false);
    test_sppf_fixture("sppf_test.bin", 5, 3, false);
    test_sppf_fixture("sppf_shortcut.bin", 5, 3, true);
    test_decode_detections();

    if (failures == 0) {
        printf("test_core: all checks passed\n");
        return 0;
    }
    fprintf(stderr, "test_core: %d failures\n", failures);
    return 1;
}
