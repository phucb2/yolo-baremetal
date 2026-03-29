/*
 * Manual layer parity vs PyTorch golden .bin dumps; prints diffs and writes NDJSON debug logs.
 * Build: make tests/verify_layers && ./tests/verify_layers
 */

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "layers.h"
#include "tensor.h"
#include "utils.h"

#define DEBUG_LOG_PATH "/Users/phucbb/Personal/random-project/.cursor/debug-5d8ee2.log"
#define MAP_MAX 64

typedef struct {
    char name[128];
    tensor_t t;
} named_tensor_entry_t;

static void agent_log(const char* hypothesis_id, const char* location, const char* message,
                      const char* data_json) {
    FILE* f = fopen(DEBUG_LOG_PATH, "a");
    if (!f) return;
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    long long ms = (long long)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
    fprintf(f,
            "{\"sessionId\":\"5d8ee2\",\"hypothesisId\":\"%s\",\"location\":\"%s\",\"message\":\"%s\",\"data\":%s,"
            "\"timestamp\":%lld}\n",
            hypothesis_id, location, message, data_json, ms);
    fclose(f);
}

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

static float max_abs_diff_tensor_idx(const tensor_t* a, const tensor_t* b, size_t* idx_out) {
    size_t na = (size_t)a->dims[0] * a->dims[1] * a->dims[2] * a->dims[3];
    size_t nb = (size_t)b->dims[0] * b->dims[1] * b->dims[2] * b->dims[3];
    if (na != nb) {
        *idx_out = 0;
        return INFINITY;
    }
    float m = 0.0f;
    size_t ix = 0;
    for (size_t i = 0; i < na; i++) {
        float d = fabsf(a->data[i] - b->data[i]);
        if (d > m) {
            m = d;
            ix = i;
        }
    }
    *idx_out = ix;
    return m;
}

static void log_dims_h2(const char* label, const tensor_t* t) {
    char buf[320];
    snprintf(buf, sizeof(buf),
             "{\"label\":\"%s\",\"n\":%d,\"c\":%d,\"h\":%d,\"w\":%d}", label, t->dims[0], t->dims[1], t->dims[2],
             t->dims[3]);
    agent_log("H2", "verify_layers.c:log_dims", "tensor_dims", buf);
}

static void log_compare_h4_h5(const char* test_tag, float max_diff, size_t idx, const tensor_t* out,
                              const tensor_t* exp) {
    int W = out->dims[3], H = out->dims[2];
    size_t plane = (size_t)W * H;
    int xi = (int)(idx % (size_t)W);
    int yi = (int)((idx / (size_t)W) % (size_t)H);
    int ci = (int)(idx / plane);
    float ov = out->data[idx], ev = exp->data[idx];
    char buf[384];
    snprintf(buf, sizeof(buf),
             "{\"tag\":\"%s\",\"max_diff\":%.9g,\"idx\":%zu,\"c\":%d,\"y\":%d,\"x\":%d,\"out\":%.9g,\"exp\":%.9g,"
             "\"is_edge\":%s}",
             test_tag, max_diff, idx, ci, yi, xi, ov, ev,
             (yi == 0 || yi == H - 1 || xi == 0 || xi == W - 1) ? "true" : "false");
    agent_log("H4", "verify_layers.c:log_compare", "max_diff_site", buf);
    agent_log("H5", "verify_layers.c:log_compare", "value_at_max_diff", buf);
}

void verify_layer(const char* label, tensor_t* output, tensor_t* expected) {
    size_t idx = 0;
    float max_diff = max_abs_diff_tensor_idx(output, expected, &idx);
    printf("%-20s Max diff: %e -> %s\n", label, max_diff, (max_diff < 1e-4f) ? "SUCCESS" : "FAILED");
}

static int run_c3k2(const char* bin_filename, const char* tag, int n_blocks, bool shortcut) {
    named_tensor_entry_t map[MAP_MAX];
    FILE* fp = open_test_data_bin(bin_filename);
    if (!fp) {
        fprintf(stderr, "SKIP: open %s\n", bin_filename);
        return 0;
    }
    int nmap = load_tensor_map_fp(fp, map, MAP_MAX);
    if (nmap <= 0) {
        free_tensor_map(map, 0);
        return 0;
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
        return 0;
    }

    int h = input->dims[2], w = input->dims[3];
    int c_total = cv1_w->dims[0];
    int c_half = c_total / 2;
    snprintf(name, sizeof(name), "%s_m0_cv1_weight", tag);
    tensor_t* m0cv1 = map_find(map, nmap, name);
    if (!m0cv1) {
        fprintf(stderr, "SKIP: m0_cv1 %s\n", bin_filename);
        free_tensor_map(map, nmap);
        return 0;
    }
    int c_mid = m0cv1->dims[0];

    {
        char buf[256];
        snprintf(buf, sizeof(buf), "{\"bin\":\"%s\",\"tag\":\"%s\",\"n_blocks\":%d,\"shortcut\":%s}", bin_filename,
                 tag, n_blocks, shortcut ? "true" : "false");
        agent_log("H3", "verify_layers.c:run_c3k2", "fixture_start", buf);
    }
    log_dims_h2("c3k2_input", input);
    log_dims_h2("c3k2_expect", expect);

    tensor_t output;
    tensor_t* buffers = (tensor_t*)calloc((size_t)(n_blocks + 3), sizeof(tensor_t));
    tensor_t* b_w = (tensor_t*)calloc((size_t)(n_blocks * 4), sizeof(tensor_t));
    if (!buffers || !b_w) {
        free(buffers);
        free(b_w);
        free_tensor_map(map, nmap);
        return 1;
    }

    tensor_allocate(&buffers[0], 1, c_total, h, w);
    for (int i = 0; i < n_blocks; i++) tensor_allocate(&buffers[1 + i], 1, c_half, h, w);
    tensor_allocate(&buffers[n_blocks + 1], 1, c_mid, h, w);
    int concat_c = c_total + n_blocks * c_half;
    tensor_allocate(&buffers[n_blocks + 2], 1, concat_c, h, w);
    tensor_allocate(&output, 1, expect->dims[1], h, w);

    for (int i = 0; i < n_blocks; i++) {
        snprintf(name, sizeof(name), "%s_m%d_cv1_weight", tag, i);
        tensor_t* p = map_find(map, nmap, name);
        if (!p) goto c3k2_fail;
        b_w[i * 4 + 0] = *p;
        snprintf(name, sizeof(name), "%s_m%d_cv1_bias", tag, i);
        p = map_find(map, nmap, name);
        if (!p) goto c3k2_fail;
        b_w[i * 4 + 1] = *p;
        snprintf(name, sizeof(name), "%s_m%d_cv2_weight", tag, i);
        p = map_find(map, nmap, name);
        if (!p) goto c3k2_fail;
        b_w[i * 4 + 2] = *p;
        snprintf(name, sizeof(name), "%s_m%d_cv2_bias", tag, i);
        p = map_find(map, nmap, name);
        if (!p) goto c3k2_fail;
        b_w[i * 4 + 3] = *p;
    }

    status_t st = c3k2_forward(&output, input, n_blocks, shortcut, cv1_w, cv1_b, cv2_w, cv2_b, b_w, buffers);
    if (st != SUCCESS) {
        fprintf(stderr, "c3k2_forward failed\n");
        goto c3k2_cleanup;
    }

    log_dims_h2("c3k2_output", &output);
    {
        size_t idx = 0;
        float md = max_abs_diff_tensor_idx(&output, expect, &idx);
        char tagbuf[64];
        snprintf(tagbuf, sizeof(tagbuf), "c3k2_%s", tag);
        log_compare_h4_h5(tagbuf, md, idx, &output, expect);
        char ibuf[280];
        snprintf(ibuf, sizeof(ibuf),
                 "{\"tag\":\"%s\",\"input0\":%.9g,\"input1\":%.9g,\"out0\":%.9g,\"exp0\":%.9g,\"max_diff\":%.9g}", tag,
                 input->data[0], input->data[1], output.data[0], expect->data[0], md);
        agent_log("H3", "verify_layers.c:run_c3k2", "sample_values", ibuf);
        verify_layer(tagbuf, &output, expect);
        int fail = (md >= 5e-4f);
        tensor_free(&output);
        for (int i = 0; i < n_blocks + 3; i++) tensor_free(&buffers[i]);
        free(buffers);
        free(b_w);
        free_tensor_map(map, nmap);
        return fail;
    }

c3k2_fail:
    fprintf(stderr, "SKIP: bottleneck tensors %s\n", bin_filename);
c3k2_cleanup:
    tensor_free(&output);
    for (int i = 0; i < n_blocks + 3; i++) tensor_free(&buffers[i]);
    free(buffers);
    free(b_w);
    free_tensor_map(map, nmap);
    return 1;
}

static int run_sppf(const char* bin_filename, int kernel_size, int n_pool, bool shortcut) {
    named_tensor_entry_t map[MAP_MAX];
    FILE* fp = open_test_data_bin(bin_filename);
    if (!fp) {
        fprintf(stderr, "SKIP: open %s\n", bin_filename);
        return 0;
    }
    int nmap = load_tensor_map_fp(fp, map, MAP_MAX);
    if (nmap <= 0) {
        free_tensor_map(map, 0);
        return 0;
    }

    tensor_t* input = map_find(map, nmap, "sppf_input");
    tensor_t* expect = map_find(map, nmap, "sppf_output");
    tensor_t* cv1_w = map_find(map, nmap, "sppf_cv1_weight");
    tensor_t* cv1_b = map_find(map, nmap, "sppf_cv1_bias");
    tensor_t* cv2_w = map_find(map, nmap, "sppf_cv2_weight");
    tensor_t* cv2_b = map_find(map, nmap, "sppf_cv2_bias");
    if (!input || !expect || !cv1_w || !cv1_b || !cv2_w || !cv2_b) {
        fprintf(stderr, "SKIP: missing SPPF tensors %s\n", bin_filename);
        free_tensor_map(map, nmap);
        return 0;
    }

    {
        char buf[200];
        snprintf(buf, sizeof(buf), "{\"bin\":\"%s\",\"k\":%d,\"n_pool\":%d,\"shortcut\":%s}", bin_filename,
                 kernel_size, n_pool, shortcut ? "true" : "false");
        agent_log("H3", "verify_layers.c:run_sppf", "fixture_start", buf);
    }
    log_dims_h2("sppf_input", input);

    int h = input->dims[2], w = input->dims[3];
    int c_ = cv1_w->dims[0];
    int concat_c = c_ * (n_pool + 1);

    tensor_t* buffers = (tensor_t*)calloc((size_t)(n_pool + 2), sizeof(tensor_t));
    if (!buffers) {
        free_tensor_map(map, nmap);
        return 1;
    }
    tensor_allocate(&buffers[0], 1, c_, h, w);
    for (int i = 1; i <= n_pool; i++) tensor_allocate(&buffers[i], 1, c_, h, w);
    tensor_allocate(&buffers[n_pool + 1], 1, concat_c, h, w);

    tensor_t output;
    tensor_allocate(&output, 1, expect->dims[1], h, w);

    status_t st = sppf_forward(&output, input, cv1_w, cv1_b, cv2_w, cv2_b, kernel_size, n_pool, shortcut,
                               buffers);
    if (st != SUCCESS) {
        fprintf(stderr, "sppf_forward failed\n");
        tensor_free(&output);
        for (int i = 0; i < n_pool + 2; i++) tensor_free(&buffers[i]);
        free(buffers);
        free_tensor_map(map, nmap);
        return 1;
    }

    size_t idx = 0;
    float md = max_abs_diff_tensor_idx(&output, expect, &idx);
    char tagbuf[96];
    snprintf(tagbuf, sizeof(tagbuf), "sppf_%s", bin_filename);
    log_compare_h4_h5(tagbuf, md, idx, &output, expect);
    verify_layer(tagbuf, &output, expect);
    int fail = (md >= 5e-4f);

    tensor_free(&output);
    for (int i = 0; i < n_pool + 2; i++) tensor_free(&buffers[i]);
    free(buffers);
    free_tensor_map(map, nmap);
    return fail;
}

static int run_c2psa(void) {
    named_tensor_entry_t map[MAP_MAX];
    FILE* fp = open_test_data_bin("c2psa_test.bin");
    if (!fp) {
        fprintf(stderr, "SKIP: c2psa_test.bin\n");
        return 0;
    }
    int nmap = load_tensor_map_fp(fp, map, MAP_MAX);
    if (nmap <= 0) {
        free_tensor_map(map, 0);
        return 0;
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
        fprintf(stderr, "SKIP: C2PSA tensors\n");
        free_tensor_map(map, nmap);
        return 0;
    }

    agent_log("H3", "verify_layers.c:run_c2psa", "fixture_start", "{\"tag\":\"c2psa\"}");
    log_dims_h2("c2psa_input", input);

    tensor_t psa_stack[32];
    char name[192];
    for (int bi = 0; bi < n_blocks; bi++) {
        snprintf(name, sizeof(name), "c2psa_m%d_qkv_weight", bi);
        tensor_t* p = map_find(map, nmap, name);
        if (!p) goto c2psa_bad;
        psa_stack[bi * 10 + 0] = *p;
        snprintf(name, sizeof(name), "c2psa_m%d_qkv_bias", bi);
        p = map_find(map, nmap, name);
        if (!p) goto c2psa_bad;
        psa_stack[bi * 10 + 1] = *p;
        snprintf(name, sizeof(name), "c2psa_m%d_proj_weight", bi);
        p = map_find(map, nmap, name);
        if (!p) goto c2psa_bad;
        psa_stack[bi * 10 + 2] = *p;
        snprintf(name, sizeof(name), "c2psa_m%d_proj_bias", bi);
        p = map_find(map, nmap, name);
        if (!p) goto c2psa_bad;
        psa_stack[bi * 10 + 3] = *p;
        snprintf(name, sizeof(name), "c2psa_m%d_pe_weight", bi);
        p = map_find(map, nmap, name);
        if (!p) goto c2psa_bad;
        psa_stack[bi * 10 + 4] = *p;
        snprintf(name, sizeof(name), "c2psa_m%d_pe_bias", bi);
        p = map_find(map, nmap, name);
        if (!p) goto c2psa_bad;
        psa_stack[bi * 10 + 5] = *p;
        snprintf(name, sizeof(name), "c2psa_m%d_ffn0_weight", bi);
        p = map_find(map, nmap, name);
        if (!p) goto c2psa_bad;
        psa_stack[bi * 10 + 6] = *p;
        snprintf(name, sizeof(name), "c2psa_m%d_ffn0_bias", bi);
        p = map_find(map, nmap, name);
        if (!p) goto c2psa_bad;
        psa_stack[bi * 10 + 7] = *p;
        snprintf(name, sizeof(name), "c2psa_m%d_ffn1_weight", bi);
        p = map_find(map, nmap, name);
        if (!p) goto c2psa_bad;
        psa_stack[bi * 10 + 8] = *p;
        snprintf(name, sizeof(name), "c2psa_m%d_ffn1_bias", bi);
        p = map_find(map, nmap, name);
        if (!p) goto c2psa_bad;
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
    if (st != SUCCESS) {
        fprintf(stderr, "c2psa_forward failed\n");
        tensor_free(&output);
        for (int i = 0; i < 3; i++) tensor_free(&buffers[i]);
        free_tensor_map(map, nmap);
        return 1;
    }

    size_t idx = 0;
    float md = max_abs_diff_tensor_idx(&output, expect, &idx);
    log_compare_h4_h5("c2psa", md, idx, &output, expect);
    verify_layer("c2psa", &output, expect);
    int fail = (md >= 5e-4f);

    tensor_free(&output);
    for (int i = 0; i < 3; i++) tensor_free(&buffers[i]);
    free_tensor_map(map, nmap);
    return fail;

c2psa_bad:
    fprintf(stderr, "SKIP: c2psa block tensors\n");
    free_tensor_map(map, nmap);
    return 1;
}

int main(void) {
    int fails = 0;
    printf("verify_layers: golden parity (threshold 5e-4)\n");
    fails += run_c3k2("c3k2_unit.bin", "unit", 2, true);
    fails += run_c3k2("c3k2_yaml.bin", "yaml", 2, false);
    fails += run_sppf("sppf_test.bin", 5, 3, false);
    fails += run_sppf("sppf_shortcut.bin", 5, 3, true);
    fails += run_c2psa();
    if (fails == 0)
        printf("verify_layers: all within tolerance\n");
    else
        printf("verify_layers: %d fixture(s) above tolerance\n", fails);
    return fails > 0 ? 1 : 0;
}
