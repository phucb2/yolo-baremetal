#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "detect.h"
#include "layers.h"

#define DETECT_STRIDE0 8
#define DETECT_STRIDE1 16
#define DETECT_STRIDE2 32

static int cmp_pair_desc(const void* a, const void* b) {
    float va = ((const float*)a)[0];
    float vb = ((const float*)b)[0];
    if (va < vb) return 1;
    if (va > vb) return -1;
    return 0;
}

static void make_anchors_for_shape(int h, int w, int stride, float* ax, float* ay, float* stride_out, int base) {
    int k = 0;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            ax[base + k] = (float)x + 0.5f;
            ay[base + k] = (float)y + 0.5f;
            stride_out[base + k] = (float)stride;
            k++;
        }
    }
}

static void dist2bbox_xyxy(const float* dist, const float* ax, const float* ay, int N, float* out_xyxy) {
    for (int j = 0; j < N; j++) {
        float l = dist[0 * N + j];
        float t = dist[1 * N + j];
        float r = dist[2 * N + j];
        float b = dist[3 * N + j];
        out_xyxy[0 * N + j] = ax[j] - l;
        out_xyxy[1 * N + j] = ay[j] - t;
        out_xyxy[2 * N + j] = ax[j] + r;
        out_xyxy[3 * N + j] = ay[j] + b;
    }
}

static void mul_xyxy_stride(const float* xyxy, const float* stride_per_anchor, int N, float* out) {
    for (int j = 0; j < N; j++) {
        float s = stride_per_anchor[j];
        out[0 * N + j] = xyxy[0 * N + j] * s;
        out[1 * N + j] = xyxy[1 * N + j] * s;
        out[2 * N + j] = xyxy[2 * N + j] * s;
        out[3 * N + j] = xyxy[3 * N + j] * s;
    }
}

static void sigmoid_ncN(float* cls, int nc, int N) {
    for (int i = 0; i < nc * N; i++) {
        float x = cls[i];
        cls[i] = 1.0f / (1.0f + expf(-x));
    }
}

status_t detect_postprocess_from_pred(const float* pred, int N, int nc, int max_det, tensor_t* out) {
    if (!pred || !out || !out->data) return ERROR_NULL_POINTER;
    if (out->dims[0] != 1 || out->dims[2] < 6 || out->dims[3] < 1) return ERROR_INVALID_DIMS;

    int cap = out->dims[1];
    int k = max_det < N ? max_det : N;
    if (k > cap) k = cap;

    float* mpa = (float*)malloc((size_t)N * sizeof(float));
    if (!mpa) return ERROR_OUT_OF_MEMORY;

    for (int a = 0; a < N; a++) {
        float m = -FLT_MAX;
        const float* row = pred + (size_t)a * (size_t)(4 + nc);
        for (int c = 0; c < nc; c++) {
            float v = row[4 + c];
            if (v > m) m = v;
        }
        mpa[a] = m;
    }

    float* pairs = (float*)malloc((size_t)N * 2 * sizeof(float));
    if (!pairs) {
        free(mpa);
        return ERROR_OUT_OF_MEMORY;
    }
    for (int i = 0; i < N; i++) {
        pairs[2 * i] = mpa[i];
        pairs[2 * i + 1] = (float)i;
    }
    qsort(pairs, (size_t)N, 2 * sizeof(float), cmp_pair_desc);
    free(mpa);

    int* ori_index = (int*)malloc((size_t)k * sizeof(int));
    if (!ori_index) {
        free(pairs);
        return ERROR_OUT_OF_MEMORY;
    }
    for (int i = 0; i < k; i++) ori_index[i] = (int)pairs[2 * i + 1];
    free(pairs);

    float* gathered = (float*)malloc((size_t)k * (size_t)nc * sizeof(float));
    if (!gathered) {
        free(ori_index);
        return ERROR_OUT_OF_MEMORY;
    }
    for (int i = 0; i < k; i++) {
        int anchor = ori_index[i];
        const float* row = pred + (size_t)anchor * (size_t)(4 + nc);
        for (int c = 0; c < nc; c++) gathered[(size_t)i * (size_t)nc + (size_t)c] = row[4 + c];
    }

    int flat_n = k * nc;
    float* flat_vals = (float*)malloc((size_t)flat_n * sizeof(float));
    if (!flat_vals) {
        free(gathered);
        free(ori_index);
        return ERROR_OUT_OF_MEMORY;
    }
    for (int i = 0; i < k; i++) {
        for (int c = 0; c < nc; c++) flat_vals[i * nc + c] = gathered[(size_t)i * (size_t)nc + (size_t)c];
    }

    float* sort2 = (float*)malloc((size_t)flat_n * 2 * sizeof(float));
    if (!sort2) {
        free(flat_vals);
        free(gathered);
        free(ori_index);
        return ERROR_OUT_OF_MEMORY;
    }
    for (int i = 0; i < flat_n; i++) {
        sort2[2 * i] = flat_vals[i];
        sort2[2 * i + 1] = (float)i;
    }
    qsort(sort2, (size_t)flat_n, 2 * sizeof(float), cmp_pair_desc);

    float* outd = out->data;
    int row_stride = out->dims[2] * out->dims[3];
    for (int t = 0; t < k; t++) {
        int flat_idx = (int)sort2[2 * t + 1];
        int anchor_slot = flat_idx / nc;
        int class_id = flat_idx % nc;
        float score = flat_vals[flat_idx];
        int idx_orig = ori_index[anchor_slot];
        const float* brow = pred + (size_t)idx_orig * (size_t)(4 + nc);
        float* row = outd + (size_t)t * (size_t)row_stride;
        row[0] = brow[0];
        row[1] = brow[1];
        row[2] = brow[2];
        row[3] = brow[3];
        row[4] = score;
        row[5] = (float)class_id;
    }
    for (int t = k; t < cap; t++) {
        float* row = outd + (size_t)t * (size_t)row_stride;
        for (int u = 0; u < 6; u++) row[u] = 0.0f;
    }

    free(sort2);
    free(flat_vals);
    free(gathered);
    free(ori_index);
    return SUCCESS;
}

static status_t run_cv2(model_t* model, int d_idx, int scale, const tensor_t* feat, tensor_t* out_box,
                                tensor_t* s1, tensor_t* s2) {
    char name[200];
    snprintf(name, sizeof name, "model.%d.cv2.%d.0.conv.weight", d_idx, scale);
    const tensor_t* w0 = model_get_weight(model, name);
    snprintf(name, sizeof name, "model.%d.cv2.%d.0.conv.bias", d_idx, scale);
    const tensor_t* b0 = model_get_weight(model, name);
    snprintf(name, sizeof name, "model.%d.cv2.%d.1.conv.weight", d_idx, scale);
    const tensor_t* w1 = model_get_weight(model, name);
    snprintf(name, sizeof name, "model.%d.cv2.%d.1.conv.bias", d_idx, scale);
    const tensor_t* b1 = model_get_weight(model, name);
    snprintf(name, sizeof name, "model.%d.cv2.%d.2.weight", d_idx, scale);
    const tensor_t* w2 = model_get_weight(model, name);
    snprintf(name, sizeof name, "model.%d.cv2.%d.2.bias", d_idx, scale);
    const tensor_t* b2 = model_get_weight(model, name);
    if (!w0 || !b0 || !w1 || !b1 || !w2 || !b2) return ERROR_FILE_NOT_FOUND;

    conv_params_t p3 = {1, 1, 1};
    status_t st = conv_block_forward(s1, feat, w0, b0, p3, true);
    if (st != SUCCESS) return st;
    st = conv_block_forward(s2, s1, w1, b1, p3, true);
    if (st != SUCCESS) return st;
    return conv2d_forward(out_box, s2, w2, b2, (conv_params_t){1, 0, 1});
}

static status_t run_cv3(model_t* model, int d_idx, int scale, const tensor_t* feat, tensor_t* out_cls,
                                tensor_t* t_dw0, tensor_t* t_pw0, tensor_t* t_dw1, tensor_t* t_pw1) {
    char name[200];
    snprintf(name, sizeof name, "model.%d.cv3.%d.0.0.conv.weight", d_idx, scale);
    const tensor_t* dw00 = model_get_weight(model, name);
    snprintf(name, sizeof name, "model.%d.cv3.%d.0.0.conv.bias", d_idx, scale);
    const tensor_t* db00 = model_get_weight(model, name);
    snprintf(name, sizeof name, "model.%d.cv3.%d.0.1.conv.weight", d_idx, scale);
    const tensor_t* pw00 = model_get_weight(model, name);
    snprintf(name, sizeof name, "model.%d.cv3.%d.0.1.conv.bias", d_idx, scale);
    const tensor_t* pb00 = model_get_weight(model, name);
    snprintf(name, sizeof name, "model.%d.cv3.%d.1.0.conv.weight", d_idx, scale);
    const tensor_t* dw10 = model_get_weight(model, name);
    snprintf(name, sizeof name, "model.%d.cv3.%d.1.0.conv.bias", d_idx, scale);
    const tensor_t* db10 = model_get_weight(model, name);
    snprintf(name, sizeof name, "model.%d.cv3.%d.1.1.conv.weight", d_idx, scale);
    const tensor_t* pw10 = model_get_weight(model, name);
    snprintf(name, sizeof name, "model.%d.cv3.%d.1.1.conv.bias", d_idx, scale);
    const tensor_t* pb10 = model_get_weight(model, name);
    snprintf(name, sizeof name, "model.%d.cv3.%d.2.weight", d_idx, scale);
    const tensor_t* wf = model_get_weight(model, name);
    snprintf(name, sizeof name, "model.%d.cv3.%d.2.bias", d_idx, scale);
    const tensor_t* bf = model_get_weight(model, name);
    if (!dw00 || !db00 || !pw00 || !pb00 || !dw10 || !db10 || !pw10 || !pb10 || !wf || !bf)
        return ERROR_FILE_NOT_FOUND;

    status_t st = dwconv3x3_same_forward(t_dw0, feat, dw00, db00);
    if (st != SUCCESS) return st;
    st = silu_forward(t_dw0);
    if (st != SUCCESS) return st;
    st = conv_block_forward(t_pw0, t_dw0, pw00, pb00, (conv_params_t){1, 0, 1}, true);
    if (st != SUCCESS) return st;
    st = dwconv3x3_same_forward(t_dw1, t_pw0, dw10, db10);
    if (st != SUCCESS) return st;
    st = silu_forward(t_dw1);
    if (st != SUCCESS) return st;
    st = conv_block_forward(t_pw1, t_dw1, pw10, pb10, (conv_params_t){1, 0, 1}, true);
    if (st != SUCCESS) return st;
    return conv2d_forward(out_cls, t_pw1, wf, bf, (conv_params_t){1, 0, 1});
}

static void copy_box_to_concat(const tensor_t* box, int N_total, int offset, float* boxes_dist) {
    int H = box->dims[2], W = box->dims[3];
    int hw = H * W;
    const float* d = box->data;
    for (int c = 0; c < 4; c++) {
        for (int j = 0; j < hw; j++) boxes_dist[(size_t)c * (size_t)N_total + (size_t)offset + (size_t)j] = d[(size_t)c * (size_t)hw + (size_t)j];
    }
}

static void copy_cls_to_concat(const tensor_t* cls, int nc, int N_total, int offset, float* cls_logits) {
    int H = cls->dims[2], W = cls->dims[3];
    int hw = H * W;
    const float* d = cls->data;
    for (int c = 0; c < nc; c++) {
        for (int j = 0; j < hw; j++)
            cls_logits[(size_t)c * (size_t)N_total + (size_t)offset + (size_t)j] = d[(size_t)c * (size_t)hw + (size_t)j];
    }
}

status_t detect_forward_one2one(model_t* model, int detect_module_idx, const tensor_t* p3, const tensor_t* p4,
                                const tensor_t* p5, tensor_t* out_postprocess) {
    if (!model || !p3 || !p4 || !p5 || !out_postprocess) return ERROR_NULL_POINTER;

    const tensor_t* feats[3] = {p3, p4, p5};
    int strides[3] = {DETECT_STRIDE0, DETECT_STRIDE1, DETECT_STRIDE2};
    int nc = model->num_classes;
    if (nc < 1) return ERROR_INVALID_DIMS;

    int H[3], W[3];
    for (int s = 0; s < 3; s++) {
        H[s] = feats[s]->dims[2];
        W[s] = feats[s]->dims[3];
    }
    int N = H[0] * W[0] + H[1] * W[1] + H[2] * W[2];

    float* ax = (float*)malloc((size_t)N * sizeof(float));
    float* ay = (float*)malloc((size_t)N * sizeof(float));
    float* stride_buf = (float*)malloc((size_t)N * sizeof(float));
    float* boxes_dist = (float*)malloc((size_t)4 * (size_t)N * sizeof(float));
    float* cls_logits = (float*)malloc((size_t)nc * (size_t)N * sizeof(float));
    float* xyxy_grid = (float*)malloc((size_t)4 * (size_t)N * sizeof(float));
    float* xyxy_px = (float*)malloc((size_t)4 * (size_t)N * sizeof(float));
    float* pred = (float*)malloc((size_t)N * (size_t)(4 + nc) * sizeof(float));
    if (!ax || !ay || !stride_buf || !boxes_dist || !cls_logits || !xyxy_grid || !xyxy_px || !pred) {
        free(ax);
        free(ay);
        free(stride_buf);
        free(boxes_dist);
        free(cls_logits);
        free(xyxy_grid);
        free(xyxy_px);
        free(pred);
        return ERROR_OUT_OF_MEMORY;
    }

    int base = 0;
    for (int s = 0; s < 3; s++) {
        make_anchors_for_shape(H[s], W[s], strides[s], ax, ay, stride_buf, base);
        base += H[s] * W[s];
    }

    int d = detect_module_idx;
    int offset = 0;
    char name[200];
    for (int s = 0; s < 3; s++) {
        const tensor_t* feat = feats[s];
        int h = H[s], w = W[s];
        int c_in = feat->dims[1];

        snprintf(name, sizeof name, "model.%d.cv2.%d.0.conv.weight", d, s);
        const tensor_t* w0 = model_get_weight(model, name);
        int c2 = w0 ? w0->dims[0] : 16;
        snprintf(name, sizeof name, "model.%d.cv3.%d.0.1.conv.weight", d, s);
        const tensor_t* pw0w = model_get_weight(model, name);
        int c3 = pw0w ? pw0w->dims[0] : nc;

        tensor_t out_box, s1, s2, out_cls, t_dw0, t_pw0, t_dw1, t_pw1;
        if (tensor_allocate(&out_box, 1, 4, h, w) != SUCCESS) goto oom;
        if (tensor_allocate(&s1, 1, c2, h, w) != SUCCESS) {
            tensor_free(&out_box);
            goto oom;
        }
        if (tensor_allocate(&s2, 1, c2, h, w) != SUCCESS) {
            tensor_free(&s1);
            tensor_free(&out_box);
            goto oom;
        }
        if (tensor_allocate(&t_dw0, 1, c_in, h, w) != SUCCESS) {
            tensor_free(&s2);
            tensor_free(&s1);
            tensor_free(&out_box);
            goto oom;
        }
        if (tensor_allocate(&t_pw0, 1, c3, h, w) != SUCCESS) {
            tensor_free(&t_dw0);
            tensor_free(&s2);
            tensor_free(&s1);
            tensor_free(&out_box);
            goto oom;
        }
        if (tensor_allocate(&t_dw1, 1, c3, h, w) != SUCCESS) {
            tensor_free(&t_pw0);
            tensor_free(&t_dw0);
            tensor_free(&s2);
            tensor_free(&s1);
            tensor_free(&out_box);
            goto oom;
        }
        if (tensor_allocate(&t_pw1, 1, c3, h, w) != SUCCESS) {
            tensor_free(&t_dw1);
            tensor_free(&t_pw0);
            tensor_free(&t_dw0);
            tensor_free(&s2);
            tensor_free(&s1);
            tensor_free(&out_box);
            goto oom;
        }
        if (tensor_allocate(&out_cls, 1, nc, h, w) != SUCCESS) {
            tensor_free(&t_pw1);
            tensor_free(&t_dw1);
            tensor_free(&t_pw0);
            tensor_free(&t_dw0);
            tensor_free(&s2);
            tensor_free(&s1);
            tensor_free(&out_box);
            goto oom;
        }

        status_t stt = run_cv2(model, d, s, feat, &out_box, &s1, &s2);
        if (stt != SUCCESS) {
            tensor_free(&out_cls);
            tensor_free(&t_pw1);
            tensor_free(&t_dw1);
            tensor_free(&t_pw0);
            tensor_free(&t_dw0);
            tensor_free(&s2);
            tensor_free(&s1);
            tensor_free(&out_box);
            goto fail;
        }
        stt = run_cv3(model, d, s, feat, &out_cls, &t_dw0, &t_pw0, &t_dw1, &t_pw1);
        if (stt != SUCCESS) {
            tensor_free(&out_cls);
            tensor_free(&t_pw1);
            tensor_free(&t_dw1);
            tensor_free(&t_pw0);
            tensor_free(&t_dw0);
            tensor_free(&s2);
            tensor_free(&s1);
            tensor_free(&out_box);
            goto fail;
        }

        copy_box_to_concat(&out_box, N, offset, boxes_dist);
        copy_cls_to_concat(&out_cls, nc, N, offset, cls_logits);

        tensor_free(&out_cls);
        tensor_free(&t_pw1);
        tensor_free(&t_dw1);
        tensor_free(&t_pw0);
        tensor_free(&t_dw0);
        tensor_free(&s2);
        tensor_free(&s1);
        tensor_free(&out_box);

        offset += h * w;
    }

    dist2bbox_xyxy(boxes_dist, ax, ay, N, xyxy_grid);
    mul_xyxy_stride(xyxy_grid, stride_buf, N, xyxy_px);
    sigmoid_ncN(cls_logits, nc, N);

    for (int j = 0; j < N; j++) {
        float* row = pred + (size_t)j * (size_t)(4 + nc);
        row[0] = xyxy_px[0 * N + j];
        row[1] = xyxy_px[1 * N + j];
        row[2] = xyxy_px[2 * N + j];
        row[3] = xyxy_px[3 * N + j];
        for (int c = 0; c < nc; c++) row[4 + c] = cls_logits[(size_t)c * (size_t)N + (size_t)j];
    }

    int max_det = out_postprocess->dims[1];
    status_t pst = detect_postprocess_from_pred(pred, N, nc, max_det, out_postprocess);

    free(pred);
    free(xyxy_px);
    free(xyxy_grid);
    free(cls_logits);
    free(boxes_dist);
    free(stride_buf);
    free(ay);
    free(ax);
    return pst;

fail:
    free(pred);
    free(xyxy_px);
    free(xyxy_grid);
    free(cls_logits);
    free(boxes_dist);
    free(stride_buf);
    free(ay);
    free(ax);
    return ERROR_FILE_NOT_FOUND;

oom:
    free(pred);
    free(xyxy_px);
    free(xyxy_grid);
    free(cls_logits);
    free(boxes_dist);
    free(stride_buf);
    free(ay);
    free(ax);
    return ERROR_OUT_OF_MEMORY;
}
