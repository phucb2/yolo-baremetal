#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "model.h"
#include "utils.h"
#include "detect.h"

status_t model_create(model_t* model, int input_w, int input_h) {
    if (!model) return ERROR_NULL_POINTER;
    model->input_w = input_w;
    model->input_h = input_h;
    model->weights = NULL;
    model->num_weights = 0;
    model->num_classes = 80;
    
    // Allocate 24 buffers for the 24 layers defined in yolo26.yaml
    model->num_buffers = 24;
    model->buffers = (tensor_t*)malloc(sizeof(tensor_t) * model->num_buffers);
    for (int i = 0; i < model->num_buffers; i++) model->buffers[i].data = NULL;
    
    // Precise buffer allocation for YOLO26n (640x640 input)
    // P1/2: 320x320
    tensor_allocate(&model->buffers[0], 1, 16, input_h/2, input_w/2);
    // P2/4: 160x160
    tensor_allocate(&model->buffers[1], 1, 32, input_h/4, input_w/4);
    tensor_allocate(&model->buffers[2], 1, 64, input_h/4, input_w/4);
    // P3/8: 80x80
    tensor_allocate(&model->buffers[3], 1, 64, input_h/8, input_w/8);
    tensor_allocate(&model->buffers[4], 1, 128, input_h/8, input_w/8);
    // P4/16: 40x40
    tensor_allocate(&model->buffers[5], 1, 128, input_h/16, input_w/16);
    tensor_allocate(&model->buffers[6], 1, 128, input_h/16, input_w/16);
    // P5/32: 20x20
    tensor_allocate(&model->buffers[7], 1, 256, input_h/32, input_w/32);
    tensor_allocate(&model->buffers[8], 1, 256, input_h/32, input_w/32);
    tensor_allocate(&model->buffers[9], 1, 256, input_h/32, input_w/32); // SPPF
    tensor_allocate(&model->buffers[10], 1, 256, input_h/32, input_w/32); // C2PSA
    
    // Head buffers
    tensor_allocate(&model->buffers[11], 1, 256, input_h/16, input_w/16); // Upsample
    tensor_allocate(&model->buffers[12], 1, 384, input_h/16, input_w/16); // Concat
    tensor_allocate(&model->buffers[13], 1, 128, input_h/16, input_w/16); // C3k2
    
    tensor_allocate(&model->buffers[14], 1, 128, input_h/8, input_w/8);  // Upsample
    tensor_allocate(&model->buffers[15], 1, 256, input_h/8, input_w/8);  // Concat
    tensor_allocate(&model->buffers[16], 1, 64, input_h/8, input_w/8);   // C3k2 (P3 small)
    
    tensor_allocate(&model->buffers[17], 1, 64, input_h/16, input_w/16); // Conv
    tensor_allocate(&model->buffers[18], 1, 192, input_h/16, input_w/16); // Concat
    tensor_allocate(&model->buffers[19], 1, 128, input_h/16, input_w/16); // C3k2 (P4 medium)
    
    tensor_allocate(&model->buffers[20], 1, 128, input_h/32, input_w/32); // Conv
    tensor_allocate(&model->buffers[21], 1, 384, input_h/32, input_w/32); // Concat
    tensor_allocate(&model->buffers[22], 1, 256, input_h/32, input_w/32); // C3k2 (P5 large)
    
    /* Final detections buffer: Ultralytics Detect.postprocess (reg_max=1) -> [1, max_det, 6] xyxy, score, class */
    tensor_allocate(&model->buffers[23], 1, 300, 6, 1);
    
    return SUCCESS;
}

status_t model_destroy(model_t* model) {
    if (!model) return ERROR_NULL_POINTER;
    for (int i = 0; i < model->num_weights; i++) tensor_free(&model->weights[i].tensor);
    free(model->weights);
    for (int i = 0; i < model->num_buffers; i++) tensor_free(&model->buffers[i]);
    free(model->buffers);
    return SUCCESS;
}

static int ends_with(const char* s, const char* suf) {
    size_t ls = strlen(s), su = strlen(suf);
    return ls >= su && strcmp(s + (ls - su), suf) == 0;
}

/* name ends with ".conv.weight" -> prefix (module path before .conv.weight). */
static void prefix_from_conv_weight_name(const char* name, char* prefix, size_t psz) {
    const char suf[] = ".conv.weight";
    size_t ln = strlen(name), ls = sizeof(suf) - 1;
    if (ln < ls || strcmp(name + ln - ls, suf) != 0) {
        prefix[0] = '\0';
        return;
    }
    size_t plen = ln - ls;
    if (plen >= psz) plen = psz - 1;
    memcpy(prefix, name, plen);
    prefix[plen] = '\0';
}

static int model_weight_index(model_t* model, const char* name) {
    for (int i = 0; i < model->num_weights; i++) {
        if (strcmp(model->weights[i].name, name) == 0) return i;
    }
    return -1;
}

static status_t model_remove_weight_by_name(model_t* model, const char* name) {
    int idx = model_weight_index(model, name);
    if (idx < 0) return ERROR_FILE_NOT_FOUND;
    tensor_free(&model->weights[idx].tensor);
    memmove(&model->weights[idx], &model->weights[idx + 1],
            (size_t)(model->num_weights - idx - 1) * sizeof(named_tensor_t));
    model->num_weights--;
    return SUCCESS;
}

static status_t model_append_weight(model_t* model, const char* name, tensor_t* tensor_owned) {
    named_tensor_t* nw =
        realloc(model->weights, (size_t)(model->num_weights + 1) * sizeof(named_tensor_t));
    if (!nw) return ERROR_OUT_OF_MEMORY;
    model->weights = nw;
    strncpy(model->weights[model->num_weights].name, name, sizeof(model->weights[model->num_weights].name) - 1);
    model->weights[model->num_weights].name[127] = '\0';
    model->weights[model->num_weights].tensor = *tensor_owned;
    model->num_weights++;
    return SUCCESS;
}

/* Fold every *.conv.weight that has Ultralytics Conv sibling *.bn.* (nested paths, any depth). */
static void fold_all_bn(model_t* model) {
    char prefix[128];
    char buf[160];

    for (;;) {
        int idx = -1;
        for (int i = 0; i < model->num_weights; i++) {
            const char* name = model->weights[i].name;
            if (!ends_with(name, ".conv.weight")) continue;
            prefix_from_conv_weight_name(name, prefix, sizeof(prefix));
            if (!prefix[0]) continue;
            snprintf(buf, sizeof buf, "%s.bn.weight", prefix);
            if (model_get_weight(model, buf)) {
                idx = i;
                break;
            }
        }
        if (idx < 0) break;

        prefix_from_conv_weight_name(model->weights[idx].name, prefix, sizeof(prefix));
        tensor_t* cw = &model->weights[idx].tensor;

        snprintf(buf, sizeof buf, "%s.bn.weight", prefix);
        tensor_t* bn_w = model_get_weight(model, buf);
        snprintf(buf, sizeof buf, "%s.bn.bias", prefix);
        tensor_t* bn_b = model_get_weight(model, buf);
        snprintf(buf, sizeof buf, "%s.bn.running_mean", prefix);
        tensor_t* bn_m = model_get_weight(model, buf);
        snprintf(buf, sizeof buf, "%s.bn.running_var", prefix);
        tensor_t* bn_v = model_get_weight(model, buf);
        if (!bn_w || !bn_b || !bn_m || !bn_v) break;

        snprintf(buf, sizeof buf, "%s.conv.bias", prefix);
        tensor_t* cb = model_get_weight(model, buf);

        if (!cb) {
            tensor_t bias;
            if (tensor_allocate(&bias, 1, cw->dims[0], 1, 1) != SUCCESS) break;
            tensor_fill(&bias, 0.0f);
            fold_bn(cw, &bias, bn_w, bn_b, bn_m, bn_v);
            if (model_append_weight(model, buf, &bias) != SUCCESS) {
                tensor_free(&bias);
                break;
            }
        } else {
            fold_bn(cw, cb, bn_w, bn_b, bn_m, bn_v);
        }

        snprintf(buf, sizeof buf, "%s.bn.weight", prefix);
        model_remove_weight_by_name(model, buf);
        snprintf(buf, sizeof buf, "%s.bn.bias", prefix);
        model_remove_weight_by_name(model, buf);
        snprintf(buf, sizeof buf, "%s.bn.running_mean", prefix);
        model_remove_weight_by_name(model, buf);
        snprintf(buf, sizeof buf, "%s.bn.running_var", prefix);
        model_remove_weight_by_name(model, buf);
        snprintf(buf, sizeof buf, "%s.bn.num_batches_tracked", prefix);
        if (model_weight_index(model, buf) >= 0) model_remove_weight_by_name(model, buf);
    }
}

status_t model_load_weights(model_t* model, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return ERROR_FILE_NOT_FOUND;
    int nc, total_params;
    fread(&nc, sizeof(int), 1, f);
    fread(&total_params, sizeof(int), 1, f);
    model->num_classes = nc;
    model->num_weights = total_params;
    model->weights = (named_tensor_t*)malloc(sizeof(named_tensor_t) * total_params);
    for (int i = 0; i < total_params; i++) load_named_tensor(f, model->weights[i].name, &model->weights[i].tensor);
    fclose(f);

    fold_all_bn(model);
    printf("Successfully loaded %d tensors (after BN fold, %d tensors in memory)\n", total_params,
           model->num_weights);
    return SUCCESS;
}


tensor_t* model_get_weight(model_t* model, const char* name) {
    for (int i = 0; i < model->num_weights; i++) {
        if (strcmp(model->weights[i].name, name) == 0) return &model->weights[i].tensor;
    }
    return NULL;
}

static status_t run_c3k2(model_t* m, int li, const tensor_t* in, tensor_t* out, int n, bool shortcut) {
    char name[200];
    snprintf(name, sizeof name, "model.%d.cv1.conv.weight", li);
    tensor_t* cv1_w = model_get_weight(m, name);
    snprintf(name, sizeof name, "model.%d.cv1.conv.bias", li);
    tensor_t* cv1_b = model_get_weight(m, name);
    snprintf(name, sizeof name, "model.%d.cv2.conv.weight", li);
    tensor_t* cv2_w = model_get_weight(m, name);
    snprintf(name, sizeof name, "model.%d.cv2.conv.bias", li);
    tensor_t* cv2_b = model_get_weight(m, name);
    if (!cv1_w || !cv1_b || !cv2_w || !cv2_b) return ERROR_FILE_NOT_FOUND;

    int c_total = cv1_w->dims[0];
    int c_half = c_total / 2;
    int h = in->dims[2], wi = in->dims[3];
    int nbuf = n + 3;

    tensor_t* bufs = (tensor_t*)calloc((size_t)nbuf, sizeof(tensor_t));
    tensor_t* bw = (tensor_t*)malloc(sizeof(tensor_t) * (size_t)n * 4);
    if (!bufs || !bw) {
        free(bufs);
        free(bw);
        return ERROR_OUT_OF_MEMORY;
    }

    int k = 0;
    status_t st = ERROR_OUT_OF_MEMORY;
    if (tensor_allocate(&bufs[0], 1, c_total, h, wi) != SUCCESS) goto c3k2_err;
    k = 1;
    for (int i = 1; i <= n; i++) {
        if (tensor_allocate(&bufs[i], 1, c_half, h, wi) != SUCCESS) goto c3k2_err;
        k = i + 1;
    }
    if (tensor_allocate(&bufs[n + 1], 1, c_half, h, wi) != SUCCESS) goto c3k2_err;
    k = n + 2;
    if (tensor_allocate(&bufs[n + 2], 1, (2 + n) * c_half, h, wi) != SUCCESS) goto c3k2_err;
    k = nbuf;

    for (int i = 0; i < n; i++) {
        snprintf(name, sizeof name, "model.%d.m.%d.cv1.conv.weight", li, i);
        tensor_t* p = model_get_weight(m, name);
        if (!p) {
            st = ERROR_FILE_NOT_FOUND;
            goto c3k2_err;
        }
        bw[i * 4 + 0] = *p;
        snprintf(name, sizeof name, "model.%d.m.%d.cv1.conv.bias", li, i);
        p = model_get_weight(m, name);
        if (!p) {
            st = ERROR_FILE_NOT_FOUND;
            goto c3k2_err;
        }
        bw[i * 4 + 1] = *p;
        snprintf(name, sizeof name, "model.%d.m.%d.cv2.conv.weight", li, i);
        p = model_get_weight(m, name);
        if (!p) {
            st = ERROR_FILE_NOT_FOUND;
            goto c3k2_err;
        }
        bw[i * 4 + 2] = *p;
        snprintf(name, sizeof name, "model.%d.m.%d.cv2.conv.bias", li, i);
        p = model_get_weight(m, name);
        if (!p) {
            st = ERROR_FILE_NOT_FOUND;
            goto c3k2_err;
        }
        bw[i * 4 + 3] = *p;
    }

    st = c3k2_forward(out, in, n, shortcut, cv1_w, cv1_b, cv2_w, cv2_b, bw, bufs);
c3k2_err:
    free(bw);
    for (int i = 0; i < k; i++) tensor_free(&bufs[i]);
    free(bufs);
    return st;
}

static status_t run_sppf(model_t* m, int li, const tensor_t* in, tensor_t* out, int kernel_size, int n_pool,
                         bool shortcut) {
    char name[200];
    snprintf(name, sizeof name, "model.%d.cv1.conv.weight", li);
    tensor_t* cv1_w = model_get_weight(m, name);
    snprintf(name, sizeof name, "model.%d.cv1.conv.bias", li);
    tensor_t* cv1_b = model_get_weight(m, name);
    snprintf(name, sizeof name, "model.%d.cv2.conv.weight", li);
    tensor_t* cv2_w = model_get_weight(m, name);
    snprintf(name, sizeof name, "model.%d.cv2.conv.bias", li);
    tensor_t* cv2_b = model_get_weight(m, name);
    if (!cv1_w || !cv1_b || !cv2_w || !cv2_b) return ERROR_FILE_NOT_FOUND;

    int c_ = cv1_w->dims[0];
    int h = in->dims[2], wi = in->dims[3];
    int nbuf = n_pool + 2;
    tensor_t* bufs = (tensor_t*)calloc((size_t)nbuf, sizeof(tensor_t));
    if (!bufs) return ERROR_OUT_OF_MEMORY;

    int k = 0;
    status_t st = ERROR_OUT_OF_MEMORY;
    if (tensor_allocate(&bufs[0], 1, c_, h, wi) != SUCCESS) goto sppf_err;
    k = 1;
    for (int i = 1; i <= n_pool; i++) {
        if (tensor_allocate(&bufs[i], 1, c_, h, wi) != SUCCESS) goto sppf_err;
        k = i + 1;
    }
    if (tensor_allocate(&bufs[n_pool + 1], 1, c_ * (n_pool + 1), h, wi) != SUCCESS) goto sppf_err;
    k = nbuf;

    st = sppf_forward(out, in, cv1_w, cv1_b, cv2_w, cv2_b, kernel_size, n_pool, shortcut, bufs);
sppf_err:
    for (int i = 0; i < k; i++) tensor_free(&bufs[i]);
    free(bufs);
    return st;
}

static status_t run_c2psa(model_t* m, int li, const tensor_t* in, tensor_t* out, int n_blocks, float e,
                          float attn_ratio) {
    char name[200];
    snprintf(name, sizeof name, "model.%d.cv1.conv.weight", li);
    tensor_t* cv1_w = model_get_weight(m, name);
    snprintf(name, sizeof name, "model.%d.cv1.conv.bias", li);
    tensor_t* cv1_b = model_get_weight(m, name);
    snprintf(name, sizeof name, "model.%d.cv2.conv.weight", li);
    tensor_t* cv2_w = model_get_weight(m, name);
    snprintf(name, sizeof name, "model.%d.cv2.conv.bias", li);
    tensor_t* cv2_b = model_get_weight(m, name);
    if (!cv1_w || !cv1_b || !cv2_w || !cv2_b) return ERROR_FILE_NOT_FOUND;

    int c1 = in->dims[1];
    int c_hidden = (int)((float)c1 * e);
    int h = in->dims[2], wi = in->dims[3];

    tensor_t bufs[3];
    for (int i = 0; i < 3; i++) bufs[i].data = NULL;

    if (tensor_allocate(&bufs[0], 1, 2 * c_hidden, h, wi) != SUCCESS) return ERROR_OUT_OF_MEMORY;
    if (tensor_allocate(&bufs[1], 1, c_hidden, h, wi) != SUCCESS) {
        tensor_free(&bufs[0]);
        return ERROR_OUT_OF_MEMORY;
    }
    if (tensor_allocate(&bufs[2], 1, 2 * c_hidden, h, wi) != SUCCESS) {
        tensor_free(&bufs[1]);
        tensor_free(&bufs[0]);
        return ERROR_OUT_OF_MEMORY;
    }

    tensor_t* psa = (tensor_t*)malloc(sizeof(tensor_t) * (size_t)n_blocks * 10);
    if (!psa) {
        tensor_free(&bufs[2]);
        tensor_free(&bufs[1]);
        tensor_free(&bufs[0]);
        return ERROR_OUT_OF_MEMORY;
    }

    status_t st = ERROR_FILE_NOT_FOUND;
    for (int bi = 0; bi < n_blocks; bi++) {
        snprintf(name, sizeof name, "model.%d.m.%d.attn.qkv.conv.weight", li, bi);
        tensor_t* p = model_get_weight(m, name);
        if (!p) goto c2psa_fail;
        psa[bi * 10 + 0] = *p;
        snprintf(name, sizeof name, "model.%d.m.%d.attn.qkv.conv.bias", li, bi);
        p = model_get_weight(m, name);
        if (!p) goto c2psa_fail;
        psa[bi * 10 + 1] = *p;
        snprintf(name, sizeof name, "model.%d.m.%d.attn.proj.conv.weight", li, bi);
        p = model_get_weight(m, name);
        if (!p) goto c2psa_fail;
        psa[bi * 10 + 2] = *p;
        snprintf(name, sizeof name, "model.%d.m.%d.attn.proj.conv.bias", li, bi);
        p = model_get_weight(m, name);
        if (!p) goto c2psa_fail;
        psa[bi * 10 + 3] = *p;
        snprintf(name, sizeof name, "model.%d.m.%d.attn.pe.conv.weight", li, bi);
        p = model_get_weight(m, name);
        if (!p) goto c2psa_fail;
        psa[bi * 10 + 4] = *p;
        snprintf(name, sizeof name, "model.%d.m.%d.attn.pe.conv.bias", li, bi);
        p = model_get_weight(m, name);
        if (!p) goto c2psa_fail;
        psa[bi * 10 + 5] = *p;
        snprintf(name, sizeof name, "model.%d.m.%d.ffn.0.conv.weight", li, bi);
        p = model_get_weight(m, name);
        if (!p) goto c2psa_fail;
        psa[bi * 10 + 6] = *p;
        snprintf(name, sizeof name, "model.%d.m.%d.ffn.0.conv.bias", li, bi);
        p = model_get_weight(m, name);
        if (!p) goto c2psa_fail;
        psa[bi * 10 + 7] = *p;
        snprintf(name, sizeof name, "model.%d.m.%d.ffn.1.conv.weight", li, bi);
        p = model_get_weight(m, name);
        if (!p) goto c2psa_fail;
        psa[bi * 10 + 8] = *p;
        snprintf(name, sizeof name, "model.%d.m.%d.ffn.1.conv.bias", li, bi);
        p = model_get_weight(m, name);
        if (!p) goto c2psa_fail;
        psa[bi * 10 + 9] = *p;
    }

    st = c2psa_forward(out, in, n_blocks, e, attn_ratio, cv1_w, cv1_b, cv2_w, cv2_b, psa, bufs);
c2psa_fail:
    free(psa);
    tensor_free(&bufs[2]);
    tensor_free(&bufs[1]);
    tensor_free(&bufs[0]);
    return st;
}

/* C3k2 with inner C3k (C3 at m.0): weights model.L.m.0.cv{1,2,3}, model.L.m.0.m.{0,1} bottlenecks. */
static status_t forward_c3k2_c3_inner(model_t* m, int li, const tensor_t* in, tensor_t* out, bool c3_bottle_shortcut) {
    char name[240];
    snprintf(name, sizeof name, "model.%d.cv1.conv.weight", li);
    tensor_t* ocv1_w = model_get_weight(m, name);
    snprintf(name, sizeof name, "model.%d.cv1.conv.bias", li);
    tensor_t* ocv1_b = model_get_weight(m, name);
    snprintf(name, sizeof name, "model.%d.cv2.conv.weight", li);
    tensor_t* ocv2_w = model_get_weight(m, name);
    snprintf(name, sizeof name, "model.%d.cv2.conv.bias", li);
    tensor_t* ocv2_b = model_get_weight(m, name);
    if (!ocv1_w || !ocv1_b || !ocv2_w || !ocv2_b) return ERROR_FILE_NOT_FOUND;

    snprintf(name, sizeof name, "model.%d.m.0.cv1.conv.weight", li);
    tensor_t* c3_cv1_w = model_get_weight(m, name);
    snprintf(name, sizeof name, "model.%d.m.0.cv1.conv.bias", li);
    tensor_t* c3_cv1_b = model_get_weight(m, name);
    snprintf(name, sizeof name, "model.%d.m.0.cv2.conv.weight", li);
    tensor_t* c3_cv2_w = model_get_weight(m, name);
    snprintf(name, sizeof name, "model.%d.m.0.cv2.conv.bias", li);
    tensor_t* c3_cv2_b = model_get_weight(m, name);
    snprintf(name, sizeof name, "model.%d.m.0.cv3.conv.weight", li);
    tensor_t* c3_cv3_w = model_get_weight(m, name);
    snprintf(name, sizeof name, "model.%d.m.0.cv3.conv.bias", li);
    tensor_t* c3_cv3_b = model_get_weight(m, name);
    if (!c3_cv1_w || !c3_cv1_b || !c3_cv2_w || !c3_cv2_b || !c3_cv3_w || !c3_cv3_b) return ERROR_FILE_NOT_FOUND;

    const int n_bottles = 2;
    tensor_t c3_b[8];
    for (int bi = 0; bi < n_bottles; bi++) {
        snprintf(name, sizeof name, "model.%d.m.0.m.%d.cv1.conv.weight", li, bi);
        tensor_t* p = model_get_weight(m, name);
        if (!p) return ERROR_FILE_NOT_FOUND;
        c3_b[bi * 4 + 0] = *p;
        snprintf(name, sizeof name, "model.%d.m.0.m.%d.cv1.conv.bias", li, bi);
        p = model_get_weight(m, name);
        if (!p) return ERROR_FILE_NOT_FOUND;
        c3_b[bi * 4 + 1] = *p;
        snprintf(name, sizeof name, "model.%d.m.0.m.%d.cv2.conv.weight", li, bi);
        p = model_get_weight(m, name);
        if (!p) return ERROR_FILE_NOT_FOUND;
        c3_b[bi * 4 + 2] = *p;
        snprintf(name, sizeof name, "model.%d.m.0.m.%d.cv2.conv.bias", li, bi);
        p = model_get_weight(m, name);
        if (!p) return ERROR_FILE_NOT_FOUND;
        c3_b[bi * 4 + 3] = *p;
    }

    int c_half = ocv1_w->dims[0] / 2;
    int h = in->dims[2], wi = in->dims[3];
    int plane = h * wi;
    conv_params_t p1 = {1, 0, 1};

    tensor_t tcv1;
    if (tensor_allocate(&tcv1, 1, 2 * c_half, h, wi) != SUCCESS) return ERROR_OUT_OF_MEMORY;
    status_t st = conv_block_forward(&tcv1, in, ocv1_w, ocv1_b, p1, true);
    if (st != SUCCESS) {
        tensor_free(&tcv1);
        return st;
    }

    tensor_t b_in;
    b_in.dims[0] = 1;
    b_in.dims[1] = c_half;
    b_in.dims[2] = h;
    b_in.dims[3] = wi;
    b_in.data = tcv1.data + (size_t)c_half * plane;
    b_in.stride[0] = c_half * plane;
    b_in.stride[1] = plane;
    b_in.stride[2] = wi;
    b_in.stride[3] = 1;
    b_in.is_owner = false;

    int c3_c_ = c3_cv1_w->dims[0];
    tensor_t c3_out;
    tensor_t c3_bufs[5];
    for (int i = 0; i < 5; i++) c3_bufs[i].data = NULL;

    if (tensor_allocate(&c3_out, 1, c3_cv3_w->dims[0], h, wi) != SUCCESS) {
        tensor_free(&tcv1);
        return ERROR_OUT_OF_MEMORY;
    }
    if (tensor_allocate(&c3_bufs[0], 1, c3_c_, h, wi) != SUCCESS) {
        tensor_free(&c3_out);
        tensor_free(&tcv1);
        return ERROR_OUT_OF_MEMORY;
    }
    if (tensor_allocate(&c3_bufs[1], 1, c3_c_, h, wi) != SUCCESS) {
        tensor_free(&c3_bufs[0]);
        tensor_free(&c3_out);
        tensor_free(&tcv1);
        return ERROR_OUT_OF_MEMORY;
    }
    if (tensor_allocate(&c3_bufs[2], 1, c3_c_, h, wi) != SUCCESS) {
        tensor_free(&c3_bufs[1]);
        tensor_free(&c3_bufs[0]);
        tensor_free(&c3_out);
        tensor_free(&tcv1);
        return ERROR_OUT_OF_MEMORY;
    }
    if (tensor_allocate(&c3_bufs[3], 1, c3_c_, h, wi) != SUCCESS) {
        tensor_free(&c3_bufs[2]);
        tensor_free(&c3_bufs[1]);
        tensor_free(&c3_bufs[0]);
        tensor_free(&c3_out);
        tensor_free(&tcv1);
        return ERROR_OUT_OF_MEMORY;
    }
    if (tensor_allocate(&c3_bufs[4], 1, c3_cv3_w->dims[1], h, wi) != SUCCESS) {
        tensor_free(&c3_bufs[3]);
        tensor_free(&c3_bufs[2]);
        tensor_free(&c3_bufs[1]);
        tensor_free(&c3_bufs[0]);
        tensor_free(&c3_out);
        tensor_free(&tcv1);
        return ERROR_OUT_OF_MEMORY;
    }

    st = c3_forward(&c3_out, &b_in, c3_cv1_w, c3_cv1_b, c3_cv2_w, c3_cv2_b, c3_cv3_w, c3_cv3_b, c3_b, n_bottles,
                    c3_bottle_shortcut, c3_bufs);
    for (int i = 0; i < 5; i++) tensor_free(&c3_bufs[i]);
    if (st != SUCCESS) {
        tensor_free(&c3_out);
        tensor_free(&tcv1);
        return st;
    }

    tensor_t comb;
    if (tensor_allocate(&comb, 1, 3 * c_half, h, wi) != SUCCESS) {
        tensor_free(&c3_out);
        tensor_free(&tcv1);
        return ERROR_OUT_OF_MEMORY;
    }
    memcpy(comb.data, tcv1.data, (size_t)c_half * plane * sizeof(float));
    memcpy(comb.data + (size_t)c_half * plane, tcv1.data + (size_t)c_half * plane,
           (size_t)c_half * plane * sizeof(float));
    memcpy(comb.data + (size_t)(2 * c_half) * plane, c3_out.data, (size_t)c_half * plane * sizeof(float));
    tensor_free(&c3_out);
    tensor_free(&tcv1);

    st = conv_block_forward(out, &comb, ocv2_w, ocv2_b, p1, true);
    tensor_free(&comb);
    return st;
}

/* Head layer 22: C3k2 with attn (Sequential: Bottleneck m.0.0 + PSABlock m.0.1); cat(chunk0, chunk1, seq(chunk1)). */
static status_t forward_c3k2_attn_head(model_t* m, int li, const tensor_t* in, tensor_t* out) {
    char name[220];
    snprintf(name, sizeof name, "model.%d.cv1.conv.weight", li);
    tensor_t* ocv1_w = model_get_weight(m, name);
    snprintf(name, sizeof name, "model.%d.cv1.conv.bias", li);
    tensor_t* ocv1_b = model_get_weight(m, name);
    snprintf(name, sizeof name, "model.%d.cv2.conv.weight", li);
    tensor_t* ocv2_w = model_get_weight(m, name);
    snprintf(name, sizeof name, "model.%d.cv2.conv.bias", li);
    tensor_t* ocv2_b = model_get_weight(m, name);
    if (!ocv1_w || !ocv1_b || !ocv2_w || !ocv2_b) return ERROR_FILE_NOT_FOUND;

    snprintf(name, sizeof name, "model.%d.m.0.0.cv1.conv.weight", li);
    tensor_t* bn_cv1_w = model_get_weight(m, name);
    snprintf(name, sizeof name, "model.%d.m.0.0.cv1.conv.bias", li);
    tensor_t* bn_cv1_b = model_get_weight(m, name);
    snprintf(name, sizeof name, "model.%d.m.0.0.cv2.conv.weight", li);
    tensor_t* bn_cv2_w = model_get_weight(m, name);
    snprintf(name, sizeof name, "model.%d.m.0.0.cv2.conv.bias", li);
    tensor_t* bn_cv2_b = model_get_weight(m, name);
    if (!bn_cv1_w || !bn_cv1_b || !bn_cv2_w || !bn_cv2_b) return ERROR_FILE_NOT_FOUND;

    snprintf(name, sizeof name, "model.%d.m.0.1.attn.qkv.conv.weight", li);
    tensor_t* qkv_w = model_get_weight(m, name);
    snprintf(name, sizeof name, "model.%d.m.0.1.attn.qkv.conv.bias", li);
    tensor_t* qkv_b = model_get_weight(m, name);
    snprintf(name, sizeof name, "model.%d.m.0.1.attn.proj.conv.weight", li);
    tensor_t* proj_w = model_get_weight(m, name);
    snprintf(name, sizeof name, "model.%d.m.0.1.attn.proj.conv.bias", li);
    tensor_t* proj_b = model_get_weight(m, name);
    snprintf(name, sizeof name, "model.%d.m.0.1.attn.pe.conv.weight", li);
    tensor_t* pe_w = model_get_weight(m, name);
    snprintf(name, sizeof name, "model.%d.m.0.1.attn.pe.conv.bias", li);
    tensor_t* pe_b = model_get_weight(m, name);
    snprintf(name, sizeof name, "model.%d.m.0.1.ffn.0.conv.weight", li);
    tensor_t* ffn0_w = model_get_weight(m, name);
    snprintf(name, sizeof name, "model.%d.m.0.1.ffn.0.conv.bias", li);
    tensor_t* ffn0_b = model_get_weight(m, name);
    snprintf(name, sizeof name, "model.%d.m.0.1.ffn.1.conv.weight", li);
    tensor_t* ffn1_w = model_get_weight(m, name);
    snprintf(name, sizeof name, "model.%d.m.0.1.ffn.1.conv.bias", li);
    tensor_t* ffn1_b = model_get_weight(m, name);
    if (!qkv_w || !qkv_b || !proj_w || !proj_b || !pe_w || !pe_b || !ffn0_w || !ffn0_b || !ffn1_w || !ffn1_b)
        return ERROR_FILE_NOT_FOUND;

    int c_half = ocv1_w->dims[0] / 2;
    int h = in->dims[2], wi = in->dims[3];
    int plane = h * wi;
    tensor_t* comb = &m->buffers[21];
    tensor_t* tcv1 = &m->buffers[22];

    conv_params_t p1 = {1, 0, 1};
    status_t st = conv_block_forward(tcv1, in, ocv1_w, ocv1_b, p1, true);
    if (st != SUCCESS) return st;

    tensor_t b_in;
    b_in.dims[0] = 1;
    b_in.dims[1] = c_half;
    b_in.dims[2] = h;
    b_in.dims[3] = wi;
    b_in.data = tcv1->data + (size_t)c_half * plane;
    b_in.stride[0] = c_half * plane;
    b_in.stride[1] = plane;
    b_in.stride[2] = wi;
    b_in.stride[3] = 1;
    b_in.is_owner = false;

    int bn_h = bn_cv1_w->dims[0];
    tensor_t bn_out;
    tensor_t bn_temp;
    if (tensor_allocate(&bn_out, 1, c_half, h, wi) != SUCCESS) return ERROR_OUT_OF_MEMORY;
    if (tensor_allocate(&bn_temp, 1, bn_h, h, wi) != SUCCESS) {
        tensor_free(&bn_out);
        return ERROR_OUT_OF_MEMORY;
    }
    st = bottleneck_forward(&bn_out, &b_in, bn_cv1_w, bn_cv1_b, bn_cv2_w, bn_cv2_b, true, &bn_temp);
    tensor_free(&bn_temp);
    if (st != SUCCESS) {
        tensor_free(&bn_out);
        return st;
    }

    tensor_t psa_out;
    if (tensor_allocate(&psa_out, 1, c_half, h, wi) != SUCCESS) {
        tensor_free(&bn_out);
        return ERROR_OUT_OF_MEMORY;
    }
    int num_heads = c_half / 64;
    if (num_heads < 1) num_heads = 1;
    st = psablock_forward(&psa_out, &bn_out, true, qkv_w, qkv_b, proj_w, proj_b, pe_w, pe_b, ffn0_w, ffn0_b, ffn1_w,
                          ffn1_b, num_heads, 0.5f);
    tensor_free(&bn_out);
    if (st != SUCCESS) {
        tensor_free(&psa_out);
        return st;
    }

    memcpy(comb->data, tcv1->data, (size_t)c_half * plane * sizeof(float));
    memcpy(comb->data + (size_t)c_half * plane, tcv1->data + (size_t)c_half * plane,
           (size_t)c_half * plane * sizeof(float));
    memcpy(comb->data + (size_t)(2 * c_half) * plane, psa_out.data, (size_t)c_half * plane * sizeof(float));
    tensor_free(&psa_out);

    st = conv_block_forward(out, comb, ocv2_w, ocv2_b, p1, true);
    return st;
}

static status_t conv_blk(model_t* m, const char* wname, const char* bname, const tensor_t* in, tensor_t* out,
                         conv_params_t p, bool act) {
    tensor_t* w = model_get_weight(m, wname);
    tensor_t* b = model_get_weight(m, bname);
    if (!w || !b) return ERROR_FILE_NOT_FOUND;
    return conv_block_forward(out, in, w, b, p, act);
}

static status_t dump_stage(FILE* f, const char* name, const tensor_t* t) {
    if (!f) return SUCCESS;
    return save_named_tensor(f, name, t);
}

static const char* const k_model_profile_names[MODEL_FORWARD_PROFILE_STEPS] = {
    "conv model.0 /2 + SiLU",
    "conv model.1 /2 + SiLU",
    "C3k2 model.2 (n=1)",
    "conv model.3 /2 + SiLU",
    "C3k2 model.4 (n=1)",
    "conv model.5 /2 + SiLU",
    "C3k2 inner model.6",
    "conv model.7 /2 + SiLU",
    "C3k2 inner model.8",
    "SPPF model.9",
    "C2PSA model.10",
    "upsample 2x → buf11",
    "concat(buf11, buf6) → buf12",
    "C3k2 inner model.13",
    "upsample 2x → buf14",
    "concat(buf14, buf4) → buf15",
    "C3k2 inner model.16 (P3)",
    "conv model.17 /2 + SiLU",
    "concat(buf17, buf13) → buf18",
    "C3k2 inner model.19 (P4)",
    "conv model.20 /2 + SiLU",
    "concat(buf20, buf10) → buf21",
    "C3k2 attn head model.22 (P5)",
    "Detect one2one → buf23",
    "tensor_copy buf23 → output",
};

void model_forward_profile_reset(model_forward_profile_t* p) {
    if (!p) return;
    memset(p->ms_sum, 0, sizeof(p->ms_sum));
    memset(p->ms_last, 0, sizeof(p->ms_last));
    p->runs = 0;
}

const char* model_forward_profile_step_name(int step_index) {
    if (step_index < 0 || step_index >= MODEL_FORWARD_PROFILE_STEPS) return "?";
    return k_model_profile_names[step_index];
}

void model_forward_profile_print_last(const model_forward_profile_t* p, FILE* fp, const char* title) {
    if (!p || !fp || p->runs == 0) return;
    fprintf(fp, "%s\n", title);
    fprintf(fp, "%-4s %-44s %12s %8s\n", "#", "step", "ms", "% fwd");
    double tot = 0;
    for (int i = 0; i < MODEL_FORWARD_PROFILE_STEPS; i++) tot += p->ms_last[i];
    for (int i = 0; i < MODEL_FORWARD_PROFILE_STEPS; i++) {
        double pct = tot > 0 ? 100.0 * p->ms_last[i] / tot : 0;
        fprintf(fp, "%2d   %-44s %12.4f %7.2f%%\n", i + 1, model_forward_profile_step_name(i), p->ms_last[i],
                pct);
    }
    fprintf(fp, "%-4s %-44s %12.4f\n", "", "total (timed steps)", tot);
}

void model_forward_profile_print_aggregate(const model_forward_profile_t* p, FILE* fp) {
    if (!p || !fp || p->runs == 0) {
        fprintf(fp, "No model_forward profile (no successful runs).\n");
        return;
    }
    fprintf(fp, "\n=== model_forward per-step (avg over %u run(s)) ===\n", p->runs);
    fprintf(fp, "%-4s %-44s %12s %8s\n", "#", "step", "avg ms", "% fwd");
    double avg[MODEL_FORWARD_PROFILE_STEPS];
    double avg_tot = 0;
    for (int i = 0; i < MODEL_FORWARD_PROFILE_STEPS; i++) {
        avg[i] = p->ms_sum[i] / (double)p->runs;
        avg_tot += avg[i];
    }
    int hot_i = 0;
    for (int i = 1; i < MODEL_FORWARD_PROFILE_STEPS; i++) {
        if (avg[i] > avg[hot_i]) hot_i = i;
    }
    for (int i = 0; i < MODEL_FORWARD_PROFILE_STEPS; i++) {
        double pct = avg_tot > 0 ? 100.0 * avg[i] / avg_tot : 0;
        fprintf(fp, "%2d   %-44s %12.4f %7.2f%%\n", i + 1, model_forward_profile_step_name(i), avg[i], pct);
    }
    fprintf(fp, "%-4s %-44s %12.4f\n", "", "total (sum of steps)", avg_tot);
    fprintf(fp, "\nHot step inside model_forward: #%d %s (~%.1f%% of forward).\n", hot_i + 1,
            model_forward_profile_step_name(hot_i), avg_tot > 0 ? 100.0 * avg[hot_i] / avg_tot : 0.0);
}

status_t model_forward(model_t* model, const tensor_t* input, tensor_t* output) {
    return model_forward_ex(model, input, output, NULL, NULL);
}

status_t model_forward_ex(model_t* model, const tensor_t* input, tensor_t* output, FILE* stage_dump,
                          model_forward_profile_t* profile) {
    if (!model || !input || !output) return ERROR_NULL_POINTER;

    double lap[MODEL_FORWARD_PROFILE_STEPS] = {0};
    timer_t tp;
    conv_params_t s2 = {2, 1, 1};
    status_t st;

#define MF_LAP(I, CODE) \
    do { \
        timer_start(&tp); \
        st = (CODE); \
        timer_stop(&tp); \
        lap[(I)] = timer_elapsed_ms(&tp); \
        if (st != SUCCESS) return st; \
    } while (0)

    st = dump_stage(stage_dump, "stage_00_input", input);
    if (st != SUCCESS) return st;

    MF_LAP(0, conv_blk(model, "model.0.conv.weight", "model.0.conv.bias", input, &model->buffers[0], s2, true));
    st = dump_stage(stage_dump, "stage_01_buf0", &model->buffers[0]);
    if (st != SUCCESS) return st;

    MF_LAP(1, conv_blk(model, "model.1.conv.weight", "model.1.conv.bias", &model->buffers[0], &model->buffers[1], s2,
                       true));
    st = dump_stage(stage_dump, "stage_02_buf1", &model->buffers[1]);
    if (st != SUCCESS) return st;

    MF_LAP(2, run_c3k2(model, 2, &model->buffers[1], &model->buffers[2], 1, true));
    st = dump_stage(stage_dump, "stage_03_buf2", &model->buffers[2]);
    if (st != SUCCESS) return st;

    MF_LAP(3, conv_blk(model, "model.3.conv.weight", "model.3.conv.bias", &model->buffers[2], &model->buffers[3], s2,
                       true));
    st = dump_stage(stage_dump, "stage_04_buf3", &model->buffers[3]);
    if (st != SUCCESS) return st;

    MF_LAP(4, run_c3k2(model, 4, &model->buffers[3], &model->buffers[4], 1, true));
    st = dump_stage(stage_dump, "stage_05_buf4", &model->buffers[4]);
    if (st != SUCCESS) return st;

    MF_LAP(5, conv_blk(model, "model.5.conv.weight", "model.5.conv.bias", &model->buffers[4], &model->buffers[5], s2,
                       true));
    st = dump_stage(stage_dump, "stage_06_buf5", &model->buffers[5]);
    if (st != SUCCESS) return st;

    MF_LAP(6, forward_c3k2_c3_inner(model, 6, &model->buffers[5], &model->buffers[6], true));
    st = dump_stage(stage_dump, "stage_07_buf6", &model->buffers[6]);
    if (st != SUCCESS) return st;

    MF_LAP(7, conv_blk(model, "model.7.conv.weight", "model.7.conv.bias", &model->buffers[6], &model->buffers[7], s2,
                       true));
    st = dump_stage(stage_dump, "stage_08_buf7", &model->buffers[7]);
    if (st != SUCCESS) return st;

    MF_LAP(8, forward_c3k2_c3_inner(model, 8, &model->buffers[7], &model->buffers[8], true));
    st = dump_stage(stage_dump, "stage_09_buf8", &model->buffers[8]);
    if (st != SUCCESS) return st;

    MF_LAP(9, run_sppf(model, 9, &model->buffers[8], &model->buffers[9], 5, 3, true));
    st = dump_stage(stage_dump, "stage_10_buf9", &model->buffers[9]);
    if (st != SUCCESS) return st;

    MF_LAP(10, run_c2psa(model, 10, &model->buffers[9], &model->buffers[10], 1, 0.5f, 0.5f));
    st = dump_stage(stage_dump, "stage_11_buf10", &model->buffers[10]);
    if (st != SUCCESS) return st;

    MF_LAP(11, upsample_nearest_forward(&model->buffers[11], &model->buffers[10], 2));
    st = dump_stage(stage_dump, "stage_12_buf11", &model->buffers[11]);
    if (st != SUCCESS) return st;

    MF_LAP(12, concat_forward(&model->buffers[12], &model->buffers[11], &model->buffers[6], 1));
    st = dump_stage(stage_dump, "stage_13_buf12", &model->buffers[12]);
    if (st != SUCCESS) return st;

    MF_LAP(13, forward_c3k2_c3_inner(model, 13, &model->buffers[12], &model->buffers[13], true));
    st = dump_stage(stage_dump, "stage_14_buf13", &model->buffers[13]);
    if (st != SUCCESS) return st;

    MF_LAP(14, upsample_nearest_forward(&model->buffers[14], &model->buffers[13], 2));
    st = dump_stage(stage_dump, "stage_15_buf14", &model->buffers[14]);
    if (st != SUCCESS) return st;

    MF_LAP(15, concat_forward(&model->buffers[15], &model->buffers[14], &model->buffers[4], 1));
    st = dump_stage(stage_dump, "stage_16_buf15", &model->buffers[15]);
    if (st != SUCCESS) return st;

    MF_LAP(16, forward_c3k2_c3_inner(model, 16, &model->buffers[15], &model->buffers[16], true));
    st = dump_stage(stage_dump, "stage_17_buf16", &model->buffers[16]);
    if (st != SUCCESS) return st;

    MF_LAP(17, conv_blk(model, "model.17.conv.weight", "model.17.conv.bias", &model->buffers[16], &model->buffers[17],
                       s2, true));
    st = dump_stage(stage_dump, "stage_18_buf17", &model->buffers[17]);
    if (st != SUCCESS) return st;

    MF_LAP(18, concat_forward(&model->buffers[18], &model->buffers[17], &model->buffers[13], 1));
    st = dump_stage(stage_dump, "stage_19_buf18", &model->buffers[18]);
    if (st != SUCCESS) return st;

    MF_LAP(19, forward_c3k2_c3_inner(model, 19, &model->buffers[18], &model->buffers[19], true));
    st = dump_stage(stage_dump, "stage_20_buf19", &model->buffers[19]);
    if (st != SUCCESS) return st;

    MF_LAP(20, conv_blk(model, "model.20.conv.weight", "model.20.conv.bias", &model->buffers[19], &model->buffers[20],
                       s2, true));
    st = dump_stage(stage_dump, "stage_21_buf20", &model->buffers[20]);
    if (st != SUCCESS) return st;

    MF_LAP(21, concat_forward(&model->buffers[21], &model->buffers[20], &model->buffers[10], 1));
    st = dump_stage(stage_dump, "stage_22_buf21", &model->buffers[21]);
    if (st != SUCCESS) return st;

    MF_LAP(22, forward_c3k2_attn_head(model, 22, &model->buffers[21], &model->buffers[22]));
    st = dump_stage(stage_dump, "stage_23_buf22", &model->buffers[22]);
    if (st != SUCCESS) return st;

    MF_LAP(23, detect_forward_one2one(model, YOLO26_DETECT_IDX, &model->buffers[16], &model->buffers[19],
                                      &model->buffers[22], &model->buffers[23]));
    st = dump_stage(stage_dump, "stage_24_buf23", &model->buffers[23]);
    if (st != SUCCESS) return st;
    st = dump_stage(stage_dump, "stage_25_detect", &model->buffers[23]);
    if (st != SUCCESS) return st;

    MF_LAP(24, tensor_copy(output, &model->buffers[23]));

#undef MF_LAP

    if (profile) {
        for (int i = 0; i < MODEL_FORWARD_PROFILE_STEPS; i++) {
            profile->ms_sum[i] += lap[i];
            profile->ms_last[i] = lap[i];
        }
        profile->runs++;
    }

    return st;
}
