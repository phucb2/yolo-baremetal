#ifdef __AVX2__
#include <immintrin.h>
#endif
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include "layers.h"

status_t silu_forward(tensor_t* tensor) {
    if (!tensor || !tensor->data) return ERROR_NULL_POINTER;
    int count = tensor->dims[0] * tensor->dims[1] * tensor->dims[2] * tensor->dims[3];
    float* data = tensor->data;
    for (int i = 0; i < count; i++) {
        float x = data[i];
        data[i] = x / (1.0f + expf(-x));
    }
    return SUCCESS;
}

status_t upsample_nearest_forward(tensor_t* output, const tensor_t* input, int scale) {
    if (!output || !input) return ERROR_NULL_POINTER;
    int n = input->dims[0], c = input->dims[1], h = input->dims[2], w = input->dims[3];
    for (int ni = 0; ni < n; ni++) {
        for (int ci = 0; ci < c; ci++) {
            for (int hi = 0; hi < h * scale; hi++) {
                for (int wi = 0; wi < w * scale; wi++) {
                    int src_h = hi / scale;
                    int src_w = wi / scale;
                    output->data[ni * output->stride[0] + ci * output->stride[1] + hi * output->stride[2] + wi] = 
                        input->data[ni * input->stride[0] + ci * input->stride[1] + src_h * input->stride[2] + src_w];
                }
            }
        }
    }
    return SUCCESS;
}

status_t conv2d_forward(tensor_t* output, const tensor_t* input, 
                       const tensor_t* weight, const tensor_t* bias, 
                       conv_params_t params) {
    if (!output || !input || !weight) return ERROR_NULL_POINTER;
    int out_c = weight->dims[0], in_c = weight->dims[1], kh = weight->dims[2], kw = weight->dims[3];
    int in_h = input->dims[2], in_w = input->dims[3], out_h = output->dims[2], out_w = output->dims[3];

    if (kh == 1 && kw == 1 && params.stride == 1 && params.padding == 0) {
        tensor_gemm(output->data, weight->data, input->data, out_c, in_h * in_w, in_c, 1.0f, 0.0f);
        if (bias) {
            for (int oc = 0; oc < out_c; oc++) {
                float b = bias->data[oc];
                float* out_ptr = output->data + oc * out_h * out_w;
                for (int i = 0; i < out_h * out_w; i++) out_ptr[i] += b;
            }
        }
        return SUCCESS;
    }

    for (int oc = 0; oc < out_c; oc++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                float sum = bias ? bias->data[oc] : 0.0f;
                for (int ic = 0; ic < in_c; ic++) {
                    for (int k_h = 0; k_h < kh; k_h++) {
                        for (int k_w = 0; k_w < kw; k_w++) {
                            int ih = oh * params.stride - params.padding + k_h;
                            int iw = ow * params.stride - params.padding + k_w;
                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                sum += input->data[ic * in_h * in_w + ih * in_w + iw] * 
                                       weight->data[oc * in_c * kh * kw + ic * kh * kw + k_h * kw + k_w];
                            }
                        }
                    }
                }
                output->data[oc * out_h * out_w + oh * out_w + ow] = sum;
            }
        }
    }
    return SUCCESS;
}

status_t conv_block_forward(tensor_t* output, const tensor_t* input, 
                           const tensor_t* weight, const tensor_t* bias,
                           conv_params_t params, bool act) {
    status_t status = conv2d_forward(output, input, weight, bias, params);
    if (status != SUCCESS) return status;
    if (act) status = silu_forward(output);
    return status;
}

status_t bottleneck_forward(tensor_t* output, const tensor_t* input,
                           const tensor_t* cv1_w, const tensor_t* cv1_b,
                           const tensor_t* cv2_w, const tensor_t* cv2_b,
                           bool shortcut, tensor_t* temp) {
    int kh1 = cv1_w->dims[2];
    int kh2 = cv2_w->dims[2];
    int pad1 = (kh1 - 1) / 2;
    int pad2 = (kh2 - 1) / 2;
    conv_params_t p1 = {1, pad1, 1}, p2 = {1, pad2, 1};
    conv_block_forward(temp, input, cv1_w, cv1_b, p1, true);
    conv_block_forward(output, temp, cv2_w, cv2_b, p2, true);
    if (shortcut) {
        size_t size = (size_t)output->dims[0] * output->dims[1] * output->dims[2] * output->dims[3];
        for (size_t i = 0; i < size; i++) output->data[i] += input->data[i];
    }
    return SUCCESS;
}

status_t c3k2_forward(tensor_t* output, const tensor_t* input, 
                     int n, bool shortcut,
                     const tensor_t* cv1_w, const tensor_t* cv1_b,
                     const tensor_t* cv2_w, const tensor_t* cv2_b,
                     const tensor_t* b_weights, tensor_t* buffers) {
    // buffers[0]: cv1_out
    // buffers[1...n]: bottleneck outputs
    // buffers[n+1]: temp for bottleneck
    // buffers[n+2]: combined buffer for final concat
    conv_params_t p1 = {1, 0, 1}, p2 = {1, 0, 1};
    conv_block_forward(&buffers[0], input, cv1_w, cv1_b, p1, true);
    
    int c_total = buffers[0].dims[1];
    int c_half = c_total / 2;
    int h = buffers[0].dims[2], w = buffers[0].dims[3];
    int plane_size = h * w;

    // a = buffers[0][0:c_half], b = buffers[0][c_half:]
    // Part 'a' is identity for concat later
    // Part 'b' goes through bottlenecks
    
    tensor_t b_in;
    b_in.dims[0] = 1; b_in.dims[1] = c_half; b_in.dims[2] = h; b_in.dims[3] = w;
    b_in.data = buffers[0].data + c_half * plane_size;
    b_in.is_owner = false;

    tensor_t* prev = &b_in;
    for (int i = 0; i < n; i++) {
        bottleneck_forward(&buffers[1+i], prev, 
                          &b_weights[i*4+0], &b_weights[i*4+1], 
                          &b_weights[i*4+2], &b_weights[i*4+3], 
                          shortcut, &buffers[n+1]);
        prev = &buffers[1+i];
    }
    
    /* Ultralytics C2f/C3k2: cat(chunk0, chunk1, m0(y1), m1(...), ...) = (2+n)*self.c channels */
    tensor_t* combined = &buffers[n+2];
    memcpy(combined->data, buffers[0].data, (size_t)c_total * plane_size * sizeof(float));
    for (int i = 0; i < n; i++) {
        memcpy(combined->data + ((size_t)c_total + (size_t)i * c_half) * plane_size,
               buffers[1+i].data, (size_t)c_half * plane_size * sizeof(float));
    }

    return conv_block_forward(output, combined, cv2_w, cv2_b, p2, true);
}

status_t c3_forward(tensor_t* output, const tensor_t* input, const tensor_t* cv1_w, const tensor_t* cv1_b,
                    const tensor_t* cv2_w, const tensor_t* cv2_b, const tensor_t* cv3_w, const tensor_t* cv3_b,
                    const tensor_t* b_weights, int n_bottles, bool shortcut, tensor_t* buffers) {
    if (!output || !input || !cv1_w || !cv1_b || !cv2_w || !cv2_b || !cv3_w || !cv3_b || !b_weights || !buffers)
        return ERROR_NULL_POINTER;
    conv_params_t p1 = {1, 0, 1};
    status_t st = conv_block_forward(&buffers[0], input, cv1_w, cv1_b, p1, true);
    if (st != SUCCESS) return st;
    st = conv_block_forward(&buffers[1], input, cv2_w, cv2_b, p1, true);
    if (st != SUCCESS) return st;

    int h = buffers[0].dims[2], w = buffers[0].dims[3];
    int plane = h * w;
    tensor_t* prev = &buffers[0];
    tensor_t* m_out = &buffers[3];
    for (int i = 0; i < n_bottles; i++) {
        st = bottleneck_forward(m_out, prev, &b_weights[i * 4 + 0], &b_weights[i * 4 + 1], &b_weights[i * 4 + 2],
                                &b_weights[i * 4 + 3], shortcut, &buffers[2]);
        if (st != SUCCESS) return st;
        prev = m_out;
    }

    int c_m = m_out->dims[1];
    int c_v2 = buffers[1].dims[1];
    tensor_t* concat = &buffers[4];
    memcpy(concat->data, m_out->data, (size_t)c_m * plane * sizeof(float));
    memcpy(concat->data + (size_t)c_m * plane, buffers[1].data, (size_t)c_v2 * plane * sizeof(float));

    return conv_block_forward(output, concat, cv3_w, cv3_b, p1, true);
}

status_t pool2d_max_forward(tensor_t* output, const tensor_t* input, int kernel_size, int stride) {
    if (!output || !input) return ERROR_NULL_POINTER;
    int n = input->dims[0], c = input->dims[1], in_h = input->dims[2], in_w = input->dims[3];
    int out_h = output->dims[2], out_w = output->dims[3];
    int pad = kernel_size / 2;
    for (int ni = 0; ni < n; ni++) {
        for (int ci = 0; ci < c; ci++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    float max_val = -FLT_MAX;
                    for (int kh = 0; kh < kernel_size; kh++) {
                        for (int kw = 0; kw < kernel_size; kw++) {
                            int ih = oh * stride - pad + kh;
                            int iw = ow * stride - pad + kw;
                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                float val = input->data[ni * input->stride[0] + ci * input->stride[1] + ih * in_w + iw];
                                if (val > max_val) max_val = val;
                            }
                        }
                    }
                    output->data[ni * output->stride[0] + ci * output->stride[1] + oh * out_w + ow] = max_val;
                }
            }
        }
    }
    return SUCCESS;
}

status_t concat_forward(tensor_t* output, const tensor_t* input1, const tensor_t* input2, int dim) {
    if (dim != 1) return ERROR_NOT_IMPLEMENTED;
    int n = input1->dims[0], h = input1->dims[2], w = input1->dims[3];
    int c1 = input1->dims[1], c2 = input2->dims[1];
    for (int ni = 0; ni < n; ni++) {
        memcpy(output->data + ni * output->stride[0], input1->data + ni * input1->stride[0], c1 * h * w * sizeof(float));
        memcpy(output->data + ni * output->stride[0] + c1 * h * w, input2->data + ni * input2->stride[0], c2 * h * w * sizeof(float));
    }
    return SUCCESS;
}

status_t sppf_forward(tensor_t* output, const tensor_t* input,
                     const tensor_t* cv1_w, const tensor_t* cv1_b,
                     const tensor_t* cv2_w, const tensor_t* cv2_b,
                     int kernel_size, int n_pool, bool shortcut,
                     tensor_t* buffers) {
    if (!output || !input || !cv1_w || !cv1_b || !cv2_w || !cv2_b || !buffers) return ERROR_NULL_POINTER;
    if (n_pool < 1) return ERROR_INVALID_DIMS;

    /* Ultralytics: cv1 uses act=False (Conv+BN fused, no SiLU). */
    conv_params_t p1 = {1, 0, 1};
    status_t st = conv_block_forward(&buffers[0], input, cv1_w, cv1_b, p1, false);
    if (st != SUCCESS) return st;

    for (int i = 0; i < n_pool; i++) {
        st = pool2d_max_forward(&buffers[i + 1], &buffers[i], kernel_size, 1);
        if (st != SUCCESS) return st;
    }

    int c = buffers[0].dims[1];
    int plane = buffers[0].dims[2] * buffers[0].dims[3];
    tensor_t* concat = &buffers[n_pool + 1];
    for (int j = 0; j <= n_pool; j++) {
        memcpy(concat->data + (size_t)j * c * plane, buffers[j].data,
               (size_t)c * plane * sizeof(float));
    }

    st = conv_block_forward(output, concat, cv2_w, cv2_b, p1, true);
    if (st != SUCCESS) return st;

    if (shortcut && input->dims[1] == output->dims[1] && input->dims[2] == output->dims[2] &&
        input->dims[3] == output->dims[3]) {
        size_t n_el = (size_t)output->dims[0] * output->dims[1] * output->dims[2] * output->dims[3];
        for (size_t i = 0; i < n_el; i++) output->data[i] += input->data[i];
    }
    return SUCCESS;
}

/* --- C2PSA: depthwise 3x3 (pe), Attention, PSABlock, C2PSA (Ultralytics) --- */

status_t dwconv3x3_same_forward(tensor_t* out, const tensor_t* in, const tensor_t* w, const tensor_t* bias) {
    if (!out || !in || !w) return ERROR_NULL_POINTER;
    int c = in->dims[1], h = in->dims[2], wi = in->dims[3];
    int pad = 1;
    for (int ci = 0; ci < c; ci++) {
        for (int oh = 0; oh < h; oh++) {
            for (int ow = 0; ow < wi; ow++) {
                float s = bias ? bias->data[ci] : 0.0f;
                for (int kh = 0; kh < 3; kh++) {
                    for (int kw = 0; kw < 3; kw++) {
                        int ih = oh - pad + kh, iw = ow - pad + kw;
                        if (ih >= 0 && ih < h && iw >= 0 && iw < wi) {
                            s += in->data[ci * h * wi + ih * wi + iw] * w->data[ci * 9 + kh * 3 + kw];
                        }
                    }
                }
                out->data[ci * h * wi + oh * wi + ow] = s;
            }
        }
    }
    return SUCCESS;
}

static void softmax_rows_nn(float* attn, int N) {
    for (int i = 0; i < N; i++) {
        float* row = attn + (size_t)i * N;
        float m = row[0];
        for (int j = 1; j < N; j++) {
            if (row[j] > m) m = row[j];
        }
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            row[j] = expf(row[j] - m);
            sum += row[j];
        }
        if (sum > 1e-12f) {
            for (int j = 0; j < N; j++) row[j] /= sum;
        }
    }
}

static status_t attention_forward(tensor_t* output, const tensor_t* input, int num_heads, float attn_ratio,
                                  const tensor_t* qkv_w, const tensor_t* qkv_b, const tensor_t* proj_w,
                                  const tensor_t* proj_b, const tensor_t* pe_w, const tensor_t* pe_b) {
    if (!output || !input || !qkv_w || !qkv_b || !proj_w || !proj_b || !pe_w || !pe_b) return ERROR_NULL_POINTER;
    int dim = input->dims[1];
    int H = input->dims[2], W = input->dims[3];
    int N = H * W;
    if (dim % num_heads != 0) return ERROR_INVALID_DIMS;
    int head_dim = dim / num_heads;
    int key_dim = (int)((float)head_dim * attn_ratio);
    if (key_dim < 1) return ERROR_INVALID_DIMS;
    float scale = powf((float)key_dim, -0.5f);
    int h_qkv = qkv_w->dims[0];
    int expect_h = dim + 2 * key_dim * num_heads;
    if (h_qkv != expect_h) return ERROR_INVALID_DIMS;

    tensor_t qkv;
    if (tensor_allocate(&qkv, 1, h_qkv, H, W) != SUCCESS) return ERROR_OUT_OF_MEMORY;
    conv_params_t p1 = {1, 0, 1};
    if (conv_block_forward(&qkv, input, qkv_w, qkv_b, p1, false) != SUCCESS) {
        tensor_free(&qkv);
        return ERROR_INVALID_DIMS;
    }

    float* attn_logits = (float*)malloc((size_t)N * (size_t)N * sizeof(float));
    float* attn_out = (float*)malloc((size_t)dim * (size_t)N * sizeof(float));
    tensor_t v_pe;
    if (!attn_logits || !attn_out || tensor_allocate(&v_pe, 1, dim, H, W) != SUCCESS) {
        free(attn_logits);
        free(attn_out);
        tensor_free(&qkv);
        if (v_pe.data) tensor_free(&v_pe);
        return ERROR_OUT_OF_MEMORY;
    }

    int block = 2 * key_dim + head_dim;
    for (int hi = 0; hi < num_heads; hi++) {
        int base = hi * block;
        const float* q_plane = qkv.data + (size_t)base * N;
        const float* k_plane = qkv.data + (size_t)(base + key_dim) * N;
        const float* v_plane = qkv.data + (size_t)(base + 2 * key_dim) * N;

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                float s = 0.0f;
                for (int d = 0; d < key_dim; d++) {
                    s += q_plane[d * N + i] * k_plane[d * N + j];
                }
                attn_logits[(size_t)i * N + j] = s * scale;
            }
        }
        softmax_rows_nn(attn_logits, N);

        for (int i = 0; i < head_dim; i++) {
            for (int j = 0; j < N; j++) {
                float s = 0.0f;
                for (int k = 0; k < N; k++) {
                    s += v_plane[i * N + k] * attn_logits[(size_t)j * N + k];
                }
                attn_out[(size_t)(hi * head_dim + i) * N + j] = s;
            }
        }

        for (int i = 0; i < head_dim; i++) {
            memcpy(v_pe.data + (size_t)(hi * head_dim + i) * N, v_plane + (size_t)i * N, (size_t)N * sizeof(float));
        }
    }

    tensor_t attn_tensor;
    tensor_t pe_out;
    if (tensor_allocate(&attn_tensor, 1, dim, H, W) != SUCCESS ||
        tensor_allocate(&pe_out, 1, dim, H, W) != SUCCESS) {
        free(attn_out);
        tensor_free(&v_pe);
        tensor_free(&qkv);
        free(attn_logits);
        return ERROR_OUT_OF_MEMORY;
    }
    memcpy(attn_tensor.data, attn_out, (size_t)dim * (size_t)N * sizeof(float));
    free(attn_out);

    dwconv3x3_same_forward(&pe_out, &v_pe, pe_w, pe_b);

    for (size_t i = 0; i < (size_t)dim * (size_t)N; i++) {
        attn_tensor.data[i] += pe_out.data[i];
    }

    status_t st = conv_block_forward(output, &attn_tensor, proj_w, proj_b, p1, false);

    tensor_free(&attn_tensor);
    tensor_free(&pe_out);
    tensor_free(&v_pe);
    tensor_free(&qkv);
    free(attn_logits);
    return st;
}

status_t psablock_forward(tensor_t* output, const tensor_t* input, bool shortcut,
                          const tensor_t* qkv_w, const tensor_t* qkv_b, const tensor_t* proj_w,
                          const tensor_t* proj_b, const tensor_t* pe_w, const tensor_t* pe_b,
                          const tensor_t* ffn0_w, const tensor_t* ffn0_b, const tensor_t* ffn1_w,
                          const tensor_t* ffn1_b, int num_heads, float attn_ratio) {
    tensor_t attn_out;
    if (tensor_allocate(&attn_out, 1, input->dims[1], input->dims[2], input->dims[3]) != SUCCESS)
        return ERROR_OUT_OF_MEMORY;
    status_t st = attention_forward(&attn_out, input, num_heads, attn_ratio, qkv_w, qkv_b, proj_w, proj_b, pe_w,
                                    pe_b);
    if (st != SUCCESS) {
        tensor_free(&attn_out);
        return st;
    }

    if (shortcut) {
        size_t n_el = (size_t)attn_out.dims[0] * attn_out.dims[1] * attn_out.dims[2] * attn_out.dims[3];
        for (size_t i = 0; i < n_el; i++) attn_out.data[i] += input->data[i];
    }

    tensor_t ffn_mid;
    int h = input->dims[2], w = input->dims[3];
    if (tensor_allocate(&ffn_mid, 1, ffn0_w->dims[0], h, w) != SUCCESS) {
        tensor_free(&attn_out);
        return ERROR_OUT_OF_MEMORY;
    }
    conv_params_t p1 = {1, 0, 1};
    st = conv_block_forward(&ffn_mid, &attn_out, ffn0_w, ffn0_b, p1, true);
    if (st != SUCCESS) {
        tensor_free(&ffn_mid);
        tensor_free(&attn_out);
        return st;
    }
    st = conv_block_forward(output, &ffn_mid, ffn1_w, ffn1_b, p1, false);
    tensor_free(&ffn_mid);
    if (st != SUCCESS) {
        tensor_free(&attn_out);
        return st;
    }
    /* Second residual: x = x + ffn(x) where x is attn_out after first residual (not original input). */
    if (shortcut) {
        size_t n_el = (size_t)output->dims[0] * output->dims[1] * output->dims[2] * output->dims[3];
        for (size_t i = 0; i < n_el; i++) output->data[i] += attn_out.data[i];
    }
    tensor_free(&attn_out);
    return SUCCESS;
}

status_t c2psa_forward(tensor_t* output, const tensor_t* input, int n_blocks, float e, float attn_ratio,
                       const tensor_t* cv1_w, const tensor_t* cv1_b, const tensor_t* cv2_w, const tensor_t* cv2_b,
                       const tensor_t* psa_weights, tensor_t* buffers) {
    if (!output || !input || !cv1_w || !cv1_b || !cv2_w || !cv2_b || !psa_weights || !buffers) return ERROR_NULL_POINTER;
    int c1 = input->dims[1];
    int c_hidden = (int)((float)c1 * e);
    if (c_hidden < 1 || n_blocks < 1) return ERROR_INVALID_DIMS;

    conv_params_t p1 = {1, 0, 1};
    status_t st = conv_block_forward(&buffers[0], input, cv1_w, cv1_b, p1, true);
    if (st != SUCCESS) return st;

    int h = buffers[0].dims[2], w = buffers[0].dims[3];
    int plane = h * w;
    int num_heads = c_hidden / 64;
    if (num_heads < 1) num_heads = 1; /* Ultralytics uses c//64; min 1 for small c */

    tensor_t b_view;
    b_view.dims[0] = 1;
    b_view.dims[1] = c_hidden;
    b_view.dims[2] = h;
    b_view.dims[3] = w;
    b_view.data = buffers[0].data + (size_t)c_hidden * plane;
    b_view.stride[0] = c_hidden * plane;
    b_view.stride[1] = plane;
    b_view.stride[2] = w;
    b_view.stride[3] = 1;
    b_view.is_owner = false;

    for (int bi = 0; bi < n_blocks; bi++) {
        const tensor_t* w = psa_weights + bi * 10;
        st = psablock_forward(&buffers[1], &b_view, true, w + 0, w + 1, w + 2, w + 3, w + 4, w + 5, w + 6, w + 7,
                              w + 8, w + 9, num_heads, attn_ratio);
        if (st != SUCCESS) return st;
        memcpy(b_view.data, buffers[1].data, (size_t)c_hidden * plane * sizeof(float));
    }

    tensor_t* concat = &buffers[2];
    memcpy(concat->data, buffers[0].data, (size_t)(2 * c_hidden) * plane * sizeof(float));

    return conv_block_forward(output, concat, cv2_w, cv2_b, p1, true);
}
