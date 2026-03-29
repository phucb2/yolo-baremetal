#ifndef LAYERS_H
#define LAYERS_H

#include "tensor.h"

typedef struct {
    int stride;
    int padding;
    int groups;
} conv_params_t;

status_t conv2d_forward(tensor_t* output, const tensor_t* input, 
                       const tensor_t* weight, const tensor_t* bias, 
                       conv_params_t params);

status_t silu_forward(tensor_t* tensor);

status_t pool2d_max_forward(tensor_t* output, const tensor_t* input, int kernel_size, int stride);

status_t upsample_nearest_forward(tensor_t* output, const tensor_t* input, int scale);

// C3K2 is a composite layer, we will implement it in model.c or as a helper here
status_t concat_forward(tensor_t* output, const tensor_t* input1, const tensor_t* input2, int dim);

// Standard YOLO conv block: Conv + Act (BN is folded)
status_t conv_block_forward(tensor_t* output, const tensor_t* input, 
                           const tensor_t* weight, const tensor_t* bias,
                           conv_params_t params, bool act);

/* Ultralytics Bottleneck: cv1/cv2 with kernel sizes from weights; padding = (k-1)/2 per axis.
 * temp: activations after cv1, allocate [N, cv1_w->dims[0], H, W] (hidden c_ = int(c2*e) in PyTorch). */
status_t bottleneck_forward(tensor_t* output, const tensor_t* input,
                           const tensor_t* cv1_w, const tensor_t* cv1_b,
                           const tensor_t* cv2_w, const tensor_t* cv2_b,
                           bool shortcut, tensor_t* temp);

/* SPPF: cv1 1x1 linear (no SiLU), n_pool chained maxpools (stride 1, pad k/2), concat, cv2 1x1+SiLU.
 * Optional residual: output += input when shortcut and shapes match (Ultralytics).
 * buffers[0]: cv1; buffers[1..n_pool]: pool chain; buffers[n_pool+1]: concat [(n_pool+1)*c_,H,W]. */
status_t sppf_forward(tensor_t* output, const tensor_t* input,
                     const tensor_t* cv1_w, const tensor_t* cv1_b,
                     const tensor_t* cv2_w, const tensor_t* cv2_b,
                     int kernel_size, int n_pool, bool shortcut,
                     tensor_t* buffers);
/* C3k2 (Ultralytics C2f path): cat(cv1 chunk0, cv1 chunk1, m0(...), ...); cv2 projects to c2.
 * buffers[0]: cv1 out [2*c_half,H,W]; buffers[1..n]: each [c_half,H,W]; buffers[n+1]: bottleneck temp
 *   [cv1_out_ch_of_bottleneck0,H,W]; buffers[n+2]: concat [c_total + n*c_half,H,W] with c_total=2*c_half.
 * b_weights: n blocks of (cv1_w, cv1_b, cv2_w, cv2_b) per bottleneck. */
status_t c3k2_forward(tensor_t* output, const tensor_t* input, 
                     int n, bool shortcut,
                     const tensor_t* cv1_w, const tensor_t* cv1_b,
                     const tensor_t* cv2_w, const tensor_t* cv2_b,
                     const tensor_t* b_weights,
                     tensor_t* buffers);

/* C2PSA: e=expansion for hidden c; self.c=int(c1*e); attn_ratio for Attention (default 0.5).
 * psa_weights: n_blocks * 10 tensors per block: qkv_w,b, proj_w,b, pe_w,b, ffn0_w,b, ffn1_w,b.
 * buffers[0]: cv1 out [2*c_h, H,W]; buffers[1]: PSABlock out [c_h,H,W]; buffers[2]: concat [2*c_h,H,W]. */
status_t c2psa_forward(tensor_t* output, const tensor_t* input, int n_blocks, float e, float attn_ratio,
                       const tensor_t* cv1_w, const tensor_t* cv1_b, const tensor_t* cv2_w, const tensor_t* cv2_b,
                       const tensor_t* psa_weights, tensor_t* buffers);

/* Single PSABlock (Ultralytics): Attention + FFN; same 10 tensors per block as C2PSA (qkv…ffn1). */
status_t psablock_forward(tensor_t* output, const tensor_t* input, bool shortcut,
                          const tensor_t* qkv_w, const tensor_t* qkv_b, const tensor_t* proj_w,
                          const tensor_t* proj_b, const tensor_t* pe_w, const tensor_t* pe_b,
                          const tensor_t* ffn0_w, const tensor_t* ffn0_b, const tensor_t* ffn1_w,
                          const tensor_t* ffn1_b, int num_heads, float attn_ratio);

/* Depthwise 3x3 same padding (Ultralytics DWConv k=3); weight [C,1,3,3], bias [C]. */
status_t dwconv3x3_same_forward(tensor_t* out, const tensor_t* in, const tensor_t* w, const tensor_t* bias);

/* C3 (Ultralytics): cv3(cat(m(cv1(x)), cv2(x))). b_weights: n Bottlenecks × 4 tensors each.
 * buffers[0] cv1 out; [1] cv2 out; [2] bottleneck temp; [3] m chain; [4] concat 2*c_ */
status_t c3_forward(tensor_t* output, const tensor_t* input, const tensor_t* cv1_w, const tensor_t* cv1_b,
                    const tensor_t* cv2_w, const tensor_t* cv2_b, const tensor_t* cv3_w, const tensor_t* cv3_b,
                    const tensor_t* b_weights, int n_bottles, bool shortcut, tensor_t* buffers);

#endif
