#include <stdio.h>
#include <stdlib.h>

#ifdef USE_CUDNN
#include <cuda_runtime.h>
#include <cudnn.h>

typedef struct Conv2DShape {
    int batch;   /* B */
    int channels;/* C */
    int width;   /* W */
    int height;  /* H */
} Conv2DShape;

static int conv2d_cudnn_bchw(const float *host_input,
                             const Conv2DShape *in_shape,
                             const float *host_kernel,
                             int out_channels,
                             int kernel_w,
                             int kernel_h,
                             int stride_w,
                             int stride_h,
                             int pad_w,
                             int pad_h,
                             float **host_output,
                             int *out_w,
                             int *out_h) {
    cudnnStatus_t cst;
    cudaError_t cet;
    cudnnHandle_t cudnn = NULL;
    cudnnTensorDescriptor_t in_desc = NULL;
    cudnnFilterDescriptor_t filt_desc = NULL;
    cudnnConvolutionDescriptor_t conv_desc = NULL;
    cudnnTensorDescriptor_t out_desc = NULL;

    float *d_input = NULL;
    float *d_kernel = NULL;
    float *d_output = NULL;
    void *workspace = NULL;

    size_t in_count = 0;
    size_t kernel_count = 0;
    size_t out_count = 0;
    size_t workspace_bytes = 0;
    int out_n = 0, out_c = 0;

    if (!host_input || !host_kernel || !in_shape || !host_output || !out_w || !out_h) {
        return -1;
    }

    *host_output = NULL;

    in_count = (size_t)in_shape->batch * (size_t)in_shape->channels *
               (size_t)in_shape->height * (size_t)in_shape->width;
    kernel_count = (size_t)out_channels * (size_t)in_shape->channels *
                   (size_t)kernel_h * (size_t)kernel_w;

    cst = cudnnCreate(&cudnn);
    if (cst != CUDNN_STATUS_SUCCESS) goto fail;

    cst = cudnnCreateTensorDescriptor(&in_desc);
    if (cst != CUDNN_STATUS_SUCCESS) goto fail;
    cst = cudnnSetTensor4dDescriptor(in_desc,
                                     CUDNN_TENSOR_NCHW,
                                     CUDNN_DATA_FLOAT,
                                     in_shape->batch,
                                     in_shape->channels,
                                     in_shape->height,
                                     in_shape->width);
    if (cst != CUDNN_STATUS_SUCCESS) goto fail;

    cst = cudnnCreateFilterDescriptor(&filt_desc);
    if (cst != CUDNN_STATUS_SUCCESS) goto fail;
    cst = cudnnSetFilter4dDescriptor(filt_desc,
                                     CUDNN_DATA_FLOAT,
                                     CUDNN_TENSOR_NCHW,
                                     out_channels,
                                     in_shape->channels,
                                     kernel_h,
                                     kernel_w);
    if (cst != CUDNN_STATUS_SUCCESS) goto fail;

    cst = cudnnCreateConvolutionDescriptor(&conv_desc);
    if (cst != CUDNN_STATUS_SUCCESS) goto fail;
    cst = cudnnSetConvolution2dDescriptor(conv_desc,
                                          pad_h,
                                          pad_w,
                                          stride_h,
                                          stride_w,
                                          1,
                                          1,
                                          CUDNN_CROSS_CORRELATION,
                                          CUDNN_DATA_FLOAT);
    if (cst != CUDNN_STATUS_SUCCESS) goto fail;

    cst = cudnnGetConvolution2dForwardOutputDim(conv_desc,
                                                in_desc,
                                                filt_desc,
                                                &out_n,
                                                &out_c,
                                                out_h,
                                                out_w);
    if (cst != CUDNN_STATUS_SUCCESS) goto fail;

    cst = cudnnCreateTensorDescriptor(&out_desc);
    if (cst != CUDNN_STATUS_SUCCESS) goto fail;
    cst = cudnnSetTensor4dDescriptor(out_desc,
                                     CUDNN_TENSOR_NCHW,
                                     CUDNN_DATA_FLOAT,
                                     out_n,
                                     out_c,
                                     *out_h,
                                     *out_w);
    if (cst != CUDNN_STATUS_SUCCESS) goto fail;

    out_count = (size_t)out_n * (size_t)out_c * (size_t)(*out_h) * (size_t)(*out_w);

    cet = cudaMalloc((void **)&d_input, in_count * sizeof(float));
    if (cet != cudaSuccess) goto fail;
    cet = cudaMalloc((void **)&d_kernel, kernel_count * sizeof(float));
    if (cet != cudaSuccess) goto fail;
    cet = cudaMalloc((void **)&d_output, out_count * sizeof(float));
    if (cet != cudaSuccess) goto fail;

    cet = cudaMemcpy(d_input, host_input, in_count * sizeof(float), cudaMemcpyHostToDevice);
    if (cet != cudaSuccess) goto fail;
    cet = cudaMemcpy(d_kernel, host_kernel, kernel_count * sizeof(float), cudaMemcpyHostToDevice);
    if (cet != cudaSuccess) goto fail;

    cst = cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                  in_desc,
                                                  filt_desc,
                                                  conv_desc,
                                                  out_desc,
                                                  CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
                                                  &workspace_bytes);
    if (cst != CUDNN_STATUS_SUCCESS) goto fail;

    if (workspace_bytes > 0) {
        cet = cudaMalloc(&workspace, workspace_bytes);
        if (cet != cudaSuccess) goto fail;
    }

    {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cst = cudnnConvolutionForward(cudnn,
                                      &alpha,
                                      in_desc,
                                      d_input,
                                      filt_desc,
                                      d_kernel,
                                      conv_desc,
                                      CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
                                      workspace,
                                      workspace_bytes,
                                      &beta,
                                      out_desc,
                                      d_output);
        if (cst != CUDNN_STATUS_SUCCESS) goto fail;
    }

    *host_output = (float *)malloc(out_count * sizeof(float));
    if (!*host_output) goto fail;

    cet = cudaMemcpy(*host_output, d_output, out_count * sizeof(float), cudaMemcpyDeviceToHost);
    if (cet != cudaSuccess) goto fail;

    cudaFree(workspace);
    cudaFree(d_output);
    cudaFree(d_kernel);
    cudaFree(d_input);
    cudnnDestroyTensorDescriptor(out_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroyFilterDescriptor(filt_desc);
    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroy(cudnn);
    return 0;

fail:
    if (*host_output) {
        free(*host_output);
        *host_output = NULL;
    }
    cudaFree(workspace);
    cudaFree(d_output);
    cudaFree(d_kernel);
    cudaFree(d_input);
    if (out_desc) cudnnDestroyTensorDescriptor(out_desc);
    if (conv_desc) cudnnDestroyConvolutionDescriptor(conv_desc);
    if (filt_desc) cudnnDestroyFilterDescriptor(filt_desc);
    if (in_desc) cudnnDestroyTensorDescriptor(in_desc);
    if (cudnn) cudnnDestroy(cudnn);
    return -2;
}
#endif

int main(void) {
#ifdef USE_CUDNN
    const Conv2DShape in_shape = {1, 1, 4, 4}; /* B, C, W, H */
    const float input[16] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    const float kernel[9] = {
        1, 0, -1,
        1, 0, -1,
        1, 0, -1
    };

    float *output = NULL;
    int out_w = 0;
    int out_h = 0;
    int rc = conv2d_cudnn_bchw(input,
                               &in_shape,
                               kernel,
                               1,
                               3,
                               3,
                               1,
                               1,
                               0,
                               0,
                               &output,
                               &out_w,
                               &out_h);
    if (rc != 0) {
        printf("conv2d_cudnn_bchw failed: %d\n", rc);
        return 1;
    }

    printf("Output shape: (B=1, C=1, W=%d, H=%d)\n", out_w, out_h);
    for (int y = 0; y < out_h; ++y) {
        for (int x = 0; x < out_w; ++x) {
            printf("%7.2f ", output[y * out_w + x]);
        }
        printf("\n");
    }
    free(output);
    return 0;
#else
    printf("Build with USE_CUDNN to run cuDNN conv2d playground.\n");
    return 0;
#endif
}