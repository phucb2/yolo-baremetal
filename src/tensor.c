#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "platform.h"
#include "tensor.h"
#include "gemm_backend.h"

#ifdef _MSC_VER
#include <malloc.h>
#endif

void* malloc_aligned(size_t size, size_t alignment) {
    void* ptr = NULL;
#ifdef _MSC_VER
    ptr = _aligned_malloc(size, alignment);
#else
    if (posix_memalign(&ptr, alignment, size) != 0) return NULL;
#endif
    return ptr;
}

void free_aligned(void* ptr) {
#ifdef _MSC_VER
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

static void tensor_clear_fields(tensor_t* tensor) {
    tensor->data = NULL;
    tensor->qdata = NULL;
    tensor->scales = NULL;
    tensor->num_scales = 0;
    tensor->dtype = TENSOR_DTYPE_FP32;
    tensor->is_owner = false;
}

status_t tensor_allocate(tensor_t* tensor, int n, int c, int h, int w) {
    if (!tensor) return ERROR_NULL_POINTER;
    tensor->dims[0] = n;
    tensor->dims[1] = c;
    tensor->dims[2] = h;
    tensor->dims[3] = w;
    tensor->stride[3] = 1;
    tensor->stride[2] = w;
    tensor->stride[1] = h * w;
    tensor->stride[0] = c * h * w;
    tensor->dtype = TENSOR_DTYPE_FP32;
    tensor->qdata = NULL;
    tensor->scales = NULL;
    tensor->num_scales = 0;
    size_t size = (size_t)n * c * h * w * sizeof(float);
    tensor->data = (float*)malloc_aligned(size, 64);
    if (!tensor->data) return ERROR_OUT_OF_MEMORY;
    tensor->is_owner = true;
    return SUCCESS;
}

status_t tensor_free(tensor_t* tensor) {
    if (!tensor) return ERROR_NULL_POINTER;
    if (tensor->is_owner) {
        if (tensor->data) {
            free_aligned(tensor->data);
            tensor->data = NULL;
        }
        if (tensor->qdata) {
            free_aligned(tensor->qdata);
            tensor->qdata = NULL;
        }
        if (tensor->scales) {
            free(tensor->scales);
            tensor->scales = NULL;
        }
    }
    tensor_clear_fields(tensor);
    return SUCCESS;
}

status_t tensor_fill(tensor_t* tensor, float value) {
    if (!tensor || !tensor->data) return ERROR_NULL_POINTER;
    size_t count = (size_t)tensor->dims[0] * tensor->dims[1] * tensor->dims[2] * tensor->dims[3];
    for (size_t i = 0; i < count; i++) {
        tensor->data[i] = value;
    }
    return SUCCESS;
}

status_t tensor_copy(tensor_t* dest, const tensor_t* src) {
    if (!dest || !src || !dest->data || !src->data) return ERROR_NULL_POINTER;
    for (int i = 0; i < 4; i++) {
        if (dest->dims[i] != src->dims[i]) return ERROR_INVALID_DIMS;
    }
    size_t count = (size_t)src->dims[0] * src->dims[1] * src->dims[2] * src->dims[3];
    memcpy(dest->data, src->data, count * sizeof(float));
    return SUCCESS;
}


void tensor_quantize_symmetric(const float* src, int8_t* dst, int count, float scale) {
    if (scale <= 0.0f) scale = 1e-8f;
    float inv = 1.0f / scale;
    int i = 0;
#if defined(__AVX2__)
    __m256 invv = _mm256_set1_ps(inv);
    __m256 minv = _mm256_set1_ps(-128.0f);
    __m256 maxv = _mm256_set1_ps(127.0f);
    for (; i <= count - 8; i += 8) {
        __m256 v = _mm256_mul_ps(_mm256_loadu_ps(src + i), invv);
        v = _mm256_round_ps(v, _MM_FROUND_CUR_DIRECTION);
        v = _mm256_min_ps(_mm256_max_ps(v, minv), maxv);
        __m256i i32 = _mm256_cvtps_epi32(v);
        __m128i lo = _mm256_castsi256_si128(i32);
        __m128i hi = _mm256_extracti128_si256(i32, 1);
        __m128i packed16 = _mm_packs_epi32(lo, hi);
        __m128i packed8 = _mm_packs_epi16(packed16, packed16);
        _mm_storel_epi64((__m128i*)(dst + i), packed8);
    }
#elif YOLO_ARCH_ARM64
    float32x4_t inv4 = vdupq_n_f32(inv);
    for (; i <= count - 4; i += 4) {
        float32x4_t v = vmulq_f32(vld1q_f32(src + i), inv4);
        int32x4_t q = vcvtnq_s32_f32(vminq_f32(vmaxq_f32(v, vdupq_n_f32(-128.0f)), vdupq_n_f32(127.0f)));
        int16x4_t p = vqmovn_s32(q);
        int8x8_t out = vqmovn_s16(vcombine_s16(p, p));
        vst1_lane_s32((int32_t*)(dst + i), vreinterpret_s32_s8(vget_low_s8(out)), 0);
    }
#endif
    for (; i < count; i++) {
        float q = roundf(src[i] * inv);
        if (q > 127.0f) q = 127.0f;
        if (q < -128.0f) q = -128.0f;
        dst[i] = (int8_t)q;
    }
}

status_t tensor_gemm(float* restrict C, const float* restrict A, const float* restrict B, int M, int N, int K,
                     float alpha, float beta) {
    return gemm_fp32(C, A, B, M, N, K, alpha, beta);
}

status_t tensor_gemm_int8(float* restrict C, const int8_t* restrict A, const int8_t* restrict B,
                          const float* weight_scales, int M, int N, int K, float input_scale) {
    return gemm_int8(C, A, B, weight_scales, M, N, K, input_scale);
}

status_t tensor_gemm_weight_int8(float* restrict C, const int8_t* restrict W, const float* restrict X,
                                 const float* weight_scales, int M, int N, int K) {
    return gemm_weight_int8(C, W, X, weight_scales, M, N, K);
}

const char* tensor_gemm_int8_backend(void) {
    return gemm_backend_int8_name();
}
