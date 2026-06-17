#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tensor.h"

#ifdef USE_OPENBLAS
#if defined(__has_include)
#if __has_include(<cblas.h>)
#include <cblas.h>
#elif __has_include(<openblas/cblas.h>)
#include <openblas/cblas.h>
#else
#include <cblas.h>
#endif
#else
#include <cblas.h>
#endif
#endif

#ifdef __x86_64__
#include <immintrin.h>
#endif
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

void* malloc_aligned(size_t size, size_t alignment) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) return NULL;
    return ptr;
}

void free_aligned(void* ptr) {
    free(ptr);
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

static float tensor_max_abs_f32(const float* src, int count) {
    float m = 0.0f;
    int i = 0;
#if defined(__AVX2__)
    __m256 vmax = _mm256_setzero_ps();
    for (; i <= count - 8; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        vmax = _mm256_max_ps(vmax, _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v));
    }
    float buf[8];
    _mm256_storeu_ps(buf, vmax);
    for (int t = 0; t < 8; t++) {
        if (buf[t] > m) m = buf[t];
    }
#elif defined(__aarch64__)
    float32x4_t vmax = vdupq_n_f32(0.0f);
    for (; i <= count - 4; i += 4) {
        float32x4_t v = vld1q_f32(src + i);
        vmax = vmaxq_f32(vmax, vabsq_f32(v));
    }
    m = vmaxvq_f32(vmax);
#endif
    for (; i < count; i++) {
        float v = fabsf(src[i]);
        if (v > m) m = v;
    }
    return m;
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
#elif defined(__aarch64__)
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
    if (!C || !A || !B) return ERROR_NULL_POINTER;

#ifdef USE_OPENBLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N, beta, C, N);
    return SUCCESS;
#else
    for (int i = 0; i < M; i++) {
        float* c_row = &C[i * N];
        if (beta == 0.0f) {
            memset(c_row, 0, N * sizeof(float));
        } else if (beta != 1.0f) {
            for (int j = 0; j < N; j++) c_row[j] *= beta;
        }

        for (int k = 0; k < K; k++) {
            float a_val = A[i * K + k] * alpha;
            if (a_val == 0.0f) continue;

            const float* b_row = &B[k * N];
            int j = 0;

#ifdef __AVX2__
            __m256 a_vec = _mm256_set1_ps(a_val);
            for (; j <= N - 16; j += 16) {
                __m256 b_vec1 = _mm256_loadu_ps(&b_row[j]);
                __m256 b_vec2 = _mm256_loadu_ps(&b_row[j + 8]);
                __m256 c_vec1 = _mm256_loadu_ps(&c_row[j]);
                __m256 c_vec2 = _mm256_loadu_ps(&c_row[j + 8]);
                _mm256_storeu_ps(&c_row[j], _mm256_fmadd_ps(a_vec, b_vec1, c_vec1));
                _mm256_storeu_ps(&c_row[j + 8], _mm256_fmadd_ps(a_vec, b_vec2, c_vec2));
            }
            if (j <= N - 8) {
                __m256 b_vec = _mm256_loadu_ps(&b_row[j]);
                __m256 c_vec = _mm256_loadu_ps(&c_row[j]);
                _mm256_storeu_ps(&c_row[j], _mm256_fmadd_ps(a_vec, b_vec, c_vec));
                j += 8;
            }
#elif defined(__AVX__)
            __m128 a_vec = _mm_set1_ps(a_val);
            for (; j <= N - 4; j += 4) {
                __m128 b_vec = _mm_loadu_ps(&b_row[j]);
                __m128 c_vec = _mm_loadu_ps(&c_row[j]);
                _mm_storeu_ps(&c_row[j], _mm_add_ps(c_vec, _mm_mul_ps(a_vec, b_vec)));
            }
#elif defined(__aarch64__)
            float32x4_t a4 = vdupq_n_f32(a_val);
            for (; j <= N - 16; j += 16) {
                float32x4_t b0 = vld1q_f32(&b_row[j]);
                float32x4_t b1 = vld1q_f32(&b_row[j + 4]);
                float32x4_t b2 = vld1q_f32(&b_row[j + 8]);
                float32x4_t b3 = vld1q_f32(&b_row[j + 12]);
                float32x4_t c0 = vld1q_f32(&c_row[j]);
                float32x4_t c1 = vld1q_f32(&c_row[j + 4]);
                float32x4_t c2 = vld1q_f32(&c_row[j + 8]);
                float32x4_t c3 = vld1q_f32(&c_row[j + 12]);
                vst1q_f32(&c_row[j], vfmaq_f32(c0, a4, b0));
                vst1q_f32(&c_row[j + 4], vfmaq_f32(c1, a4, b1));
                vst1q_f32(&c_row[j + 8], vfmaq_f32(c2, a4, b2));
                vst1q_f32(&c_row[j + 12], vfmaq_f32(c3, a4, b3));
            }
            for (; j <= N - 4; j += 4) {
                float32x4_t bv = vld1q_f32(&b_row[j]);
                float32x4_t cv = vld1q_f32(&c_row[j]);
                vst1q_f32(&c_row[j], vfmaq_f32(cv, a4, bv));
            }
#endif
            for (; j < N; j++) {
                c_row[j] += a_val * b_row[j];
            }
        }
    }
    return SUCCESS;
#endif
}

/* Scalar reference: one output column. */
static int32_t gemm_int8_dot_col(const int8_t* a_row, const int8_t* B, int K, int N, int j) {
    int32_t acc = 0;
    for (int k = 0; k < K; k++) {
        acc += (int32_t)a_row[k] * (int32_t)B[k * N + j];
    }
    return acc;
}

#if defined(__x86_64__) && defined(__AVX2__)

#include <cpuid.h>

/* 0 = AVX2 only, 1 = AVX-VNNI (256-bit dpbusds), 2 = AVX512-VNNI */
static int gemm_int8_vnni_level(void) {
    static int level = -1;
    if (level >= 0) return level;
    if (__get_cpuid_max(0, NULL) < 7) {
        level = 0;
        return level;
    }
    unsigned eax, ebx, ecx, edx;
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
#ifdef GEMM_PREFER_AVX512_VNNI
    if (ecx & (1u << 11)) {
        level = 2;
    } else if (ecx & (1u << 10)) {
        level = 1;
    } else {
        level = 0;
    }
#else
    /* Prefer 256-bit AVX-VNNI on client CPUs (AVX-512 often throttles clock). */
    if (ecx & (1u << 10)) {
        level = 1;
    } else if (ecx & (1u << 11)) {
        level = 2;
    } else {
        level = 0;
    }
#endif
    return level;
}

static inline __m256i gemm_int8_pack_b_k4x8(const int8_t* B, int k, int N, int j0) {
    const int8_t* b0 = B + (k + 0) * N + j0;
    const int8_t* b1 = B + (k + 1) * N + j0;
    const int8_t* b2 = B + (k + 2) * N + j0;
    const int8_t* b3 = B + (k + 3) * N + j0;
    const __m128i r0 = _mm_loadl_epi64((const __m128i*)b0);
    const __m128i r1 = _mm_loadl_epi64((const __m128i*)b1);
    const __m128i r2 = _mm_loadl_epi64((const __m128i*)b2);
    const __m128i r3 = _mm_loadl_epi64((const __m128i*)b3);
    const __m128i t01l = _mm_unpacklo_epi8(r0, r1);
    const __m128i t23l = _mm_unpacklo_epi8(r2, r3);
    const __m128i t01h = _mm_unpackhi_epi8(r0, r1);
    const __m128i t23h = _mm_unpackhi_epi8(r2, r3);
    const __m128i lo = _mm_unpacklo_epi16(t01l, t23l);
    const __m128i hi = _mm_unpacklo_epi16(t01h, t23h);
    return _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
}

__attribute__((target("avx512f,avx512vnni")))
static inline __m512i gemm_int8_pack_b_k4x16(const int8_t* B, int k, int N, int j0) {
    const int8_t* b0 = B + (k + 0) * N + j0;
    const int8_t* b1 = B + (k + 1) * N + j0;
    const int8_t* b2 = B + (k + 2) * N + j0;
    const int8_t* b3 = B + (k + 3) * N + j0;
    const __m128i r0 = _mm_loadu_si128((const __m128i*)b0);
    const __m128i r1 = _mm_loadu_si128((const __m128i*)b1);
    const __m128i r2 = _mm_loadu_si128((const __m128i*)b2);
    const __m128i r3 = _mm_loadu_si128((const __m128i*)b3);
    const __m128i t01l = _mm_unpacklo_epi8(r0, r1);
    const __m128i t23l = _mm_unpacklo_epi8(r2, r3);
    const __m128i t01h = _mm_unpackhi_epi8(r0, r1);
    const __m128i t23h = _mm_unpackhi_epi8(r2, r3);
    const __m128i lo = _mm_unpacklo_epi16(t01l, t23l);
    const __m128i hi = _mm_unpacklo_epi16(t01h, t23h);
    const __m256i p0 = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);

    const __m128i r0h = _mm_loadu_si128((const __m128i*)(b0 + 8));
    const __m128i r1h = _mm_loadu_si128((const __m128i*)(b1 + 8));
    const __m128i r2h = _mm_loadu_si128((const __m128i*)(b2 + 8));
    const __m128i r3h = _mm_loadu_si128((const __m128i*)(b3 + 8));
    const __m128i u01l = _mm_unpacklo_epi8(r0h, r1h);
    const __m128i u23l = _mm_unpacklo_epi8(r2h, r3h);
    const __m128i u01h = _mm_unpackhi_epi8(r0h, r1h);
    const __m128i u23h = _mm_unpackhi_epi8(r2h, r3h);
    const __m128i ulo = _mm_unpacklo_epi16(u01l, u23l);
    const __m128i uhi = _mm_unpacklo_epi16(u01h, u23h);
    const __m256i p1 = _mm256_inserti128_si256(_mm256_castsi128_si256(ulo), uhi, 1);
    return _mm512_inserti64x4(_mm512_castsi256_si512(p0), p1, 1);
}

static void gemm_int8_row_block_avx2(float* c_row, const int8_t* a_row, const int8_t* B, int K, int N, int j0,
                                      int jb, float scale) {
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();

    for (int k = 0; k < K; k++) {
        const int32_t ak = (int32_t)a_row[k];
        if (ak == 0) continue;
        const __m256i akv = _mm256_set1_epi32(ak);

        if (jb > 0) {
            __m128i b8 = _mm_loadl_epi64((const __m128i*)(B + k * N + j0));
            acc0 = _mm256_add_epi32(acc0, _mm256_mullo_epi32(akv, _mm256_cvtepi8_epi32(b8)));
        }
        if (jb > 8) {
            __m128i b8 = _mm_loadl_epi64((const __m128i*)(B + k * N + j0 + 8));
            acc1 = _mm256_add_epi32(acc1, _mm256_mullo_epi32(akv, _mm256_cvtepi8_epi32(b8)));
        }
    }

    const __m256 sf = _mm256_set1_ps(scale);
    if (jb >= 8) {
        _mm256_storeu_ps(c_row + j0, _mm256_mul_ps(_mm256_cvtepi32_ps(acc0), sf));
    }
    if (jb > 8) {
        _mm256_storeu_ps(c_row + j0 + 8, _mm256_mul_ps(_mm256_cvtepi32_ps(acc1), sf));
    }
}

static void gemm_int8_row_finish_tail(float* c_row, int32_t* ibuf, const int8_t* a_row, const int8_t* B, int k,
                                      int K, int N, int j0, int cols, float scale) {
    for (; k < K; k++) {
        const int32_t ak = (int32_t)a_row[k];
        if (ak == 0) continue;
        for (int c = 0; c < cols; c++) {
            ibuf[c] += ak * (int32_t)B[k * N + j0 + c];
        }
    }
    for (int c = 0; c < cols; c++) {
        c_row[j0 + c] = scale * (float)ibuf[c];
    }
}

__attribute__((target("avx512f,avx512vnni")))
static void gemm_int8_row_block_vnni512(float* c_row, const int8_t* a_row, const int8_t* B, int K, int N, int j0,
                                        int jb, float scale) {
    int32_t ibuf[16] __attribute__((aligned(64)));
    memset(ibuf, 0, sizeof(ibuf));
    int k = 0;

    if (jb >= 16) {
        __m512i acc = _mm512_setzero_si512();
        for (; k + 3 < K; k += 4) {
            const __m512i av = _mm512_set1_epi32(*(const int32_t*)(a_row + k));
            acc = _mm512_dpbusds_epi32(acc, av, gemm_int8_pack_b_k4x16(B, k, N, j0));
        }
        _mm512_storeu_si512(ibuf, acc);
        gemm_int8_row_finish_tail(c_row, ibuf, a_row, B, k, K, N, j0, 16, scale);
        return;
    }

    if (jb >= 8) {
        __m256i acc = _mm256_setzero_si256();
        for (; k + 3 < K; k += 4) {
            const __m256i av = _mm256_set1_epi32(*(const int32_t*)(a_row + k));
            acc = _mm256_dpbusds_epi32(acc, av, gemm_int8_pack_b_k4x8(B, k, N, j0));
        }
        _mm256_storeu_si256((__m256i*)ibuf, acc);
        gemm_int8_row_finish_tail(c_row, ibuf, a_row, B, k, K, N, j0, 8, scale);
    }
}

__attribute__((target("avx2,avx512vnni")))
static void gemm_int8_row_block_vnni256(float* c_row, const int8_t* a_row, const int8_t* B, int K, int N, int j0,
                                        int jb, float scale) {
    int32_t ibuf[16] __attribute__((aligned(64)));
    memset(ibuf, 0, sizeof(ibuf));
    int k = 0;

    if (jb >= 16) {
        __m256i acc0 = _mm256_setzero_si256();
        __m256i acc1 = _mm256_setzero_si256();
        for (; k + 3 < K; k += 4) {
            const __m256i av = _mm256_set1_epi32(*(const int32_t*)(a_row + k));
            acc0 = _mm256_dpbusds_epi32(acc0, av, gemm_int8_pack_b_k4x8(B, k, N, j0));
            acc1 = _mm256_dpbusds_epi32(acc1, av, gemm_int8_pack_b_k4x8(B, k, N, j0 + 8));
        }
        _mm256_storeu_si256((__m256i*)ibuf, acc0);
        _mm256_storeu_si256((__m256i*)(ibuf + 8), acc1);
        gemm_int8_row_finish_tail(c_row, ibuf, a_row, B, k, K, N, j0, 16, scale);
        return;
    }

    if (jb >= 8) {
        __m256i acc = _mm256_setzero_si256();
        for (; k + 3 < K; k += 4) {
            const __m256i av = _mm256_set1_epi32(*(const int32_t*)(a_row + k));
            acc = _mm256_dpbusds_epi32(acc, av, gemm_int8_pack_b_k4x8(B, k, N, j0));
        }
        _mm256_storeu_si256((__m256i*)ibuf, acc);
        gemm_int8_row_finish_tail(c_row, ibuf, a_row, B, k, K, N, j0, 8, scale);
    }
}

static void gemm_int8_row_block(float* c_row, const int8_t* a_row, const int8_t* B, int K, int N, int j0, int jb,
                                float scale) {
    const int vnni = gemm_int8_vnni_level();
    if (vnni == 2) {
        gemm_int8_row_block_vnni512(c_row, a_row, B, K, N, j0, jb, scale);
    } else if (vnni == 1) {
        gemm_int8_row_block_vnni256(c_row, a_row, B, K, N, j0, jb, scale);
    } else {
        gemm_int8_row_block_avx2(c_row, a_row, B, K, N, j0, jb, scale);
    }
}

#endif /* __x86_64__ && __AVX2__ */

#if defined(__aarch64__)
static void gemm_int8_row_block_neon(float* c_row, const int8_t* a_row, const int8_t* B, int K, int N, int j0,
                                     int jb, float scale) {
    int32x4_t acc0 = vdupq_n_s32(0);
    int32x4_t acc1 = vdupq_n_s32(0);
    int32x4_t acc2 = vdupq_n_s32(0);
    int32x4_t acc3 = vdupq_n_s32(0);

    for (int k = 0; k < K; k++) {
        const int32_t ak = (int32_t)a_row[k];
        if (ak == 0) continue;
        if (jb > 0) {
            int8x8_t b0 = vld1_s8(B + k * N + j0);
            int16x8_t b16 = vmovl_s8(b0);
            acc0 = vmlaq_n_s32(acc0, vmovl_s16(vget_low_s16(b16)), ak);
            if (jb > 4) acc1 = vmlaq_n_s32(acc1, vmovl_s16(vget_high_s16(b16)), ak);
        }
        if (jb > 8) {
            int8x8_t b1 = vld1_s8(B + k * N + j0 + 8);
            int16x8_t b16 = vmovl_s8(b1);
            acc2 = vmlaq_n_s32(acc2, vmovl_s16(vget_low_s16(b16)), ak);
            if (jb > 12) acc3 = vmlaq_n_s32(acc3, vmovl_s16(vget_high_s16(b16)), ak);
        }
    }

    const float32x4_t sf = vdupq_n_f32(scale);
    if (jb > 0) vst1q_f32(c_row + j0, vmulq_f32(vcvtq_f32_s32(acc0), sf));
    if (jb > 4) vst1q_f32(c_row + j0 + 4, vmulq_f32(vcvtq_f32_s32(acc1), sf));
    if (jb > 8) vst1q_f32(c_row + j0 + 8, vmulq_f32(vcvtq_f32_s32(acc2), sf));
    if (jb > 12) vst1q_f32(c_row + j0 + 12, vmulq_f32(vcvtq_f32_s32(acc3), sf));
}
#endif

status_t tensor_gemm_int8(float* restrict C, const int8_t* restrict A, const int8_t* restrict B,
                          const float* weight_scales, int M, int N, int K, float input_scale) {
    if (!C || !A || !B || !weight_scales) return ERROR_NULL_POINTER;

    for (int i = 0; i < M; i++) {
        const float scale = weight_scales[i] * input_scale;
        const int8_t* a_row = A + i * K;
        float* c_row = C + i * N;
        int j = 0;

#if defined(__x86_64__) && defined(__AVX2__)
        for (; j <= N - 16; j += 16) {
            gemm_int8_row_block(c_row, a_row, B, K, N, j, 16, scale);
        }
        if (j <= N - 8) {
            gemm_int8_row_block(c_row, a_row, B, K, N, j, 8, scale);
            j += 8;
        }
#elif defined(__aarch64__)
        for (; j <= N - 16; j += 16) {
            gemm_int8_row_block_neon(c_row, a_row, B, K, N, j, 16, scale);
        }
        if (j <= N - 8) {
            gemm_int8_row_block_neon(c_row, a_row, B, K, N, j, 8, scale);
            j += 8;
        }
#endif
        for (; j < N; j++) {
            c_row[j] = scale * (float)gemm_int8_dot_col(a_row, B, K, N, j);
        }
    }
    return SUCCESS;
}

status_t tensor_gemm_weight_int8(float* restrict C, const int8_t* restrict W, const float* restrict X,
                                 const float* weight_scales, int M, int N, int K) {
    if (!C || !W || !X || !weight_scales) return ERROR_NULL_POINTER;

    const int count = K * N;
    float act_max = tensor_max_abs_f32(X, count);
    if (act_max < 1e-8f) act_max = 1e-8f;
    const float input_scale = act_max / 127.0f;

    int8_t* xq = (int8_t*)malloc_aligned((size_t)count, 64);
    if (!xq) return ERROR_OUT_OF_MEMORY;
    tensor_quantize_symmetric(X, xq, count, input_scale);

    status_t st = tensor_gemm_int8(C, W, xq, weight_scales, M, N, K, input_scale);
    free_aligned(xq);
    return st;
}

const char* tensor_gemm_int8_backend(void) {
#if defined(__x86_64__) && defined(__AVX2__)
    switch (gemm_int8_vnni_level()) {
        case 2:
            return "avx512-vnni";
        case 1:
            return "avx-vnni";
        default:
            return "avx2";
    }
#elif defined(__aarch64__)
    return "neon";
#else
    return "scalar";
#endif
}
