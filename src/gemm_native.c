#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "platform.h"
#include "tensor.h"
#include "gemm_native.h"

#ifdef _MSC_VER
#include <malloc.h>
#endif

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

static float gemm_max_abs_f32(const float* src, int count) {
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
#elif YOLO_ARCH_ARM64
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

status_t gemm_native_fp32(float* restrict C, const float* restrict A, const float* restrict B, int M, int N, int K,
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
#elif YOLO_ARCH_ARM64
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

#if YOLO_ARCH_X64 && defined(__AVX2__)

#ifdef _MSC_VER
#include <intrin.h>
static void yolo_cpuid_ex(int info[4], int leaf, int subleaf) {
    __cpuidex(info, leaf, subleaf);
}
static int yolo_cpuid_max_leaf(void) {
    int info[4];
    __cpuid(info, 0);
    return info[0];
}
#else
#include <cpuid.h>
static void yolo_cpuid_ex(int info[4], int leaf, int subleaf) {
    __cpuid_count(leaf, subleaf, (unsigned int*)info, (unsigned int*)info + 1, (unsigned int*)info + 2,
                  (unsigned int*)info + 3);
}
static int yolo_cpuid_max_leaf(void) {
    return (int)__get_cpuid_max(0, NULL);
}
#endif

/* 0 = AVX2 only, 1 = AVX-VNNI (256-bit dpbusds), 2 = AVX512-VNNI */
static int gemm_int8_vnni_level(void) {
    static int level = -1;
    if (level >= 0) return level;
#ifdef _MSC_VER
    level = 0;
    return level;
#else
    if (yolo_cpuid_max_leaf() < 7) {
        level = 0;
        return level;
    }
    int info[4];
    yolo_cpuid_ex(info, 7, 0);
    const unsigned int ecx = (unsigned int)info[2];
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
#endif
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

#if !defined(_MSC_VER)
YOLO_GCC_TARGET("avx512f,avx512vnni")
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
#endif /* !defined(_MSC_VER) */

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

#if !defined(_MSC_VER)
YOLO_GCC_TARGET("avx512f,avx512vnni")
static void gemm_int8_row_block_vnni512(float* c_row, const int8_t* a_row, const int8_t* B, int K, int N, int j0,
                                        int jb, float scale) {
    int32_t ibuf[16] YOLO_ALIGNED(64);
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

YOLO_GCC_TARGET("avx2,avx512vnni")
static void gemm_int8_row_block_vnni256(float* c_row, const int8_t* a_row, const int8_t* B, int K, int N, int j0,
                                        int jb, float scale) {
    int32_t ibuf[16] YOLO_ALIGNED(64);
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
#endif /* !defined(_MSC_VER) */

static void gemm_int8_row_block(float* c_row, const int8_t* a_row, const int8_t* B, int K, int N, int j0, int jb,
                                float scale) {
#if !defined(_MSC_VER)
    const int vnni = gemm_int8_vnni_level();
    if (vnni == 2) {
        gemm_int8_row_block_vnni512(c_row, a_row, B, K, N, j0, jb, scale);
    } else if (vnni == 1) {
        gemm_int8_row_block_vnni256(c_row, a_row, B, K, N, j0, jb, scale);
    } else
#endif
    {
        gemm_int8_row_block_avx2(c_row, a_row, B, K, N, j0, jb, scale);
    }
}

#endif /* YOLO_ARCH_X64 && __AVX2__ */

#if YOLO_ARCH_ARM64
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

status_t gemm_native_int8(float* restrict C, const int8_t* restrict A, const int8_t* restrict B,
                          const float* weight_scales, int M, int N, int K, float input_scale) {
    if (!C || !A || !B || !weight_scales) return ERROR_NULL_POINTER;

    for (int i = 0; i < M; i++) {
        const float scale = weight_scales[i] * input_scale;
        const int8_t* a_row = A + i * K;
        float* c_row = C + i * N;
        int j = 0;

#if YOLO_ARCH_X64 && defined(__AVX2__)
        for (; j <= N - 16; j += 16) {
            gemm_int8_row_block(c_row, a_row, B, K, N, j, 16, scale);
        }
        if (j <= N - 8) {
            gemm_int8_row_block(c_row, a_row, B, K, N, j, 8, scale);
            j += 8;
        }
#elif YOLO_ARCH_ARM64
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

status_t gemm_native_weight_int8(float* restrict C, const int8_t* restrict W, const float* restrict X,
                                 const float* weight_scales, int M, int N, int K) {
    if (!C || !W || !X || !weight_scales) return ERROR_NULL_POINTER;

    const int count = K * N;
    float act_max = gemm_max_abs_f32(X, count);
    if (act_max < 1e-8f) act_max = 1e-8f;
    const float input_scale = act_max / 127.0f;

    int8_t* xq = (int8_t*)malloc_aligned((size_t)count, 64);
    if (!xq) return ERROR_OUT_OF_MEMORY;
    tensor_quantize_symmetric(X, xq, count, input_scale);

    status_t st = gemm_native_int8(C, W, xq, weight_scales, M, N, K, input_scale);
    free_aligned(xq);
    return st;
}

const char* gemm_native_int8_backend_name(void) {
#if YOLO_ARCH_X64 && defined(__AVX2__)
    switch (gemm_int8_vnni_level()) {
        case 2:
            return "avx512-vnni";
        case 1:
            return "avx-vnni";
        default:
            return "avx2";
    }
#elif YOLO_ARCH_ARM64
    return "neon";
#else
    return "scalar";
#endif
}