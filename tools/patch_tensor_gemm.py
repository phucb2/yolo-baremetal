from pathlib import Path

p = Path("src/tensor.c")
text = p.read_text(encoding="utf-8")

start_ob = text.index("#ifdef USE_OPENBLAS")
depth = 0
i = start_ob
while i < len(text):
    if text.startswith("#if", i) and not text.startswith("#ifdef USE_OPENBLAS", i) and not text.startswith("#ifndef", i):
        pass
    if text.startswith("#ifdef USE_OPENBLAS", i) or text.startswith("#if defined(__has_include)", i) or text.startswith("#if __has_include", i):
        depth += 1
    if text.startswith("#endif", i):
        depth -= 1
        if depth == 0:
            end_ob = i + len("#endif")
            break
    i += 1
text = text[:start_ob] + text[end_ob + 1 :]

text = text.replace('#include "tensor.h"\n', '#include "tensor.h"\n#include "gemm_backend.h"\n')

s = text.index("static float tensor_max_abs_f32")
marker = "const char* tensor_gemm_int8_backend(void)"
e = text.index(marker)
e = text.index("}", e) + 1

wrappers = r"""
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
"""

text = text[:s] + wrappers
p.write_text(text, encoding="utf-8")
print("tensor.c lines:", len(text.splitlines()))
