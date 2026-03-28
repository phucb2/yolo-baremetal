#include <stdlib.h>
#include <string.h>
#include "tensor.h"

#ifdef __x86_64__
#include <immintrin.h>
#endif

void* malloc_aligned(size_t size, size_t alignment) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) return NULL;
    return ptr;
}

void free_aligned(void* ptr) {
    free(ptr);
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
    size_t size = (size_t)n * c * h * w * sizeof(float);
    tensor->data = (float*)malloc_aligned(size, 64);
    if (!tensor->data) return ERROR_OUT_OF_MEMORY;
    tensor->is_owner = true;
    return SUCCESS;
}

status_t tensor_free(tensor_t* tensor) {
    if (!tensor) return ERROR_NULL_POINTER;
    if (tensor->is_owner && tensor->data) {
        free_aligned(tensor->data);
        tensor->data = NULL;
    }
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

status_t tensor_gemm(float* restrict C, const float* restrict A, const float* restrict B, 
                    int M, int N, int K, float alpha, float beta) {
    if (!C || !A || !B) return ERROR_NULL_POINTER;

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
#endif
            for (; j < N; j++) {
                c_row[j] += a_val * b_row[j];
            }
        }
    }
    return SUCCESS;
}
