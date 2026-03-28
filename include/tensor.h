#ifndef TENSOR_H
#define TENSOR_H

#include <stdbool.h>
#include <stddef.h>
#include "status.h"

typedef struct {
    float* restrict data;
    int dims[4]; // N, C, H, W
    int stride[4];
    bool is_owner;
} tensor_t;

status_t tensor_allocate(tensor_t* tensor, int n, int c, int h, int w);
status_t tensor_free(tensor_t* tensor);
status_t tensor_copy(tensor_t* dest, const tensor_t* src);
status_t tensor_fill(tensor_t* tensor, float value);

// Optimized GEMM: C = alpha * A * B + beta * C
// A: M x K, B: K x N, C: M x N
status_t tensor_gemm(float* C, const float* A, const float* B, 
                    int M, int N, int K, 
                    float alpha, float beta);

// Memory alignment for SIMD (NEON)
void* malloc_aligned(size_t size, size_t alignment);
void free_aligned(void* ptr);

#endif
