#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>
#include <stdio.h>
#include "status.h"
#include "tensor.h"

#ifdef __APPLE__
#include <mach/mach_time.h>
#endif

typedef struct {
    uint64_t start;
    uint64_t end;
} timer_t;

void timer_start(timer_t* timer);
void timer_stop(timer_t* timer);
double timer_elapsed_ms(const timer_t* timer);

// Binary tensor loading
status_t load_named_tensor(FILE* f, char* name, tensor_t* tensor);

// BatchNorm folding
void fold_bn(tensor_t* conv_w, tensor_t* conv_b, 
             const tensor_t* bn_w, const tensor_t* bn_b, 
             const tensor_t* bn_rm, const tensor_t* bn_rv);

// Logging with performance benchmarking
#define BENCH_START(name) timer_t _timer_##name; timer_start(&_timer_##name);
#define BENCH_STOP(name) timer_stop(&_timer_##name); \
    printf("[BENCH] %-20s: %8.4f ms\n", #name, timer_elapsed_ms(&_timer_##name));

#endif
