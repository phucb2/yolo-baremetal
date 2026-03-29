#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include "status.h"
#include "tensor.h"

#ifndef UTIL_DEBUG_LOG_PATH
#define UTIL_DEBUG_LOG_PATH "/Users/phucbb/Personal/random-project/.cursor/debug-5d8ee2.log"
#endif

#ifdef NO_LOGGING
#define UTIL_DEBUG_LOG_TENSOR_LOAD(name_str, expected_sz, got_sz) ((void)0)
#else
#define UTIL_DEBUG_LOG_TENSOR_LOAD(name_str, expected_sz, got_sz)                                                      \
    do {                                                                                                               \
        FILE* _df = fopen(UTIL_DEBUG_LOG_PATH, "a");                                                                   \
        if (_df) {                                                                                                     \
            struct timespec _tss;                                                                                       \
            clock_gettime(CLOCK_REALTIME, &_tss);                                                                      \
            long long _ts = (long long)_tss.tv_sec * 1000 + _tss.tv_nsec / 1000000;                                      \
            fprintf(_df,                                                                                               \
                    "{\"sessionId\":\"utils\",\"hypothesisId\":\"H1\",\"location\":\"utils.c:load_named_tensor\","       \
                    "\"message\":\"fread_floats\",\"data\":{\"tensorName\":\"%s\",\"expected\":%zu,\"got\":%zu},"        \
                    "\"timestamp\":%lld}\n",                                                                            \
                    (name_str), (size_t)(expected_sz), (size_t)(got_sz), _ts);                                         \
            fclose(_df);                                                                                               \
        }                                                                                                              \
    } while (0)
#endif

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

// Binary tensor loading / saving (same layout as tools/generate_layer_tests.py save_tensor)
status_t load_named_tensor(FILE* f, char* name, tensor_t* tensor);
status_t save_named_tensor(FILE* f, const char* name, const tensor_t* tensor);

// BatchNorm folding
void fold_bn(tensor_t* conv_w, tensor_t* conv_b, 
             const tensor_t* bn_w, const tensor_t* bn_b, 
             const tensor_t* bn_rm, const tensor_t* bn_rv);

// Logging with performance benchmarking
#define BENCH_START(name) timer_t _timer_##name; timer_start(&_timer_##name);
#define BENCH_STOP(name) timer_stop(&_timer_##name); \
    printf("[BENCH] %-20s: %8.4f ms\n", #name, timer_elapsed_ms(&_timer_##name));

#endif
