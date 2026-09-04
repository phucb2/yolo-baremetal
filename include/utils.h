#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include "status.h"
#include "tensor.h"

#ifndef UTIL_DEBUG_LOG_PATH
#define UTIL_DEBUG_LOG_PATH "debug-yolo.log"
#endif

static inline long long util_debug_timestamp_ms(void) {
#ifdef _WIN32
    return (long long)time(NULL) * 1000;
#else
    struct timespec tss;
    clock_gettime(CLOCK_REALTIME, &tss);
    return (long long)tss.tv_sec * 1000 + tss.tv_nsec / 1000000;
#endif
}

#ifdef NO_LOGGING
#define UTIL_DEBUG_LOG_TENSOR_LOAD(name_str, expected_sz, got_sz) ((void)0)
#else
#define UTIL_DEBUG_LOG_TENSOR_LOAD(name_str, expected_sz, got_sz)                                                      \
    do {                                                                                                               \
        FILE* _df = fopen(UTIL_DEBUG_LOG_PATH, "a");                                                                   \
        if (_df) {                                                                                                     \
            const long long _ts = util_debug_timestamp_ms();                                                           \
            fprintf(_df,                                                                                               \
                    "{\"sessionId\":\"utils\",\"hypothesisId\":\"H1\",\"location\":\"utils.c:load_named_tensor\","     \
                    "\"message\":\"fread_floats\",\"data\":{\"tensorName\":\"%s\",\"expected\":%zu,\"got\":%zu},"      \
                    "\"timestamp\":%lld}\n",                                                                           \
                    (name_str), (size_t)(expected_sz), (size_t)(got_sz), _ts);                                         \
            fclose(_df);                                                                                               \
        }                                                                                                              \
    } while (0)
#endif

typedef struct {
    uint64_t start;
    uint64_t end;
} yolo_timer_t;

void timer_start(yolo_timer_t* timer);
void timer_stop(yolo_timer_t* timer);
double timer_elapsed_ms(const yolo_timer_t* timer);

// Binary tensor loading / saving (v1 FP32 or v2 with dtype; file_version 0/1 = legacy)
status_t load_named_tensor(FILE* f, char* name, tensor_t* tensor, int file_version);
status_t save_named_tensor(FILE* f, const char* name, const tensor_t* tensor);

// BatchNorm folding
void fold_bn(tensor_t* conv_w, tensor_t* conv_b, 
             const tensor_t* bn_w, const tensor_t* bn_b, 
             const tensor_t* bn_rm, const tensor_t* bn_rv);

// Logging with performance benchmarking
#define BENCH_START(name) yolo_timer_t _timer_##name; timer_start(&_timer_##name);
#define BENCH_STOP(name) timer_stop(&_timer_##name); \
    printf("[BENCH] %-20s: %8.4f ms\n", #name, timer_elapsed_ms(&_timer_##name));

#endif
