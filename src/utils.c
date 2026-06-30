#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "utils.h"
#include "tensor.h"

#ifdef __APPLE__
#include <mach/mach_time.h>
static mach_timebase_info_data_t timebase_info;
static void init_timebase(void) {
    if (timebase_info.denom == 0) {
        mach_timebase_info(&timebase_info);
    }
}
#endif

#ifdef _WIN32
#include <windows.h>
static LARGE_INTEGER timer_freq;
static void init_timer_freq(void) {
    if (timer_freq.QuadPart == 0) {
        QueryPerformanceFrequency(&timer_freq);
    }
}
#endif

void timer_start(yolo_timer_t* timer) {
#ifdef __APPLE__
    init_timebase();
    timer->start = mach_absolute_time();
#elif defined(_WIN32)
    init_timer_freq();
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    timer->start = (uint64_t)t.QuadPart;
#else
    (void)timer;
#endif
}

void timer_stop(yolo_timer_t* timer) {
#ifdef __APPLE__
    timer->end = mach_absolute_time();
#elif defined(_WIN32)
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    timer->end = (uint64_t)t.QuadPart;
#else
    (void)timer;
#endif
}

double timer_elapsed_ms(const yolo_timer_t* timer) {
#ifdef __APPLE__
    uint64_t elapsed = timer->end - timer->start;
    double nanoseconds = (double)elapsed * timebase_info.numer / timebase_info.denom;
    return nanoseconds / 1000000.0;
#elif defined(_WIN32)
    if (timer_freq.QuadPart == 0) return 0.0;
    const double elapsed = (double)(timer->end - timer->start);
    return elapsed * 1000.0 / (double)timer_freq.QuadPart;
#else
    (void)timer;
    return 0.0;
#endif
}

status_t load_named_tensor(FILE* f, char* name, tensor_t* tensor, int file_version) {
    if (!f || !name || !tensor) return ERROR_NULL_POINTER;

    memset(tensor, 0, sizeof(*tensor));
    tensor->dtype = TENSOR_DTYPE_FP32;

    int name_len;
    if (fread(&name_len, sizeof(int), 1, f) != 1) return ERROR_FILE_NOT_FOUND;
    if (name_len <= 0 || name_len >= 128) return ERROR_INVALID_FORMAT;
    if (fread(name, 1, (size_t)name_len, f) != (size_t)name_len) return ERROR_FILE_NOT_FOUND;
    name[name_len] = '\0';

    int dim_count;
    if (fread(&dim_count, sizeof(int), 1, f) != 1) return ERROR_INVALID_FORMAT;
    if (dim_count < 1 || dim_count > 4) return ERROR_INVALID_FORMAT;
    int dims[4] = {1, 1, 1, 1};
    for (int d = 0; d < dim_count; d++) {
        if (fread(&dims[d], sizeof(int), 1, f) != 1) return ERROR_INVALID_FORMAT;
    }

    int dtype = TENSOR_DTYPE_FP32;
    if (file_version >= 2) {
        if (fread(&dtype, sizeof(int), 1, f) != 1) return ERROR_INVALID_FORMAT;
    }

    size_t total_elements = (size_t)dims[0] * (size_t)dims[1] * (size_t)dims[2] * (size_t)dims[3];

    if (dtype == TENSOR_DTYPE_FP32) {
        status_t status = tensor_allocate(tensor, dims[0], dims[1], dims[2], dims[3]);
        if (status != SUCCESS) return status;
        size_t nread = fread(tensor->data, sizeof(float), total_elements, f);
        UTIL_DEBUG_LOG_TENSOR_LOAD(name, total_elements, nread);
        if (nread != total_elements) {
            tensor_free(tensor);
            return ERROR_INVALID_FORMAT;
        }
        return SUCCESS;
    }

    if (dtype == TENSOR_DTYPE_INT8) {
        int num_scales = 0;
        if (fread(&num_scales, sizeof(int), 1, f) != 1) return ERROR_INVALID_FORMAT;
        if (num_scales <= 0) return ERROR_INVALID_FORMAT;

        tensor->dims[0] = dims[0];
        tensor->dims[1] = dims[1];
        tensor->dims[2] = dims[2];
        tensor->dims[3] = dims[3];
        tensor->stride[3] = 1;
        tensor->stride[2] = dims[3];
        tensor->stride[1] = dims[2] * dims[3];
        tensor->stride[0] = dims[1] * dims[2] * dims[3];
        tensor->dtype = TENSOR_DTYPE_INT8;
        tensor->num_scales = num_scales;
        tensor->scales = (float*)malloc((size_t)num_scales * sizeof(float));
        if (!tensor->scales) return ERROR_OUT_OF_MEMORY;
        if (fread(tensor->scales, sizeof(float), (size_t)num_scales, f) != (size_t)num_scales) {
            tensor_free(tensor);
            return ERROR_INVALID_FORMAT;
        }
        tensor->qdata = (int8_t*)malloc_aligned(total_elements, 64);
        if (!tensor->qdata) {
            tensor_free(tensor);
            return ERROR_OUT_OF_MEMORY;
        }
        tensor->is_owner = true;
        size_t nread = fread(tensor->qdata, 1, total_elements, f);
        UTIL_DEBUG_LOG_TENSOR_LOAD(name, total_elements, nread);
        if (nread != total_elements) {
            tensor_free(tensor);
            return ERROR_INVALID_FORMAT;
        }
        return SUCCESS;
    }

    return ERROR_INVALID_FORMAT;
}

status_t save_named_tensor(FILE* f, const char* name, const tensor_t* tensor) {
    if (!f || !name || !tensor || !tensor->data) return ERROR_NULL_POINTER;
    int name_len = (int)strlen(name);
    if (fwrite(&name_len, sizeof(int), 1, f) != 1) return ERROR_FILE_NOT_FOUND;
    if (fwrite(name, 1, (size_t)name_len, f) != (size_t)name_len) return ERROR_FILE_NOT_FOUND;
    int dim_count = 4;
    if (fwrite(&dim_count, sizeof(int), 1, f) != 1) return ERROR_FILE_NOT_FOUND;
    for (int d = 0; d < 4; d++) {
        if (fwrite(&tensor->dims[d], sizeof(int), 1, f) != 1) return ERROR_FILE_NOT_FOUND;
    }
    size_t n = (size_t)tensor->dims[0] * (size_t)tensor->dims[1] * (size_t)tensor->dims[2] * (size_t)tensor->dims[3];
    if (fwrite(tensor->data, sizeof(float), n, f) != n) return ERROR_FILE_NOT_FOUND;
    return SUCCESS;
}

void fold_bn(tensor_t* conv_w, tensor_t* conv_b, 
             const tensor_t* bn_w, const tensor_t* bn_b, 
             const tensor_t* bn_rm, const tensor_t* bn_rv) {
    int out_c = conv_w->dims[0];
    int in_c = conv_w->dims[1];
    int kh = conv_w->dims[2];
    int kw = conv_w->dims[3];
    float eps = 1e-5f;

    for (int i = 0; i < out_c; i++) {
        float gamma = bn_w->data[i];
        float beta = bn_b->data[i];
        float mean = bn_rm->data[i];
        float var = bn_rv->data[i];
        float scale = gamma / sqrtf(var + eps);

        for (int j = 0; j < in_c * kh * kw; j++) {
            conv_w->data[i * in_c * kh * kw + j] *= scale;
        }

        if (conv_b) {
            float b = conv_b->data[i];
            conv_b->data[i] = (b - mean) * scale + beta;
        }
    }
}
