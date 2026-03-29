#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "utils.h"
#include "tensor.h"

#ifdef __APPLE__
static mach_timebase_info_data_t timebase_info;
static void init_timebase() {
    if (timebase_info.denom == 0) {
        mach_timebase_info(&timebase_info);
    }
}
#endif

void timer_start(timer_t* timer) {
#ifdef __APPLE__
    init_timebase();
    timer->start = mach_absolute_time();
#endif
}

void timer_stop(timer_t* timer) {
#ifdef __APPLE__
    timer->end = mach_absolute_time();
#endif
}

double timer_elapsed_ms(const timer_t* timer) {
#ifdef __APPLE__
    uint64_t elapsed = timer->end - timer->start;
    double nanoseconds = (double)elapsed * timebase_info.numer / timebase_info.denom;
    return nanoseconds / 1000000.0;
#else
    return 0.0;
#endif
}

status_t load_named_tensor(FILE* f, char* name, tensor_t* tensor) {
    int name_len;
    if (fread(&name_len, sizeof(int), 1, f) != 1) return ERROR_FILE_NOT_FOUND;
    fread(name, 1, name_len, f);
    name[name_len] = '\0';
    
    int dim_count;
    fread(&dim_count, sizeof(int), 1, f);
    int dims[4] = {1, 1, 1, 1};
    for (int d = 0; d < dim_count; d++) {
        fread(&dims[d], sizeof(int), 1, f);
    }
    
    status_t status = tensor_allocate(tensor, dims[0], dims[1], dims[2], dims[3]);
    if (status != SUCCESS) return status;
    
    size_t total_elements = (size_t)dims[0] * dims[1] * dims[2] * dims[3];
    size_t nread = fread(tensor->data, sizeof(float), total_elements, f);
    UTIL_DEBUG_LOG_TENSOR_LOAD(name, total_elements, nread);
    if (nread != total_elements) {
        tensor_free(tensor);
        return ERROR_INVALID_FORMAT;
    }
    return SUCCESS;
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
