#ifndef MODEL_H
#define MODEL_H

#include <stdio.h>
#include "tensor.h"
#include "layers.h"
#include "detection.h"

#define MAX_MODEL_LAYERS 32
/* Timed substeps inside model_forward_ex (conv / C3k2 / neck / detect / copy). */
#define MODEL_FORWARD_PROFILE_STEPS 25
/* Detect sequential index for the wired yolo26 graph (model.23.*). Fixed; not configurable at runtime. */
#define YOLO26_DETECT_IDX 23

typedef struct {
    double ms_sum[MODEL_FORWARD_PROFILE_STEPS];
    double ms_last[MODEL_FORWARD_PROFILE_STEPS];
    unsigned runs;
} model_forward_profile_t;

void model_forward_profile_reset(model_forward_profile_t* p);
const char* model_forward_profile_step_name(int step_index);
void model_forward_profile_print_last(const model_forward_profile_t* p, FILE* fp, const char* title);
void model_forward_profile_print_aggregate(const model_forward_profile_t* p, FILE* fp);

typedef struct {
    char name[128];
    tensor_t tensor;
} named_tensor_t;

typedef struct {
    named_tensor_t* weights;
    int num_weights;
    
    tensor_t* buffers; // Intermediate layer buffers
    int num_buffers;
    
    int input_w, input_h;
    int num_classes;
} model_t;

status_t model_create(model_t* model, int input_w, int input_h);
status_t model_destroy(model_t* model);
status_t model_forward(model_t* model, const tensor_t* input, tensor_t* output);
/**
 * Same as model_forward; if stage_dump is non-NULL, append stage tensors.
 * If profile is non-NULL, accumulates per-step times (successful full forwards only).
 */
status_t model_forward_ex(model_t* model, const tensor_t* input, tensor_t* output, FILE* stage_dump,
                          model_forward_profile_t* profile);
status_t model_load_weights(model_t* model, const char* path);
tensor_t* model_get_weight(model_t* model, const char* name);

#endif
