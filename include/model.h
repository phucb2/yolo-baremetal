#ifndef MODEL_H
#define MODEL_H

#include <stdio.h>
#include "tensor.h"
#include "layers.h"
#include "detection.h"

#define MAX_MODEL_LAYERS 32
/* Detect sequential index for the wired yolo26 graph (model.23.*). Fixed; not configurable at runtime. */
#define YOLO26_DETECT_IDX 23

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
/** Same as model_forward; if stage_dump is non-NULL, append stage tensors (stage_00_input, stage_01_buf0 … stage_24_buf23). */
status_t model_forward_ex(model_t* model, const tensor_t* input, tensor_t* output, FILE* stage_dump);
status_t model_load_weights(model_t* model, const char* path);
tensor_t* model_get_weight(model_t* model, const char* name);

#endif
