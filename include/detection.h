#ifndef DETECTION_H
#define DETECTION_H

#include "tensor.h"

typedef struct {
    float x1, y1, x2, y2;
    float score;
    int class_id;
} detection_t;

typedef struct {
    detection_t* detections;
    int count;
    int capacity;
} detection_results_t;

status_t decode_detections(detection_results_t* results, const tensor_t* head_output, 
                          float threshold, float img_w, float img_h);

#endif
