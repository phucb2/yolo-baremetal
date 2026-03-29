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

/* head_output: NCHW with N=1, C=max_det, H>=6, W=1 — same memory as Ultralytics
 * Detect.postprocess(): each row is x1,y1,x2,y2 (pixels, model input space), score, class as float.
 * reg_max==1; threshold filters on score. */
status_t decode_detections(detection_results_t* results, const tensor_t* head_output,
                          float threshold);

#endif
