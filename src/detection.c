#include "detection.h"

status_t decode_detections(detection_results_t* results, const tensor_t* head_output,
                          float threshold) {
    if (!results || !head_output) return ERROR_NULL_POINTER;

    if (head_output->dims[0] < 1) return ERROR_INVALID_DIMS;
    int num_candidates = head_output->dims[1];
    int feat = head_output->dims[2];
    int w = head_output->dims[3];
    if (feat < 6 || w < 1) return ERROR_INVALID_DIMS;

    const float* data = head_output->data;
    int row_stride = feat * w;

    results->count = 0;

    for (int i = 0; i < num_candidates; i++) {
        const float* row = data + i * row_stride;
        float score = row[4];
        if (score <= threshold) continue;
        if (results->count >= results->capacity) break;

        detection_t* det = &results->detections[results->count++];
        det->x1 = row[0];
        det->y1 = row[1];
        det->x2 = row[2];
        det->y2 = row[3];
        det->score = score;
        det->class_id = (int)row[5];
    }

    return SUCCESS;
}
