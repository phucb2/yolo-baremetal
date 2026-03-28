#include <float.h>
#include "detection.h"

static float find_max_class_score(const float* class_scores, int num_classes, int* class_id) {
    float max_score = -FLT_MAX;
    *class_id = -1;
    for (int i = 0; i < num_classes; i++) {
        if (class_scores[i] > max_score) {
            max_score = class_scores[i];
            *class_id = i;
        }
    }
    return max_score;
}

status_t decode_detections(detection_results_t* results, const tensor_t* head_output, 
                          float threshold, float img_w, float img_h) {
    if (!results || !head_output) return ERROR_NULL_POINTER;
    
    int num_candidates = head_output->dims[1]; // 300
    int total_elements = head_output->dims[2]; // 4 + C
    int num_classes = total_elements - 4;
    const float* data = head_output->data;
    
    results->count = 0;
    
    for (int i = 0; i < num_candidates; i++) {
        const float* row = data + i * total_elements;
        int class_id;
        float score = find_max_class_score(row + 4, num_classes, &class_id);
        
        if (score > threshold) {
            if (results->count >= results->capacity) break;
            
            float cx = row[0];
            float cy = row[1];
            float w = row[2];
            float h = row[3];
            
            detection_t* det = &results->detections[results->count++];
            det->score = score;
            det->class_id = class_id;
            
            // Convert to x1, y1, x2, y2 absolute coordinates
            det->x1 = (cx - w / 2.0f) * img_w;
            det->y1 = (cy - h / 2.0f) * img_h;
            det->x2 = (cx + w / 2.0f) * img_w;
            det->y2 = (cy + h / 2.0f) * img_h;
        }
    }
    
    return SUCCESS;
}
