#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "tensor.h"
#include "layers.h"
#include "detection.h"
#include "model.h"
#include "camera.h"
#include "utils.h"

void preprocess(tensor_t* input_tensor, const uint8_t* rgb_buffer, int w, int h) {
    float* data = input_tensor->data;
    int c_stride = h * w;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            const uint8_t* pixel = rgb_buffer + (y * w + x) * 3;
            data[0 * c_stride + y * w + x] = pixel[0] / 255.0f; // R
            data[1 * c_stride + y * w + x] = pixel[1] / 255.0f; // G
            data[2 * c_stride + y * w + x] = pixel[2] / 255.0f; // B
        }
    }
}

int main() {
    const int W = 640, H = 640;
    model_t model;
    model_create(&model, W, H);
    
    if (model_load_weights(&model, "weights/yolo26.bin") != SUCCESS) {
        printf("Failed to load weights\n");
        return 1;
    }
    
    camera_t* cam = NULL;
    if (camera_create(&cam, W, H) != SUCCESS) {
        printf("Failed to create camera\n");
        return 1;
    }
    
    camera_start(cam);
    
    uint8_t* rgb_buffer = malloc(W * H * 3);
    tensor_t input_tensor;
    tensor_allocate(&input_tensor, 1, 3, H, W);
    
    // Dummy detection head output for testing [1, 300, 85] (COCO)
    tensor_t head_output;
    tensor_allocate(&head_output, 1, 300, 85, 1);
    tensor_fill(&head_output, 0.0f);
    
    // Inject a dummy detection
    head_output.data[0 * 85 + 0] = 0.5f; // cx
    head_output.data[0 * 85 + 1] = 0.5f; // cy
    head_output.data[0 * 85 + 2] = 0.2f; // w
    head_output.data[0 * 85 + 3] = 0.2f; // h
    head_output.data[0 * 85 + 4 + 0] = 0.9f; // class 0 score
    
    detection_results_t results;
    results.capacity = 100;
    results.detections = malloc(sizeof(detection_t) * results.capacity);
    
    printf("Starting inference loop. Press Ctrl+C to stop.\n");
    
    for (int i = 0; i < 5; i++) {
        printf("\n--- Frame %d ---\n", i);
        
        BENCH_START(capture);
        camera_capture(cam, rgb_buffer);
        BENCH_STOP(capture);
        
        BENCH_START(preprocess);
        preprocess(&input_tensor, rgb_buffer, W, H);
        BENCH_STOP(preprocess);
        
        BENCH_START(inference_dummy);
        // In a real scenario, we'd run the full YOLO26 layers here
        // For now, we simulate with a dummy latency
        usleep(20000); 
        BENCH_STOP(inference_dummy);
        
        BENCH_START(decode);
        decode_detections(&results, &head_output, 0.5f, (float)W, (float)H);
        BENCH_STOP(decode);
        
        printf("Detections: %d\n", results.count);
        for (int d = 0; d < results.count; d++) {
            detection_t* det = &results.detections[d];
            printf("  [%d] Class %d: %.2f @ (%.1f, %.1f, %.1f, %.1f)\n", 
                   d, det->class_id, det->score, det->x1, det->y1, det->x2, det->y2);
        }
    }
    
    camera_stop(cam);
    camera_destroy(cam);
    free(rgb_buffer);
    tensor_free(&input_tensor);
    tensor_free(&head_output);
    free(results.detections);
    
    return 0;
}
