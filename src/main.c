#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "tensor.h"
#include "layers.h"
#include "detection.h"
#include "model.h"
#include "camera.h"
#include "utils.h"
#include "visualize.h"

static void resize_rgb_bilinear(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh) {
    if (sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0) {
        return;
    }
    for (int dy = 0; dy < dh; dy++) {
        for (int dx = 0; dx < dw; dx++) {
            float sx = (dw == 1) ? 0.f : (float)dx * (float)(sw - 1) / (float)(dw - 1);
            float sy = (dh == 1) ? 0.f : (float)dy * (float)(sh - 1) / (float)(dh - 1);
            int x0 = (int)floorf(sx);
            int y0 = (int)floorf(sy);
            int x1 = x0 + 1 < sw ? x0 + 1 : x0;
            int y1 = y0 + 1 < sh ? y0 + 1 : y0;
            float fx = sx - (float)x0;
            float fy = sy - (float)y0;
            for (int c = 0; c < 3; c++) {
                float v00 = (float)src[(y0 * sw + x0) * 3 + c];
                float v10 = (float)src[(y0 * sw + x1) * 3 + c];
                float v01 = (float)src[(y1 * sw + x0) * 3 + c];
                float v11 = (float)src[(y1 * sw + x1) * 3 + c];
                float v0 = v00 * (1.f - fx) + v10 * fx;
                float v1 = v01 * (1.f - fx) + v11 * fx;
                float v = v0 * (1.f - fy) + v1 * fy;
                int iv = (int)(v + 0.5f);
                if (iv < 0) {
                    iv = 0;
                }
                if (iv > 255) {
                    iv = 255;
                }
                dst[(dy * dw + dx) * 3 + c] = (uint8_t)iv;
            }
        }
    }
}

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

static int run_image_mode(const char* img_path, const char* out_bmp, int W, int H) {
    model_t model;
    model_create(&model, W, H);
    if (model_load_weights(&model, "weights/yolo26.bin") != SUCCESS) {
        printf("Failed to load weights\n");
        model_destroy(&model);
        return 1;
    }

    int iw = 0, ih = 0, ic = 0;
    unsigned char* raw = stbi_load(img_path, &iw, &ih, &ic, 3);
    if (!raw || iw <= 0 || ih <= 0) {
        fprintf(stderr, "Failed to decode image: %s\n", img_path);
        if (raw) {
            stbi_image_free(raw);
        }
        model_destroy(&model);
        return 1;
    }

    uint8_t* rgb_buffer = malloc((size_t)W * H * 3);
    if (!rgb_buffer) {
        stbi_image_free(raw);
        model_destroy(&model);
        return 1;
    }
    resize_rgb_bilinear(raw, iw, ih, rgb_buffer, W, H);
    stbi_image_free(raw);

    tensor_t input_tensor;
    tensor_allocate(&input_tensor, 1, 3, H, W);
    tensor_t head_output;
    tensor_allocate(&head_output, 1, 300, 6, 1);

    detection_results_t results;
    results.capacity = 100;
    results.detections = malloc(sizeof(detection_t) * results.capacity);
    if (!results.detections) {
        free(rgb_buffer);
        tensor_free(&input_tensor);
        tensor_free(&head_output);
        model_destroy(&model);
        return 1;
    }

    preprocess(&input_tensor, rgb_buffer, W, H);

    model_forward_profile_t layer_prof;
    model_forward_profile_reset(&layer_prof);

    timer_t t_inf;
    timer_start(&t_inf);
    status_t inf_st = model_forward_ex(&model, &input_tensor, &head_output, NULL, &layer_prof);
    timer_stop(&t_inf);
    double ms_inf = timer_elapsed_ms(&t_inf);

    if (inf_st != SUCCESS) {
        fprintf(stderr, "model_forward failed (%d)\n", (int)inf_st);
        free(rgb_buffer);
        tensor_free(&input_tensor);
        tensor_free(&head_output);
        free(results.detections);
        model_destroy(&model);
        return 1;
    }

    model_forward_profile_print_last(&layer_prof, stdout, "  model_forward steps, ms:");
    model_forward_profile_print_aggregate(&layer_prof, stdout);

    timer_t t_dec;
    timer_start(&t_dec);
    decode_detections(&results, &head_output, 0.2f);
    timer_stop(&t_dec);
    double ms_dec = timer_elapsed_ms(&t_dec);

    double ms_viz = 0.0;
    if (out_bmp) {
        timer_t t_viz;
        timer_start(&t_viz);
        status_t viz_st = visualize_save_frame_bmp(out_bmp, rgb_buffer, W, H, &results, 2);
        timer_stop(&t_viz);
        ms_viz = timer_elapsed_ms(&t_viz);
        if (viz_st != SUCCESS) {
            fprintf(stderr, "visualize_save_frame_bmp failed (%d)\n", (int)viz_st);
        }
    }

    printf("  timings (ms): inference=%.4f decode=%.4f", ms_inf, ms_dec);
    if (out_bmp) {
        printf(" visualize=%.4f", ms_viz);
    }
    printf("\n");

    printf("Detections: %d\n", results.count);
    for (int d = 0; d < results.count; d++) {
        detection_t* det = &results.detections[d];
        printf("  [%d] Class %d: %.2f @ (%.1f, %.1f, %.1f, %.1f)\n", d, det->class_id, det->score,
               det->x1, det->y1, det->x2, det->y2);
    }

    free(rgb_buffer);
    tensor_free(&input_tensor);
    tensor_free(&head_output);
    free(results.detections);
    model_destroy(&model);
    return 0;
}

int main(int argc, char** argv) {
    const int W = 640, H = 640;

    if (argc >= 2 && strcmp(argv[1], "--image") == 0) {
        if (argc < 3) {
            fprintf(stderr, "usage: %s --image <path> [annotated.bmp]\n", argv[0]);
            return 1;
        }
        if (argc > 4) {
            fprintf(stderr, "usage: %s --image <path> [annotated.bmp]\n", argv[0]);
            return 1;
        }
        const char* out_bmp = (argc >= 4) ? argv[3] : NULL;
        return run_image_mode(argv[2], out_bmp, W, H);
    }

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
    
    /* Postprocess layout [1, max_det, 6]: xyxy (pixels), score, class — filled by model_forward → detect. */
    tensor_t head_output;
    tensor_allocate(&head_output, 1, 300, 6, 1);
    
    detection_results_t results;
    results.capacity = 100;
    results.detections = malloc(sizeof(detection_t) * results.capacity);
    
    printf("Starting inference loop. Press Ctrl+C to stop.\n");
    const int save_bmp = argc >= 2;
    if (save_bmp) {
        printf("Saving annotated frames to %s (last frame wins).\n", argv[1]);
    }

    const int frames_total = 5;
    model_forward_profile_t layer_prof;
    model_forward_profile_reset(&layer_prof);

    for (int i = 0; i < frames_total; i++) {
        printf("\n--- Frame %d ---\n", i);

        timer_t t_cap;
        timer_start(&t_cap);
        camera_capture(cam, rgb_buffer);
        timer_stop(&t_cap);
        double ms_cap = timer_elapsed_ms(&t_cap);

        timer_t t_pre;
        timer_start(&t_pre);
        preprocess(&input_tensor, rgb_buffer, W, H);
        timer_stop(&t_pre);
        double ms_pre = timer_elapsed_ms(&t_pre);

        timer_t t_inf;
        timer_start(&t_inf);
        status_t inf_st = model_forward_ex(&model, &input_tensor, &head_output, NULL, &layer_prof);
        timer_stop(&t_inf);
        double ms_inf = timer_elapsed_ms(&t_inf);

        if (inf_st != SUCCESS) {
            fprintf(stderr, "model_forward failed (%d)\n", (int)inf_st);
            results.count = 0;
            printf("  timings (ms): capture=%.4f preprocess=%.4f inference=%.4f (failed)\n", ms_cap,
                   ms_pre, ms_inf);
            continue;
        }

        char layer_title[64];
        snprintf(layer_title, sizeof layer_title, "  model_forward steps (frame %d), ms:", i);
        model_forward_profile_print_last(&layer_prof, stdout, layer_title);

        timer_t t_dec;
        timer_start(&t_dec);
        decode_detections(&results, &head_output, 0.2f);
        timer_stop(&t_dec);
        double ms_dec = timer_elapsed_ms(&t_dec);

        double ms_viz = 0.0;
        if (save_bmp) {
            timer_t t_viz;
            timer_start(&t_viz);
            status_t viz_st = visualize_save_frame_bmp(argv[1], rgb_buffer, W, H, &results, 2);
            timer_stop(&t_viz);
            ms_viz = timer_elapsed_ms(&t_viz);
            if (viz_st != SUCCESS) {
                fprintf(stderr, "visualize_save_frame_bmp failed (%d)\n", (int)viz_st);
            }
        }

        printf("  timings (ms): capture=%.4f preprocess=%.4f inference=%.4f decode=%.4f", ms_cap,
               ms_pre, ms_inf, ms_dec);
        if (save_bmp) {
            printf(" visualize=%.4f", ms_viz);
        }
        printf("\n");

        printf("Detections: %d\n", results.count);
        for (int d = 0; d < results.count; d++) {
            detection_t* det = &results.detections[d];
            printf("  [%d] Class %d: %.2f @ (%.1f, %.1f, %.1f, %.1f)\n", 
                   d, det->class_id, det->score, det->x1, det->y1, det->x2, det->y2);
        }
    }

    model_forward_profile_print_aggregate(&layer_prof, stdout);

    camera_stop(cam);
    camera_destroy(cam);
    free(rgb_buffer);
    tensor_free(&input_tensor);
    tensor_free(&head_output);
    free(results.detections);
    
    return 0;
}
