/*
 * Benchmark C inference on a set of images with model loaded once.
 * Build: make bench-coco8
 * Usage: ./tests/bench_coco8 --weights weights/yolo26_int8.bin --runs 5 --warmup 2 img1.jpg img2.jpg ...
 */

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "detection.h"
#include "model.h"
#include "tensor.h"
#include "utils.h"

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

static void preprocess(tensor_t* input_tensor, const uint8_t* rgb_buffer, int w, int h) {
    float* data = input_tensor->data;
    int c_stride = h * w;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            const uint8_t* pixel = rgb_buffer + (y * w + x) * 3;
            data[0 * c_stride + y * w + x] = pixel[0] / 255.0f;
            data[1 * c_stride + y * w + x] = pixel[1] / 255.0f;
            data[2 * c_stride + y * w + x] = pixel[2] / 255.0f;
        }
    }
}

typedef struct {
    char* path;
} image_path_t;

static int path_has_image_suffix(const char* path) {
    const char* dot = strrchr(path, '.');
    if (!dot) {
        return 0;
    }
    return strcmp(dot, ".jpg") == 0 || strcmp(dot, ".jpeg") == 0 || strcmp(dot, ".png") == 0 ||
           strcmp(dot, ".bmp") == 0 || strcmp(dot, ".webp") == 0;
}

static int collect_images_from_dir(const char* dir_path, image_path_t** out_paths, int* out_count) {
    DIR* dir = opendir(dir_path);
    if (!dir) {
        return 0;
    }
    int cap = 16;
    int count = 0;
    image_path_t* paths = (image_path_t*)malloc((size_t)cap * sizeof(image_path_t));
    if (!paths) {
        closedir(dir);
        return 0;
    }
    struct dirent* ent;
    while ((ent = readdir(dir)) != NULL) {
        if (ent->d_name[0] == '.') {
            continue;
        }
        char full[4096];
        snprintf(full, sizeof full, "%s/%s", dir_path, ent->d_name);
        if (!path_has_image_suffix(full)) {
            continue;
        }
        if (count >= cap) {
            cap *= 2;
            image_path_t* grown = (image_path_t*)realloc(paths, (size_t)cap * sizeof(image_path_t));
            if (!grown) {
                for (int i = 0; i < count; i++) {
                    free(paths[i].path);
                }
                free(paths);
                closedir(dir);
                return 0;
            }
            paths = grown;
        }
        paths[count].path = strdup(full);
        if (!paths[count].path) {
            for (int i = 0; i < count; i++) {
                free(paths[i].path);
            }
            free(paths);
            closedir(dir);
            return 0;
        }
        count++;
    }
    closedir(dir);
    *out_paths = paths;
    *out_count = count;
    return 1;
}

static int cmp_path(const void* a, const void* b) {
    const image_path_t* pa = (const image_path_t*)a;
    const image_path_t* pb = (const image_path_t*)b;
    return strcmp(pa->path, pb->path);
}

static void free_paths(image_path_t* paths, int count) {
    if (!paths) {
        return;
    }
    for (int i = 0; i < count; i++) {
        free(paths[i].path);
    }
    free(paths);
}

static double mean_ms(const double* xs, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) {
        s += xs[i];
    }
    return n > 0 ? s / (double)n : 0.0;
}

int main(int argc, char** argv) {
    const int W = 640, H = 640;
    const char* weights_path = "weights/yolo26_int8.bin";
    float conf_threshold = 0.001f;
    int runs = 5;
    int warmup = 2;

    image_path_t* image_paths = NULL;
    int image_count = 0;

    for (int a = 1; a < argc; a++) {
        if (strcmp(argv[a], "--weights") == 0) {
            if (a + 1 >= argc) {
                fprintf(stderr, "--weights requires a path\n");
                return 1;
            }
            weights_path = argv[++a];
        } else if (strcmp(argv[a], "--conf") == 0) {
            if (a + 1 >= argc) {
                fprintf(stderr, "--conf requires a float\n");
                return 1;
            }
            conf_threshold = (float)atof(argv[++a]);
        } else if (strcmp(argv[a], "--runs") == 0) {
            if (a + 1 >= argc) {
                fprintf(stderr, "--runs requires an int\n");
                return 1;
            }
            runs = atoi(argv[++a]);
        } else if (strcmp(argv[a], "--warmup") == 0) {
            if (a + 1 >= argc) {
                fprintf(stderr, "--warmup requires an int\n");
                return 1;
            }
            warmup = atoi(argv[++a]);
        } else if (strcmp(argv[a], "--dir") == 0) {
            if (a + 1 >= argc) {
                fprintf(stderr, "--dir requires a path\n");
                return 1;
            }
            image_path_t* dir_paths = NULL;
            int dir_count = 0;
            if (!collect_images_from_dir(argv[++a], &dir_paths, &dir_count)) {
                fprintf(stderr, "Failed to read image dir: %s\n", argv[a]);
                return 1;
            }
            image_paths = (image_path_t*)realloc(image_paths, (size_t)(image_count + dir_count) * sizeof(image_path_t));
            if (!image_paths) {
                free_paths(dir_paths, dir_count);
                return 1;
            }
            for (int i = 0; i < dir_count; i++) {
                image_paths[image_count++] = dir_paths[i];
            }
            free(dir_paths);
        } else if (argv[a][0] != '-') {
            image_paths = (image_path_t*)realloc(image_paths, (size_t)(image_count + 1) * sizeof(image_path_t));
            if (!image_paths) {
                return 1;
            }
            image_paths[image_count].path = strdup(argv[a]);
            if (!image_paths[image_count].path) {
                return 1;
            }
            image_count++;
        } else {
            fprintf(stderr,
                    "usage: %s [--weights w.bin] [--conf thr] [--runs N] [--warmup N] [--dir path] [images...]\n",
                    argv[0]);
            free_paths(image_paths, image_count);
            return 1;
        }
    }

    if (image_count == 0) {
        fprintf(stderr, "No images provided (use paths or --dir)\n");
        free_paths(image_paths, image_count);
        return 1;
    }
    qsort(image_paths, (size_t)image_count, sizeof(image_path_t), cmp_path);

    model_t model;
    model_create(&model, W, H);

    timer_t t_load;
    timer_start(&t_load);
    if (model_load_weights(&model, weights_path) != SUCCESS) {
        fprintf(stderr, "model_load_weights failed: %s\n", weights_path);
        model_destroy(&model);
        free_paths(image_paths, image_count);
        return 1;
    }
    timer_stop(&t_load);
    double ms_load = timer_elapsed_ms(&t_load);

    tensor_t input_tensor;
    tensor_allocate(&input_tensor, 1, 3, H, W);
    tensor_t head_output;
    tensor_allocate(&head_output, 1, 300, 6, 1);

    detection_results_t results;
    results.capacity = 100;
    results.detections = malloc(sizeof(detection_t) * results.capacity);
    if (!results.detections) {
        tensor_free(&input_tensor);
        tensor_free(&head_output);
        model_destroy(&model);
        free_paths(image_paths, image_count);
        return 1;
    }

    uint8_t* rgb_buffer = malloc((size_t)W * H * 3);
    if (!rgb_buffer) {
        free(results.detections);
        tensor_free(&input_tensor);
        tensor_free(&head_output);
        model_destroy(&model);
        free_paths(image_paths, image_count);
        return 1;
    }

    const int total_iters = (warmup + runs) * image_count;
    double* ms_load_img = calloc((size_t)total_iters, sizeof(double));
    double* ms_resize = calloc((size_t)total_iters, sizeof(double));
    double* ms_pre = calloc((size_t)total_iters, sizeof(double));
    double* ms_inf = calloc((size_t)total_iters, sizeof(double));
    double* ms_dec = calloc((size_t)total_iters, sizeof(double));
    if (!ms_load_img || !ms_resize || !ms_pre || !ms_inf || !ms_dec) {
        free(ms_load_img);
        free(ms_resize);
        free(ms_pre);
        free(ms_inf);
        free(ms_dec);
        free(rgb_buffer);
        free(results.detections);
        tensor_free(&input_tensor);
        tensor_free(&head_output);
        model_destroy(&model);
        free_paths(image_paths, image_count);
        return 1;
    }

    int iter = 0;
    for (int r = 0; r < warmup + runs; r++) {
        for (int i = 0; i < image_count; i++) {
            const char* img_path = image_paths[i].path;

            timer_t t0;
            timer_start(&t0);
            int iw = 0, ih = 0, ic = 0;
            unsigned char* raw = stbi_load(img_path, &iw, &ih, &ic, 3);
            timer_stop(&t0);
            ms_load_img[iter] = timer_elapsed_ms(&t0);
            if (!raw || iw <= 0 || ih <= 0) {
                fprintf(stderr, "Failed to decode: %s\n", img_path);
                goto cleanup;
            }

            timer_start(&t0);
            resize_rgb_bilinear(raw, iw, ih, rgb_buffer, W, H);
            timer_stop(&t0);
            ms_resize[iter] = timer_elapsed_ms(&t0);
            stbi_image_free(raw);

            timer_start(&t0);
            preprocess(&input_tensor, rgb_buffer, W, H);
            timer_stop(&t0);
            ms_pre[iter] = timer_elapsed_ms(&t0);

            timer_start(&t0);
            status_t inf_st = model_forward_ex(&model, &input_tensor, &head_output, NULL, NULL);
            timer_stop(&t0);
            ms_inf[iter] = timer_elapsed_ms(&t0);
            if (inf_st != SUCCESS) {
                fprintf(stderr, "model_forward failed on %s\n", img_path);
                goto cleanup;
            }

            timer_start(&t0);
            decode_detections(&results, &head_output, conf_threshold);
            timer_stop(&t0);
            ms_dec[iter] = timer_elapsed_ms(&t0);

            iter++;
        }
    }

    const int measured = runs * image_count;
    const int skip = warmup * image_count;
    printf("bench_coco8: images=%d runs=%d warmup=%d input=%dx%d conf=%.4g\n", image_count, runs, warmup, W, H,
           conf_threshold);
    printf("model_load_weights (once): %.3f ms\n", ms_load);
    printf("per-image mean (excluding warmup):\n");
    printf("  load_image:   %.3f ms\n", mean_ms(ms_load_img + skip, measured));
    printf("  resize:       %.3f ms\n", mean_ms(ms_resize + skip, measured));
    printf("  preprocess:   %.3f ms\n", mean_ms(ms_pre + skip, measured));
    printf("  inference:    %.3f ms\n", mean_ms(ms_inf + skip, measured));
    printf("  decode:       %.3f ms\n", mean_ms(ms_dec + skip, measured));
    printf("  pipeline:     %.3f ms\n",
           mean_ms(ms_load_img + skip, measured) + mean_ms(ms_resize + skip, measured) +
               mean_ms(ms_pre + skip, measured) + mean_ms(ms_inf + skip, measured) + mean_ms(ms_dec + skip, measured));

cleanup:
    free(ms_load_img);
    free(ms_resize);
    free(ms_pre);
    free(ms_inf);
    free(ms_dec);
    free(rgb_buffer);
    free(results.detections);
    tensor_free(&input_tensor);
    tensor_free(&head_output);
    model_destroy(&model);
    free_paths(image_paths, image_count);
    return 0;
}
