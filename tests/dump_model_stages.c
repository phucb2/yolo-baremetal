/*
 * Dump full-model stage tensors to a single .bin (same format as tools/generate_layer_tests.py).
 * Default weights: weights/yolo26.bin (export from yolo26n.pt via tools/converter.py — same checkpoint as Python dump).
 *
 * argv[1] may be an image (jpg/png/...) or a single-tensor .bin from:
 *   python tools/dump_model_stages.py --write-shared-input runs/shared_input.bin --image tests/data/zidane.jpg
 * so C and Python use identical stage_00_input (OpenCV preprocess on the Python side).
 *
 * Build (from repo root, after `make` produces CORE objects):
 *   $(CC) $(CFLAGS) -DNO_LOGGING -Iinclude -Ithird_party tests/dump_model_stages.c \
 *     build/tensor.o build/utils.o build/layers.o build/detection.o build/detect.o build/model.o -o tests/dump_model_stages -lm
 */

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"
#include "tensor.h"
#include "utils.h"

static int path_ends_with_bin(const char* path) {
    size_t n = strlen(path);
    return n >= 4 && strcmp(path + n - 4, ".bin") == 0;
}

static status_t load_input_from_tensor_bin(const char* path, tensor_t* out_input) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return ERROR_FILE_NOT_FOUND;
    char name[256];
    status_t st = load_named_tensor(fp, name, out_input);
    fclose(fp);
    if (st != SUCCESS) return st;
    if (out_input->dims[0] != 1 || out_input->dims[1] != 3 || out_input->dims[2] != 640 ||
        out_input->dims[3] != 640) {
        tensor_free(out_input);
        return ERROR_INVALID_FORMAT;
    }
    return SUCCESS;
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

static uint8_t* load_rgb_resized(const char* path, int out_w, int out_h) {
    int iw = 0, ih = 0, n = 0;
    unsigned char* img = stbi_load(path, &iw, &ih, &n, 3);
    if (!img) {
        fprintf(stderr, "stbi_load failed: %s\n", path);
        return NULL;
    }
    uint8_t* out = malloc((size_t)out_w * out_h * 3);
    if (!out) {
        stbi_image_free(img);
        return NULL;
    }
    for (int y = 0; y < out_h; y++) {
        for (int x = 0; x < out_w; x++) {
            float fx = (x + 0.5f) * (float)iw / (float)out_w - 0.5f;
            float fy = (y + 0.5f) * (float)ih / (float)out_h - 0.5f;
            int sx = (int)fx;
            int sy = (int)fy;
            if (sx < 0) sx = 0;
            if (sy < 0) sy = 0;
            if (sx >= iw) sx = iw - 1;
            if (sy >= ih) sy = ih - 1;
            int sx1 = sx + 1 < iw ? sx + 1 : sx;
            int sy1 = sy + 1 < ih ? sy + 1 : sy;
            float tx = fx - (float)sx;
            float ty = fy - (float)sy;
            for (int c = 0; c < 3; c++) {
                float v00 = img[(sy * iw + sx) * 3 + c];
                float v10 = img[(sy * iw + sx1) * 3 + c];
                float v01 = img[(sy1 * iw + sx) * 3 + c];
                float v11 = img[(sy1 * iw + sx1) * 3 + c];
                float v = v00 * (1 - tx) * (1 - ty) + v10 * tx * (1 - ty) + v01 * (1 - tx) * ty + v11 * tx * ty;
                out[(y * out_w + x) * 3 + c] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
            }
        }
    }
    stbi_image_free(img);
    return out;
}

int main(int argc, char** argv) {
    const int W = 640, H = 640;
    const char* weights = "weights/yolo26.bin";
    const char* image_path = "tests/data/zidane.jpg";
    const char* out_path = "runs/zidane_c_stages.bin";

    if (argc >= 2) image_path = argv[1];
    if (argc >= 3) out_path = argv[2];
    if (argc >= 4) weights = argv[3];

    model_t model;
    tensor_t input_tensor;
    memset(&input_tensor, 0, sizeof(input_tensor));

    int use_bin_input = path_ends_with_bin(image_path);
    uint8_t* rgb = NULL;

    if (use_bin_input) {
        if (model_create(&model, W, H) != SUCCESS) {
            fprintf(stderr, "model_create failed\n");
            return 1;
        }
        if (load_input_from_tensor_bin(image_path, &input_tensor) != SUCCESS) {
            fprintf(stderr, "load_input_from_tensor_bin failed: %s\n", image_path);
            model_destroy(&model);
            return 1;
        }
    } else {
        rgb = load_rgb_resized(image_path, W, H);
        if (!rgb) return 1;
        if (model_create(&model, W, H) != SUCCESS) {
            free(rgb);
            fprintf(stderr, "model_create failed\n");
            return 1;
        }
        if (tensor_allocate(&input_tensor, 1, 3, H, W) != SUCCESS) {
            free(rgb);
            model_destroy(&model);
            return 1;
        }
        preprocess(&input_tensor, rgb, W, H);
        free(rgb);
        rgb = NULL;
    }

    if (model_load_weights(&model, weights) != SUCCESS) {
        tensor_free(&input_tensor);
        model_destroy(&model);
        fprintf(stderr, "model_load_weights failed\n");
        return 1;
    }

    tensor_t head_output;
    if (tensor_allocate(&head_output, 1, 300, 6, 1) != SUCCESS) {
        tensor_free(&input_tensor);
        model_destroy(&model);
        return 1;
    }

    FILE* out = fopen(out_path, "wb");
    if (!out) {
        fprintf(stderr, "fopen %s failed\n", out_path);
        tensor_free(&input_tensor);
        tensor_free(&head_output);
        model_destroy(&model);
        return 1;
    }

    status_t st = model_forward_ex(&model, &input_tensor, &head_output, out, NULL);
    fclose(out);

    tensor_free(&input_tensor);
    tensor_free(&head_output);
    model_destroy(&model);

    if (st != SUCCESS) {
        fprintf(stderr, "model_forward_ex failed (%d)\n", (int)st);
        return 1;
    }
    printf("Wrote %s\n", out_path);
    return 0;
}
