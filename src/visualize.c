#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include "visualize.h"

static int clampi(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static void set_rgb(uint8_t* rgb, int w, int h, int x, int y, uint8_t r, uint8_t g, uint8_t b) {
    if (x < 0 || x >= w || y < 0 || y >= h) return;
    size_t i = ((size_t)y * (size_t)w + (size_t)x) * 3;
    rgb[i] = r;
    rgb[i + 1] = g;
    rgb[i + 2] = b;
}

/* Simple class-dependent hue: distinct colors for a few ids (COCO 80). */
static void class_color(int class_id, uint8_t* r, uint8_t* g, uint8_t* b) {
    static const uint8_t pal[][3] = {
        {255, 0, 0},   {0, 255, 0},   {0, 128, 255}, {255, 128, 0}, {255, 0, 255},
        {0, 255, 255}, {128, 0, 255}, {255, 255, 0}, {128, 255, 0}, {255, 192, 203},
    };
    int k = class_id % 10;
    *r = pal[k][0];
    *g = pal[k][1];
    *b = pal[k][2];
}

static void draw_rect_outline(uint8_t* rgb, int w, int h, int x1, int y1, int x2, int y2, int lw,
                              uint8_t cr, uint8_t cg, uint8_t cb) {
    if (x2 < x1) {
        int t = x1;
        x1 = x2;
        x2 = t;
    }
    if (y2 < y1) {
        int t = y1;
        y1 = y2;
        y2 = t;
    }
    x1 = clampi(x1, 0, w - 1);
    x2 = clampi(x2, 0, w - 1);
    y1 = clampi(y1, 0, h - 1);
    y2 = clampi(y2, 0, h - 1);
    if (lw < 1) lw = 1;

    for (int y = y1; y <= y2; y++) {
        for (int x = x1; x <= x2; x++) {
            int left = x < x1 + lw;
            int right = x > x2 - lw;
            int top = y < y1 + lw;
            int bottom = y > y2 - lw;
            if (left || right || top || bottom) set_rgb(rgb, w, h, x, y, cr, cg, cb);
        }
    }
}

void visualize_draw_boxes_rgb(uint8_t* rgb, int w, int h, const detection_results_t* results, int line_width) {
    if (!rgb || !results || w < 1 || h < 1) return;
    if (line_width < 1) line_width = 2;

    for (int i = 0; i < results->count; i++) {
        const detection_t* d = &results->detections[i];
        int x1 = (int)(d->x1 + 0.5f);
        int y1 = (int)(d->y1 + 0.5f);
        int x2 = (int)(d->x2 + 0.5f);
        int y2 = (int)(d->y2 + 0.5f);
        uint8_t r, g, b;
        class_color(d->class_id, &r, &g, &b);
        draw_rect_outline(rgb, w, h, x1, y1, x2, y2, line_width, r, g, b);
    }
}

status_t visualize_write_bmp_rgb24(const char* path, const uint8_t* rgb, int w, int h) {
    if (!path || !rgb || w < 1 || h < 1) return ERROR_NULL_POINTER;

    int row_padded = (w * 3 + 3) & ~3;
    int image_size = row_padded * h;
    unsigned int file_size = 54 + (unsigned int)image_size;

    FILE* f = fopen(path, "wb");
    if (!f) return ERROR_FILE_NOT_FOUND;

    unsigned char file_hdr[14] = {
        'B', 'M',
        (unsigned char)(file_size & 0xff), (unsigned char)((file_size >> 8) & 0xff),
        (unsigned char)((file_size >> 16) & 0xff), (unsigned char)((file_size >> 24) & 0xff),
        0, 0, 0, 0,
        54, 0, 0, 0,
    };

    int32_t dib_w = w;
    int32_t dib_h = -h;
    unsigned char info_hdr[40] = {
        40, 0, 0, 0,
        (unsigned char)(dib_w & 0xff), (unsigned char)((dib_w >> 8) & 0xff),
        (unsigned char)((dib_w >> 16) & 0xff), (unsigned char)((dib_w >> 24) & 0xff),
        (unsigned char)(dib_h & 0xff), (unsigned char)((dib_h >> 8) & 0xff),
        (unsigned char)((dib_h >> 16) & 0xff), (unsigned char)((dib_h >> 24) & 0xff),
        1, 0,
        24, 0,
        0, 0, 0, 0,
        (unsigned char)(image_size & 0xff), (unsigned char)((image_size >> 8) & 0xff),
        (unsigned char)((image_size >> 16) & 0xff), (unsigned char)((image_size >> 24) & 0xff),
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };

    if (fwrite(file_hdr, 1, 14, f) != 14 || fwrite(info_hdr, 1, 40, f) != 40) {
        fclose(f);
        return ERROR_FILE_NOT_FOUND;
    }

    for (int y = 0; y < h; y++) {
        const uint8_t* row = rgb + (size_t)y * (size_t)w * 3;
        for (int x = 0; x < w; x++) {
            uint8_t rr = row[x * 3 + 0], gg = row[x * 3 + 1], bb = row[x * 3 + 2];
            if (fputc(bb, f) == EOF || fputc(gg, f) == EOF || fputc(rr, f) == EOF) {
                fclose(f);
                return ERROR_FILE_NOT_FOUND;
            }
        }
        for (int p = w * 3; p < row_padded; p++) {
            if (fputc(0, f) == EOF) {
                fclose(f);
                return ERROR_FILE_NOT_FOUND;
            }
        }
    }

    fclose(f);
    return SUCCESS;
}

status_t visualize_save_frame_bmp(const char* path, const uint8_t* rgb, int w, int h,
                                  const detection_results_t* results, int line_width) {
    if (!path || !rgb || !results || w < 1 || h < 1) return ERROR_NULL_POINTER;

    size_t n = (size_t)w * (size_t)h * 3;
    uint8_t* copy = (uint8_t*)malloc(n);
    if (!copy) return ERROR_OUT_OF_MEMORY;
    memcpy(copy, rgb, n);
    visualize_draw_boxes_rgb(copy, w, h, results, line_width);
    status_t st = visualize_write_bmp_rgb24(path, copy, w, h);
    free(copy);
    return st;
}
