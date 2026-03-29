#ifndef VISUALIZE_H
#define VISUALIZE_H

#include <stdint.h>
#include "detection.h"
#include "status.h"

/* Draw detection boxes on RGB interleaved buffer [H*W*3], row-major R,G,B per pixel. */
void visualize_draw_boxes_rgb(uint8_t* rgb, int w, int h, const detection_results_t* results, int line_width);

/* Write buffer as 24-bit uncompressed BMP (top-down). */
status_t visualize_write_bmp_rgb24(const char* path, const uint8_t* rgb, int w, int h);

/* Copy rgb, draw boxes, write BMP. Does not modify original rgb. */
status_t visualize_save_frame_bmp(const char* path, const uint8_t* rgb, int w, int h,
                                  const detection_results_t* results, int line_width);

#endif
