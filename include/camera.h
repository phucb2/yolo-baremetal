#ifndef CAMERA_H
#define CAMERA_H

#include <stdint.h>
#include "status.h"

typedef struct camera camera_t;

status_t camera_create(camera_t** cam, int width, int height);
status_t camera_destroy(camera_t* cam);
status_t camera_start(camera_t* cam);
status_t camera_stop(camera_t* cam);
status_t camera_capture(camera_t* cam, uint8_t* buffer);

#endif
