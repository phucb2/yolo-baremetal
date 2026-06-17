#include "camera.h"

status_t camera_create(camera_t** cam, int width, int height) {
    (void)cam;
    (void)width;
    (void)height;
    return ERROR_NOT_IMPLEMENTED;
}

status_t camera_destroy(camera_t* cam) {
    (void)cam;
    return SUCCESS;
}

status_t camera_start(camera_t* cam) {
    (void)cam;
    return ERROR_NOT_IMPLEMENTED;
}

status_t camera_stop(camera_t* cam) {
    (void)cam;
    return SUCCESS;
}

status_t camera_capture(camera_t* cam, uint8_t* buffer) {
    (void)cam;
    (void)buffer;
    return ERROR_NOT_IMPLEMENTED;
}
