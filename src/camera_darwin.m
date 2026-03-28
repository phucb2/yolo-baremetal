#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>
#import <CoreVideo/CoreVideo.h>
#include <unistd.h>
#include "camera.h"

@interface CameraDelegate : NSObject <AVCaptureVideoDataOutputSampleBufferDelegate>
@property (assign) uint8_t* buffer;
@property (assign) int width;
@property (assign) int height;
@property (assign) BOOL hasFrame;
@end

@implementation CameraDelegate
- (void)captureOutput:(AVCaptureOutput *)output didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection {
    CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    CVPixelBufferLockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);
    
    uint8_t *baseAddress = (uint8_t *)CVPixelBufferGetBaseAddress(imageBuffer);
    size_t bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer);
    size_t width = CVPixelBufferGetWidth(imageBuffer);
    size_t height = CVPixelBufferGetHeight(imageBuffer);
    
    // Convert BGRA to RGB if needed, or just copy first 3 channels
    // For simplicity, let's assume we want RGB
    for (int y = 0; y < MIN(height, self.height); y++) {
        for (int x = 0; x < MIN(width, self.width); x++) {
            uint8_t *src = baseAddress + y * bytesPerRow + x * 4;
            uint8_t *dst = self.buffer + (y * self.width + x) * 3;
            dst[0] = src[2]; // R
            dst[1] = src[1]; // G
            dst[2] = src[0]; // B
        }
    }
    
    self.hasFrame = YES;
    CVPixelBufferUnlockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);
}
@end

struct camera {
    AVCaptureSession *session;
    CameraDelegate *delegate;
    int width, height;
};

status_t camera_create(camera_t** cam, int width, int height) {
    *cam = malloc(sizeof(struct camera));
    (*cam)->width = width;
    (*cam)->height = height;
    (*cam)->session = [[AVCaptureSession alloc] init];
    (*cam)->delegate = [[CameraDelegate alloc] init];
    (*cam)->delegate.buffer = NULL;
    (*cam)->delegate.width = width;
    (*cam)->delegate.height = height;
    
    AVCaptureDevice *device = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
    NSError *error = nil;
    AVCaptureDeviceInput *input = [AVCaptureDeviceInput deviceInputWithDevice:device error:&error];
    if (!input) return ERROR_FILE_NOT_FOUND;
    
    [(*cam)->session addInput:input];
    
    AVCaptureVideoDataOutput *output = [[AVCaptureVideoDataOutput alloc] init];
    dispatch_queue_t queue = dispatch_queue_create("cameraQueue", NULL);
    [output setSampleBufferDelegate:(*cam)->delegate queue:queue];
    
    // Configure output format: 32BGRA
    output.videoSettings = @{(id)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA)};
    
    [(*cam)->session addOutput:output];
    return SUCCESS;
}

status_t camera_destroy(camera_t* cam) {
    if (!cam) return ERROR_NULL_POINTER;
    [cam->session stopRunning];
    free(cam);
    return SUCCESS;
}

status_t camera_start(camera_t* cam) {
    [cam->session startRunning];
    return SUCCESS;
}

status_t camera_stop(camera_t* cam) {
    [cam->session stopRunning];
    return SUCCESS;
}

status_t camera_capture(camera_t* cam, uint8_t* buffer) {
    cam->delegate.buffer = buffer;
    cam->delegate.hasFrame = NO;
    
    // Busy wait for frame (not ideal but simple for this project)
    int timeout = 1000000;
    while (!cam->delegate.hasFrame && timeout-- > 0) {
        usleep(1);
    }
    
    return cam->delegate.hasFrame ? SUCCESS : ERROR_FILE_NOT_FOUND;
}
