#ifndef PLATFORM_H
#define PLATFORM_H

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64)
#define YOLO_ARCH_X64 1
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#define YOLO_ARCH_ARM64 1
#endif

#if defined(_MSC_VER)
#define YOLO_ALIGNED(n) __declspec(align(n))
#define YOLO_UNUSED
#define YOLO_GCC_TARGET(...)
#else
#define YOLO_ALIGNED(n) __attribute__((aligned(n)))
#define YOLO_UNUSED __attribute__((unused))
#define YOLO_GCC_TARGET(...) __attribute__((target(__VA_ARGS__)))
#endif

#if YOLO_ARCH_X64
#include <immintrin.h>
#endif
#if YOLO_ARCH_ARM64
#include <arm_neon.h>
#endif

#endif
