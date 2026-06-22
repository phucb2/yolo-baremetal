cmake_minimum_required(VERSION 3.20)
project(yolo-baremetal C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

option(USE_INT8 "Enable INT8 inference" ON)
option(USE_OPENBLAS "Link OpenBLAS for FP32 GEMM" OFF)
option(USE_ONEDNN "Route FP32/INT8 conv GEMM through oneDNN (yolo_core)" OFF)
option(USE_CUDNN "Enable cuDNN conv2d path in playground" OFF)
option(GEMM_PREFER_AVX512_VNNI "Prefer AVX512-VNNI over AVX-VNNI" OFF)

if(APPLE)
    set(_use_camera_default ON)
else()
    set(_use_camera_default OFF)
endif()
option(USE_CAMERA "Enable live camera capture (macOS AVFoundation)" ${_use_camera_default})

if(USE_INT8)
    add_compile_definitions(USE_INT8=1)
endif()
if(USE_CAMERA)
    add_compile_definitions(YOLO_USE_CAMERA)
endif()
if(GEMM_PREFER_AVX512_VNNI)
    add_compile_definitions(GEMM_PREFER_AVX512_VNNI)
endif()

if(MSVC)
    add_compile_options(/W4 /O2 /std:c11)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8 AND NOT CMAKE_GENERATOR_PLATFORM MATCHES "ARM")
        add_compile_options(/arch:AVX2)
    endif()
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(-O3 -Wall -Wextra)
    if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
        add_compile_options(-mcpu=apple-m1)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
        add_compile_options(-mavx2 -mfma)
        if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
            add_compile_options(-march=native)
        endif()
    endif()
endif()

include_directories(${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/third_party)

if(USE_OPENBLAS AND USE_ONEDNN)
    message(WARNING "USE_OPENBLAS and USE_ONEDNN are both ON; FP32 GEMM uses oneDNN (OpenBLAS only affects tiny-shape fallback).")
endif()

set(YOLO_CORE_SOURCES
    src/tensor.c
    src/gemm_backend.c
    src/gemm_native.c
    src/utils.c
    src/layers.c
    src/detection.c
    src/detect.c
    src/model.c
    src/visualize.c
)

add_library(yolo_core STATIC ${YOLO_CORE_SOURCES})

if(USE_ONEDNN)
    find_package(dnnl CONFIG REQUIRED)
    target_sources(yolo_core PRIVATE src/gemm_onednn.c)
    target_compile_definitions(yolo_core PUBLIC USE_ONEDNN=1)
    target_link_libraries(yolo_core PUBLIC DNNL::dnnl)
endif()

if(USE_CUDNN)
    find_package(CUDAToolkit REQUIRED)
    find_package(CUDNN REQUIRED)
endif()

if(USE_OPENBLAS)
    if(DEFINED OPENBLAS_ROOT)
        list(APPEND CMAKE_PREFIX_PATH ${OPENBLAS_ROOT})
    endif()
    find_path(OPENBLAS_INCLUDE_DIR NAMES cblas.h openblas/cblas.h
        HINTS ${OPENBLAS_ROOT}/include
        PATH_SUFFIXES openblas)
    find_library(OPENBLAS_LIBRARY NAMES openblas openblas64
        HINTS ${OPENBLAS_ROOT}/lib ${OPENBLAS_ROOT}/lib64)
    if(NOT OPENBLAS_INCLUDE_DIR OR NOT OPENBLAS_LIBRARY)
        message(FATAL_ERROR "OpenBLAS not found. Set OPENBLAS_ROOT (e.g. vcpkg install openblas:x64-windows).")
    endif()
    target_compile_definitions(yolo_core PUBLIC USE_OPENBLAS)
    target_include_directories(yolo_core PUBLIC ${OPENBLAS_INCLUDE_DIR})
    set(YOLO_BLAS_LIBS ${OPENBLAS_LIBRARY})
else()
    set(YOLO_BLAS_LIBS)
endif()

if(USE_CAMERA AND APPLE)
    enable_language(OBJC)
    add_library(yolo_camera STATIC src/camera_darwin.m)
    target_compile_options(yolo_camera PRIVATE -fobjc-arc)
    set(YOLO_CAMERA_LIBS yolo_camera)
else()
    add_library(yolo_camera STATIC src/camera_stub.c)
    set(YOLO_CAMERA_LIBS yolo_camera)
endif()

function(yolo_link_math target)
    if(NOT WIN32)
        target_link_libraries(${target} PRIVATE m)
    endif()
endfunction()

function(yolo_link_core target)
    target_link_libraries(${target} PRIVATE yolo_core ${YOLO_BLAS_LIBS})
    yolo_link_math(${target})
endfunction()

add_executable(yolo26_bench src/main.c)
target_link_libraries(yolo26_bench PRIVATE yolo_core ${YOLO_CAMERA_LIBS} ${YOLO_BLAS_LIBS})
yolo_link_math(yolo26_bench)
if(USE_CAMERA AND APPLE)
    target_link_libraries(yolo26_bench PRIVATE
        "-framework Foundation"
        "-framework AVFoundation"
        "-framework CoreVideo"
        "-framework CoreMedia")
endif()

add_executable(test_core tests/test_core.c)
yolo_link_core(test_core)

add_executable(verify_layers tests/verify_layers.c)
yolo_link_core(verify_layers)

add_executable(bench_gemm tests/bench_gemm.c)
target_link_libraries(bench_gemm PRIVATE yolo_core ${YOLO_BLAS_LIBS})
yolo_link_math(bench_gemm)

if(USE_ONEDNN)
    add_executable(bench_gemm_compare tests/bench_gemm_compare.c)
    target_link_libraries(bench_gemm_compare PRIVATE yolo_core ${YOLO_BLAS_LIBS})
    yolo_link_math(bench_gemm_compare)
endif()

add_executable(playground src/playground.c)
target_link_libraries(playground PRIVATE yolo_core ${YOLO_BLAS_LIBS})
yolo_link_math(playground)
if(USE_CUDNN)
    target_compile_definitions(playground PRIVATE USE_CUDNN=1)
    target_link_libraries(playground PRIVATE CUDA::cudart CuDNN::CuDNN)
endif()

add_executable(bench_coco8 tests/bench_coco8.c)
yolo_link_core(bench_coco8)

enable_testing()
add_test(NAME test_core COMMAND test_core)
add_test(NAME verify_layers COMMAND verify_layers)
