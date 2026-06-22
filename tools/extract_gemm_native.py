#!/usr/bin/env python3
"""One-off: extract GEMM from tensor.c into gemm_native.c."""
from pathlib import Path

src = Path("src/tensor.c").read_text(encoding="utf-8")
start = src.index("static float tensor_max_abs_f32")
end = src.index("const char* tensor_gemm_int8_backend")
end = src.index("}", end) + 1
chunk = src[start:end]
chunk = chunk.replace("tensor_max_abs_f32", "gemm_max_abs_f32")
chunk = chunk.replace("status_t tensor_gemm(", "status_t gemm_native_fp32(")
chunk = chunk.replace("status_t tensor_gemm_int8(", "status_t gemm_native_int8(")
chunk = chunk.replace("status_t tensor_gemm_weight_int8(", "status_t gemm_native_weight_int8(")
chunk = chunk.replace("tensor_gemm_int8(C, W, xq", "gemm_native_int8(C, W, xq")
chunk = chunk.replace(
    "const char* tensor_gemm_int8_backend(void)",
    "const char* gemm_native_int8_backend_name(void)",
)

header = """\
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "platform.h"
#include "tensor.h"
#include "gemm_native.h"

#ifdef _MSC_VER
#include <malloc.h>
#endif

#ifdef USE_OPENBLAS
#if defined(__has_include)
#if __has_include(<cblas.h>)
#include <cblas.h>
#elif __has_include(<openblas/cblas.h>)
#include <openblas/cblas.h>
#else
#include <cblas.h>
#endif
#else
#include <cblas.h>
#endif
#endif

"""

Path("src/gemm_native.c").write_text(header + chunk, encoding="utf-8")
print("wrote src/gemm_native.c", len(header + chunk), "bytes")
