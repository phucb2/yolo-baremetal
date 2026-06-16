# Python tools (torch/ultralytics): conda env py39 or tools/with_py39.sh

CC = clang
ifeq ($(shell uname -s),Darwin)
  XCRUN := $(shell xcrun -f clang 2>/dev/null)
  ifneq ($(XCRUN),)
    CC := $(XCRUN)
  endif
endif

TARGET = yolo26_bench
BUILD_DIR = build
CFLAGS = -O3 -Iinclude -Ithird_party -Wall -Wextra -std=c11
LDFLAGS = -framework Foundation -framework AVFoundation -framework CoreVideo -framework CoreMedia -lm

# Without -isysroot, clang may not find system headers (e.g. stdlib.h) while parsing intrinsics.
ifeq ($(shell uname -s),Darwin)
  MACOS_SDK := $(shell xcrun --show-sdk-path 2>/dev/null)
  ifneq ($(MACOS_SDK),)
    CFLAGS += -isysroot $(MACOS_SDK)
    LDFLAGS += -isysroot $(MACOS_SDK)
  endif
endif

UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),arm64)
	CFLAGS += -mcpu=apple-m1
endif
ifeq ($(UNAME_M),x86_64)
	CFLAGS += -mavx2 -mfma -march=native
endif

# OpenBLAS: USE_OPENBLAS=1. Set OPENBLAS_PREFIX if headers/libs are not found (e.g. brew install openblas).
# Prefix: try `brew --prefix openblas`, then arm64 / Intel Homebrew defaults.
USE_OPENBLAS ?= 0
TENSOR_OBJ = $(BUILD_DIR)/tensor_u$(USE_OPENBLAS).o
BLAS_LDFLAGS :=
ifeq ($(USE_OPENBLAS),1)
	CFLAGS += -DUSE_OPENBLAS
	OPENBLAS_PREFIX ?= $(shell brew --prefix openblas 2>/dev/null)
	ifeq ($(OPENBLAS_PREFIX),)
		ifeq ($(UNAME_M),arm64)
			OPENBLAS_PREFIX := /opt/homebrew/opt/openblas
		else
			OPENBLAS_PREFIX := /usr/local/opt/openblas
		endif
	endif
	CFLAGS += -I$(OPENBLAS_PREFIX)/include
	# Some Homebrew layouts only expose cblas.h here:
	ifneq ($(wildcard $(OPENBLAS_PREFIX)/include/openblas),)
		CFLAGS += -I$(OPENBLAS_PREFIX)/include/openblas
	endif
	BLAS_LDFLAGS := -L$(OPENBLAS_PREFIX)/lib -lopenblas
endif

OBJ = $(TENSOR_OBJ) $(BUILD_DIR)/utils.o $(BUILD_DIR)/layers.o $(BUILD_DIR)/detection.o $(BUILD_DIR)/detect.o $(BUILD_DIR)/model.o $(BUILD_DIR)/visualize.o $(BUILD_DIR)/main.o $(BUILD_DIR)/camera_darwin.o
CORE_OBJ = $(TENSOR_OBJ) $(BUILD_DIR)/utils.o $(BUILD_DIR)/layers.o $(BUILD_DIR)/detection.o $(BUILD_DIR)/detect.o $(BUILD_DIR)/model.o
TEST_CORE = tests/test_core
VERIFY_LAYERS = tests/verify_layers
BENCH_GEMM = tests/bench_gemm
PT_MODEL ?= weights/yolo26n.pt
C_WEIGHTS ?= weights/yolo26.bin
PYTHON_EVAL ?= uv run python

$(TARGET): $(OBJ)
	$(CC) $(OBJ) -o $(TARGET) $(LDFLAGS) $(BLAS_LDFLAGS)

$(TEST_CORE): tests/test_core.c $(CORE_OBJ)
	$(CC) $(CFLAGS) tests/test_core.c $(CORE_OBJ) -o $(TEST_CORE) -lm $(BLAS_LDFLAGS)

$(VERIFY_LAYERS): tests/verify_layers.c $(CORE_OBJ)
	$(CC) $(CFLAGS) tests/verify_layers.c $(CORE_OBJ) -o $(VERIFY_LAYERS) -lm $(BLAS_LDFLAGS)

verify: $(TEST_CORE) $(TARGET)
	./$(TEST_CORE)
	python3 -m py_compile tools/converter.py tools/generate_layer_tests.py tools/inference_py.py

# Unit tests + golden layer parity (fixtures under tests/data/ are optional; missing bins SKIP).
.PHONY: e2e
e2e: $(TEST_CORE) $(TARGET) $(VERIFY_LAYERS)
	./$(TEST_CORE)
	python3 -m py_compile tools/converter.py tools/generate_layer_tests.py tools/inference_py.py
	./$(VERIFY_LAYERS)

$(BENCH_GEMM): tests/bench_gemm.c $(TENSOR_OBJ) $(BUILD_DIR)/utils.o | $(BUILD_DIR)
	$(CC) $(CFLAGS) tests/bench_gemm.c $(TENSOR_OBJ) $(BUILD_DIR)/utils.o -o $(BENCH_GEMM) -lm $(BLAS_LDFLAGS)

.PHONY: bench
bench: $(BENCH_GEMM)
	./$(BENCH_GEMM)

.PHONY: eval-coco8
eval-coco8: $(TARGET)
	$(PYTHON_EVAL) tools/eval_coco8.py --pt "$(PT_MODEL)" --c-bin "./$(TARGET)" --c-weights "$(C_WEIGHTS)"

regenerate-golden:
	bash tools/with_py39.sh python tools/generate_layer_tests.py

$(BUILD_DIR)/tensor_u0.o $(BUILD_DIR)/tensor_u1.o: src/tensor.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: src/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: src/%.m | $(BUILD_DIR)
	$(CC) $(CFLAGS) -fobjc-arc -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR) $(TARGET) $(TEST_CORE) $(VERIFY_LAYERS) $(BENCH_GEMM)
