# Python tools that need torch + ultralytics: use conda env py39, e.g.
#   source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate py39
# or: bash tools/with_py39.sh python tools/generate_layer_tests.py

CC = clang
# On macOS, prefer Xcode/CLT clang (xcrun). Conda-shim clang can fail on SDK math.h (_Float16) with project CFLAGS.
ifeq ($(shell uname -s),Darwin)
  XCRUN_CLANG := $(shell xcrun -f clang 2>/dev/null)
  ifneq ($(XCRUN_CLANG),)
    CC := $(XCRUN_CLANG)
  endif
endif
TARGET = yolo26_bench
BUILD_DIR = build
CFLAGS = -O3 -Iinclude -Wall -Wextra -std=c11
LDFLAGS = -framework Foundation -framework AVFoundation -framework CoreVideo -framework CoreMedia

UNAME_M := $(shell uname -m)

ifeq ($(UNAME_M), arm64)
	CFLAGS += -mcpu=apple-m1
endif

ifeq ($(UNAME_M), x86_64)
	CFLAGS += -mavx2 -mfma -march=native
endif

OBJ = $(BUILD_DIR)/tensor.o $(BUILD_DIR)/utils.o $(BUILD_DIR)/layers.o $(BUILD_DIR)/detection.o $(BUILD_DIR)/detect.o $(BUILD_DIR)/model.o $(BUILD_DIR)/visualize.o $(BUILD_DIR)/main.o $(BUILD_DIR)/camera_darwin.o

CORE_OBJ = $(BUILD_DIR)/tensor.o $(BUILD_DIR)/utils.o $(BUILD_DIR)/layers.o $(BUILD_DIR)/detection.o $(BUILD_DIR)/detect.o $(BUILD_DIR)/model.o
TEST_CORE = tests/test_core
VERIFY_LAYERS = tests/verify_layers

$(TARGET): $(OBJ)
	$(CC) $(OBJ) -o $(TARGET) $(LDFLAGS)

$(TEST_CORE): tests/test_core.c $(CORE_OBJ)
	$(CC) $(CFLAGS) tests/test_core.c $(CORE_OBJ) -o $(TEST_CORE) -lm

$(VERIFY_LAYERS): tests/verify_layers.c $(CORE_OBJ)
	$(CC) $(CFLAGS) tests/verify_layers.c $(CORE_OBJ) -o $(VERIFY_LAYERS) -lm

verify: $(TEST_CORE) $(TARGET)
	./$(TEST_CORE)
	python3 -m py_compile tools/converter.py tools/generate_layer_tests.py tools/inference_py.py

# Regenerate tests/data/*.bin goldens (requires conda activate py39).
regenerate-golden:
	bash tools/with_py39.sh python tools/generate_layer_tests.py

$(BUILD_DIR)/%.o: src/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: src/%.m | $(BUILD_DIR)
	$(CC) $(CFLAGS) -fobjc-arc -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR) $(TARGET) $(TEST_CORE) $(VERIFY_LAYERS)
