# Python tools that need torch + ultralytics: use conda env py39, e.g.
#   source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate py39
# or: bash tools/with_py39.sh python tools/generate_layer_tests.py

CC = clang
TARGET = yolo26_bench
CFLAGS = -O3 -Iinclude -Wall -Wextra -std=c11
LDFLAGS = -framework Foundation -framework AVFoundation -framework CoreVideo -framework CoreMedia

UNAME_M := $(shell uname -m)

ifeq ($(UNAME_M), arm64)
	CFLAGS += -mcpu=apple-m1
endif

ifeq ($(UNAME_M), x86_64)
	CFLAGS += -mavx2 -mfma -march=native
endif

SRC = src/tensor.c src/utils.c src/layers.c src/detection.c src/detect.c src/model.c src/visualize.c src/main.c src/camera_darwin.m
OBJ = src/tensor.o src/utils.o src/layers.o src/detection.o src/detect.o src/model.o src/visualize.o src/main.o src/camera_darwin.o

CORE_OBJ = src/tensor.o src/utils.o src/layers.o src/detection.o src/detect.o src/model.o
TEST_CORE = tests/test_core

$(TARGET): $(OBJ)
	$(CC) $(OBJ) -o $(TARGET) $(LDFLAGS)

$(TEST_CORE): tests/test_core.c $(CORE_OBJ)
	$(CC) $(CFLAGS) tests/test_core.c $(CORE_OBJ) -o $(TEST_CORE) -lm

verify: $(TEST_CORE) $(TARGET)
	./$(TEST_CORE)
	python3 -m py_compile tools/converter.py tools/generate_layer_tests.py

# Regenerate tests/data/*.bin goldens (requires conda activate py39).
regenerate-golden:
	bash tools/with_py39.sh python tools/generate_layer_tests.py

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.m
	$(CC) $(CFLAGS) -fobjc-arc -c $< -o $@

clean:
	rm -f src/*.o $(TARGET) $(TEST_CORE)
