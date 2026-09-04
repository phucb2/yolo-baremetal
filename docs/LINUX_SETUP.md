<!--
Linux Development Environment Setup Guide
This file documents the required setup steps for developing the YOLO26 C inference project on Linux systems.
Use this guide for initial environment configuration.
-->

# Linux Development Environment Setup

This guide covers setting up the development environment for the YOLO26 C inference project on Linux systems.

## Prerequisites

The following tools and libraries are required:

### Build Tools
- **GCC** 11+ or **Clang** 14+ (C compiler with C11 support)
- **CMake** 3.20 or higher
- **Make** (GNU Make)
- **Git** (for version control)

### Python Environment (Optional - for tooling)
- **Python** 3.8 or higher
- **uv** - Fast Python package manager (recommended for running Python tools)

### Optional Dependencies
- **OpenBLAS** - For optimized FP32 GEMM operations (`USE_OPENBLAS=ON`)
- **oneDNN** - For GEMM benchmarking comparisons (`USE_ONEDNN=ON`)
- **vcpkg** - C++ package manager (required for oneDNN on some systems)

## Quick Setup

### 1. Install Build Tools

On Ubuntu/Debian:
```bash
sudo apt update
sudo apt install -y build-essential cmake git clang
```

On Fedora/RHEL:
```bash
sudo dnf install -y gcc gcc-c++ cmake make git clang
```

### 2. Install uv (Python Package Manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

Verify installation:
```bash
uv --version
```

### 3. Clone and Setup Repository

```bash
git clone <repository-url>
cd yolo-baremetal
mkdir -p weights
```

## Build Instructions

### Using CMake (Recommended for Linux)

#### Standard Build (INT8, no camera support)

```bash
# Configure with Linux preset
cmake --preset linux

# Build all targets
cmake --build --preset linux -j$(nproc)
```

#### Build with Custom Options

```bash
# Configure with specific options
cmake -B build -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_INT8=ON \
    -DUSE_CAMERA=OFF \
    -DUSE_OPENBLAS=OFF

# Build
cmake --build build -j$(nproc)
```

#### Available CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `USE_INT8` | ON | Enable INT8 quantized inference |
| `USE_CAMERA` | OFF (Linux) | Enable camera support (macOS only) |
| `USE_OPENBLAS` | OFF | Use OpenBLAS for FP32 GEMM operations |
| `USE_ONEDNN` | OFF | Build GEMM comparison benchmarks with oneDNN |
| `GEMM_PREFER_AVX512_VNNI` | OFF | Prefer AVX512-VNNI over AVX-VNNI (x86_64 only) |

### Using Make (Alternative)

The Makefile is primarily designed for macOS but can be adapted for Linux:

```bash
# Note: Makefile defaults assume macOS with camera support
# For Linux, you would need to modify CFLAGS and LDFLAGS
make
```

## Running Tests

### Core Unit Tests

```bash
./build/test_core
```

### Layer Verification Tests

```bash
./build/verify_layers
```

### GEMM Benchmarks

```bash
./build/bench_gemm
```

### Using CTest

```bash
cd build
ctest --output-on-failure
```

## Python Tools Setup

For weight conversion and evaluation tools:

### 1. Create Python Environment

```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate
uv pip install torch torchvision ultralytics numpy
```

### 2. Convert Model Weights

```bash
# Convert PyTorch model to binary format
uv run python tools/converter.py --model <checkpoint.pt> --output weights/yolo26.bin
```

### 3. Run Evaluations

```bash
# Evaluate on COCO8 dataset
uv run python tools/eval_coco8.py --pt weights/yolo26n.pt --c-bin ./build/yolo26_bench --c-weights weights/yolo26_int8.bin
```

## Troubleshooting

### Build Errors

**Error: `timer_t` conflicts with system type**
- This has been fixed in the codebase by renaming to `yolo_timer_t`
- If you see this error, ensure you have the latest code

**Error: `__cpuid_count` lvalue issues**
- This has been fixed by using proper unsigned int variables
- Update to the latest code version

### Missing Dependencies

**OpenBLAS not found:**
```bash
# Ubuntu/Debian
sudo apt install libopenblas-dev

# Fedora/RHEL
sudo dnf install openblas-devel
```

**oneDNN for benchmarking:**
```bash
# Install via vcpkg
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh
./vcpkg install onednn

# Then configure with:
cmake --preset linux -DUSE_ONEDNN=ON -DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
```

## Running Inference

### Image Inference

```bash
# Run inference on a single image
./build/yolo26_bench --image input.jpg output.bmp
```

### Camera Inference

Camera support is not available on Linux. Use image inference mode instead.

## Build Artifacts

After a successful build, you'll find:

- `build/yolo26_bench` - Main inference binary
- `build/test_core` - Core unit tests
- `build/verify_layers` - Layer verification tests
- `build/bench_gemm` - GEMM performance benchmarks
- `build/bench_coco8` - COCO8 benchmark utility
- `build/libyolo_core.a` - Core static library
- `build/libyolo_camera.a` - Camera stub library (Linux)

## Architecture-Specific Notes

### x86_64 (Intel/AMD)

The build automatically enables AVX2, FMA, and march=native optimizations.

For AVX512-VNNI systems:
```bash
cmake --preset linux -DGEMM_PREFER_AVX512_VNNI=ON
```

### ARM64 (Aarch64)

ARM NEON optimizations are automatically enabled for ARM64 builds.

## Environment Variables

### For Python Tools

Add to your shell profile (`~/.bashrc` or `~/.zshrc`):

```bash
export PATH="$HOME/.local/bin:$PATH"  # For uv
```

## Development Workflow

1. **Make changes** to source files in `src/` or `include/`
2. **Rebuild** with `cmake --build build -j$(nproc)`
3. **Run tests** with `./build/test_core` and `./build/verify_layers`
4. **Benchmark** with `./build/bench_gemm`
5. **Test inference** with sample images

## Additional Resources

- Main README: [`README.md`](../README.md)
- Project plan: [`docs/plan.md`](plan.md)
- Performance notes: [`docs/plan_layers_perf.md`](plan_layers_perf.md)

## Notes

- Camera support (`USE_CAMERA=ON`) requires macOS with AVFoundation framework
- For Linux, use image-based inference only
- The Python environment setup is optional but required for weight conversion and evaluation tools
- Build time is approximately 2-3 seconds on modern hardware with `-j$(nproc)`
