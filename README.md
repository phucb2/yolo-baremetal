# YOLO26 C inference

**What:** Minimal C inference stack for Ultralytics-style YOLO26 (CPU, SIMD). **When:** Use this repo to run or benchmark the model from exported weights; see `plan.md` for roadmap and status.

## Requirements

- Weights at `weights/yolo26.bin` (export from `.pt` first)
- **Linux:** GCC/Clang, CMake 3.20+; see [Linux Setup Guide](docs/LINUX_SETUP.md)
- **macOS:** Clang + AVFoundation for live camera (Makefile or CMake with `USE_CAMERA=ON`)
- **Windows:** Visual Studio 2022+ (MSVC), CMake 3.20+; image inference only (`USE_CAMERA=OFF` by default)

## Build & test (macOS, Makefile)

```bash
make              # builds yolo26_bench
make verify       # runs tests/test_core + Python syntax check on tools/
```

## Build (CMake, macOS or Windows)

Presets are in [CMakePresets.json](CMakePresets.json).

**Linux (Clang, image mode):**

```bash
cmake --preset linux
cmake --build --preset linux
./build/yolo26_bench --image path/to/img.jpg out.bmp
```

**Windows (MSVC, image mode):**

```powershell
cmake --preset win
cmake --build --preset win
.\build\Release\yolo26_bench.exe --image path\to\img.jpg out.bmp
```

**macOS (Clang, live camera):**

```bash
cmake --preset mac
cmake --build --preset mac
./build/yolo26_bench
```

### CMake options

| Option | Default | Description |
|--------|---------|-------------|
| `USE_INT8` | ON | INT8 inference (`-DUSE_INT8=1`) |
| `USE_CAMERA` | ON on macOS, OFF on Windows | Live camera via AVFoundation |
| `USE_OPENBLAS` | OFF | Link OpenBLAS for FP32 GEMM (`-DOPENBLAS_ROOT=...`) |
| `USE_ONEDNN` | OFF | Build `bench_gemm_compare` (oneDNN vs `tensor_gemm`; requires vcpkg) |
| `GEMM_PREFER_AVX512_VNNI` | OFF | Prefer AVX512-VNNI over AVX-VNNI (GCC/Clang only) |

On Windows, INT8 GEMM uses AVX2 (VNNI kernels are GCC/Clang-only for now).

### GEMM benchmark vs oneDNN (optional)

Compare `tensor_gemm` / `tensor_gemm_weight_int8` against oneDNN matmul on the same shapes as `bench_gemm`:

```powershell
vcpkg install onednn:x64-windows
$env:VCPKG_ROOT = "C:\path\to\vcpkg"
cmake --preset win-onednn
cmake --build --preset win-onednn
.\build\Release\bench_gemm_compare.exe
```

Set `OMP_NUM_THREADS=1` if oneDNN was built with OpenMP and thread count was not pinned via API.

INT8 cases use oneDNN `s8 × s8` matmul with per-row scale applied after the kernel (fused scale attrs are not available on all CPU builds).

Run C tests after build:

```powershell
.\build\Release\test_core.exe
```

## Run (live camera, macOS only)

```bash
./yolo26_bench              # 5 frames, stdout
./yolo26_bench out.bmp      # also writes last annotated frame
```

## Run (single image, all platforms)

```bash
./yolo26_bench --image photo.jpg annotated.bmp
```

## Export weights (Python)

Use conda env with PyTorch (e.g. `conda activate py39`):

```bash
python tools/converter.py --model <checkpoint.pt> --output weights/yolo26.bin
```

Regenerate layer goldens: `make regenerate-golden` (uses `tools/with_py39.sh`).

## Layout

| Path | Role |
|------|------|
| `src/` | Tensor, layers, model forward, detect, camera, visualize |
| `include/` | Headers |
| `tests/` | C unit / parity tests |
| `tools/` | `.pt` → `.bin` converter, golden generation |
