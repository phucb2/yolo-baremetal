# Linux Benchmark Results

## Test Environment

- **CPU**: Intel(R) Xeon(R) Processor (4 cores, 1 thread/core)
- **Architecture**: x86_64
- **ISA Extensions**: AVX2, FMA, AVX-VNNI, AVX512F, AVX512-VNNI, AVX512-BF16, AMX-INT8
- **Compiler**: Clang 18.1.3
- **Build Flags**: `-O3 -mavx2 -mfma -march=native`
- **OS**: Ubuntu 24.04 LTS
- **Date**: 2026-06-30

## GEMM Micro-Benchmarks

### FP32 Performance (Hand-rolled AVX2)

```
tensor_gemm micro-benchmark - scalar/AVX (hand-rolled)

  tiny (unit-test-like)        M=   5 N=     7 K=   4 reps=50000  total=    6.73 ms   2.080 GFLOP/s
  medium                       M= 128 N=   320 K= 128 reps=  500  total=  192.86 ms  27.185 GFLOP/s
  large-N (conv-like)          M= 256 N=  4096 K= 256 reps=   20  total=  579.82 ms  18.518 GFLOP/s
```

**Per-iteration latency:**
- Tiny: 0.135 µs/iter
- Medium: 0.386 ms/iter
- Large: 29.0 ms/iter

### INT8 Performance (AVX-VNNI)

```
tensor_gemm_weight_int8 - backend: avx-vnni

  int8 medium                  M= 128 N=   320 K= 128 reps=  500  total=   56.30 ms  93.132 Gop/s
  int8 large-N (conv-like)     M= 256 N=  4096 K= 256 reps=   20  total=  275.46 ms  38.980 Gop/s
```

**Per-iteration latency:**
- Medium: 0.113 ms/iter (70% faster than FP32)
- Large: 13.8 ms/iter (53% faster than FP32)

## Performance Analysis

### INT8 vs FP32 Speedup

| Workload | FP32 (GFLOP/s) | INT8 (Gop/s) | Speedup |
|----------|----------------|--------------|---------|
| Medium   | 27.2           | 93.1         | **3.43x** |
| Large    | 18.5           | 39.0         | **2.10x** |

### Key Findings

1. **AVX-VNNI Acceleration**: INT8 kernels automatically detect and utilize AVX-VNNI instructions (vpd pbusds)
2. **Latency Reduction**: INT8 provides 53-70% latency reduction compared to FP32
3. **Throughput Scaling**: 2-3.4x throughput improvement with INT8 quantization
4. **Hardware Utilization**: Efficient use of modern Intel CPU SIMD extensions

## Build Performance

- **Configuration time**: ~0.2s
- **Build time (clean)**: ~2.9s with `-j4`
- **Incremental rebuild**: ~0.6s

## Test Results

### Unit Tests
```
$ ./build/test_core
Successfully loaded 6 tensors (v1, quantized=no, 2 tensors in memory)
Successfully loaded 1 tensors (v1, quantized=no, 1 tensors in memory)
test_core: all checks passed
```

### Layer Verification
```
$ ./build/verify_layers
verify_layers: golden parity (threshold 5e-4)
c3k2_unit            Max diff: 3.576279e-07 -> SUCCESS
c3k2_yaml            Max diff: 2.384186e-07 -> SUCCESS
sppf_sppf_test.bin   Max diff: 1.549721e-06 -> SUCCESS
sppf_sppf_shortcut.bin Max diff: 1.668930e-06 -> SUCCESS
c2psa                Max diff: 3.874302e-07 -> SUCCESS
verify_layers: all within tolerance
```

## Weight Conversion

Successfully converted YOLO26n model to binary format:
- Input: 5.35 MB (yolo26n.pt)
- Output: 10.0 MB (yolo26.bin, v2 format with dtype)
- Tensors: 175 FP32 parameters
- Conv+BN fusion: 499 → 175 tensors after fusion

## Conclusions

1. **INT8 Optimization**: INT8 quantization with AVX-VNNI provides significant performance benefits (2-3.4x) for CNN inference workloads
2. **Build System**: CMake-based build system works reliably on Linux with both Clang and GCC
3. **Timer Accuracy**: POSIX `clock_gettime(CLOCK_MONOTONIC)` provides microsecond-precision timing
4. **Layer Correctness**: All layer implementations match golden references within tolerance

## Future Work

- End-to-end inference benchmarking with production YOLO26 models
- OpenBLAS integration for FP32 optimization comparison
- INT8 quantization with per-channel scales for full model quantization
- oneDNN benchmark comparisons on AVX512 systems
