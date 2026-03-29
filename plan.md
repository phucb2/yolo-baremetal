# YOLO26 C-Inference Framework Plan & Status

> **Purpose:** Single source of truth for roadmap and implementation status. Use when planning work or checking what is implemented; run `make verify` to re-check “DONE” items that have automated tests.

## 1. Project Objective
Build a standalone, zero-dependency C framework for YOLO26 inference on x86_64 CPU (Darwin), optimized with AVX2/FMA, featuring real-time camera integration.

## 2. Current Progress Status

| Component | Status | Description |
| :--- | :--- | :--- |
| **Tensor Core** | ✅ DONE | Aligned memory, AVX2 + FMA optimized GEMM. |
| **SIMD Optimizations**| ✅ DONE | Hardcoded x86 optimization flags and vectorized loops. |
| **Weight Loader** | ✅ DONE | Binary `.bin` loader with named tensor lookup. |
| **BN Folding** | ✅ DONE | BatchNorm fusion into Conv weights during load time. |
| **Basic Layers** | ✅ DONE | Conv2D, SiLU, MaxPool, Upsample, Concat. |
| **Complex Modules** | 🔄 WIP | `C3k2`, `SPPF`, and `C2PSA` implemented in `src/layers.c`; wire into `model_forward` pending. |
| **Camera Shim** | ✅ DONE | AVFoundation (macOS) wrapper for real-time RGB frames. |
| **Detection Head** | ✅ DONE | NMS-free box decoding logic. |
| **Weight Converter** | ✅ DONE | Python script to export `.pt` to custom `.bin`. |
| **Verification** | 🔄 WIP | Core C path + loader + BN + decode covered by `make verify`; layer parity vs PyTorch and full graph still pending. |

## 2b. How to verify “DONE” items

| Item | Command / note |
| :--- | :--- |
| Tensor + SIMD GEMM, layers (sample conv), BN fold, binary loader, decode | `make verify` runs `tests/test_core` and `py_compile` on `tools/converter.py`. |
| SIMD flags (x86_64) | Observed in `Makefile` (`-mavx2 -mfma`); ARM uses `-mcpu=apple-m1`. |
| C3k2 vs PyTorch | See [plan_c3k2.md](plan_c3k2.md): `make verify` runs `tests/test_core` C3k2 parity on `tests/data/c3k2_*.bin`; regenerate with `conda activate py39` then `python tools/generate_layer_tests.py`, or `make regenerate-golden` (uses [tools/with_py39.sh](tools/with_py39.sh)). |
| SPPF vs PyTorch | `sppf_forward` in `src/layers.c` (cv1 linear, chained pools, optional shortcut); parity in `tests/test_core` on `tests/data/sppf_test.bin` and `sppf_shortcut.bin` (regenerate with `python tools/generate_layer_tests.py`). |
| C2PSA vs PyTorch | See [plan_c2psa.md](plan_c2psa.md): `c2psa_forward` in `src/layers.c`; parity on `tests/data/c2psa_test.bin` from `generate_layer_tests.py`. |
| Camera shim | Build includes `src/camera_darwin.m`; runtime needs macOS + camera. Smoke: build `yolo26_bench` and run (expects `weights/yolo26.bin`). |
| `.pt` → `.bin` export | Requires PyTorch + checkpoint in conda `py39`: `conda activate py39` then `python tools/converter.py --model <path.pt> --output weights/yolo26.bin`. |

`tests/verify_layers.c` is a stub; use `tests/test_core.c` for automated checks (including C3k2 parity — [plan_c3k2.md](plan_c3k2.md)).

## 3. Concrete Implementation Plan

### Phase 1: Finish Layer Implementations (Immediate)
1. **Implement `C2PSA`**: Add the Position-Sensitive Attention block to `src/layers.c`.
2. **Refine `C3k2`**: Ensure the multi-concatenation logic exactly matches the `yolo26n` width scaling (0.25).

### Phase 2: Full Architecture Mapping
1. **Manual Forward Pass**: In `src/model.c`, explicitly map all 24 layers from `yolo26.yaml`.
2. **Buffer Management**: Finalize the intermediate tensor buffer indices to minimize memory footprint.
3. **Detect Head**: Implement the logic to merge P3, P4, and P5 scales into the final NMS-free candidate list.

### Phase 3: Performance Tuning
1. **Convolution Optimization**: Replace the direct 3x3 loop with an `im2col + GEMM` approach to leverage AVX2 throughput.
2. **Multi-threading**: Integrate `pthreads` to parallelize independent branches in the Neck and Head.

### Phase 4: Final Validation
1. **End-to-End Test**: Run a static image through both Python and C to ensure bit-exact parity.
2. **Camera Benchmarking**: Achieve < 50ms total latency for `yolo26n` on modern x86 CPU.

## 4. Coding Style Checklist
- [x] No project-wide prefixes.
- [x] Descriptive variable names (e.g., `plane_size` vs `ps`).
- [x] No globals; state passed via `model_t` or `context`.
- [ ] Performance benchmarking on every pipeline step (capture/preprocess/decode are instrumented in `main.c`; full `model_forward` / inference path is still a stub).
- [x] Minimal comments (Self-explanatory code).
