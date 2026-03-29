# Layers performance plan

**Use:** Roadmap to optimize inference hot paths in `src/layers.c`, `src/tensor.c`, and build/link options (e.g. OpenBLAS). **When:** Before implementing perf work; revisit after profiling or when adding new layer paths.

## Goals

- Improve end-to-end bench time (`yolo26_bench`) without changing numerical semantics beyond documented tolerances (keep `tests/test_core`, `verify_layers`, parity tooling green).
- Prefer **low integration cost** first (library GEMM, optional compile flag), then larger refactors (im2col, buffer reuse).

## Non-goals (for this plan)

- Changing model architecture or weight layout on disk.
- GPU / Metal paths (out of scope unless listed later).

## Current hotspots (reference)

| Area | Location | Issue |
| :--- | :--- | :--- |
| 1×1 conv | `conv2d_forward` → `tensor_gemm` | Dominates many blocks; `tensor_gemm` is hand-rolled; **no AVX path on arm64** (scalar inner loops). |
| Attention | `attention_forward` | `N×N` logits and V projection are **triple nested scalar loops**; `O(N²)` per head in FLOPs and memory traffic. |
| SiLU | `silu_forward` | Scalar `expf` per element; runs after many conv blocks. |
| k>1 conv | `conv2d_forward` (general branch) | Naive nested loops; poor cache/blocking vs GEMM-based conv. |
| Depthwise / pool | `dwconv3x3_same_forward`, `pool2d_max_forward` | Scalar inner loops; SIMD-friendly. |
| Upsample | `upsample_nearest_forward` | Per-output indexing; can often be row/column `memcpy` patterns. |
| Allocations | `attention_forward` | `malloc`/`free` for logits and intermediate buffers each call; may matter if attention runs often at large `N`. |

## Phase 1 — OpenBLAS `SGEMM` (highest leverage / low hanging)

1. **Wire optional OpenBLAS** in the Makefile (e.g. `USE_OPENBLAS=1`): include paths, `-lopenblas`, document `OPENBLAS_NUM_THREADS` for repeatability vs throughput.
2. **Replace or branch `tensor_gemm`** in `src/tensor.c` with `cblas_sgemm` under `CblasRowMajor`, matching current layout: `M = out_c`, `N = H*W`, `K = in_c`, leading dimensions `lda = in_c`, `ldb = H*W`, `ldc = H*W`.
3. **Verification:** existing `test_tensor_gemm` and all conv-heavy tests; optional micro-bench on a few shapes (tiny vs large `N`) to decide threshold for “always BLAS” vs “fallback scalar” if needed.
4. **Bias add after 1×1:** optional follow-up (`cblas_saxpy` per channel or SIMD); likely small next to GEMM.

## Phase 2 — Attention matmuls via BLAS

1. **Logits:** express current `q_plane` / `k_plane` layout as one or two `sgemm` calls with correct `Trans`/`NoTrans` so result remains `N×N` row-major, then apply `scale` (fuse into `alpha` if valid).
2. **Value projection:** replace the `v_plane` × attention weights loop with `sgemm` (again matching memory layout).
3. **Correctness:** C2PSA / PSABlock fixtures (`tests/data`, `verify_layers`); watch float ordering differences — keep tolerance explicit in tests if needed.
4. **Memory:** consider reusing a **scratch buffer** owned by the model or `c2psa_forward` instead of per-call `malloc` in `attention_forward` once shapes are stable.

## Phase 3 — SIMD and small kernels (no BLAS)

1. **`silu_forward`:** vectorize (AVX2 on x86_64 per existing project flags; NEON on arm64); evaluate `-ffast-math` impact on max diff vs PyTorch.
2. **`pool2d_max_forward` / `dwconv3x3_same_forward`:** SIMD inner loops or small unrolled tiles.
3. **`upsample_nearest_forward`:** rewrite hot path using repeated source rows / bulk copy to cut index arithmetic.

## Phase 4 — General convolution (larger effort)

1. **im2col (or col2im) + `sgemm`** for `kh,kw > 1` (and/or grouped conv if introduced): tune tile sizes; consider **only** medium/large spatial sizes first.
2. Optional: Winograd or depthwise-specific paths later; treat as separate sub-plans if pursued.

## Profiling and acceptance

- Baseline: `yolo26_bench` (same input), optional Instruments / `perf` on `tensor_gemm`, `attention_forward`, `conv2d_forward`.
- Regression gate: `make verify` (or documented subset), no silent widening of parity without a note in [C_PY_PARITY_REPORT.md](C_PY_PARITY_REPORT.md) if applicable.

## Open questions

- Single build always-on OpenBLAS vs optional flag for minimal dependencies.
- Whether tiny GEMMs should stay scalar to avoid BLAS fixed overhead (measure on target CPU).
- Threading: OpenBLAS threads vs single-thread BLAS + parallel outer loops (avoid oversubscription).

## Related files

- [src/layers.c](../src/layers.c), [src/tensor.c](../src/tensor.c), [include/tensor.h](../include/tensor.h), [Makefile](../Makefile), [tests/test_core.c](../tests/test_core.c), [tests/verify_layers.c](../tests/verify_layers.c).
