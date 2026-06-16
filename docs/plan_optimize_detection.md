> **Use:** Track detection-head performance work (Detect forward + postprocess).  
> **When:** Profiling, tuning, or reviewing what is left to optimize.

# Detection head performance plan

## Status (implementation)

| Item | State |
| :--- | :--- |
| Coalesce `ax` / `ay` / `stride_buf` into one allocation | Done (`detect_forward_one2one`) |
| Postprocess: one scratch `malloc`, drop redundant `mpa` / `flat_vals` copy | Done (`detect_postprocess_from_pred`) |
| Fuse DW + SiLU after depthwise | Done (`dwconv3x3_same_forward_fuse_silu` in `run_cv3`) |
| Per-scale head tensor reuse (`detect_workspace_t` / `model_t`) | Not started |
| Precompute anchors when `H,W` fixed across forwards | Done (`model_t.detect_anchor_block` at `model_create`; `detect_forward_one2one` skips rebuild when P3/P4/P5 match `input_w`/`input_h`) |
| Partial sort / SIMD postprocess | Not started |

## Scope

- **In scope:** Ultralytics-style **one2one** Detect on P3/P4/P5 (`run_cv2` / `run_cv3`), bbox+cls concat, dist→bbox, sigmoid, **top‑K postprocess** (`detect_postprocess_from_pred`).
- **Out of scope:** Changing checkpoint layout, anchor strides (8/16/32), or `num_classes` / `max_det` semantics unless explicitly validated against PyTorch.

## Current pipeline (reference)

1. **Per scale** (`s = 0..2`): allocate temp tensors → `run_cv2` (two 3×3 `conv_block` + SiLU, then 1×1 box) → `run_cv3` (DW 3×3 + SiLU + 1×1 + SiLU ×2, final 1×1 cls) → `copy_box_to_concat` / `copy_cls_to_concat` → free temps.
2. **Fused geometry:** `dist2bbox_xyxy` → `mul_xyxy_stride` → `sigmoid_ncN` on full `N×nc` cls grid.
3. **Pack** rows into `pred` `[N, 4+nc]`.
4. **Postprocess:** per-anchor class max → `qsort` by score → top‑`k` indices → gather → **second** `qsort` over flat `k×nc` scores → write `[1, max_det, 6]`.

Hot spots tend to be **many convs** (same kernels as backbone) plus **allocation churn** and **postprocess** (sorts + `expf`).

## Phase 1 — Quick wins (low risk)

| Area | Issue | Direction |
| :--- | :--- | :--- |
| **Allocations** | Per-scale `tensor_allocate` / `tensor_free` for box/cls temps | **Reuse buffers** from `model_t` or a `detect_workspace_t` sized once for fixed `H,W` at each scale (same as `model_create` input size). |
| **Anchors / strides** | `make_anchors_for_shape` + buffers every forward | **Done:** precompute at `model_create` into `detect_anchor_block` (3×N floats). Fallback: dynamic alloc + fill if feature sizes diverge from `model->input_w`/`input_h`. |
| **SiLU after DW** | `run_cv3` calls `dwconv3x3_same_forward` then `silu_forward` | Fuse **DW + SiLU** in one pass (same idea as conv+fused SiLU) to cut a full tensor pass per block. **Implemented:** `dwconv3x3_same_forward_fuse_silu`. |
| **GEMM** | Many 1×1 convs | Optional **`USE_OPENBLAS=1`** for `tensor_gemm` (see `docs/plan_layers_perf.md`). |

## Phase 2 — `detect_postprocess_from_pred`

| Issue | Direction |
| :--- | :--- |
| **Many `malloc`s** | Single scratch arena (`malloc` once, or stack for small fixed caps) for `mpa`, `pairs`, `ori_index`, `gathered`, `flat_vals`, `sort2`. **Partial:** one block for `pairs`…`sort2`; removed duplicate max buffer and `flat_vals` copy. |
| **Full `qsort` on `N`** | Replace with **partial sort / `nth_element`-style** top‑`k` by score (or `std::partial_sort` equivalent in C) when `k ≪ N`. |
| **Two-stage sorting** | Second sort over `k×nc` flattens; consider **one ranking** of `(anchor, class)` pairs if semantics match Ultralytics. |
| **Per-row max** | **SIMD** (NEON/AVX) for argmax over `nc` logits per anchor. |
| **`sigmoid_ncN`** | Vectorized **1/(1+exp(-x))**; watch numerical tolerance vs current `expf` loop. |

**Acceptance:** Bit-exact or bounded max-diff vs current C path on fixed `pred` tensors; keep `tests/test_core` / decode tests green.

## Phase 3 — Head convs (`run_cv2` / `run_cv3`)

| Path | Notes |
| :--- | :--- |
| **`run_cv2`** | Two stride‑1 pad‑1 3×3 stacks → im2col+GEMM already; ensure **no redundant** `silu_forward` where `conv_block_forward(..., true)` suffices end-to-end. |
| **`run_cv3`** | **Depthwise 3×3** dominates; optimize **`dwconv3x3_same_forward`** (SIMD, tiling) per `plan_layers_perf.md` Phase 3. |
| **`copy_*_to_concat`** | Large copies: **memcpy** per channel row or SIMD if layout allows. |

## Phase 4 — Measurement

- **Isolate:** Time `run_cv2`+`run_cv3` per scale vs `detect_postprocess_from_pred` with `MF_LAP`-style splits or `clock_gettime` around each block.
- **Baseline:** `yolo26_bench` / `model_forward` aggregate; compare before/after on same machine and `OPENBLAS_NUM_THREADS` if using BLAS.
- **Regression:** `make e2e` or image inference parity vs PyTorch export if available.

## Related files

- [`src/detect.c`](../src/detect.c) — `detect_forward_one2one`, `run_cv2`, `run_cv3`, `detect_postprocess_from_pred`
- [`include/detect.h`](../include/detect.h)
- [`src/detection.c`](../src/detection.c) — `decode_detections` (consumer of head output)
- [`docs/plan_layers_perf.md`](plan_layers_perf.md) — GEMM, DW conv, SiLU
