# C3k2 (Ultralytics) — C parity and verification

> **Purpose:** Document C3k2 vs PyTorch semantics, golden-file layout, and how to re-verify. **When:** Implementing or debugging `c3k2_forward`, regenerating test vectors, or matching [yolo26.yaml](yolo26.yaml) blocks.

## Module in YOLO26

C3k2 appears in the backbone and head with YAML args `[c2, shortcut, e, …]` — see [yolo26.yaml](yolo26.yaml) (e.g. lines 24–30 backbone, 38–50 head). The Ultralytics constructor is:

`C3k2(c1, c2, n, c3k=False, e=0.5, attn=False, g=1, shortcut=True)`.

The **4th argument is `c3k`**, not `shortcut`. Use `c3k=False` for the standard `Bottleneck` stack that matches this C implementation.

## C API

- `c3k2_forward` — see comments in [include/layers.h](include/layers.h): concat order matches PyTorch `C2f`: **full `cv1` output (two chunks)** plus each bottleneck output; then `cv2` 1×1.
- `bottleneck_forward` — padding is derived from weight kernel height so 3×3 convs use padding 1 (same as PyTorch `Conv`).

## Python reference (fixtures)

| Fixture | File | Ultralytics call | Notes |
| :--- | :--- | :--- | :--- |
| Unit | `tests/data/c3k2_unit.bin` | `C3k2(64, 64, 2, c3k=False, e=0.5, shortcut=True)` | Default-like hidden width |
| YAML-style | `tests/data/c3k2_yaml.bin` | `C3k2(256, 256, 2, c3k=False, e=0.25, shortcut=False)` | Backbone-style `e`, residual off |

Weights in the `.bin` files are **fused Conv+BN** using the same math as [src/utils.c](src/utils.c) `fold_bn` (implemented in [tools/generate_layer_tests.py](tools/generate_layer_tests.py)).

### Tensor names (stable order)

Each file contains, in order:

1. `{tag}_input`, `{tag}_output`
2. `{tag}_cv1_weight`, `{tag}_cv1_bias`, `{tag}_cv2_weight`, `{tag}_cv2_bias`
3. For each block `i` in `0 .. n-1`: `{tag}_m{i}_cv1_weight`, `{tag}_m{i}_cv1_bias`, `{tag}_m{i}_cv2_weight`, `{tag}_m{i}_cv2_bias`

`tag` is `unit` or `yaml`.

### Mapping to `c3k2_forward`

| Role | C argument / layout |
| :--- | :--- |
| Top conv | `cv1_w`, `cv1_b` |
| Final 1×1 | `cv2_w`, `cv2_b` |
| Block `i` | `b_weights[4*i+0..3]` = `m{i}` cv1 w/b, cv2 w/b |

## How to verify

1. **Environment:** PyTorch + Ultralytics (e.g. `conda activate py39` — use your env that has `ultralytics`).
2. **Regenerate goldens** from repo root:  
   `python tools/generate_layer_tests.py`  
   This writes `tests/data/c3k2_unit.bin`, `c3k2_yaml.bin`, plus conv/sppf samples.
3. **Run C tests:**  
   `make verify`  
   or `./tests/test_core` (paths resolve `tests/data/…` from project root or from `tests/`).

Parity threshold in [tests/test_core.c](tests/test_core.c): max absolute difference **&lt; 5e-4** vs PyTorch `float32` output.

### Troubleshooting

- **Large diff after a PyTorch / Ultralytics upgrade:** Re-run the generator; layer names or fusion rules may change.
- **Wrong module type:** `C3k2(..., c3k=True)` uses `C3k` blocks, not this `Bottleneck` path.
- **Missing `.bin` files:** Tests skip with `SKIP:` and still pass; add generated files under `tests/data/` for real parity checks in CI.
