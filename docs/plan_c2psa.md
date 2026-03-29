# C2PSA — C API and verification

> **Purpose:** C2PSA / PSABlock parity notes. **When:** Wiring `c2psa_forward` in `model.c` or regenerating goldens.

## Ultralytics behavior

- `C2PSA(c1, c2, n, e=0.5)`: requires `c1 == c2`; `self.c = int(c1 * e)`; `cv1: c1 → 2·self.c`; split `a | b` (each `self.c` ch); `b` through `n` stacked `PSABlock`s; `cv2: cat(a,b) → c1`.
- `PSABlock`: `x += attn(x)` then `x += ffn(x)` (both residuals; second add uses **x after first residual**, not the original input).
- `Attention`: qkv 1×1 (no act), softmax attention, `+ pe(v)` depthwise 3×3, proj 1×1 (no act).

## C API

`c2psa_forward(output, input, n_blocks, e, attn_ratio, cv1_w, cv1_b, cv2_w, cv2_b, psa_weights, buffers)`

- `psa_weights`: `n_blocks × 10` tensors per block: `qkv_w,b`, `proj_w,b`, `pe_w,b`, `ffn0_w,b`, `ffn1_w,b`.
- `buffers[0]`: cv1 `[2·c_h,H,W]`; `buffers[1]`: PSABlock out `[c_h,H,W]`; `buffers[2]`: concat `[2·c_h,H,W]`.
- `num_heads = max(1, (int)(c1*e) / 64)` to match Ultralytics when `c ≥ 64`.

## Verify

`conda activate py39` (or env with ultralytics), then `python tools/generate_layer_tests.py` → `tests/data/c2psa_test.bin`; `make verify` runs parity (max abs diff &lt; 5e-4).
