# C / Python stage parity — root causes

**What:** Why C vs Python stage dumps and activations diverged, and what we changed. **When:** Read this when debugging `compare_stage_bins.py`, `dump_model_stages`, or fused-weight parity.

## 1. Python dump: fused weights vs module graph

**Symptom:** `stage_01+` huge errors while `compare_weights_bin_pt.py` showed matching tensors.

**Causes:**

1. **Fused bias not loaded** — Ultralytics `Conv` uses `Conv2d(bias=False)` + `BatchNorm`. After `fuse_conv_bn_state_dict`, biases exist in the state dict as `*.conv.bias`, but PyTorch cannot load them onto a `bias=False` conv. Inference used **weight-only** convs → large activation errors.
2. **BN still in the forward path** — Fused weights already embed BN; running real `BatchNorm2d` afterward **double-applies** normalization → exploding activations.
3. **`model.fuse()` vs exporter** — Ultralytics `fuse()` folds with different numerics than `tools/converter.py`; C follows the exporter. Do not rely on `fuse()` for parity with `weights/yolo26.bin`.

**Fix (Python):** After loading the fused dict, **`ensure_fused_conv_biases`** (rebuild `Conv2d` with `bias=True`) and **`replace_bn2d_with_identity`**. See `tools/dump_model_stages.py`.

## 2. C backbone: `C3k2` shortcut flags

**Symptom:** First mismatch at `stage_03_buf2` after fixing Python.

**Cause:** `run_c3k2(..., shortcut=false)` for YAML layers 2 and 4, but Ultralytics `C3k2` passes **`shortcut=True`** into `C2f` / `Bottleneck` by default. Residuals inside bottlenecks were wrong.

**Fix:** `run_c3k2` for indices **2** and **4** with **`shortcut=true`**. See `src/model.c`.

## 3. C `c3_forward`: second bottleneck in-place alias

**Symptom:** Mismatch from `stage_07_buf6` (first `C3k` inner block with **two** bottlenecks).

**Cause:** For `n_bottles == 2`, the second `bottleneck_forward` used the **same** tensor as input and output → undefined overwrite while reading inputs.

**Fix:** **Ping-pong** between two buffers; concat from the buffer that holds the last bottleneck output. See `src/layers.c` (`c3_forward`).

## 4. Detect head: `cv2`/`cv3` vs `one2one_cv2`/`one2one_cv3`

**Symptom:** `stage_24_buf23` / `stage_25_detect` large errors despite matching neck.

**Cause:** YOLO26 `end2end=True` — PyTorch **`Detect.forward`** uses **`one2one_cv2` / `one2one_cv3`** (deep copies). C **`detect_forward_one2one`** used **`cv2` / `cv3`**, which are **different trained weights** than the one-to-one inference path.

**Fix:** If `model.{d}.one2one_cv2.0.0.conv.weight` exists, load **`one2one_cv2` / `one2one_cv3`**; else fall back to **`cv2` / `cv3`**. See `src/detect.c`.

## 5. Residual float noise

After the above, backbone/neck diffs are ~**1e−5–1e−4**; detect postprocess ~**6e−4** max abs — acceptable vs strict bitwise identity.

## Checklist

| Check | Command / note |
|--------|----------------|
| Shared input | `dump_model_stages.py --write-shared-input …` then same `.bin` for C and Py |
| Weights | `converter.py` export + C `fold_all_bn` consistent with `fuse_conv_bn_state_dict` |
| Stage compare | `tools/compare_stage_bins.py … --include-detect` for full graph including detect |
