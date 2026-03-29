"""
Step-by-step parity: (1) weights in weights/yolo26.bin vs fused yolo26n.pt after C-style fold_all_bn;
(2) optional first Conv+SiLU output vs C stage dump (isolates conv logic).

Run from repo root with conda py39:
  python tools/compare_weights_bin_pt.py weights/yolo26.bin yolo26n.pt
  python tools/compare_weights_bin_pt.py weights/yolo26.bin yolo26n.pt \\
    --shared-input runs/shared_input.bin --c-stages runs/zidane_c_stages.bin
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import struct
import sys

import numpy as np
import torch
import torch.nn.functional as F

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve(p: str) -> str:
    if os.path.isfile(p):
        return p
    alt = os.path.join(_REPO, p)
    return alt if os.path.isfile(alt) else p


def _load_tool(rel: str, mod_name: str):
    path = os.path.join(_REPO, rel)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_converter = _load_tool("tools/converter.py", "converter")
fold_bn_into_conv_numpy = _converter.fold_bn_into_conv_numpy
fuse_conv_bn_state_dict = _converter.fuse_conv_bn_state_dict


def load_yolo26_bin(path: str) -> tuple[int, dict[str, np.ndarray]]:
    """Load .bin tensors with **exact** rank from file (same as C). Do not pad to 4D — padding [16]→(16,1,1,1) breaks BN fold broadcasting."""
    with open(path, "rb") as f:
        nc, n_tensors = struct.unpack("ii", f.read(8))
        tensors: dict[str, np.ndarray] = {}
        for _ in range(n_tensors):
            (nl,) = struct.unpack("i", f.read(4))
            name = f.read(nl).decode("ascii")
            (nd,) = struct.unpack("i", f.read(4))
            if nd == 0:
                raw = f.read(4)
                arr = np.frombuffer(raw, dtype=np.float32).copy().reshape(())
            else:
                dims = [struct.unpack("i", f.read(4))[0] for _ in range(nd)]
                count = int(np.prod(dims)) if dims else 0
                raw = f.read(4 * count)
                arr = np.frombuffer(raw, dtype=np.float32).copy().reshape(dims)
            tensors[name] = arr
    return nc, tensors


def fold_all_bn_dict(d: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Mirror src/model.c fold_all_bn on the in-memory weight table."""
    d = {k: np.array(v, dtype=np.float32, copy=True) for k, v in d.items()}
    suffix = ".conv.weight"
    while True:
        found = None
        for name in d:
            if not name.endswith(suffix):
                continue
            prefix = name[: -len(suffix)]
            if f"{prefix}.bn.weight" in d:
                found = name
                break
        if found is None:
            break
        prefix = found[: -len(suffix)]
        cw = d[found].copy()
        bn_w = d.pop(f"{prefix}.bn.weight")
        bn_b = d.pop(f"{prefix}.bn.bias")
        rm = d.pop(f"{prefix}.bn.running_mean")
        rv = d.pop(f"{prefix}.bn.running_var")
        d.pop(f"{prefix}.bn.num_batches_tracked", None)
        cb_key = f"{prefix}.conv.bias"
        if cb_key in d:
            cb = d[cb_key].copy().astype(np.float32)
        else:
            cb = np.zeros(cw.shape[0], dtype=np.float32)
        w_new, b_new = fold_bn_into_conv_numpy(cw, cb, bn_w, bn_b, rm, rv)
        d[found] = w_new
        d[cb_key] = b_new
    return d


def fused_pt_state_dict(pt_path: str) -> dict[str, np.ndarray]:
    """Match tools/converter.py: float32 + fuse_conv_bn only (no model.fuse — keeps cv2/cv3 for C parity)."""
    from ultralytics import YOLO

    ym = YOLO(_resolve(pt_path))
    net = ym.model.float().eval()
    sd = {k: v for k, v in net.state_dict().items() if isinstance(v, torch.Tensor)}
    sd = fuse_conv_bn_state_dict(sd)
    return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in sd.items()}


def load_stage_tensor(stages_bin: str, stage_name: str) -> np.ndarray | None:
    path = _resolve(stages_bin)
    with open(path, "rb") as f:
        while True:
            hdr = f.read(4)
            if len(hdr) < 4:
                return None
            (nl,) = struct.unpack("i", hdr)
            name = f.read(nl).decode("ascii")
            (nd,) = struct.unpack("i", f.read(4))
            dims = [struct.unpack("i", f.read(4))[0] for _ in range(nd)]
            while len(dims) < 4:
                dims.append(1)
            n, c, h, w = dims
            count = n * c * h * w
            raw = f.read(4 * count)
            if len(raw) != 4 * count:
                return None
            if name == stage_name:
                return np.frombuffer(raw, dtype=np.float32).copy().reshape(dims)


def load_first_named_tensor(path: str) -> np.ndarray:
    path = _resolve(path)
    with open(path, "rb") as f:
        (nl,) = struct.unpack("i", f.read(4))
        f.read(nl)
        (nd,) = struct.unpack("i", f.read(4))
        dims = [struct.unpack("i", f.read(4))[0] for _ in range(nd)]
        while len(dims) < 4:
            dims.append(1)
        count = int(np.prod(dims))
        raw = f.read(4 * count)
        return np.frombuffer(raw, dtype=np.float32).copy().reshape(dims)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare .bin weights vs fused .pt (C fold simulation + optional layer-0 op)")
    ap.add_argument("bin_path", help="weights/yolo26.bin")
    ap.add_argument("pt_path", help="yolo26n.pt")
    ap.add_argument("--atol", type=float, default=1e-5, help="per-weight max abs diff tolerance")
    ap.add_argument("--shared-input", metavar="PATH", help="runs/shared_input.bin for first-conv check")
    ap.add_argument("--c-stages", metavar="PATH", help="C stage dump; compare stage_01_buf0 to PyTorch conv0")
    args = ap.parse_args()

    bin_path = _resolve(args.bin_path)
    pt_path = _resolve(args.pt_path)
    if not os.path.isfile(bin_path):
        print(f"Bin not found: {bin_path}", file=sys.stderr)
        sys.exit(2)
    if not os.path.isfile(pt_path):
        print(f"PT not found: {pt_path}", file=sys.stderr)
        sys.exit(2)

    print("=== Step 1: load .bin (as written on disk) ===")
    nc_bin, raw_bin = load_yolo26_bin(bin_path)
    print(f"  nc={nc_bin}, tensor_count={len(raw_bin)}")

    print("=== Step 2: apply C-style fold_all_bn (Python port) ===")
    folded = fold_all_bn_dict(raw_bin)
    print(f"  tensor_count_after_fold={len(folded)}")

    print("=== Step 3: fused PyTorch state_dict (YOLO.fuse + fuse_conv_bn_state_dict) ===")
    ref = fused_pt_state_dict(pt_path)
    print(f"  tensor_count_ref={len(ref)}")

    print("=== Step 4: compare folded-bin vs ref (by name) ===")
    ref_keys = sorted(ref.keys())
    only_bin = sorted(set(folded.keys()) - set(ref.keys()))
    only_ref = sorted(set(ref.keys()) - set(folded.keys()))
    if only_bin:
        print(f"  ONLY_IN_BIN_AFTER_FOLD ({len(only_bin)}): {only_bin[:20]}{' ...' if len(only_bin) > 20 else ''}")
    if only_ref:
        print(f"  ONLY_IN_PT_REF ({len(only_ref)}): {only_ref[:20]}{' ...' if len(only_ref) > 20 else ''}")

    failed = False
    weight_failures = 0
    checked = 0
    max_diff_overall = 0.0
    for k in ref_keys:
        if k not in folded:
            print(f"  MISSING_IN_BIN {k}")
            failed = True
            continue
        a, b = folded[k], ref[k]
        if a.shape != b.shape:
            print(f"  SHAPE {k}: bin{a.shape} pt{b.shape}")
            failed = True
            continue
        mad = float(np.max(np.abs(a - b)))
        checked += 1
        max_diff_overall = max(max_diff_overall, mad)
        if mad > args.atol:
            print(f"  WEIGHT_FAIL {k} max_abs_diff={mad:.6g}")
            failed = True
            weight_failures += 1
    if not failed and checked:
        print(f"  all {checked} shared tensors within atol={args.atol}")
    else:
        print(f"  compared {checked} tensors present in both ({weight_failures} above atol)")

    if args.shared_input and args.c_stages:
        print("=== Step 5: first Conv+SiLU (stride=2 pad=1) vs C stage_01_buf0 ===")
        x = torch.from_numpy(load_first_named_tensor(args.shared_input))
        w_name, b_name = "model.0.conv.weight", "model.0.conv.bias"
        if w_name not in ref or b_name not in ref:
            print("  skip: missing model.0 conv weights in ref", file=sys.stderr)
        else:
            w = torch.from_numpy(ref[w_name])
            b = torch.from_numpy(ref[b_name])
            y = F.conv2d(x, w, b, stride=2, padding=1)
            y = F.silu(y)
            c_stage = load_stage_tensor(args.c_stages, "stage_01_buf0")
            if c_stage is None:
                print("  skip: stage_01_buf0 not found in C stages bin", file=sys.stderr)
            else:
                py = y.detach().numpy()
                if py.shape != c_stage.shape:
                    print(f"  SHAPE pytorch{tuple(py.shape)} vs c_stage{tuple(c_stage.shape)}")
                    failed = True
                else:
                    mad = float(np.max(np.abs(py - c_stage)))
                    print(f"  PyTorch Conv+SiLU vs C stage_01_buf0: max_abs_diff={mad:.6g}")
                    if mad > 1e-3:
                        print("  -> If Step 4 weights matched: likely C conv/SiLU/layout bug. If Step 4 failed: fix weights first.")
                        failed = True
    elif args.shared_input or args.c_stages:
        print("  (Step 5 skipped: pass both --shared-input and --c-stages for first-conv vs C dump)")

    print("=== Summary (weights vs module logic) ===")
    if only_bin:
        print(
            f"  • {len(only_bin)} tensors only in folded .bin (e.g. model.23.cv2/cv3.*): "
            "extra Detect branches in the exported checkpoint; current PyTorch graph uses one2one_* for end2end — naming mismatch only."
        )
    if weight_failures == 0 and checked:
        print("  • Weights: folded .bin matches fused .pt within atol for all shared tensors.")
    elif weight_failures:
        print(
            f"  • Weights: {weight_failures} mismatches (max_abs_diff across compared tensors ≈ {max_diff_overall:.6g}). "
            "Most likely yolo26.bin was **not** exported from this yolo26n.pt. Fix: "
            "`python tools/converter.py --model yolo26n.pt --output weights/yolo26.bin`, then re-run dumps."
        )
    if args.shared_input and args.c_stages:
        if weight_failures:
            print("  • Activations: do **not** interpret Step 5 until weights match — large diffs are expected from wrong weights.")
        else:
            print("  • Activations: if Step 5 still fails with matching weights, then inspect C conv2d / SiLU / layout in src/layers.c vs PyTorch.")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
