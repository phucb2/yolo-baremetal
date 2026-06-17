"""
Use: Calibrate static INT8 activation scales for YOLO26 C inference.
When: Before exporting weights with converter.py --quantize.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

KEEP_FP32_SUBSTR = (
    "c2psa",
    "psa",
    "qkv",
    "proj",
    "pe",
    "ffn",
    "attn",
    "attention",
)

DETECT_FP32_RE = re.compile(r"one2one_cv[23]\.\d\.2\.(weight|bias)$")


def should_keep_fp32_weight(name: str) -> bool:
    low = name.lower()
    if DETECT_FP32_RE.search(name):
        return True
    return any(s in low for s in KEEP_FP32_SUBSTR)


def load_model(model_path: str) -> nn.Module:
    try:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(model_path, map_location="cpu")

    if isinstance(ckpt, dict):
        if "model" in ckpt:
            model = ckpt["model"]
        elif "ema" in ckpt:
            model = ckpt["ema"]
        else:
            model = ckpt
    else:
        model = ckpt

    if hasattr(model, "float"):
        model = model.float()
    model.eval()
    return model


def preprocess_bgr_chw(bgr: np.ndarray, w: int, h: int) -> torch.Tensor:
    resized = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return t


def weight_prefix(name: str) -> str:
    for suf in (".conv.weight", ".weight"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def load_fp32_bin_tensors(bin_path: str) -> dict[str, np.ndarray]:
    """Read v1 or v2 .bin; return name -> float32 numpy array (INT8 dequantized)."""
    tensors: dict[str, np.ndarray] = {}
    with open(bin_path, "rb") as f:
        nc = struct.unpack("i", f.read(4))[0]
        n_tensors = struct.unpack("i", f.read(4))[0]
        _ = nc
        pos = f.tell()
        maybe_magic = struct.unpack("i", f.read(4))[0]
        file_version = 2 if maybe_magic == 0xBF01 else 1
        if file_version == 1:
            f.seek(pos)

        for _ in range(n_tensors):
            name_len = struct.unpack("i", f.read(4))[0]
            name = f.read(name_len).decode("ascii")
            ndim = struct.unpack("i", f.read(4))[0]
            dims = [struct.unpack("i", f.read(4))[0] for _ in range(ndim)]
            while len(dims) < 4:
                dims.append(1)
            dtype = 0
            if file_version >= 2:
                dtype = struct.unpack("i", f.read(4))[0]
            count = int(np.prod(dims))
            if dtype == 1:
                num_scales = struct.unpack("i", f.read(4))[0]
                scales = np.frombuffer(f.read(4 * num_scales), dtype=np.float32)
                q = np.frombuffer(f.read(count), dtype=np.int8).reshape(dims)
                flat = q.reshape(num_scales, -1).astype(np.float32) * scales.reshape(num_scales, 1)
                tensors[name] = flat.reshape(dims).astype(np.float32)
            else:
                data = np.frombuffer(f.read(4 * count), dtype=np.float32).reshape(dims)
                tensors[name] = data
    return tensors


def calibrate_from_bin(bin_path: str, percentile: float) -> dict:
    tensors = load_fp32_bin_tensors(bin_path)
    weight_scales: dict[str, list[float]] = {}
    keep_fp32: list[str] = []
    act_scales: dict[str, float] = {}

    for name, arr in sorted(tensors.items()):
        if not name.endswith((".conv.weight", ".weight")):
            continue
        if should_keep_fp32_weight(name):
            keep_fp32.append(name)
            continue
        w = arr.astype(np.float32)
        out_c = w.shape[0]
        flat = w.reshape(out_c, -1)
        amax = np.abs(flat).max(axis=1)
        amax = np.maximum(amax, 1e-8)
        weight_scales[name] = (amax / 127.0).tolist()
        prefix = weight_prefix(name)
        # Post-SiLU activations are typically O(0..6); uniform conservative scale when hooks unavailable.
        act_scales[prefix] = 6.0 / 127.0

    return {
        "input_size": 640,
        "percentile": percentile,
        "act_scales": act_scales,
        "weight_scales": weight_scales,
        "keep_fp32_weights": sorted(keep_fp32),
        "act_scale_keys": {name: f"__act_scale.{name}" for name in act_scales},
        "weight_prefixes": {name: weight_prefix(name) for name in weight_scales},
        "source": "bin_heuristic",
    }


def calibrate(model_path: str, image_dir: Path | None, input_size: int, percentile: float) -> dict:
    act_vals: dict[str, list[float]] = {}

    def hook(_m: nn.Module, inp: tuple, _out, mod_name: str) -> None:
        x = inp[0].detach()
        flat = x.reshape(-1).float()
        if flat.numel() == 0:
            return
        if percentile >= 100.0:
            v = float(flat.abs().max().item())
        else:
            v = float(torch.quantile(flat.abs(), percentile / 100.0).item())
        act_vals.setdefault(mod_name, []).append(v)

    model = load_model(model_path)

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(lambda m, i, o, n=name: hook(m, i, o, n)))

    if image_dir is not None:
        images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            raise RuntimeError(f"No images in {image_dir}")
        with torch.no_grad():
            for img_path in images:
                bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                x = preprocess_bgr_chw(bgr, input_size, input_size)
                try:
                    model(x)
                except Exception as exc:
                    print(f"Warning: forward failed on {img_path.name}: {exc}")

    for h in hooks:
        h.remove()

    act_scales: dict[str, float] = {}
    for mod_name, vals in act_vals.items():
        peak = max(vals) if vals else 1e-6
        peak = max(peak, 1e-6)
        act_scales[mod_name] = peak / 127.0

    sd = model.state_dict() if hasattr(model, "state_dict") else {}
    weight_scales: dict[str, list[float]] = {}
    keep_fp32: list[str] = []

    for name, param in sd.items():
        if not name.endswith((".conv.weight", ".weight")):
            continue
        if should_keep_fp32_weight(name):
            keep_fp32.append(name)
            continue
        w = param.detach().cpu().float().numpy()
        if w.ndim < 1:
            continue
        out_c = w.shape[0]
        flat = w.reshape(out_c, -1)
        amax = np.abs(flat).max(axis=1)
        amax = np.maximum(amax, 1e-8)
        weight_scales[name] = (amax / 127.0).tolist()

    return {
        "input_size": input_size,
        "percentile": percentile,
        "act_scales": act_scales,
        "weight_scales": weight_scales,
        "keep_fp32_weights": sorted(keep_fp32),
        "act_scale_keys": {name: f"__act_scale.{name}" for name in act_scales},
        "weight_prefixes": {name: weight_prefix(name) for name in weight_scales},
        "source": "pt_forward",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate INT8 scales for YOLO26 C export")
    parser.add_argument("--model", type=str, default=None, help="Path to .pt checkpoint")
    parser.add_argument("--weights-bin", type=str, default=None, help="FP32 .bin fallback when .pt cannot load")
    parser.add_argument("--images", type=str, default=None, help="Calibration image directory (optional with --weights-bin)")
    parser.add_argument("--out", type=str, default="runs/quant_scales.json", help="Output JSON path")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.9,
        help="Activation abs percentile (100 = max)",
    )
    args = parser.parse_args()

    if args.weights_bin:
        meta = calibrate_from_bin(args.weights_bin, args.percentile)
    elif args.model:
        image_dir = Path(args.images) if args.images else None
        if image_dir is None or not image_dir.is_dir():
            raise SystemExit("--images is required when calibrating from --model")
        try:
            meta = calibrate(args.model, image_dir, args.input_size, args.percentile)
        except Exception as exc:
            print(f"PT calibration failed ({exc}); use --weights-bin for heuristic calibration.")
            raise
    else:
        raise SystemExit("Provide --model or --weights-bin")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} ({len(meta['act_scales'])} act scales, {len(meta['weight_scales'])} weight tensors)")


if __name__ == "__main__":
    main()
