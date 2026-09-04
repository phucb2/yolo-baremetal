"""
Export Ultralytics YOLO .pt weights to project .bin format.
Fuses Conv+BatchNorm into conv weights/bias (matches src/utils.c fold_bn) before export so C inference matches fused PyTorch.
v2 (--quantize): per-channel INT8 conv weights + __act_scale.* sidecar tensors.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path

import numpy as np
import torch

from calibrate_quant import load_fp32_bin_tensors, should_keep_fp32_weight, weight_prefix

WEIGHT_BIN_MAGIC_V2 = 0xBF01
TENSOR_DTYPE_FP32 = 0
TENSOR_DTYPE_INT8 = 1


def fold_bn_into_conv_numpy(
    w: np.ndarray,
    b: np.ndarray | None,
    bn_w: np.ndarray,
    bn_b: np.ndarray,
    rm: np.ndarray,
    rv: np.ndarray,
    eps: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Match src/utils.c fold_bn: scale conv weights; bias = (b - rm) * scale + bn_b with b=0 if missing."""
    out_c = w.shape[0]
    scale = bn_w / np.sqrt(rv + eps)
    w_out = w * scale.reshape(out_c, *([1] * (w.ndim - 1)))
    if b is None:
        b = np.zeros(out_c, dtype=np.float32)
    else:
        b = b.astype(np.float32, copy=True)
    b = (b - rm) * scale + bn_b
    return w_out.astype(np.float32), b.astype(np.float32)


def fuse_conv_bn_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    For every *.conv.weight with sibling *.bn.{weight,bias,running_mean,running_var}, fold BN into conv
    and drop BN tensors. Handles nested names (e.g. model.2.cv1.conv.weight).
    """
    out: dict[str, torch.Tensor] = dict(sd)

    for k in list(out.keys()):
        if not k.endswith(".conv.weight"):
            continue
        prefix = k[: -len(".conv.weight")]
        k_bn_w = f"{prefix}.bn.weight"
        k_bn_b = f"{prefix}.bn.bias"
        k_rm = f"{prefix}.bn.running_mean"
        k_rv = f"{prefix}.bn.running_var"
        if k_bn_w not in out or k_bn_b not in out or k_rm not in out or k_rv not in out:
            continue

        w = out[k].detach().cpu().numpy()
        bn_w = out[k_bn_w].detach().cpu().numpy()
        bn_b = out[k_bn_b].detach().cpu().numpy()
        rm = out[k_rm].detach().cpu().numpy()
        rv = out[k_rv].detach().cpu().numpy()
        k_bias = f"{prefix}.conv.bias"
        b = out[k_bias].detach().cpu().numpy() if k_bias in out else None

        w_new, b_new = fold_bn_into_conv_numpy(w, b, bn_w, bn_b, rm, rv)
        out[k] = torch.from_numpy(w_new)
        out[k_bias] = torch.from_numpy(b_new)

        for bk in (k_bn_w, k_bn_b, k_rm, k_rv):
            out.pop(bk, None)
        nb = f"{prefix}.bn.num_batches_tracked"
        out.pop(nb, None)
    return out


def quantize_weight_per_channel(w: np.ndarray, scales: list[float]) -> tuple[np.ndarray, np.ndarray]:
    out_c = w.shape[0]
    sc = np.asarray(scales, dtype=np.float32).reshape(out_c)
    sc = np.maximum(sc, 1e-8)
    flat = w.reshape(out_c, -1)
    q = np.round(flat / sc.reshape(out_c, 1)).astype(np.int32)
    q = np.clip(q, -128, 127).astype(np.int8)
    return q.reshape(w.shape), sc


def _write_tensor_v2(
    f,
    name: str,
    shape: tuple[int, ...],
    dtype: int,
    payload: bytes,
    scales: np.ndarray | None = None,
) -> None:
    name_bytes = name.encode("ascii")
    f.write(struct.pack("i", len(name_bytes)))
    f.write(name_bytes)
    f.write(struct.pack("i", len(shape)))
    for d in shape:
        f.write(struct.pack("i", int(d)))
    f.write(struct.pack("i", dtype))
    if dtype == TENSOR_DTYPE_FP32:
        f.write(payload)
    elif dtype == TENSOR_DTYPE_INT8:
        assert scales is not None
        f.write(struct.pack("i", int(scales.size)))
        f.write(scales.astype(np.float32).tobytes())
        f.write(payload)
    else:
        raise ValueError(f"unsupported dtype {dtype}")


def export_yolo26_to_bin(
    model_path: str | None,
    output_path: str,
    no_fuse: bool = False,
    quant_json: str | None = None,
    source_bin: str | None = None,
) -> None:
    nc = 80
    state_dict: dict[str, torch.Tensor] = {}

    if source_bin:
        print(f"Loading FP32 tensors from {source_bin}...")
        arrays = load_fp32_bin_tensors(source_bin)
        for name, arr in arrays.items():
            state_dict[name] = torch.from_numpy(arr)
        if state_dict:
            nc = 80
    else:
        if not model_path:
            raise ValueError("model_path or source_bin required")
        print(f"Loading weights from {model_path}...")

        try:
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(model_path, map_location="cpu")

        if "model" in ckpt:
            model = ckpt["model"]
        elif "ema" in ckpt:
            model = ckpt["ema"]
        else:
            model = ckpt

        if hasattr(model, "state_dict"):
            if hasattr(model, "float"):
                model = model.float()
                print("Cast model to float32 before export.")
            model.eval()
            state_dict = model.state_dict()
            nc = int(getattr(model, "nc", 80))
        else:
            state_dict = model  # type: ignore[assignment]

        if not no_fuse:
            before = len(state_dict)
            state_dict = fuse_conv_bn_state_dict(state_dict)
            after = len(state_dict)
            print(f"Conv+BN fuse: {before} -> {after} tensors (nested names included).")
        else:
            print("Skipping Conv+BN fuse (--no-fuse).")

    quant_meta: dict | None = None
    if quant_json:
        quant_meta = json.loads(Path(quant_json).read_text(encoding="utf-8"))
        print(f"Quantize export using {quant_json}")

    items = sorted(
        [(k, v) for k, v in state_dict.items() if isinstance(v, torch.Tensor)], key=lambda x: x[0]
    )

    act_scales: dict[str, float] = {}
    weight_scales: dict[str, list[float]] = {}
    if quant_meta:
        act_scales = quant_meta.get("act_scales", {})
        weight_scales = quant_meta.get("weight_scales", {})

    sidecars: list[tuple[str, float]] = []
    if quant_meta:
        for mod_name, scale in sorted(act_scales.items()):
            sidecars.append((f"__act_scale.{mod_name}", float(scale)))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    file_version = 2  # Always use v2 format (includes dtype field)
    total_tensors = len(items) + len(sidecars)

    int8_count = 0
    with open(output_path, "wb") as f:
        f.write(struct.pack("i", nc))
        f.write(struct.pack("i", total_tensors))
        if file_version:
            f.write(struct.pack("i", WEIGHT_BIN_MAGIC_V2))

        for name, param in items:
            w_np = param.detach().cpu().numpy().astype(np.float32)
            shape = tuple(int(d) for d in param.shape)

            use_int8 = (
                quant_meta is not None
                and name in weight_scales
                and not should_keep_fp32_weight(name)
                and name.endswith((".conv.weight", ".weight"))
            )

            if use_int8:
                q, sc = quantize_weight_per_channel(w_np, weight_scales[name])
                _write_tensor_v2(f, name, shape, TENSOR_DTYPE_INT8, q.tobytes(), sc)
                int8_count += 1
                print(f"Exported INT8: {name:.<55} {list(shape)}")
            else:
                _write_tensor_v2(f, name, shape, TENSOR_DTYPE_FP32, w_np.tobytes())
                print(f"Exported FP32: {name:.<55} {list(shape)}")

        for sname, sval in sidecars:
            sc_arr = np.array([sval], dtype=np.float32)
            _write_tensor_v2(f, sname, (1, 1, 1, 1), TENSOR_DTYPE_FP32, sc_arr.tobytes())
            print(f"Exported act scale: {sname} = {sval:.6g}")

    print(f"\nSuccess! Weights saved to {output_path}")
    if quant_meta:
        print(f"  INT8 tensors: {int8_count}/{len(items)}, act sidecars: {len(sidecars)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Path to .pt model")
    parser.add_argument("--source-bin", type=str, default=None, help="FP32 .bin source when .pt cannot load")
    parser.add_argument("--output", type=str, default="weights/yolo26.bin", help="Output .bin path")
    parser.add_argument(
        "--no-fuse",
        action="store_true",
        help="Export raw state_dict (for testing C-side fold_all_bn on unfused bins).",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        default=None,
        metavar="JSON",
        help="Path to calibrate_quant.py output JSON for INT8 v2 export",
    )
    args = parser.parse_args()

    if not args.model and not args.source_bin:
        raise SystemExit("Provide --model or --source-bin")
    export_yolo26_to_bin(
        args.model,
        args.output,
        no_fuse=args.no_fuse,
        quant_json=args.quantize,
        source_bin=args.source_bin,
    )
