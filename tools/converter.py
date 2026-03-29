"""
Export Ultralytics YOLO .pt weights to project .bin format.
Fuses Conv+BatchNorm into conv weights/bias (matches src/utils.c fold_bn) before export so C inference matches fused PyTorch.
"""

from __future__ import annotations

import argparse
import os
import struct

import numpy as np
import torch


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
    fused_prefixes: list[str] = []

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


def export_yolo26_to_bin(model_path: str, output_path: str, no_fuse: bool = False) -> None:
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

    nc = 80
    state_dict: dict[str, torch.Tensor]

    if hasattr(model, "state_dict"):
        # FP16 checkpoints: cast to FP32 so fuse_conv_bn runs on CPU tensors.
        if hasattr(model, "float"):
            model = model.float()
            print("Cast model to float32 before export.")
        model.eval()
        # Do NOT call model.fuse(): it drops Detect cv2/cv3 tensors. C loads one2one_cv2/cv3 when present (end2end), else cv2/cv3.
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

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Stable order for diffs; tensors only (matches C loader)
    items = sorted(
        [(k, v) for k, v in state_dict.items() if isinstance(v, torch.Tensor)], key=lambda x: x[0]
    )

    with open(output_path, "wb") as f:
        f.write(struct.pack("i", nc))
        f.write(struct.pack("i", len(items)))
        for name, param in items:
            data = param.detach().cpu().numpy().astype("float32").flatten()
            name_bytes = name.encode("ascii")
            f.write(struct.pack("i", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("i", len(param.shape)))
            for d in param.shape:
                f.write(struct.pack("i", int(d)))
            f.write(data.tobytes())
            print(f"Exported: {name:.<60} {list(param.shape)}")

    print(f"\nSuccess! Weights saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to .pt model")
    parser.add_argument("--output", type=str, default="weights/yolo26.bin", help="Output .bin path")
    parser.add_argument(
        "--no-fuse",
        action="store_true",
        help="Export raw state_dict (for testing C-side fold_all_bn on unfused bins).",
    )
    args = parser.parse_args()

    export_yolo26_to_bin(args.model, args.output, no_fuse=args.no_fuse)
