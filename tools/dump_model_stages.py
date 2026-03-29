"""
Dump YOLO26 stage tensors matching C model_forward_ex names (fused Conv+BN like weights export).
Pair with C dump using weights/yolo26.bin exported from the same checkpoint.

Shared input (recommended for parity vs C): one preprocess path (OpenCV), same float tensor for both sides:
  python tools/dump_model_stages.py --write-shared-input runs/shared_input.bin --image tests/data/zidane.jpg
  ./tests/dump_model_stages runs/shared_input.bin runs/zidane_c_stages.bin weights/yolo26.bin
  python tools/dump_model_stages.py --input-bin runs/shared_input.bin --out runs/zidane_py_stages.bin
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import struct
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_PT = os.path.join(_REPO, "yolo26n.pt")


def _resolve_path(path: str) -> str:
    """If path is relative and missing cwd, try repo root (for yolo26n.pt / weights/...)."""
    if os.path.isfile(path):
        return path
    alt = os.path.join(_REPO, path)
    if os.path.isfile(alt):
        return alt
    return path


def _load_tool_module(mod_name: str, rel_path: str):
    path = os.path.join(_REPO, rel_path)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_gl = _load_tool_module("generate_layer_tests", "tools/generate_layer_tests.py")
save_tensor = _gl.save_tensor
_converter = _load_tool_module("converter", "tools/converter.py")
fuse_conv_bn_state_dict = _converter.fuse_conv_bn_state_dict


def read_first_named_tensor(path: str) -> torch.Tensor:
    """Read one record (same layout as save_tensor / C load_named_tensor)."""
    with open(path, "rb") as f:
        (nl,) = struct.unpack("i", f.read(4))
        f.read(nl)  # name
        (nd,) = struct.unpack("i", f.read(4))
        dims = [struct.unpack("i", f.read(4))[0] for _ in range(nd)]
        while len(dims) < 4:
            dims.append(1)
        n, c, h, w = dims[0], dims[1], dims[2], dims[3]
        count = n * c * h * w
        raw = f.read(4 * count)
        if len(raw) != 4 * count:
            raise ValueError("truncated tensor payload")
    arr = np.frombuffer(raw, dtype=np.float32).copy().reshape((n, c, h, w))
    return torch.from_numpy(arr)


def preprocess_rgb_chw01(bgr_uint8: np.ndarray, w: int, h: int) -> torch.Tensor:
    """Linear resize, BGR->RGB, [1,3,H,W] float32 /255 — canonical shared preprocess (OpenCV)."""
    if bgr_uint8.ndim != 3 or bgr_uint8.shape[2] != 3:
        raise ValueError(f"expected HxWx3 BGR, got {bgr_uint8.shape}")
    resized = cv2.resize(bgr_uint8, (w, h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0).contiguous()


def ensure_fused_conv_biases(net: nn.Module, fused_sd: dict[str, torch.Tensor]) -> int:
    """
    Ultralytics Conv uses Conv2d(..., bias=False) + separate BatchNorm. After fuse_conv_bn_state_dict, fused biases
    exist in fused_sd as *.conv.bias but load_state_dict cannot attach them without a bias Parameter. Rebuild those
    Conv2d layers with bias=True and copy weights + fused bias (matches C inference after fold_all_bn).
    """
    upgraded = 0
    for k, bias_tensor in fused_sd.items():
        if not k.endswith(".conv.bias"):
            continue
        path = k[: -len(".bias")]  # e.g. model.0.conv
        try:
            conv = net.get_submodule(path)
        except Exception:
            continue
        if not isinstance(conv, nn.Conv2d) or conv.bias is not None:
            continue
        new_c = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
            padding_mode=conv.padding_mode,
        )
        new_c.weight.data.copy_(conv.weight.data)
        new_c.bias.data.copy_(bias_tensor.data.to(device=new_c.weight.device, dtype=new_c.weight.dtype))
        parent_path, _, name = path.rpartition(".")
        parent = net.get_submodule(parent_path) if parent_path else net
        setattr(parent, name, new_c)
        upgraded += 1
    return upgraded


def replace_bn2d_with_identity(module: nn.Module) -> int:
    """Fused weights already include BN; remove running BatchNorm from the forward path (else activations diverge)."""
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.Identity())
            n += 1
        else:
            n += replace_bn2d_with_identity(child)
    return n


def apply_fused_state_dict(net: torch.nn.Module, verbose: bool = False) -> None:
    """Match tools/converter.py + C runtime: fuse_conv_bn, load weights, add conv biases, strip BN from graph."""
    if hasattr(net, "float"):
        net.float()
    net.eval()
    sd = {k: v for k, v in net.state_dict().items() if isinstance(v, torch.Tensor)}
    before = len(sd)
    fused = fuse_conv_bn_state_dict(sd)
    if verbose:
        print(f"fuse_conv_bn_state_dict: {before} -> {len(fused)} tensors")
    missing, unexpected = net.load_state_dict(fused, strict=False)
    if verbose and (missing or unexpected):
        print("load_state_dict strict=False:", "missing", len(missing), "unexpected", len(unexpected))
    nb = ensure_fused_conv_biases(net, fused)
    if verbose:
        print(f"ensure_fused_conv_biases: upgraded {nb} Conv2d layers to bias=True")
    nid = replace_bn2d_with_identity(net)
    if verbose:
        print(f"replace_bn2d_with_identity: replaced {nid} BatchNorm2d with Identity")


def _tensor_out(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (list, tuple)) and x:
        return x[0] if isinstance(x[0], torch.Tensor) else _tensor_out(x[0])
    return None


def predict_once_dump(net: torch.nn.Module, x: torch.Tensor, f) -> None:
    """Mirror ultralytics BaseModel._predict_once; write stage tensors after each layer (names match C)."""
    y: list = []
    save_tensor(f, "stage_00_input", x.detach().cpu())
    for m in net.model:
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        x = m(x)
        y.append(x if m.i in net.save else None)
        out_t = _tensor_out(x)
        if out_t is not None:
            save_tensor(f, f"stage_{m.i + 1:02d}_buf{m.i}", out_t.detach().cpu())
    final = _tensor_out(x)
    if final is not None:
        save_tensor(f, "stage_25_detect", final.detach().cpu())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default=_DEFAULT_PT,
        help="Ultralytics yolo26n.pt (repo root by default; must match weights/yolo26.bin export)",
    )
    p.add_argument("--image", default="tests/data/zidane.jpg")
    p.add_argument("--out", default="runs/zidane_py_stages.bin")
    p.add_argument(
        "--input-bin",
        default=None,
        metavar="PATH",
        help="Load stage_00_input from this .bin (single save_tensor record); use with --write-shared-input output",
    )
    p.add_argument(
        "--write-shared-input",
        default=None,
        metavar="PATH",
        help="Only write shared preprocess (OpenCV) to this .bin as stage_00_input, then exit",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    image_path = _resolve_path(args.image)

    if args.write_shared_input:
        if not os.path.isfile(image_path):
            print(f"Image not found: {args.image} (tried {image_path})", file=sys.stderr)
            sys.exit(1)
        bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"cv2.imread failed: {image_path}", file=sys.stderr)
            sys.exit(1)
        x = preprocess_rgb_chw01(bgr, 640, 640)
        out_shared = args.write_shared_input
        if not os.path.isabs(out_shared):
            out_shared = os.path.join(_REPO, out_shared)
        os.makedirs(os.path.dirname(out_shared) or ".", exist_ok=True)
        with open(out_shared, "wb") as f:
            save_tensor(f, "stage_00_input", x.detach().cpu())
        print(f"Wrote shared input {out_shared} (use as first arg to tests/dump_model_stages, and --input-bin here)")
        return

    model_path = _resolve_path(args.model)
    out_path = args.out if os.path.isabs(args.out) else os.path.join(_REPO, args.out)

    if not os.path.isfile(model_path):
        print(f"Model not found: {args.model} (tried {model_path})", file=sys.stderr)
        sys.exit(1)

    if args.input_bin:
        input_bin = _resolve_path(args.input_bin)
        if not os.path.isfile(input_bin):
            print(f"Input bin not found: {args.input_bin} (tried {input_bin})", file=sys.stderr)
            sys.exit(1)
        x = read_first_named_tensor(input_bin)
        if tuple(x.shape) != (1, 3, 640, 640):
            print(f"Expected input shape (1,3,640,640), got {tuple(x.shape)}", file=sys.stderr)
            sys.exit(1)
    else:
        if not os.path.isfile(image_path):
            print(f"Image not found: {args.image} (tried {image_path})", file=sys.stderr)
            sys.exit(1)
        bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"cv2.imread failed: {image_path}", file=sys.stderr)
            sys.exit(1)
        x = preprocess_rgb_chw01(bgr, 640, 640)

    from ultralytics import YOLO

    ym = YOLO(model_path)
    inner = ym.model
    inner.eval()
    apply_fused_state_dict(inner, verbose=args.verbose)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        with torch.no_grad():
            predict_once_dump(inner, x, f)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
