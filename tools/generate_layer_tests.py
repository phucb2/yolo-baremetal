"""
Generate golden .bin tensors for C layer parity tests. Requires conda env py39 with torch + ultralytics.
Run from repo root: conda activate py39 && python tools/generate_layer_tests.py
Or: bash tools/with_py39.sh python tools/generate_layer_tests.py
"""

from __future__ import annotations

import os
import struct
from typing import Tuple

import torch
import torch.nn as nn
from ultralytics.nn.modules import C3k2, SPPF
from ultralytics.nn.modules.block import C2PSA


def save_tensor(f, name: str, tensor: torch.Tensor) -> None:
    name_bytes = name.encode("ascii")
    f.write(struct.pack("i", len(name_bytes)))
    f.write(name_bytes)
    f.write(struct.pack("i", len(tensor.shape)))
    for d in tensor.shape:
        f.write(struct.pack("i", d))
    f.write(tensor.detach().cpu().numpy().astype("float32").tobytes())


def fold_bn_into_conv(
    w: torch.Tensor,
    bn_w: torch.Tensor,
    bn_b: torch.Tensor,
    rm: torch.Tensor,
    rv: torch.Tensor,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Match src/utils.c fold_bn: fused conv weight and bias (no pre-existing conv bias)."""
    out_c = w.shape[0]
    scale = bn_w / torch.sqrt(rv + eps)
    w_out = w * scale.view(out_c, *([1] * (w.dim() - 1)))
    b_out = (-rm) * scale + bn_b
    return w_out, b_out


def fuse_c3k2_for_c(m: C3k2) -> dict[str, torch.Tensor]:
    """Fused Conv+BN tensors with stable names for c3k2_forward."""
    sd = m.state_dict()
    out: dict[str, torch.Tensor] = {}

    def add_pair(prefix: str, conv_prefix: str):
        cw = sd[f"{conv_prefix}.conv.weight"]
        bw, bb = sd[f"{conv_prefix}.bn.weight"], sd[f"{conv_prefix}.bn.bias"]
        rm, rv = sd[f"{conv_prefix}.bn.running_mean"], sd[f"{conv_prefix}.bn.running_var"]
        fw, fb = fold_bn_into_conv(cw, bw, bb, rm, rv)
        out[f"{prefix}_weight"] = fw
        out[f"{prefix}_bias"] = fb

    add_pair("cv1", "cv1")
    add_pair("cv2", "cv2")
    for i in range(len(m.m)):
        add_pair(f"m{i}_cv1", f"m.{i}.cv1")
        add_pair(f"m{i}_cv2", f"m.{i}.cv2")
    return out


def fuse_c2psa_for_c(m: C2PSA) -> dict[str, torch.Tensor]:
    """Fused Conv+BN for C2PSA (cv1, cv2, each PSABlock attn + ffn)."""
    sd = m.state_dict()
    out: dict[str, torch.Tensor] = {}

    def add_pair(prefix: str, conv_prefix: str) -> None:
        cw = sd[f"{conv_prefix}.conv.weight"]
        bw, bb = sd[f"{conv_prefix}.bn.weight"], sd[f"{conv_prefix}.bn.bias"]
        rm, rv = sd[f"{conv_prefix}.bn.running_mean"], sd[f"{conv_prefix}.bn.running_var"]
        fw, fb = fold_bn_into_conv(cw, bw, bb, rm, rv)
        out[f"{prefix}_weight"] = fw
        out[f"{prefix}_bias"] = fb

    add_pair("cv1", "cv1")
    add_pair("cv2", "cv2")
    for i in range(len(m.m)):
        add_pair(f"m{i}_qkv", f"m.{i}.attn.qkv")
        add_pair(f"m{i}_proj", f"m.{i}.attn.proj")
        add_pair(f"m{i}_pe", f"m.{i}.attn.pe")
        add_pair(f"m{i}_ffn0", f"m.{i}.ffn.0")
        add_pair(f"m{i}_ffn1", f"m.{i}.ffn.1")
    return out


def fuse_sppf_for_c(m: SPPF) -> dict[str, torch.Tensor]:
    """Fused Conv+BN for cv1/cv2 (same layout as C fold_bn)."""
    sd = m.state_dict()
    out: dict[str, torch.Tensor] = {}
    for prefix in ("cv1", "cv2"):
        cw = sd[f"{prefix}.conv.weight"]
        bw, bb = sd[f"{prefix}.bn.weight"], sd[f"{prefix}.bn.bias"]
        rm, rv = sd[f"{prefix}.bn.running_mean"], sd[f"{prefix}.bn.running_var"]
        fw, fb = fold_bn_into_conv(cw, bw, bb, rm, rv)
        out[f"{prefix}_weight"] = fw
        out[f"{prefix}_bias"] = fb
    return out


def write_c3k2_fixture(path: str, tag: str, m: C3k2, x: torch.Tensor, y: torch.Tensor) -> None:
    fused = fuse_c3k2_for_c(m)
    n = len(m.m)
    with open(path, "wb") as f:
        save_tensor(f, f"{tag}_input", x)
        save_tensor(f, f"{tag}_output", y)
        save_tensor(f, f"{tag}_cv1_weight", fused["cv1_weight"])
        save_tensor(f, f"{tag}_cv1_bias", fused["cv1_bias"])
        save_tensor(f, f"{tag}_cv2_weight", fused["cv2_weight"])
        save_tensor(f, f"{tag}_cv2_bias", fused["cv2_bias"])
        for i in range(n):
            save_tensor(f, f"{tag}_m{i}_cv1_weight", fused[f"m{i}_cv1_weight"])
            save_tensor(f, f"{tag}_m{i}_cv1_bias", fused[f"m{i}_cv1_bias"])
            save_tensor(f, f"{tag}_m{i}_cv2_weight", fused[f"m{i}_cv2_weight"])
            save_tensor(f, f"{tag}_m{i}_cv2_bias", fused[f"m{i}_cv2_bias"])


def generate_conv_test(path: str) -> None:
    print("Generating Conv test...")
    m = nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False)
    m.eval()
    x = torch.randn(1, 32, 64, 64)
    with torch.no_grad():
        y = m(x)

    with open(os.path.join(path, "conv_test.bin"), "wb") as f:
        save_tensor(f, "input", x)
        save_tensor(f, "weight", m.weight)
        save_tensor(f, "output", y)


def generate_c3k2_tests(path: str) -> None:
    """Unit: C3k2(64,64,2) with c3k=False (Bottleneck), e=0.5. YAML-like: 256 ch, e=0.25, shortcut=False."""
    print("Generating C3k2 tests...")
    torch.manual_seed(42)
    # Ultralytics C3k2(c1,c2,n,c3k=False,...): 4th arg is c3k, NOT shortcut
    m_unit = C3k2(64, 64, 2, c3k=False, e=0.5, shortcut=True)
    m_unit.eval()
    x_unit = torch.randn(1, 64, 16, 16)
    with torch.no_grad():
        y_unit = m_unit(x_unit)

    write_c3k2_fixture(os.path.join(path, "c3k2_unit.bin"), "unit", m_unit, x_unit, y_unit)

    torch.manual_seed(43)
    m_yaml = C3k2(256, 256, 2, c3k=False, e=0.25, shortcut=False)
    m_yaml.eval()
    x_yaml = torch.randn(1, 256, 8, 8)
    with torch.no_grad():
        y_yaml = m_yaml(x_yaml)

    write_c3k2_fixture(os.path.join(path, "c3k2_yaml.bin"), "yaml", m_yaml, x_yaml, y_yaml)


def generate_c2psa_test(path: str) -> None:
    """C2PSA(128,128,n=2,e=0.5); requires self.c//64>=1 (here c_hidden=64)."""
    print("Generating C2PSA test...")
    torch.manual_seed(46)
    m = C2PSA(128, 128, 2, 0.5)
    m.eval()
    x = torch.randn(1, 128, 8, 8)
    with torch.no_grad():
        y = m(x)
    fused = fuse_c2psa_for_c(m)
    n = len(m.m)
    with open(os.path.join(path, "c2psa_test.bin"), "wb") as f:
        save_tensor(f, "c2psa_input", x)
        save_tensor(f, "c2psa_output", y)
        save_tensor(f, "c2psa_cv1_weight", fused["cv1_weight"])
        save_tensor(f, "c2psa_cv1_bias", fused["cv1_bias"])
        save_tensor(f, "c2psa_cv2_weight", fused["cv2_weight"])
        save_tensor(f, "c2psa_cv2_bias", fused["cv2_bias"])
        for i in range(n):
            save_tensor(f, f"c2psa_m{i}_qkv_weight", fused[f"m{i}_qkv_weight"])
            save_tensor(f, f"c2psa_m{i}_qkv_bias", fused[f"m{i}_qkv_bias"])
            save_tensor(f, f"c2psa_m{i}_proj_weight", fused[f"m{i}_proj_weight"])
            save_tensor(f, f"c2psa_m{i}_proj_bias", fused[f"m{i}_proj_bias"])
            save_tensor(f, f"c2psa_m{i}_pe_weight", fused[f"m{i}_pe_weight"])
            save_tensor(f, f"c2psa_m{i}_pe_bias", fused[f"m{i}_pe_bias"])
            save_tensor(f, f"c2psa_m{i}_ffn0_weight", fused[f"m{i}_ffn0_weight"])
            save_tensor(f, f"c2psa_m{i}_ffn0_bias", fused[f"m{i}_ffn0_bias"])
            save_tensor(f, f"c2psa_m{i}_ffn1_weight", fused[f"m{i}_ffn1_weight"])
            save_tensor(f, f"c2psa_m{i}_ffn1_bias", fused[f"m{i}_ffn1_bias"])


def generate_sppf_test(path: str) -> None:
    """SPPF(128,128,k=5,n=3); optional YAML-style with shortcut (matches yolo26 SPPF tail)."""
    print("Generating SPPF test...")
    torch.manual_seed(44)
    m = SPPF(128, 128, 5, 3, shortcut=False)
    m.eval()
    x = torch.randn(1, 128, 16, 16)
    with torch.no_grad():
        y = m(x)
    fused = fuse_sppf_for_c(m)

    with open(os.path.join(path, "sppf_test.bin"), "wb") as f:
        save_tensor(f, "sppf_input", x)
        save_tensor(f, "sppf_output", y)
        save_tensor(f, "sppf_cv1_weight", fused["cv1_weight"])
        save_tensor(f, "sppf_cv1_bias", fused["cv1_bias"])
        save_tensor(f, "sppf_cv2_weight", fused["cv2_weight"])
        save_tensor(f, "sppf_cv2_bias", fused["cv2_bias"])

    torch.manual_seed(45)
    m_sc = SPPF(256, 256, 5, 3, shortcut=True)
    m_sc.eval()
    x_sc = torch.randn(1, 256, 8, 8)
    with torch.no_grad():
        y_sc = m_sc(x_sc)
    fused_sc = fuse_sppf_for_c(m_sc)
    with open(os.path.join(path, "sppf_shortcut.bin"), "wb") as f:
        save_tensor(f, "sppf_input", x_sc)
        save_tensor(f, "sppf_output", y_sc)
        save_tensor(f, "sppf_cv1_weight", fused_sc["cv1_weight"])
        save_tensor(f, "sppf_cv1_bias", fused_sc["cv1_bias"])
        save_tensor(f, "sppf_cv2_weight", fused_sc["cv2_weight"])
        save_tensor(f, "sppf_cv2_bias", fused_sc["cv2_bias"])


if __name__ == "__main__":
    test_path = "tests/data"
    os.makedirs(test_path, exist_ok=True)
    generate_conv_test(test_path)
    generate_c3k2_tests(test_path)
    generate_c2psa_test(test_path)
    generate_sppf_test(test_path)
    print("Test data generated in", test_path)
