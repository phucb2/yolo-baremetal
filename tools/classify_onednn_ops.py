"""Classify YOLO26 forward ops by oneDNN primitive groups (% of compute/time)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from calibrate_quant import load_fp32_bin_tensors

WEIGHTS = ROOT / "weights" / "yolo26.bin"

# Spatial H=W after each model index (640 input).
SPATIAL = {
    0: 320,
    1: 160,
    2: 160,
    3: 80,
    4: 80,
    5: 40,
    6: 40,
    7: 20,
    8: 20,
    9: 20,
    10: 20,
    11: 40,
    12: 40,
    13: 40,
    14: 80,
    15: 80,
    16: 80,
    17: 40,
    18: 40,
    19: 40,
    20: 20,
    21: 20,
    22: 20,
}

# Measured model_forward profile (ms), 640x640, FP32, Windows AVX2.
PROFILE_MS = {
    "conv_s2_silu": 14.19 + 20.73 + 26.84 + 18.87 + 4.92 + 4.63 + 2.91,
    "c3k2_blocks": 35.43 + 20.96 + 10.45 + 7.90 + 13.23 + 24.99 + 11.43 + 18.54,
    "sppf": 5.49,
    "c2psa": 14.63,
    "upsample": 0.92 + 1.82,
    "concat": 0.32 + 0.81 + 0.21 + 0.11,
    "detect": 48.25,
    "copy": 0.001,
}


def conv_flops(oc: int, ic: int, kh: int, kw: int, h: int, w: int, groups: int = 1) -> float:
    return 2.0 * oc * (ic // groups) * kh * kw * h * w


def is_depthwise(name: str, oc: int, ic: int, kh: int, kw: int) -> bool:
    if kh != 3 or kw != 3 or oc != ic:
        return False
    if "pe.conv.weight" in name:
        return True
    if "cv3" in name and (".0.0.conv.weight" in name or ".1.0.conv.weight" in name):
        return True
    return False


def spatial_for_weight(name: str) -> int:
    m = re.search(r"model\.(\d+)", name)
    if not m:
        return 20
    li = int(m.group(1))
    if li == 23:
        if re.search(r"\.(0\.|cv2\.0|cv3\.0)", name):
            return 80
        if re.search(r"\.(1\.|cv2\.1|cv3\.1)", name):
            return 40
        return 20
    return SPATIAL.get(li, 20)


def flop_breakdown(tensors: dict[str, object]) -> dict[str, float]:
    cats = {
        "matmul_1x1": 0.0,
        "conv_3x3": 0.0,
        "depthwise_conv": 0.0,
        "pooling_max": 0.0,
        "matmul_attn": 0.0,
        "softmax": 0.0,
        "eltwise": 0.0,
        "resampling": 0.0,
        "concat": 0.0,
        "custom_post": 0.0,
    }

    for name, arr in tensors.items():
        dims = tuple(int(x) for x in arr.shape)
        if not name.endswith(".weight") or len(dims) != 4:
            continue
        oc, ic, kh, kw = dims
        s = spatial_for_weight(name)
        fuse_silu = kh == 1 or (kh == 3 and "detect" not in name and not name.endswith(".2.weight"))
        if is_depthwise(name, oc, ic, kh, kw):
            cats["depthwise_conv"] += conv_flops(oc, ic, kh, kw, s, s, oc)
            if fuse_silu:
                cats["eltwise"] += 3.0 * oc * s * s
        elif kh == 1 and kw == 1:
            cats["matmul_1x1"] += conv_flops(oc, ic, kh, kw, s, s)
            if fuse_silu:
                cats["eltwise"] += 3.0 * oc * s * s
        elif kh == 3 and kw == 3:
            cats["conv_3x3"] += conv_flops(oc, ic, kh, kw, s, s)
            if fuse_silu:
                cats["eltwise"] += 3.0 * oc * s * s

    # SPPF: 3× maxpool k=5 @ 20×20, ~256 channels
    cats["pooling_max"] += 3 * 256 * 20 * 20 * 25

    def attn_flops(dim: int, heads: int, attn_ratio: float, h: int) -> tuple[float, float]:
        n = h * h
        kd = max(1, int((dim // heads) * attn_ratio))
        hd = dim // heads
        matmul = heads * (2 * n * n * kd + 2 * n * n * hd)
        softmax = heads * 3 * n * n
        return matmul, softmax

    m, sm = attn_flops(256, 8, 0.5, 20)
    cats["matmul_attn"] += m
    cats["softmax"] += sm
    m, sm = attn_flops(128, 2, 0.5, 20)
    cats["matmul_attn"] += m
    cats["softmax"] += sm

    cats["resampling"] += 512 * 20 * 20 + 256 * 40 * 40
    cats["concat"] += (256 + 256) * 40 * 40 + (128 + 128) * 80 * 80
    cats["concat"] += (128 + 128) * 40 * 40 + (256 + 256) * 20 * 20
    cats["custom_post"] += 8400 * 80
    cats["eltwise"] += 30 * 256 * 80 * 80  # residual adds (rough)

    return cats


def timing_to_onednn(profile: dict[str, float]) -> dict[str, float]:
    """Map profile step groups to oneDNN primitives using in-block structure ratios."""
    p = profile
    return {
        "Convolution 3x3 (dnnl_convolution)": p["conv_s2_silu"] * 0.85 + p["c3k2_blocks"] * 0.42 + p["detect"] * 0.22,
        "MatMul / Conv 1x1 (dnnl_matmul)": p["c3k2_blocks"] * 0.38
        + p["sppf"] * 0.35
        + p["c2psa"] * 0.28
        + p["detect"] * 0.38,
        "Depthwise conv (dnnl_deconvolution/dw)": p["detect"] * 0.22 + p["c2psa"] * 0.08,
        "Pooling max (dnnl_pooling)": p["sppf"] * 0.40,
        "MatMul attention (dnnl_matmul)": p["c2psa"] * 0.35 + p["c3k2_blocks"] * 0.05,
        "Softmax (dnnl_softmax)": p["c2psa"] * 0.12 + p["c3k2_blocks"] * 0.02,
        "Eltwise SiLU/add (dnnl_eltwise)": p["conv_s2_silu"] * 0.15
        + p["c3k2_blocks"] * 0.08
        + p["sppf"] * 0.10
        + p["c2psa"] * 0.05
        + p["detect"] * 0.08,
        "Resampling nearest (dnnl_resampling)": p["upsample"],
        "Concat (dnnl_concat)": p["concat"],
        "Custom detect post (not oneDNN)": p["detect"] * 0.30,
        "Memory copy (not oneDNN)": p["copy"],
    }


def pct_table(items: dict[str, float]) -> list[tuple[str, float, float]]:
    total = sum(items.values())
    rows = [(k, v, 100.0 * v / total) for k, v in items.items()]
    rows.sort(key=lambda x: -x[1])
    return rows


def main() -> None:
    if not WEIGHTS.exists():
        raise SystemExit(f"weights not found: {WEIGHTS}")

    tensors = load_fp32_bin_tensors(str(WEIGHTS))
    flops = flop_breakdown(tensors)
    odnn_time = timing_to_onednn(PROFILE_MS)

    print("YOLO26 @ 640x640 — ops grouped by oneDNN primitive\n")

    print("A) Static FLOP mix (from weight shapes + graph structure)")
    print(f"{'oneDNN group':<40} {'%':>7}")
    print("-" * 49)
    for name, _, pct in pct_table(flops):
        label = {
            "matmul_1x1": "MatMul / Conv 1x1 (dnnl_matmul)",
            "conv_3x3": "Convolution 3x3 (dnnl_convolution)",
            "depthwise_conv": "Depthwise conv",
            "pooling_max": "Pooling max (dnnl_pooling)",
            "matmul_attn": "MatMul attention (dnnl_matmul)",
            "softmax": "Softmax (dnnl_softmax)",
            "eltwise": "Eltwise SiLU/add (dnnl_eltwise)",
            "resampling": "Resampling (dnnl_resampling)",
            "concat": "Concat (dnnl_concat)",
            "custom_post": "Custom detect post (not oneDNN)",
        }[name]
        print(f"{label:<40} {pct:6.2f}%")

    conv_total = flops["matmul_1x1"] + flops["conv_3x3"] + flops["depthwise_conv"]
    flop_total = sum(flops.values())
    print(f"\n  Conv/GEMM family total: {100 * conv_total / flop_total:.1f}%")

    print("\nB) Measured time mix (model_forward profile, ~309 ms)")
    print(f"{'oneDNN group':<40} {'ms':>8} {'%':>7}")
    print("-" * 57)
    for name, ms, pct in pct_table(odnn_time):
        print(f"{name:<40} {ms:8.2f} {pct:6.2f}%")

    dnnl_core = sum(
        v
        for k, v in odnn_time.items()
        if "not oneDNN" not in k and "Memory copy" not in k
    )
    total_ms = sum(odnn_time.values())
    print(f"\n  oneDNN-mappable compute: {100 * dnnl_core / total_ms:.1f}%")
    print(f"  Custom / memory:         {100 * (total_ms - dnnl_core) / total_ms:.1f}%")


if __name__ == "__main__":
    main()
