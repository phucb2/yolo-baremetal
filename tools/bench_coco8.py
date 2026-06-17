"""
Use: Compare PyTorch vs C inference speed on Ultralytics COCO8.
When: Fair timing with model loaded once per backend (no per-image subprocess reload).
"""

from __future__ import annotations

import argparse
import re
import statistics
import subprocess
import time
from pathlib import Path

import cv2
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _resolve_coco8_val_dir() -> Path:
    data = check_det_dataset("coco8.yaml")
    val_dir = Path(data["val"])
    if not val_dir.is_absolute():
        val_dir = Path(data["path"]) / val_dir
    if not val_dir.is_dir():
        raise RuntimeError(f"COCO8 val dir not found: {val_dir}")
    return val_dir


def _list_images(val_dir: Path) -> list[Path]:
    return sorted([p for p in val_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES])


def _bench_pytorch(images: list[Path], model_path: str, input_size: int, conf: float, warmup: int, runs: int) -> dict[str, float]:
    model = YOLO(model_path)
    try:
        model.fuse()
    except Exception:
        pass

    bgr_resized: list = []
    for p in images:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read image: {p}")
        bgr_resized.append(cv2.resize(bgr, (input_size, input_size), interpolation=cv2.INTER_LINEAR))

    for _ in range(warmup):
        model.predict(source=bgr_resized[0], imgsz=input_size, conf=conf, verbose=False, device="cpu", half=False)

    inf_times: list[float] = []
    full_times: list[float] = []
    for _ in range(runs):
        for p, bgr in zip(images, bgr_resized):
            t0 = time.perf_counter()
            model.predict(source=bgr, imgsz=input_size, conf=conf, verbose=False, device="cpu", half=False)
            inf_times.append((time.perf_counter() - t0) * 1000.0)

        for p in images:
            t0 = time.perf_counter()
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            bgr = cv2.resize(bgr, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
            model.predict(source=bgr, imgsz=input_size, conf=conf, verbose=False, device="cpu", half=False)
            full_times.append((time.perf_counter() - t0) * 1000.0)

    return {
        "inference_mean_ms": statistics.mean(inf_times),
        "inference_median_ms": statistics.median(inf_times),
        "full_pipeline_mean_ms": statistics.mean(full_times),
        "full_pipeline_median_ms": statistics.median(full_times),
    }


def _parse_c_bench(stdout: str) -> dict[str, float | str]:
    keys = {
        "model_load_weights (once)": "load_once_ms",
        "load_image": "load_image_ms",
        "resize": "resize_ms",
        "preprocess": "preprocess_ms",
        "inference": "inference_ms",
        "decode": "decode_ms",
        "pipeline": "pipeline_ms",
    }
    out: dict[str, float | str] = {}
    for line in stdout.splitlines():
        qm = re.search(r"quantized=(yes|no)", line)
        if qm:
            out["quantized"] = qm.group(1)
        for prefix, name in keys.items():
            if prefix in line:
                m = re.search(r"([0-9.]+)\s*ms", line)
                if m:
                    out[name] = float(m.group(1))
    required = ("inference_ms", "pipeline_ms", "load_once_ms")
    if not all(k in out for k in required):
        raise RuntimeError(f"Failed to parse C bench output:\n{stdout}")
    return out


def _bench_c(
    images: list[Path],
    c_bin: str,
    c_weights: str,
    conf: float,
    warmup: int,
    runs: int,
) -> dict[str, float | str]:
    cmd = [
        c_bin,
        "--weights",
        c_weights,
        "--conf",
        str(conf),
        "--runs",
        str(runs),
        "--warmup",
        str(warmup),
        "--dir",
        str(images[0].parent),
    ]
    run = subprocess.run(cmd, capture_output=True, text=True)
    if run.returncode != 0:
        raise RuntimeError(f"C bench failed\nstdout:\n{run.stdout}\nstderr:\n{run.stderr}")
    return _parse_c_bench(run.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs C on COCO8 (model loaded once).")
    parser.add_argument("--pt", default="weights/yolo26n.pt", help="PyTorch .pt model")
    parser.add_argument("--c-bin", default="./tests/bench_coco8", help="C bench binary")
    parser.add_argument("--c-weights", default="weights/yolo26_int8.bin", help="C .bin weights (default: INT8)")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    val_dir = _resolve_coco8_val_dir()
    images = _list_images(val_dir)
    if not images:
        raise SystemExit(f"No images in {val_dir}")

    print(f"COCO8 val dir: {val_dir}")
    print(f"Images: {len(images)} | runs={args.runs} warmup={args.warmup} | input={args.input_size} CPU")
    print(f"C weights: {args.c_weights}")
    print()

    pt = _bench_pytorch(images, args.pt, args.input_size, args.conf, args.warmup, args.runs)
    c = _bench_c(images, args.c_bin, args.c_weights, args.conf, args.warmup, args.runs)

    print("PyTorch (model loaded once)")
    print(f"  inference (pre-resized): {pt['inference_mean_ms']:.2f} ms mean, {pt['inference_median_ms']:.2f} ms median")
    print(f"  full pipeline:           {pt['full_pipeline_mean_ms']:.2f} ms mean, {pt['full_pipeline_median_ms']:.2f} ms median")
    print()
    print("C (model loaded once)")
    if "quantized" in c:
        print(f"  quantized:               {c['quantized']}")
    print(f"  model_load_weights:      {c['load_once_ms']:.2f} ms (one-time)")
    print(f"  inference:               {c['inference_ms']:.2f} ms mean")
    print(f"  decode:                  {c.get('decode_ms', 0.0):.2f} ms mean")
    print(f"  full pipeline:           {c['pipeline_ms']:.2f} ms mean")
    print()
    print("Comparison (mean)")
    ratio_inf = float(c["inference_ms"]) / pt["inference_mean_ms"]
    ratio_full = float(c["pipeline_ms"]) / pt["full_pipeline_mean_ms"]
    print(f"  C inference / PT inference:      {ratio_inf:.2f}x ({'faster' if ratio_inf < 1 else 'slower'})")
    print(f"  C pipeline  / PT full pipeline:  {ratio_full:.2f}x ({'faster' if ratio_full < 1 else 'slower'})")
    print(f"  PT throughput (inference):       {1000.0 / pt['inference_mean_ms']:.2f} img/s")
    print(f"  C throughput (inference):        {1000.0 / float(c['inference_ms']):.2f} img/s")


if __name__ == "__main__":
    main()
