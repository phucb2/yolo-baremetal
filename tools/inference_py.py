"""
Run Ultralytics YOLO26 .pt inference (camera or image) for comparison with yolo26_bench.
Uses stretch resize to 640×640, BGR→RGB, /255 — aligned with C preprocess in src/main.c.
Requires: torch, ultralytics, opencv-python (e.g. conda env py39). Example:
  bash tools/with_py39.sh python tools/inference_py.py --model weights/yolo26n.pt --image a.bmp --out-dir runs/py_cmp
  bash tools/with_py39.sh python tools/inference_py.py --model weights/yolo26n.pt --camera --frames 5 --out-dir runs/py_cmp
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from ultralytics import YOLO


def resize_bgr_like_c(bgr: np.ndarray, w: int, h: int) -> np.ndarray:
    """Match C camera path: linear stretch to WxH; keep BGR uint8 (Ultralytics /255 matches main.c)."""
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"expected HxWx3 BGR image, got shape {bgr.shape}")
    return cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)


def load_bgr(path: str) -> np.ndarray:
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return im


def print_detections_like_c(results, frame_idx: int | None = None) -> int:
    """Print in the same style as yolo26_bench (decode_detections @ conf threshold already applied by Ultralytics)."""
    if frame_idx is not None:
        print(f"\n--- Frame {frame_idx} ---")
    r = results[0]
    boxes = r.boxes
    n = len(boxes) if boxes is not None else 0
    print(f"Detections: {n}")
    if n == 0:
        return 0
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(np.int32)
    for d in range(n):
        x1, y1, x2, y2 = xyxy[d]
        print(
            f"  [{d}] Class {int(cls[d])}: {float(conf[d]):.2f} @ "
            f"({float(x1):.1f}, {float(y1):.1f}, {float(x2):.1f}, {float(y2):.1f})"
        )
    return n


def save_frame_outputs(
    results,
    out_dir: str,
    frame_index: int,
    image_path: str | None,
    single_image: bool,
) -> str:
    """Write annotated BGR image; return path written."""
    os.makedirs(out_dir, exist_ok=True)
    plot_bgr = results[0].plot()
    if single_image and image_path:
        stem = Path(image_path).stem
        name = f"{stem}_annotated.png"
    else:
        name = f"frame_{frame_index:04d}.png"
    path = os.path.join(out_dir, name)
    if not cv2.imwrite(path, plot_bgr):
        raise OSError(f"failed to write image: {path}")
    return path


def run_predict(
    model: YOLO,
    bgr_640: np.ndarray,
    conf: float,
    imgsz: int,
    bench: bool,
) -> Tuple[object, float | None]:
    """Run inference on BGR uint8 HxWx3 (640²); Ultralytics scales by 1/255 like src/main.c preprocess."""
    t0 = time.perf_counter()
    results = model.predict(
        source=bgr_640,
        imgsz=imgsz,
        conf=conf,
        verbose=False,
        device="cpu",
        half=False,
    )
    elapsed = time.perf_counter() - t0
    if bench:
        print(f"inference: {elapsed * 1000:.3f} ms")
    return results, elapsed


def main() -> None:
    p = argparse.ArgumentParser(description="YOLO26 PyTorch inference vs C bench (640², /255).")
    p.add_argument("--model", type=str, required=True, help="Path to .pt checkpoint (same as tools/converter.py --model)")
    p.add_argument("--image", type=str, default=None, help="Image path (bmp, jpg, …). Mutually exclusive with --camera.")
    p.add_argument("--camera", action="store_true", help="Use default webcam (OpenCV)")
    p.add_argument("--camera-index", type=int, default=0, help="VideoCapture device index (default 0)")
    p.add_argument("--frames", type=int, default=None, help="Number of frames (camera default 5; image always 1)")
    p.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (matches C decode_detections threshold)")
    p.add_argument("--imgsz", type=int, default=640, help="Square size W=H (must match C model)")
    p.add_argument("--bench", action="store_true", help="Print inference wall time per frame")
    p.add_argument(
        "--out-dir",
        type=str,
        default="py_inference_out",
        help="Directory for annotated PNGs (created if missing). Use --no-save to skip files.",
    )
    p.add_argument("--no-save", action="store_true", help="Do not write images; print detections only")
    args = p.parse_args()

    if bool(args.image) == bool(args.camera):
        p.error("specify exactly one of --image PATH or --camera")

    imgsz = int(args.imgsz)
    if imgsz <= 0:
        p.error("--imgsz must be positive")

    nframes = 1 if args.image else (args.frames if args.frames is not None else 5)
    if nframes < 1:
        p.error("--frames must be >= 1")

    do_save = not args.no_save
    out_dir = args.out_dir
    if do_save and not (out_dir and out_dir.strip()):
        p.error("--out-dir must be non-empty when saving (or pass --no-save)")

    print(f"Loading {args.model}...")
    model = YOLO(args.model)
    try:
        model.fuse()
        print("Applied fuse() (matches fused weights in yolo26.bin).")
    except Exception as e:
        print(f"fuse() skipped: {e}")

    cap = None
    if args.camera:
        cap = cv2.VideoCapture(args.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"could not open camera index {args.camera_index}")

    try:
        for i in range(nframes):
            if args.image:
                bgr = load_bgr(args.image)
            else:
                assert cap is not None
                t_cap0 = time.perf_counter()
                ok, bgr = cap.read()
                t_cap = time.perf_counter() - t_cap0
                if not ok or bgr is None:
                    raise RuntimeError("camera read failed")
                if args.bench:
                    print(f"capture: {t_cap * 1000:.3f} ms")

            t_pre0 = time.perf_counter()
            bgr_640 = resize_bgr_like_c(bgr, imgsz, imgsz)
            t_pre = time.perf_counter() - t_pre0
            if args.bench:
                print(f"preprocess: {t_pre * 1000:.3f} ms")

            results, _ = run_predict(model, bgr_640, conf=args.conf, imgsz=imgsz, bench=args.bench)
            print_detections_like_c(results, frame_idx=i if args.camera or nframes > 1 else None)
            if do_save:
                assert out_dir is not None
                path = save_frame_outputs(
                    results,
                    out_dir,
                    frame_index=i,
                    image_path=args.image,
                    single_image=bool(args.image) and nframes == 1,
                )
                print(f"Saved: {path}")
    finally:
        if cap is not None:
            cap.release()


if __name__ == "__main__":
    main()
