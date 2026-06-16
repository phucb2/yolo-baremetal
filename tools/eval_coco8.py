"""
Use: Evaluate PyTorch and C detections on Ultralytics COCO8.
When: Compare accuracy parity between .pt and C reimplementation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _resolve_coco8() -> tuple[Path, Path, list[str]]:
    try:
        from ultralytics.data.utils import check_det_dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "ultralytics is required. Install it in your Python env, e.g. `uv add ultralytics` or `pip install ultralytics`."
        ) from exc
    data = check_det_dataset("coco8.yaml")
    names = data.get("names")
    if isinstance(names, dict):
        class_names = [names[i] for i in range(len(names))]
    elif isinstance(names, list):
        class_names = names
    else:
        raise RuntimeError("Could not parse class names from coco8.yaml")

    data_root = Path(data["path"])
    val_dir = Path(data["val"])
    if not val_dir.is_absolute():
        val_dir = data_root / val_dir
    labels_dir = data_root / "labels" / "val"
    if not val_dir.is_dir() or not labels_dir.is_dir():
        raise RuntimeError(f"Unexpected COCO8 layout: val={val_dir} labels={labels_dir}")
    return val_dir, labels_dir, class_names


def _build_gt_coco(val_dir: Path, labels_dir: Path, class_names: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    categories = [{"id": i + 1, "name": class_names[i], "supercategory": "none"} for i in range(len(class_names))]
    ann_id = 1

    image_files = sorted([p for p in val_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES])
    image_id_to_meta: list[dict[str, Any]] = []
    for image_id, img_path in enumerate(image_files, start=1):
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        h, w = bgr.shape[:2]
        images.append({"id": image_id, "file_name": img_path.name, "width": w, "height": h})
        image_id_to_meta.append({"id": image_id, "path": img_path, "width": w, "height": h})

        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
        for line in label_path.read_text(encoding="utf-8").splitlines():
            row = line.strip()
            if not row:
                continue
            parts = row.split()
            if len(parts) != 5:
                continue
            cls, cx, cy, bw, bh = map(float, parts)
            cls_i = int(cls)
            x = (cx - bw / 2.0) * w
            y = (cy - bh / 2.0) * h
            ww = bw * w
            hh = bh * h
            if ww <= 0.0 or hh <= 0.0:
                continue
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cls_i + 1,
                    "bbox": [x, y, ww, hh],
                    "area": ww * hh,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    gt = {"images": images, "annotations": annotations, "categories": categories}
    return gt, image_id_to_meta


def _xyxy_to_coco_bbox(x1: float, y1: float, x2: float, y2: float) -> list[float]:
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return [x1, y1, w, h]


def _eval_predictions(gt_dict: dict[str, Any], predictions: list[dict[str, Any]]) -> dict[str, float]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pycocotools is required. Install it in your Python env, e.g. `uv add pycocotools` or `pip install pycocotools`."
        ) from exc

    with tempfile.NamedTemporaryFile(suffix="_gt.json", delete=False) as f_gt:
        gt_path = Path(f_gt.name)
    with tempfile.NamedTemporaryFile(suffix="_pred.json", delete=False) as f_pred:
        pred_path = Path(f_pred.name)

    try:
        gt_path.write_text(json.dumps(gt_dict), encoding="utf-8")
        pred_path.write_text(json.dumps(predictions), encoding="utf-8")

        coco_gt = COCO(str(gt_path))
        coco_dt = coco_gt.loadRes(str(pred_path))
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats = coco_eval.stats
        return {"map": float(stats[0]), "map50": float(stats[1]), "map75": float(stats[2])}
    finally:
        gt_path.unlink(missing_ok=True)
        pred_path.unlink(missing_ok=True)


def _run_pt_predictions(
    model_path: str,
    image_meta: list[dict[str, Any]],
    input_size: int,
    conf: float,
) -> list[dict[str, Any]]:
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "ultralytics is required. Install it in your Python env, e.g. `uv add ultralytics` or `pip install ultralytics`."
        ) from exc
    model = YOLO(model_path)
    preds: list[dict[str, Any]] = []
    for meta in image_meta:
        image_id = meta["id"]
        img_path = Path(meta["path"])
        orig_w = float(meta["width"])
        orig_h = float(meta["height"])
        sx = orig_w / float(input_size)
        sy = orig_h / float(input_size)

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        resized = cv2.resize(bgr, (input_size, input_size), interpolation=cv2.INTER_LINEAR)

        res = model.predict(
            source=resized,
            imgsz=input_size,
            conf=conf,
            verbose=False,
            device="cpu",
            half=False,
        )[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy()
        score = res.boxes.conf.cpu().numpy()
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i].tolist()
            x1 *= sx
            x2 *= sx
            y1 *= sy
            y2 *= sy
            preds.append(
                {
                    "image_id": image_id,
                    "category_id": int(cls[i]) + 1,
                    "bbox": _xyxy_to_coco_bbox(x1, y1, x2, y2),
                    "score": float(score[i]),
                }
            )
    return preds


def _run_c_predictions(
    c_bin: str,
    c_weights: str,
    image_meta: list[dict[str, Any]],
    input_size: int,
    conf: float,
) -> list[dict[str, Any]]:
    preds: list[dict[str, Any]] = []
    for meta in image_meta:
        image_id = meta["id"]
        img_path = str(meta["path"])
        orig_w = float(meta["width"])
        orig_h = float(meta["height"])
        sx = orig_w / float(input_size)
        sy = orig_h / float(input_size)

        with tempfile.NamedTemporaryFile(suffix="_det.json", delete=False) as tmp:
            tmp_json = tmp.name
        try:
            cmd = [
                c_bin,
                "--image",
                img_path,
                "--json",
                tmp_json,
                "--weights",
                c_weights,
                "--conf",
                str(conf),
                "--no-layer-profile",
            ]
            run = subprocess.run(cmd, capture_output=True, text=True)
            if run.returncode != 0:
                raise RuntimeError(f"C inference failed for {img_path}\nstdout:\n{run.stdout}\nstderr:\n{run.stderr}")
            payload = json.loads(Path(tmp_json).read_text(encoding="utf-8"))
            for det in payload.get("detections", []):
                x1 = float(det["x1"]) * sx
                y1 = float(det["y1"]) * sy
                x2 = float(det["x2"]) * sx
                y2 = float(det["y2"]) * sy
                preds.append(
                    {
                        "image_id": image_id,
                        "category_id": int(det["class_id"]) + 1,
                        "bbox": _xyxy_to_coco_bbox(x1, y1, x2, y2),
                        "score": float(det["score"]),
                    }
                )
        finally:
            Path(tmp_json).unlink(missing_ok=True)
    return preds


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PyTorch and C implementations on Ultralytics COCO8.")
    parser.add_argument("--pt", required=True, help="Path to PyTorch .pt model")
    parser.add_argument("--c-bin", default="./yolo26_bench", help="Path to C inference binary")
    parser.add_argument("--c-weights", default="weights/yolo26.bin", help="Path to C .bin weights")
    parser.add_argument("--input-size", type=int, default=640, help="Model input size (square)")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold for both backends")
    parser.add_argument("--max-images", type=int, default=0, help="Optional cap for smoke tests (0 = all)")
    args = parser.parse_args()

    val_dir, labels_dir, class_names = _resolve_coco8()
    gt_dict, image_meta = _build_gt_coco(val_dir, labels_dir, class_names)
    if args.max_images > 0:
        image_meta = image_meta[: args.max_images]
        keep_ids = {m["id"] for m in image_meta}
        gt_dict["images"] = [im for im in gt_dict["images"] if im["id"] in keep_ids]
        gt_dict["annotations"] = [ann for ann in gt_dict["annotations"] if ann["image_id"] in keep_ids]

    print(f"COCO8 val dir: {val_dir}")
    print(f"Evaluating {len(image_meta)} image(s)")

    pt_preds = _run_pt_predictions(args.pt, image_meta, args.input_size, args.conf)
    print(f"PyTorch predictions: {len(pt_preds)}")
    pt_metrics = _eval_predictions(gt_dict, pt_preds)

    c_preds = _run_c_predictions(args.c_bin, args.c_weights, image_meta, args.input_size, args.conf)
    print(f"C predictions: {len(c_preds)}")
    c_metrics = _eval_predictions(gt_dict, c_preds)

    print("\nResults")
    print(
        f"PyTorch: mAP@[.5:.95]={pt_metrics['map']:.4f} mAP50={pt_metrics['map50']:.4f} mAP75={pt_metrics['map75']:.4f}"
    )
    print(f"C      : mAP@[.5:.95]={c_metrics['map']:.4f} mAP50={c_metrics['map50']:.4f} mAP75={c_metrics['map75']:.4f}")
    print(
        f"Delta  : mAP={c_metrics['map'] - pt_metrics['map']:+.4f} "
        f"mAP50={c_metrics['map50'] - pt_metrics['map50']:+.4f} "
        f"mAP75={c_metrics['map75'] - pt_metrics['map75']:+.4f}"
    )


if __name__ == "__main__":
    main()
