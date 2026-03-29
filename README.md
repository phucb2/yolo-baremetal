# YOLO26 C inference

**What:** Minimal C inference stack for Ultralytics-style YOLO26 (CPU, SIMD). **When:** Use this repo to run or benchmark the model from exported weights; see `plan.md` for roadmap and status.

## Requirements

- **macOS** (camera + bench use AVFoundation / Darwin frameworks)
- **Clang**, weights at `weights/yolo26.bin` (export from `.pt` first)

## Build & test

```bash
make              # builds yolo26_bench
make verify       # runs tests/test_core + Python syntax check on tools/
```

## Run (live camera)

```bash
./yolo26_bench              # 5 frames, stdout
./yolo26_bench out.bmp      # also writes last annotated frame
```

## Export weights (Python)

Use conda env with PyTorch (e.g. `conda activate py39`):

```bash
python tools/converter.py --model <checkpoint.pt> --output weights/yolo26.bin
```

Regenerate layer goldens: `make regenerate-golden` (uses `tools/with_py39.sh`).

## Layout

| Path | Role |
|------|------|
| `src/` | Tensor, layers, model forward, detect, camera, visualize |
| `include/` | Headers |
| `tests/` | C unit / parity tests |
| `tools/` | `.pt` → `.bin` converter, golden generation |
