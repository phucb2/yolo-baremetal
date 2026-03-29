"""
Compare two stage dump .bin files (named tensors, same format as generate_layer_tests).
For identical stage_00_input, generate runs/shared_input.bin once:
  python tools/dump_model_stages.py --write-shared-input runs/shared_input.bin --image tests/data/zidane.jpg
Then: tests/dump_model_stages runs/shared_input.bin ...  and  python tools/dump_model_stages.py --input-bin runs/shared_input.bin ...
Skips stage_24_buf23 / stage_25_detect by default (optional; C and Py align when C loads one2one_cv2/cv3).
Usage: python tools/compare_stage_bins.py runs/zidane_c_stages.bin runs/zidane_py_stages.bin [--atol 1e-3] [--include-detect]
"""

from __future__ import annotations

import argparse
import struct
import sys
from typing import BinaryIO


def read_one_tensor(f: BinaryIO):
    hdr = f.read(4)
    if len(hdr) < 4:
        return None, None
    nl = struct.unpack("i", hdr)[0]
    name = f.read(nl).decode("ascii")
    nd = struct.unpack("i", f.read(4))[0]
    dims = []
    for _ in range(nd):
        dims.append(struct.unpack("i", f.read(4))[0])
    while len(dims) < 4:
        dims.append(1)
    n, c, h, w = dims[0], dims[1], dims[2], dims[3]
    count = n * c * h * w
    raw = f.read(4 * count)
    if len(raw) != 4 * count:
        return None, None
    import array

    arr = array.array("f")
    arr.frombytes(raw)
    return name, (n, c, h, w, arr)


def load_all(path: str) -> dict[str, tuple]:
    out: dict[str, tuple] = {}
    with open(path, "rb") as f:
        while True:
            name, payload = read_one_tensor(f)
            if name is None:
                break
            out[name] = payload
    return out


def max_abs_diff(a: array.array, b: array.array, shape: tuple[int, int, int, int]) -> tuple[float, int]:
    n, c, h, w = shape
    count = n * c * h * w
    if len(a) != count or len(b) != count:
        return float("inf"), 0
    m = 0.0
    ix = 0
    for i in range(count):
        d = abs(a[i] - b[i])
        if d > m:
            m = d
            ix = i
    return m, ix


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("c_bin")
    p.add_argument("py_bin")
    p.add_argument("--atol", type=float, default=1e-3)
    p.add_argument("--include-detect", action="store_true", help="also compare stage_24_buf23 / stage_25_detect")
    args = p.parse_args()

    c_map = load_all(args.c_bin)
    py_map = load_all(args.py_bin)
    names = sorted(c_map.keys())
    skip_detect = frozenset({"stage_24_buf23", "stage_25_detect"})

    failed = False
    for name in names:
        if not args.include_detect and name in skip_detect:
            print(f"SKIP {name} (use --include-detect to compare)")
            continue
        if name not in py_map:
            print(f"MISSING in py: {name}")
            failed = True
            continue
        c_pl = c_map[name]
        py_pl = py_map[name]
        if c_pl[0:4] != py_pl[0:4]:
            print(f"SHAPE mismatch {name}: c{c_pl[0:4]} py{py_pl[0:4]}")
            failed = True
            continue
        mad, ix = max_abs_diff(c_pl[4], py_pl[4], c_pl[0:4])
        if mad > args.atol:
            print(f"FAIL {name}: max_abs_diff={mad:.6g} at flat_index={ix} atol={args.atol}")
            failed = True
        else:
            print(f"OK   {name}: max_abs_diff={mad:.6g}")

    for name in sorted(py_map.keys()):
        if name not in c_map:
            print(f"EXTRA in py (not in c): {name}")
            failed = True

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
