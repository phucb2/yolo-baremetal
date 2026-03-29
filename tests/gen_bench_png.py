#!/usr/bin/env python3
"""Emit tests/fixtures/bench_640.png (solid RGB) for yolo26_bench --image. Stdlib only."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


def write_png_rgb(path: Path, w: int, h: int, r: int, g: int, b: int) -> None:
    raw = bytearray()
    row = bytes([r, g, b]) * w
    for _ in range(h):
        raw.append(0)  # filter type 0
        raw.extend(row)
    comp = zlib.compress(bytes(raw), 9)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", comp)
        + _png_chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "fixtures" / "bench_640.png"
    write_png_rgb(out, 640, 640, 42, 128, 200)
    print(out)
