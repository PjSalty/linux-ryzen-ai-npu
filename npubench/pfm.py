"""Minimal PFM (Portable Float Map) reader and writer, the format oidnDenoise uses."""

from __future__ import annotations

import numpy as np


def read_pfm(path: str) -> np.ndarray:
    """Returns an HWC float32 image (C is 1 or 3), top row first."""
    with open(path, "rb") as f:
        kind = f.readline().strip()
        if kind not in (b"PF", b"Pf"):
            raise ValueError(f"{path}: not a PFM file")
        width, height = map(int, f.readline().split())
        scale = float(f.readline().strip())
        channels = 3 if kind == b"PF" else 1
        dtype = "<f4" if scale < 0 else ">f4"
        data = np.frombuffer(f.read(width * height * channels * 4), dtype=dtype)
    img = data.reshape(height, width, channels)[::-1]
    return np.ascontiguousarray(img, dtype=np.float32)


def write_pfm(path: str, img: np.ndarray) -> None:
    img = np.asarray(img, dtype="<f4")
    if img.ndim == 2:
        img = img[..., None]
    height, width, channels = img.shape
    kind = b"PF" if channels == 3 else b"Pf"
    with open(path, "wb") as f:
        f.write(kind + b"\n" + f"{width} {height}\n-1.0\n".encode())
        f.write(np.ascontiguousarray(img[::-1]).tobytes())
