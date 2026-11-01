# Copyright 2018 Intel Corporation
# Copyright 2026 the linux-ryzen-ai-npu contributors
# SPDX-License-Identifier: Apache-2.0
#
# The UNet definition, the TZA reader, the PU transfer function and the
# autoexposure are adapted from Intel Open Image Denoise 2.5.1, training/
# (https://github.com/RenderKit/oidn/tree/v2.5.1/training), Apache-2.0.
# The weights come from https://github.com/RenderKit/oidn-weights (Apache-2.0).
"""Open Image Denoise's RT UNet in PyTorch, its ONNX export, and OIDN's exact
input and output processing in NumPy, so the network can run anywhere ONNX
Runtime runs (CPU, or the NPU through the Vitis AI execution provider)."""

from __future__ import annotations

import os
import struct

import numpy as np

# -----------------------------------------------------------------------------
# TZA weight files
# -----------------------------------------------------------------------------

_TZA_MAGIC = 0x41D7
_TZA_DTYPES = {"f": np.float32, "h": np.float16, "b": np.int8, "B": np.uint8}


def read_tza(path: str) -> dict[str, np.ndarray]:
    """Returns every tensor in a TZA file as a float32 array, keyed by name."""
    with open(path, "rb") as f:
        data = f.read()
    magic, major, _minor, table_offset = struct.unpack_from("<HBBQ", data, 0)
    if magic != _TZA_MAGIC or major != 2:
        raise ValueError(f"{path}: not a TZA v2 file")
    pos = table_offset
    (count,) = struct.unpack_from("<I", data, pos)
    pos += 4
    tensors = {}
    for _ in range(count):
        (name_len,) = struct.unpack_from("<H", data, pos)
        pos += 2
        name = data[pos : pos + name_len].decode()
        pos += name_len
        (ndims,) = struct.unpack_from("<B", data, pos)
        pos += 1
        shape = struct.unpack_from(f"<{ndims}I", data, pos)
        pos += 4 * ndims
        pos += ndims  # layout string, "oihw" for conv weights and "x" for biases
        dtype = _TZA_DTYPES[chr(data[pos])]
        pos += 1
        (offset,) = struct.unpack_from("<Q", data, pos)
        pos += 8
        count_elems = int(np.prod(shape))
        arr = np.frombuffer(data, dtype=dtype, count=count_elems, offset=offset)
        tensors[name] = arr.reshape(shape).astype(np.float32)
    return tensors


# -----------------------------------------------------------------------------
# UNet (needs PyTorch; imported lazily so inference-only users do not need it)
# -----------------------------------------------------------------------------

ALIGNMENT = 16  # input height and width must be multiples of this


def build_unet(tza_path: str, small: bool = False, upsample: str = "resize"):
    """Builds OIDN's UNet in float32 and loads the weights from a TZA file.

    upsample="resize" is OIDN's nearest-neighbour upsampling (ONNX Resize).
    The other two modes compute exactly the same values with ops that Ryzen AI
    A16W8 supports (it has no Resize): "convtranspose" as a stride-2
    ConvTranspose with identity weights, and "depthtospace" by repeating every
    channel four times (Reshape, Expand, Reshape) and DepthToSpace (CRD)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    if upsample not in ("resize", "convtranspose", "depthtospace"):
        raise ValueError(f"unknown upsample mode {upsample!r}")

    def conv(ic: int, oc: int) -> nn.Conv2d:
        return nn.Conv2d(ic, oc, 3, padding=1)

    class Upsample2x(nn.Module):
        """Nearest-neighbour 2x upsampling as a fixed ConvTranspose."""

        def __init__(self, c: int):
            super().__init__()
            self.ct = nn.ConvTranspose2d(c, c, 2, stride=2, bias=False)
            with torch.no_grad():
                self.ct.weight.zero_()
                idx = torch.arange(c)
                self.ct.weight[idx, idx] = 1.0
            self.ct.weight.requires_grad_(False)

        def forward(self, t):
            return self.ct(t)

    class UNet(nn.Module):
        def __init__(self, ic: int = 9, oc: int = 3):
            super().__init__()
            if small:
                ec1, ec2, ec3, ec4, ec5 = 32, 32, 32, 32, 32
                dc4, dc3, dc2a, dc2b, dc1a, dc1b = 64, 64, 64, 32, 32, 32
            else:
                ec1, ec2, ec3, ec4, ec5 = 32, 48, 64, 80, 96
                dc4, dc3, dc2a, dc2b, dc1a, dc1b = 112, 96, 64, 64, 64, 32
            self.enc_conv0 = conv(ic, ec1)
            self.enc_conv1 = conv(ec1, ec1)
            self.enc_conv2 = conv(ec1, ec2)
            self.enc_conv3 = conv(ec2, ec3)
            self.enc_conv4 = conv(ec3, ec4)
            self.enc_conv5a = conv(ec4, ec5)
            self.enc_conv5b = conv(ec5, ec5)
            self.dec_conv4a = conv(ec5 + ec3, dc4)
            self.dec_conv4b = conv(dc4, dc4)
            self.dec_conv3a = conv(dc4 + ec2, dc3)
            self.dec_conv3b = conv(dc3, dc3)
            self.dec_conv2a = conv(dc3 + ec1, dc2a)
            self.dec_conv2b = conv(dc2a, dc2b)
            self.dec_conv1a = conv(dc2b + ic, dc1a)
            self.dec_conv1b = conv(dc1a, dc1b)
            self.dec_conv0 = conv(dc1b, oc)
            if upsample == "convtranspose":
                self.up4, self.up3 = Upsample2x(ec5), Upsample2x(dc4)
                self.up2, self.up1 = Upsample2x(dc3), Upsample2x(dc2b)

        def up(self, t, level: int):
            if upsample == "convtranspose":
                return getattr(self, f"up{level}")(t)
            if upsample == "depthtospace":
                n, c, h, w = t.shape
                rep = t.reshape(n, c, 1, h, w).expand(n, c, 4, h, w).reshape(n, 4 * c, h, w)
                return F.pixel_shuffle(rep, 2)
            return F.interpolate(t, scale_factor=2, mode="nearest")

        def forward(self, inp):
            relu, pool = F.relu, (lambda t: F.max_pool2d(t, 2, 2))
            x = relu(self.enc_conv0(inp))
            x = pool1 = pool(relu(self.enc_conv1(x)))
            x = pool2 = pool(relu(self.enc_conv2(x)))
            x = pool3 = pool(relu(self.enc_conv3(x)))
            x = pool(relu(self.enc_conv4(x)))
            x = relu(self.enc_conv5a(x))
            x = relu(self.enc_conv5b(x))
            x = torch.cat((self.up(x, 4), pool3), 1)
            x = relu(self.dec_conv4b(relu(self.dec_conv4a(x))))
            x = torch.cat((self.up(x, 3), pool2), 1)
            x = relu(self.dec_conv3b(relu(self.dec_conv3a(x))))
            x = torch.cat((self.up(x, 2), pool1), 1)
            x = relu(self.dec_conv2b(relu(self.dec_conv2a(x))))
            x = torch.cat((self.up(x, 1), inp), 1)
            x = relu(self.dec_conv1b(relu(self.dec_conv1a(x))))
            return self.dec_conv0(x)

    model = UNet()
    weights = read_tza(tza_path)
    state = {k: torch.from_numpy(v) for k, v in weights.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Only the fixed upsampling layers may be absent from the TZA file.
    if unexpected or any(not k.startswith("up") for k in missing):
        raise ValueError(f"weights do not match the UNet: missing={missing} unexpected={unexpected}")
    return model.eval()


def export_onnx(
    tza_path: str,
    onnx_path: str,
    height: int,
    width: int,
    small: bool = False,
    opset: int = 17,
    upsample: str = "resize",
) -> None:
    """Exports the UNet with a static input shape [1, 9, height, width]."""
    import torch

    if height % ALIGNMENT or width % ALIGNMENT:
        raise ValueError(f"height and width must be multiples of {ALIGNMENT}")
    model = build_unet(tza_path, small=small, upsample=upsample)
    dummy = torch.zeros(1, 9, height, width, dtype=torch.float32)
    torch.onnx.export(
        model,
        (dummy,),
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        dynamo=False,
    )
    if upsample == "depthtospace":
        fold_constants(onnx_path)


def fold_constants(onnx_path: str) -> None:
    """Folds the shape arithmetic torch emits around Expand into constants,
    with ONNX Runtime's hardware-independent basic optimizations."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.optimized_model_filepath = onnx_path + ".folded"
    ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    os.replace(onnx_path + ".folded", onnx_path)


# -----------------------------------------------------------------------------
# Tiled inference, with OIDN's overlap rule
# -----------------------------------------------------------------------------

# OIDN (core/unet_filter.cpp): overlap = round_up(receptive field / 2, 16); the
# UNet's receptive field is 174 pixels.
TILE_OVERLAP = 96


def tile_plan(total: int, tile: int, overlap: int = TILE_OVERLAP) -> list[tuple[int, int, int]]:
    """Splits one axis into tiles. Returns (tile_start, out_start, out_end):
    each tile is read at [tile_start, tile_start + tile) and supplies output
    [out_start, out_end), so every output pixel keeps at least `overlap`
    pixels of context inside its tile."""
    if tile >= total:
        return [(0, 0, total)]
    if tile <= 2 * overlap:
        raise ValueError("tile must be larger than twice the overlap")
    n = 1 + -(-(total - tile) // (tile - 2 * overlap))
    starts = [round(i * (total - tile) / (n - 1)) for i in range(n)]
    plan = []
    for i, s in enumerate(starts):
        lo = 0 if i == 0 else (starts[i - 1] + tile + s) // 2
        hi = total if i == n - 1 else (s + tile + starts[i + 1]) // 2
        plan.append((s, lo, hi))
    return plan


def run_tiled(run, x: np.ndarray, tile_h: int, tile_w: int) -> np.ndarray:
    """Runs run(tile) -> output tile over a [1, C, H, W] input in overlapping
    tiles and stitches the outputs; run takes and returns NCHW arrays."""
    _, _, h, w = x.shape
    out = None
    for ys, ylo, yhi in tile_plan(h, tile_h):
        for xs, xlo, xhi in tile_plan(w, tile_w):
            y = run(np.ascontiguousarray(x[:, :, ys : ys + tile_h, xs : xs + tile_w]))
            if out is None:
                out = np.empty((1, y.shape[1], h, w), dtype=y.dtype)
            out[:, :, ylo:yhi, xlo:xhi] = y[:, :, ylo - ys : yhi - ys, xlo - xs : xhi - xs]
    assert out is not None
    return out


# -----------------------------------------------------------------------------
# OIDN input and output processing for HDR color + albedo + normal
# -----------------------------------------------------------------------------

HDR_Y_MAX = 65504.0
PU_A, PU_B, PU_C = 1.41283765e03, 1.64593172e00, 4.31384981e-01
PU_D, PU_E, PU_F = -2.94139609e-03, 1.92653254e-01, 6.26026094e-03
PU_G, PU_Y0, PU_Y1 = 9.98620152e-01, 1.57945760e-06, 3.22087631e-02
PU_X0, PU_X1 = 2.23151711e-03, 3.70974749e-01


def _pu_forward(y: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            y <= PU_Y0,
            PU_A * y,
            np.where(y <= PU_Y1, PU_B * np.power(y, PU_C) + PU_D, PU_E * np.log(y + PU_F) + PU_G),
        )


def _pu_inverse(x: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return np.where(
            x <= PU_X0,
            x / PU_A,
            np.where(x <= PU_X1, np.power((x - PU_D) / PU_B, 1.0 / PU_C), np.exp((x - PU_G) / PU_E) - PU_F),
        )


PU_NORM_SCALE = float(1.0 / _pu_forward(np.array(HDR_Y_MAX, dtype=np.float64)))


def autoexposure(color: np.ndarray) -> float:
    """OIDN's autoexposure: log-average luminance over 16x16 bins."""
    lum = 0.212671 * color[..., 0] + 0.715160 * color[..., 1] + 0.072169 * color[..., 2]
    h, w = lum.shape
    nh, nw = (h + 15) // 16, (w + 15) // 16
    rows = [i * h // nh for i in range(nh + 1)]
    cols = [j * w // nw for j in range(nw + 1)]
    bins = np.array(
        [[lum[rows[i] : rows[i + 1], cols[j] : cols[j + 1]].mean() for j in range(nw)] for i in range(nh)]
    )
    bins = bins[bins > 1e-8]
    if bins.size == 0:
        return 1.0
    return float(0.18 / np.exp2(np.log2(bins).mean()))


def preprocess(
    color: np.ndarray, albedo: np.ndarray, normal: np.ndarray, exposure: float | None = None
) -> tuple[np.ndarray, float]:
    """HWC float images in, a [1, 9, H, W] float32 network input out."""
    color = np.nan_to_num(color.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    color = np.clip(color, 0.0, HDR_Y_MAX)
    if exposure is None:
        exposure = autoexposure(color)
    c = (_pu_forward(color * exposure) * PU_NORM_SCALE).astype(np.float32)
    a = np.clip(albedo.astype(np.float32), 0.0, 1.0)
    n = np.clip(normal.astype(np.float32), -1.0, 1.0) * 0.5 + 0.5
    x = np.concatenate([c, a, n], axis=-1).transpose(2, 0, 1)[None]
    return np.ascontiguousarray(x, dtype=np.float32), exposure


def postprocess(output: np.ndarray, exposure: float) -> np.ndarray:
    """[1, 3, H, W] network output in, an HWC float32 HDR image out."""
    y = np.maximum(output[0].transpose(1, 2, 0).astype(np.float64), 0.0)
    return (_pu_inverse(y / PU_NORM_SCALE) / exposure).astype(np.float32)
