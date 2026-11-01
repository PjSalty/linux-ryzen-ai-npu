"""Checks that the OIDN UNet gives the same picture on the NPU as on the CPU.

Runs the exported ONNX model on the CPU in float32 and on the NPU (BF16 inside
the Vitis AI EP), with OIDN's own input and output processing, and compares
both with the output of the real OIDN library when one is given. Reports PSNR
and SSIM on the display-referred image and writes a visual diff:

    python -m npubench.oidn_check --onnx models/oidn_rt_hdr_alb_nrm_1920x1280.onnx \\
        --color data/color.pfm --albedo data/albedo.pfm --normal data/normal.pfm \\
        --oidn data/oidn-cpu-out.pfm --save-dir results/oidn
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from npubench.bench import DEFAULT_CONFIG, make_session
from npubench.oidn_unet import postprocess, preprocess, run_tiled
from npubench.pfm import read_pfm, write_pfm


def to_display(hdr: np.ndarray, exposure: float) -> np.ndarray:
    """Exposure, clamp to [0, 1] and the sRGB curve: the image a viewer sees."""
    x = np.clip(hdr.astype(np.float64) * exposure, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM (Wang et al. 2004): 11x11 Gaussian window, sigma 1.5, data range 1,
    computed per channel and averaged."""
    import torch
    import torch.nn.functional as F

    g = torch.exp(-((torch.arange(11, dtype=torch.float64) - 5) ** 2) / (2 * 1.5**2))
    g = g / g.sum()
    win = (g[:, None] * g[None, :])[None, None]
    x = torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))[:, None]
    y = torch.from_numpy(np.ascontiguousarray(b.transpose(2, 0, 1)))[:, None]

    def blur(t):
        return F.conv2d(t, win)

    mx, my = blur(x), blur(y)
    sxx = blur(x * x) - mx * mx
    syy = blur(y * y) - my * my
    sxy = blur(x * y) - mx * my
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return float(s.mean())


def compare(a: np.ndarray, b: np.ndarray, exposure: float) -> dict:
    da, db = to_display(a, exposure), to_display(b, exposure)
    rel = np.abs(a - b) / (np.abs(b) + 1e-3)
    return {
        "psnr_db": round(psnr(da, db), 2),
        "ssim": round(ssim(da, db), 5),
        "max_abs_display_err": round(float(np.abs(da - db).max()), 4),
        "mean_rel_hdr_err": float(np.mean(rel)),
    }


def save_png(path: pathlib.Path, img: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onnx", required=True)
    p.add_argument("--color", required=True)
    p.add_argument("--albedo", required=True)
    p.add_argument("--normal", required=True)
    p.add_argument("--oidn", help="output of the OIDN library for the same input")
    p.add_argument(
        "--display-exposure",
        type=float,
        help="exposure for the display transform (default: OIDN autoexposure)",
    )
    p.add_argument("--cache-dir", default="cache/npu")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--devices", default="cpu,npu", help="comma list of cpu and npu (empty: none)")
    p.add_argument(
        "--also",
        action="append",
        default=[],
        metavar="NAME=PFM",
        help="another denoised output to score the same way, e.g. oidn_hip=out.pfm",
    )
    p.add_argument("--save-dir")
    args = p.parse_args()

    color, albedo, normal = (read_pfm(f) for f in (args.color, args.albedo, args.normal))
    x, exposure = preprocess(color, albedo, normal)
    shown = args.display_exposure or exposure

    outs = {}
    for device in filter(None, args.devices.split(",")):
        sess = make_session(args.onnx, device, args.threads, args.cache_dir, args.config)
        inp = sess.get_inputs()[0]
        th, tw = inp.shape[2], inp.shape[3]
        if (th, tw) == x.shape[2:]:
            y = sess.run(None, {inp.name: x})[0]
        else:  # a tile-sized model: OIDN-style overlapping tiles
            y = run_tiled(lambda t, s=sess, n=inp.name: s.run(None, {n: t})[0], x, th, tw)
        outs[device] = postprocess(y, exposure)
    for item in args.also:
        name, _, path = item.partition("=")
        outs[name] = read_pfm(path)

    report = {"oidn_autoexposure": exposure, "display_exposure": shown}
    ref = read_pfm(args.oidn) if args.oidn else None
    if "npu" in outs and "cpu" in outs:
        report["npu_vs_cpu_fp32"] = compare(outs["npu"], outs["cpu"], shown)
    for device, img in outs.items():
        if ref is not None:
            report[f"{device}_vs_oidn_library"] = compare(img, ref, shown)
    if ref is not None:
        report["noisy_input_vs_oidn_library"] = compare(color, ref, shown)
    if args.save_dir:
        out = pathlib.Path(args.save_dir)
        out.mkdir(parents=True, exist_ok=True)
        for device, img in outs.items():
            write_pfm(str(out / f"denoised_{device}.pfm"), img)
            save_png(out / f"denoised_{device}.png", to_display(img, shown))
        save_png(out / "noisy_input.png", to_display(color, shown))
        if ref is not None:
            save_png(out / "oidn_library.png", to_display(ref, shown))
        base = ref if ref is not None else outs.get("cpu")
        for device, img in outs.items():
            if base is not None and img is not base:
                diff = np.abs(to_display(img, shown) - to_display(base, shown))
                save_png(out / f"diff_{device}_x10.png", np.clip(diff * 10.0, 0.0, 1.0))
        (out / "accuracy.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
