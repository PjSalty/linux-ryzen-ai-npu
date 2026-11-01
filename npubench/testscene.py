"""Renders a noisy path-traced input for the denoiser with Mitsuba 3 (BSD-3):
HDR color, albedo and shading normals as PFM files, the inputs OIDN expects.

    python -m npubench.testscene --width 1920 --height 1280 --spp 8 --out data/cbox
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

from npubench.oidn_unet import preprocess
from npubench.pfm import write_pfm


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1280)
    p.add_argument("--spp", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--camera-x", type=float, default=0.0, help="camera offset, for varied samples")
    p.add_argument("--camera-y", type=float, default=0.0)
    p.add_argument("--out", default="data/cbox")
    args = p.parse_args()

    import mitsuba as mi

    mi.set_variant("scalar_rgb")
    mi.set_log_level(mi.LogLevel.Warn)
    from mitsuba import ScalarTransform4f as T

    scene_dict = mi.cornell_box()
    scene_dict["sensor"]["film"]["width"] = args.width
    scene_dict["sensor"]["film"]["height"] = args.height
    scene_dict["sensor"]["film"]["rfilter"] = {"type": "box"}
    scene_dict["sensor"]["to_world"] = T().look_at(
        origin=[args.camera_x, args.camera_y, 4.2], target=[0, 0, 0], up=[0, 1, 0]
    )
    scene_dict["integrator"] = {
        "type": "aov",
        "aovs": "albedo:albedo,nn:sh_normal",
        "img": {"type": "path", "max_depth": 8},
    }
    scene = mi.load_dict(scene_dict)
    mi.render(scene, spp=args.spp, seed=args.seed)

    # The film holds the path tracer's color plus one layer per AOV.
    layers = dict(scene.sensors()[0].film().bitmap().split())
    color = np.array(layers["<root>"], dtype=np.float32)[..., 0:3]
    albedo = np.array(layers["albedo"], dtype=np.float32)[..., 0:3]
    normal = np.array(layers["nn"], dtype=np.float32)[..., 0:3]
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_pfm(str(out / "color.pfm"), color)
    write_pfm(str(out / "albedo.pfm"), albedo)
    write_pfm(str(out / "normal.pfm"), normal)
    # The network input after OIDN's own processing: a calibration sample.
    x, exposure = preprocess(color, albedo, normal)
    np.savez(out / "input.npz", input=x)
    print(f"wrote {out}: color {color.shape}, layers {sorted(layers)}, autoexposure {exposure:.3f}")


if __name__ == "__main__":
    main()
