"""Exports the three benchmark models to static-shape ONNX (the NPU compiler
needs fixed shapes).

    python -m npubench.export oidn --weights rt_hdr_alb_nrm.tza --height 1280 --width 1920
    python -m npubench.export siglip2 --model google/siglip2-base-patch16-224
    python -m npubench.export da3 --exporter third_party/Depth-Anything-3-Onnx \\
        --model depth-anything/DA3-SMALL
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

# Hugging Face revisions the benchmarks were run with (all Apache-2.0, checked 2026-10-31).
REVISIONS = {
    "google/siglip2-base-patch16-224": "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
    "depth-anything/DA3-SMALL": "e08cab65ca0ec38e7826075418411ab90cab4da3",
    "depth-anything/DA3-BASE": "f4a6c9b3c95e41c82048423d3493a81ec3fa810e",
}


def export_oidn(args: argparse.Namespace) -> pathlib.Path:
    from npubench.oidn_unet import export_onnx

    name = pathlib.Path(args.weights).stem
    suffix = {"resize": "", "convtranspose": "_ct", "depthtospace": "_d2s"}[args.upsample]
    out = pathlib.Path(args.out or f"models/oidn_{name}_{args.width}x{args.height}{suffix}.onnx")
    out.parent.mkdir(parents=True, exist_ok=True)
    export_onnx(
        args.weights,
        str(out),
        args.height,
        args.width,
        small=name.endswith("_small"),
        opset=args.opset,
        upsample=args.upsample,
    )
    return out


def export_siglip2(args: argparse.Namespace) -> pathlib.Path:
    import torch
    from transformers import AutoConfig, SiglipVisionModel

    rev = args.revision or REVISIONS.get(args.model)
    cfg = AutoConfig.from_pretrained(args.model, revision=rev)
    size = cfg.vision_config.image_size
    model = SiglipVisionModel.from_pretrained(args.model, revision=rev, attn_implementation="eager").eval()

    class ImageEmbed(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, pixel_values):
            return self.inner(pixel_values=pixel_values).pooler_output

    tag = args.model.split("/")[-1].replace("-", "_")
    out = pathlib.Path(args.out or f"models/{tag}_vision.onnx")
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, size, size, dtype=torch.float32)
    with torch.inference_mode():
        torch.onnx.export(
            ImageEmbed(model),
            (dummy,),
            str(out),
            input_names=["pixel_values"],
            output_names=["image_embeds"],
            opset_version=args.opset,
            dynamo=False,
        )
    return out


def export_da3(args: argparse.Namespace) -> pathlib.Path:
    from huggingface_hub import snapshot_download

    tag = args.model.split("/")[-1]
    out = pathlib.Path(args.out or f"models/{tag}_{args.res}.onnx").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    exporter = pathlib.Path(args.exporter).resolve()
    # The model config names the classes to build, so load a pinned snapshot.
    local = snapshot_download(args.model, revision=args.revision or REVISIONS.get(args.model))
    cmd = [
        sys.executable,
        "export_onnx.py",
        "--model",
        local,
        "--process-res",
        str(args.res),
        "--output",
        str(out),
        "--static",
        "--opset",
        str(args.opset),
    ]
    subprocess.run(cmd, cwd=exporter, check=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="what", required=True)

    o = sub.add_parser("oidn", help="Open Image Denoise RT UNet from a .tza file")
    o.add_argument("--weights", required=True)
    o.add_argument("--height", type=int, default=1280)
    o.add_argument("--width", type=int, default=1920)
    o.add_argument(
        "--upsample",
        choices=["resize", "convtranspose", "depthtospace"],
        default="resize",
        help="exact equivalents of nearest upsampling built from ops Ryzen AI A16W8 supports",
    )

    s = sub.add_parser("siglip2", help="SigLIP 2 image tower (pooled embedding)")
    s.add_argument("--model", default="google/siglip2-base-patch16-224")

    d = sub.add_parser("da3", help="Depth Anything 3 Small or Base, monocular depth")
    d.add_argument("--exporter", required=True, help="checkout of devin-lai/Depth-Anything-3-Onnx")
    d.add_argument("--model", default="depth-anything/DA3-SMALL")
    d.add_argument("--res", type=int, default=504)

    for sp in (o, s, d):
        sp.add_argument("--opset", type=int, default=17)
        sp.add_argument("--out")
    for sp in (s, d):
        sp.add_argument("--revision", help="Hugging Face revision (default: the pinned one)")

    args = p.parse_args()
    out = {"oidn": export_oidn, "siglip2": export_siglip2, "da3": export_da3}[args.what](args)
    print(out)


if __name__ == "__main__":
    main()
