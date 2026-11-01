"""Compares a model's NPU output (BF16 inside the Vitis AI EP) with its CPU
output (float32) on real images, with each model's own preprocessing.

    python -m npubench.compare --kind siglip2 --model models/siglip2_base_patch16_224_vision.onnx img/*.png
    python -m npubench.compare --kind da3 --model models/DA3-SMALL_504.onnx img/*.png
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from npubench.bench import DEFAULT_CONFIG, make_session

KINDS = {
    # google/siglip2-base-patch16-224 preprocessor_config.json: bilinear, mean/std 0.5.
    "siglip2": {"size": 224, "mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
    # Depth Anything 3 InputProcessor: fixed square resize, ImageNet mean/std.
    "da3": {"size": 504, "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
}


def load_image(path: str, size: int, mean, std) -> np.ndarray:
    from PIL import Image

    img = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    x = (np.asarray(img, dtype=np.float32) / 255.0 - np.array(mean, np.float32)) / np.array(std, np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)


def embed_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return {"cosine": round(cos, 6), "rel_l2": float(np.linalg.norm(a - b) / np.linalg.norm(b))}


def retrieval_agreement(npu: np.ndarray, cpu: np.ndarray) -> dict:
    """What a search over these embeddings would see: the image-to-image
    similarity matrices, and whether each image keeps its nearest neighbour."""

    def sims(e: np.ndarray) -> np.ndarray:
        e = e.astype(np.float64) / np.linalg.norm(e, axis=1, keepdims=True)
        s = e @ e.T
        np.fill_diagonal(s, -np.inf)
        return s

    s_npu, s_cpu = sims(npu), sims(cpu)
    off = ~np.eye(len(npu), dtype=bool)
    return {
        "max_similarity_diff": float(np.abs(s_npu[off] - s_cpu[off]).max()),
        "nearest_neighbour_agreement": float(np.mean(s_npu.argmax(1) == s_cpu.argmax(1))),
    }


def depth_metrics(npu: np.ndarray, cpu: np.ndarray) -> dict:
    n, c = npu.reshape(-1).astype(np.float64), cpu.reshape(-1).astype(np.float64)
    ok = np.abs(c) > 1e-6
    ratio = np.maximum(n[ok] / c[ok], c[ok] / n[ok])
    return {
        "abs_rel": float(np.mean(np.abs(n[ok] - c[ok]) / np.abs(c[ok]))),
        "delta_1.25": float(np.mean(ratio < 1.25)),
        "pearson": float(np.corrcoef(n, c)[0, 1]),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kind", choices=sorted(KINDS), required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--cache-dir", default="cache/npu")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--out")
    p.add_argument("images", nargs="+")
    args = p.parse_args()

    spec = KINDS[args.kind]
    cpu = make_session(args.model, "cpu", args.threads, args.cache_dir, args.config)
    npu = make_session(args.model, "npu", args.threads, args.cache_dir, args.config)
    name = cpu.get_inputs()[0].name
    metric = embed_metrics if args.kind == "siglip2" else depth_metrics

    per_image, emb_cpu, emb_npu = [], [], []
    for path in args.images:
        x = load_image(path, spec["size"], spec["mean"], spec["std"])
        y_cpu = cpu.run(None, {name: x})[0]
        y_npu = npu.run(None, {name: x})[0]
        per_image.append({"image": pathlib.Path(path).name, **metric(y_npu, y_cpu)})
        emb_cpu.append(y_cpu.reshape(-1))
        emb_npu.append(y_npu.reshape(-1))

    keys = [k for k in per_image[0] if k != "image"]
    summary = {
        k: {"min": min(r[k] for r in per_image), "mean": float(np.mean([r[k] for r in per_image]))}
        for k in keys
    }
    report = {
        "model": pathlib.Path(args.model).name,
        "kind": args.kind,
        "images": len(per_image),
        "npu_bf16_vs_cpu_fp32": summary,
        "per_image": per_image,
    }
    if args.kind == "siglip2" and len(emb_cpu) > 2:
        report["retrieval"] = retrieval_agreement(np.stack(emb_npu), np.stack(emb_cpu))
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
