"""Quantizes an ONNX model to INT8 QDQ for the Ryzen AI INT8 compiler, with
ONNX Runtime's own quantization tool (AMD documents that any tool producing
QDQ works for its A8W8 and A16W8 schemes).

    python -m npubench.quantize --fp32 models/m_640x640.onnx --out models/m_a16w8.onnx \\
        --scheme a16w8 --crop 640 640 --set-shape 1280 1920 data/calib/*/input.npz

Each .npz holds one sample of every model input, keyed by input name. Use
calibration data that is not your test data, or the accuracy you measure
afterwards will look better than it is.

--crop H W cuts each [1, C, H, W] sample into crops of that size (the --fp32
model must be exported at it), so calibration needs a fraction of the memory
of a full-size pass. For a fully convolutional graph without Reshape (OIDN's
UNet with Resize or ConvTranspose upsampling), --set-shape H W then rewrites
the quantized model's input and output size for deployment.
"""

from __future__ import annotations

import argparse
import pathlib
from collections.abc import Iterator

import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

# AMD Ryzen AI INT8 schemes: symmetric activations and weights with float scales.
SCHEMES = {
    "a16w8": (QuantType.QInt16, QuantType.QInt8),
    "a8w8": (QuantType.QInt8, QuantType.QInt8),
}


def samples(paths: list[str], crop: tuple[int, int] | None) -> Iterator[dict[str, np.ndarray]]:
    for path in paths:
        with np.load(path) as data:
            feeds = {k: data[k] for k in data.files}
        if not crop:
            yield feeds
            continue
        ch, cw = crop
        name, x = next(iter(feeds.items()))
        h, w = x.shape[2], x.shape[3]
        # A grid of crops that covers the image, edges included.
        ys = np.linspace(0, h - ch, max(1, round(h / ch))).astype(int)
        xs = np.linspace(0, w - cw, max(1, round(w / cw))).astype(int)
        for y in ys:
            for x0 in xs:
                yield {name: np.ascontiguousarray(x[:, :, y : y + ch, x0 : x0 + cw])}


class SampleReader(CalibrationDataReader):
    def __init__(self, paths: list[str], crop: tuple[int, int] | None):
        self._it = samples(paths, crop)
        self.count = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        feeds = next(self._it, None)
        if feeds is not None:
            self.count += 1
        return feeds


def merge_ranges_before_clearing() -> None:
    """Works around ONNX Runtime 1.30's MinMaxCalibrater: once
    CalibMaxIntermediateOutputs outputs are collected it clears them without
    merging their ranges first, so every batch is lost and calibration ends in
    "No data is collected". This merges (compute_data) before each clear."""
    from onnxruntime.quantization.calibrate import MinMaxCalibrater

    def collect_data(self, data_reader: CalibrationDataReader) -> None:
        while True:
            inputs = data_reader.get_next()
            if not inputs:
                break
            outputs = self.infer_session.run(None, inputs)
            self.intermediate_outputs.append(
                [
                    value if sess_o.name not in self.model_original_outputs else None
                    for sess_o, value in zip(self.infer_session.get_outputs(), outputs, strict=True)
                ]
            )
            if (
                self.max_intermediate_outputs
                and len(self.intermediate_outputs) >= self.max_intermediate_outputs
            ):
                self.compute_data()
                self.clear_collected_data()
        if not self.intermediate_outputs and self.calibrate_tensors_range is None:
            raise ValueError("No data is collected.")
        self.compute_data()
        self.clear_collected_data()

    MinMaxCalibrater.collect_data = collect_data


def set_spatial_shape(path: str, height: int, width: int) -> None:
    """Rewrites the H and W of every graph input and output and drops the
    stale intermediate shapes; valid only for fully convolutional graphs."""
    import onnx

    model = onnx.load(path)
    if any(n.op_type == "Reshape" for n in model.graph.node):
        raise ValueError("the graph has Reshape nodes with fixed sizes; quantize at the deployment size")
    for vi in list(model.graph.input) + list(model.graph.output):
        dims = vi.type.tensor_type.shape.dim
        dims[2].dim_value, dims[3].dim_value = height, width
    del model.graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model)
    onnx.save(model, path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fp32", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--scheme", choices=sorted(SCHEMES), default="a16w8")
    p.add_argument("--method", choices=["minmax", "percentile"], default="minmax")
    p.add_argument("--per-channel", action="store_true", help="per-channel weight scales")
    p.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        metavar="NODE",
        help="nodes to keep in float (e.g. /enc_conv0/Conv)",
    )
    p.add_argument(
        "--crop", type=int, nargs=2, metavar=("H", "W"), help="calibrate on H x W crops of each sample"
    )
    p.add_argument("--set-shape", type=int, nargs=2, metavar=("H", "W"), help="deployment input size")
    p.add_argument("calib", nargs="+", help=".npz calibration samples")
    args = p.parse_args()

    act, wgt = SCHEMES[args.scheme]
    method = CalibrationMethod.MinMax if args.method == "minmax" else CalibrationMethod.Percentile
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if method == CalibrationMethod.MinMax:
        merge_ranges_before_clearing()
    reader = SampleReader(args.calib, tuple(args.crop) if args.crop else None)
    quantize_static(
        args.fp32,
        args.out,
        reader,
        quant_format=QuantFormat.QDQ,
        activation_type=act,
        weight_type=wgt,
        per_channel=args.per_channel,
        nodes_to_exclude=args.exclude,
        calibrate_method=method,
        # Reduce ranges after every sample to bound memory.
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
            "CalibMaxIntermediateOutputs": 1,
        },
    )
    if args.set_shape:
        set_spatial_shape(args.out, *args.set_shape)
    print(f"{args.out} ({reader.count} calibration samples)")


if __name__ == "__main__":
    main()
