"""Times one ONNX model on one device (NPU through the Vitis AI EP, or CPU)
and records SoC power from gpu_metrics while it runs.

    python -m npubench.bench --model models/oidn_1920x1280.onnx --device npu --out results/x.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import statistics
import time

import numpy as np
import onnxruntime as ort

from npubench.power import PowerSampler

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "vitisai_config_bf16.json"


def make_session(
    model: str, device: str, threads: int, cache_dir: str, config_file: str, log_level: int = 3
) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.log_severity_level = log_level
    if device == "npu":
        opts = {
            "cache_dir": cache_dir,
            "cache_key": pathlib.Path(model).stem,
            "enable_cache_file_io_in_mem": "0",
        }
        if config_file != "default":  # "default": AMD's built-in config (INT8 models)
            opts["config_file"] = config_file
        sess = ort.InferenceSession(
            model, sess_options=so, providers=["VitisAIExecutionProvider"], provider_options=[opts]
        )
        # When the EP fails to load, ONNX Runtime silently runs everything on
        # the CPU. Never report those numbers as NPU numbers.
        if "VitisAIExecutionProvider" not in sess.get_providers():
            raise SystemExit("Vitis AI EP did not load; refusing to benchmark the CPU as the NPU")
        return sess
    if device == "igpu":
        # ROCm image only: MIGraphX compiles the graph for the iGPU, in FP16.
        mgx = {"device_id": 0, "migraphx_fp16_enable": True}
        sess = ort.InferenceSession(
            model, sess_options=so, providers=["MIGraphXExecutionProvider"], provider_options=[mgx]
        )
        if "MIGraphXExecutionProvider" not in sess.get_providers():
            raise SystemExit("MIGraphX EP did not load; refusing to benchmark the CPU as the iGPU")
        return sess
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model, sess_options=so, providers=["CPUExecutionProvider"])


def random_inputs(sess: ort.InferenceSession, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    feeds = {}
    for i in sess.get_inputs():
        shape = [d if isinstance(d, int) else 1 for d in i.shape]
        if "int64" in i.type:
            feeds[i.name] = rng.integers(0, 100, size=shape, dtype=np.int64)
        else:
            feeds[i.name] = rng.random(shape, dtype=np.float32)
    return feeds


def npu_partition(cache_dir: str, model: str) -> dict | None:
    """Reads the EP report: how many graph nodes the NPU took versus the CPU."""
    report = pathlib.Path(cache_dir) / pathlib.Path(model).stem / "vitisai_ep_report.json"
    if not report.exists():
        return None
    data = json.loads(report.read_text(encoding="utf-8"))
    out = {}
    for stat in data.get("deviceStat", []):
        out[stat.get("name")] = {"nodes": stat.get("nodeNum"), "op_types": stat.get("supportedOpType")}
    return out


def run(args: argparse.Namespace) -> dict:
    os.environ.setdefault("XLNX_ONNX_EP_REPORT_FILE", "vitisai_ep_report.json")
    t_load = time.perf_counter()
    sess = make_session(args.model, args.device, args.threads, args.cache_dir, args.config, args.log_level)
    load_s = time.perf_counter() - t_load
    providers = sess.get_providers()
    feeds = random_inputs(sess) if not args.input else dict(np.load(args.input))

    if args.frame:
        # One run = one whole frame, processed as tiles of the model's input size.
        from npubench.oidn_unet import run_tiled

        ((name, x),) = feeds.items()
        th, tw = x.shape[2], x.shape[3]
        frame = np.random.default_rng(1).random((1, x.shape[1], *args.frame), dtype=np.float32)

        def one() -> None:
            run_tiled(lambda t: sess.run(None, {name: t})[0], frame, th, tw)
    else:

        def one() -> None:
            sess.run(None, feeds)

    for _ in range(args.warmup):
        one()

    sampler = PowerSampler(interval_s=0.05)
    lat = []
    with sampler:
        time.sleep(args.idle_s)
        idle_t1 = time.monotonic()
        t_end = time.monotonic() + args.seconds
        run_t0 = time.monotonic()
        while time.monotonic() < t_end or len(lat) < args.min_runs:
            t0 = time.perf_counter()
            one()
            lat.append(time.perf_counter() - t0)
        run_t1 = time.monotonic()

    idle = sampler.log.summary(idle_t1 - args.idle_s, idle_t1) if args.idle_s else None
    busy = sampler.log.summary(run_t0 + 0.5, run_t1)  # skip filter ramp-up
    mean_s = statistics.fmean(lat)
    result = {
        "model": pathlib.Path(args.model).name,
        "frame": args.frame,
        "device": args.device,
        "threads": args.threads if args.device == "cpu" else None,
        "providers": providers,
        "session_create_s": round(load_s, 3),
        "runs": len(lat),
        "latency_ms": {
            "mean": round(mean_s * 1e3, 2),
            "median": round(statistics.median(lat) * 1e3, 2),
            "p90": round(float(np.percentile(lat, 90)) * 1e3, 2),
            "min": round(min(lat) * 1e3, 2),
        },
        "throughput_per_s": round(len(lat) / sum(lat), 3),
        "power_idle": idle,
        "power_busy": busy,
        "host": platform.node(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if busy.get("socket_w") is not None:
        result["energy_per_run_j"] = round(busy["socket_w"] * mean_s, 3)
        if idle and idle.get("socket_w") is not None:
            result["energy_above_idle_per_run_j"] = round((busy["socket_w"] - idle["socket_w"]) * mean_s, 3)
    if args.device == "npu":
        result["npu_partition"] = npu_partition(args.cache_dir, args.model)
        # Second guard: the NPU's own busy counter must have moved.
        busy_pct = busy.get("npu_busy_pct")
        result["npu_verified"] = None if busy_pct is None else busy_pct >= 1.0
    if args.device == "igpu":
        # Same idea for the iGPU: the GFX busy counter must have moved.
        gfx_pct = busy.get("gfx_busy_pct")
        result["igpu_verified"] = None if gfx_pct is None else gfx_pct >= 5.0
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument(
        "--device", choices=["npu", "cpu", "igpu"], required=True, help="igpu needs the ROCm image"
    )
    p.add_argument("--threads", type=int, default=12, help="CPU threads (cpu device)")
    p.add_argument("--input", help=".npz with named inputs; random data if omitted")
    p.add_argument(
        "--frame",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        help="time whole H x W frames, run as overlapping tiles of the model's input size",
    )
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--min-runs", type=int, default=10)
    p.add_argument("--idle-s", type=float, default=5.0, help="idle power window first")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help='Vitis AI EP config file (BF16 by default), or "default" for INT8 models',
    )
    p.add_argument("--log-level", type=int, default=3, help="ONNX Runtime: 1 info, 3 errors")
    p.add_argument("--out")
    args = p.parse_args()
    result = run(args)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(text + "\n", encoding="utf-8")
    if result.get("npu_verified") is False:
        raise SystemExit("NPU busy counter stayed at 0: this run did not use the NPU")
    if result.get("igpu_verified") is False:
        raise SystemExit("GFX busy counter stayed near 0: this run did not use the iGPU")


if __name__ == "__main__":
    main()
