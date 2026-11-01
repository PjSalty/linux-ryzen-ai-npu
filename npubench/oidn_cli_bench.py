"""Times Intel Open Image Denoise's own oidnDenoise (CPU or HIP device) and
records SoC power while it runs, so the library's numbers sit in the same
table as the ONNX Runtime ones:

    python -m npubench.oidn_cli_bench --out results/oidn_hip.json -- \\
        oidnDenoise -d hip -f RT -q high --hdr color.pfm --alb albedo.pfm --nrm normal.pfm -n 30
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics
import subprocess
import time

from npubench.power import PowerSampler


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--idle-s", type=float, default=5.0)
    p.add_argument("--skip", type=int, default=2, help="warm-up runs to drop from the timings")
    p.add_argument("--out")
    p.add_argument("cmd", nargs="+", help="the oidnDenoise command line (after --)")
    args = p.parse_args()

    sampler = PowerSampler(interval_s=0.05)
    with sampler:
        time.sleep(args.idle_s)
        idle_t1 = time.monotonic()
        t0 = time.monotonic()
        proc = subprocess.run(args.cmd, capture_output=True, text=True, check=True)
        t1 = time.monotonic()

    # oidnDenoise -v 2 prints "Denoising" then "msec=<time>, hash=..." per run.
    runs = [float(m) for m in re.findall(r"^\s*msec=([0-9.]+), hash=", proc.stdout + proc.stderr, re.M)]
    lat = runs[args.skip :] if len(runs) > args.skip else runs
    busy = sampler.log.summary(t0 + 0.5, t1)
    idle = sampler.log.summary(idle_t1 - args.idle_s, idle_t1)
    mean_s = statistics.fmean(lat) / 1000.0
    result = {
        "cmd": args.cmd,
        "runs": len(lat),
        "latency_ms": {"median": round(statistics.median(lat), 2), "min": round(min(lat), 2)},
        "power_idle": idle,
        "power_busy": busy,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if busy.get("socket_w") is not None and idle.get("socket_w") is not None:
        result["energy_per_run_j"] = round(busy["socket_w"] * mean_s, 3)
        result["energy_above_idle_per_run_j"] = round((busy["socket_w"] - idle["socket_w"]) * mean_s, 3)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
