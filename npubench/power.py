"""SoC power telemetry for AMD Ryzen AI laptops, read from amdgpu gpu_metrics.

On Strix Point and newer APUs, amdgpu exposes the SMU metrics table as
/sys/class/drm/card*/device/gpu_metrics in format 3.0 (struct gpu_metrics_v3_0
in the kernel's drivers/gpu/drm/amd/include/kgd_pp_interface.h). It reports
separate, time-filtered power for the NPU (called IPU there), the CPU cores,
the GFX block and the whole socket, in milliwatts. Reading it needs no root
and does not disturb a running GPU workload.
"""

from __future__ import annotations

import ctypes
import glob
import statistics
import threading
import time
from dataclasses import dataclass, field

U16_NA = 0xFFFF
U32_NA = 0xFFFFFFFF


class MetricsHeader(ctypes.Structure):
    _fields_ = [
        ("structure_size", ctypes.c_uint16),
        ("format_revision", ctypes.c_uint8),
        ("content_revision", ctypes.c_uint8),
    ]


class GpuMetricsV3_0(ctypes.Structure):
    # Field order and natural alignment match struct gpu_metrics_v3_0.
    _fields_ = [
        ("common_header", MetricsHeader),
        ("temperature_gfx", ctypes.c_uint16),
        ("temperature_soc", ctypes.c_uint16),
        ("temperature_core", ctypes.c_uint16 * 16),
        ("temperature_skin", ctypes.c_uint16),
        ("average_gfx_activity", ctypes.c_uint16),
        ("average_vcn_activity", ctypes.c_uint16),
        ("average_ipu_activity", ctypes.c_uint16 * 8),
        ("average_core_c0_activity", ctypes.c_uint16 * 16),
        ("average_dram_reads", ctypes.c_uint16),
        ("average_dram_writes", ctypes.c_uint16),
        ("average_ipu_reads", ctypes.c_uint16),
        ("average_ipu_writes", ctypes.c_uint16),
        ("system_clock_counter", ctypes.c_uint64),
        ("average_socket_power", ctypes.c_uint32),
        ("average_ipu_power", ctypes.c_uint16),
        ("average_apu_power", ctypes.c_uint32),
        ("average_gfx_power", ctypes.c_uint32),
        ("average_dgpu_power", ctypes.c_uint32),
        ("average_all_core_power", ctypes.c_uint32),
        ("average_core_power", ctypes.c_uint16 * 16),
        ("average_sys_power", ctypes.c_uint16),
        ("stapm_power_limit", ctypes.c_uint16),
        ("current_stapm_power_limit", ctypes.c_uint16),
        ("average_gfxclk_frequency", ctypes.c_uint16),
        ("average_socclk_frequency", ctypes.c_uint16),
        ("average_vpeclk_frequency", ctypes.c_uint16),
        ("average_ipuclk_frequency", ctypes.c_uint16),
        ("average_fclk_frequency", ctypes.c_uint16),
        ("average_vclk_frequency", ctypes.c_uint16),
        ("average_uclk_frequency", ctypes.c_uint16),
        ("average_mpipu_frequency", ctypes.c_uint16),
        ("current_coreclk", ctypes.c_uint16 * 16),
        ("current_core_maxfreq", ctypes.c_uint16),
        ("current_gfx_maxfreq", ctypes.c_uint16),
        ("throttle_residency_prochot", ctypes.c_uint32),
        ("throttle_residency_spl", ctypes.c_uint32),
        ("throttle_residency_fppt", ctypes.c_uint32),
        ("throttle_residency_sppt", ctypes.c_uint32),
        ("throttle_residency_thm_core", ctypes.c_uint32),
        ("throttle_residency_thm_gfx", ctypes.c_uint32),
        ("throttle_residency_thm_soc", ctypes.c_uint32),
        ("time_filter_alphavalue", ctypes.c_uint32),
    ]


assert ctypes.sizeof(GpuMetricsV3_0) == 264, ctypes.sizeof(GpuMetricsV3_0)


def find_metrics_file() -> str | None:
    """Return the first gpu_metrics file in format 3.0, or None."""
    for path in sorted(glob.glob("/sys/class/drm/card*/device/gpu_metrics")):
        try:
            with open(path, "rb") as f:
                head = f.read(4)
        except OSError:
            continue
        if len(head) == 4:
            hdr = MetricsHeader.from_buffer_copy(head)
            if hdr.format_revision == 3 and hdr.content_revision == 0:
                return path
    return None


def _na(value: int, width: int) -> int | None:
    return None if value == (U16_NA if width == 16 else U32_NA) else value


@dataclass(slots=True, frozen=True)
class Sample:
    t: float
    socket_mw: int | None
    npu_mw: int | None
    cores_mw: int | None
    gfx_mw: int | None
    sys_mw: int | None
    npu_busy_pct: float | None
    gfx_busy_pct: int | None


def read_sample(path: str) -> Sample:
    with open(path, "rb") as f:
        raw = f.read(264)
    m = GpuMetricsV3_0.from_buffer_copy(raw)
    cols = [c for c in m.average_ipu_activity if c != U16_NA]
    return Sample(
        t=time.monotonic(),
        socket_mw=_na(m.average_socket_power, 32),
        npu_mw=_na(m.average_ipu_power, 16),
        cores_mw=_na(m.average_all_core_power, 32),
        gfx_mw=_na(m.average_gfx_power, 32),
        sys_mw=_na(m.average_sys_power, 16),
        npu_busy_pct=(sum(cols) / len(cols)) if cols else None,
        gfx_busy_pct=_na(m.average_gfx_activity, 16),
    )


@dataclass
class PowerLog:
    samples: list[Sample] = field(default_factory=list)

    def window(self, t0: float, t1: float) -> list[Sample]:
        return [s for s in self.samples if t0 <= s.t <= t1]

    @staticmethod
    def mean_w(samples: list[Sample], attr: str) -> float | None:
        vals = [getattr(s, attr) for s in samples if getattr(s, attr) is not None]
        return (statistics.fmean(vals) / 1000.0) if vals else None

    def summary(self, t0: float, t1: float) -> dict:
        win = self.window(t0, t1)
        out: dict[str, float | int | None] = {"samples": len(win)}
        for attr, key in (
            ("socket_mw", "socket_w"),
            ("npu_mw", "npu_w"),
            ("cores_mw", "cpu_cores_w"),
            ("gfx_mw", "gfx_w"),
            ("sys_mw", "system_w"),
        ):
            out[key] = self.mean_w(win, attr)
        for attr in ("npu_busy_pct", "gfx_busy_pct"):
            vals = [getattr(s, attr) for s in win if getattr(s, attr) is not None]
            out[attr] = statistics.fmean(vals) if vals else None
        return out


class PowerSampler:
    """Samples gpu_metrics on a background thread at a fixed interval."""

    def __init__(self, interval_s: float = 0.05, path: str | None = None):
        self.path = path or find_metrics_file()
        self.interval_s = interval_s
        self.log = PowerLog()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def available(self) -> bool:
        return self.path is not None

    def _run(self) -> None:
        path = self.path
        if path is None:
            return
        while not self._stop.is_set():
            try:
                self.log.samples.append(read_sample(path))
            except OSError:
                pass
            self._stop.wait(self.interval_s)

    def __enter__(self) -> PowerSampler:
        if self.available:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()


if __name__ == "__main__":
    p = find_metrics_file()
    if not p:
        raise SystemExit("no gpu_metrics v3.0 file found")
    print(p)
    s = read_sample(p)
    print(s)
