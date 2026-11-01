"""Tests that need neither the NPU nor AMD's software: telemetry parsing, file
formats and OIDN's input and output processing.

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import ctypes
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from npubench import oidn_unet, power
from npubench.pfm import read_pfm, write_pfm


class GpuMetricsTest(unittest.TestCase):
    def test_struct_matches_kernel_layout(self):
        # struct gpu_metrics_v3_0 is 264 bytes; these offsets follow from it.
        m = power.GpuMetricsV3_0
        self.assertEqual(ctypes.sizeof(m), 264)
        self.assertEqual(m.system_clock_counter.offset, 104)
        self.assertEqual(m.average_socket_power.offset, 112)
        self.assertEqual(m.average_ipu_power.offset, 116)
        self.assertEqual(m.average_all_core_power.offset, 132)
        self.assertEqual(m.time_filter_alphavalue.offset, 256)

    def test_read_sample_decodes_fields(self):
        m = power.GpuMetricsV3_0()
        m.common_header.structure_size = 264
        m.common_header.format_revision = 3
        m.average_socket_power = 12_345
        m.average_ipu_power = 2_500
        m.average_all_core_power = 4_000
        m.average_gfx_power = power.U32_NA  # "not available" must become None
        for i in range(8):
            m.average_ipu_activity[i] = 50 if i < 4 else power.U16_NA
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "gpu_metrics"
            path.write_bytes(bytes(m))
            s = power.read_sample(str(path))
        self.assertEqual(s.socket_mw, 12_345)
        self.assertEqual(s.npu_mw, 2_500)
        self.assertEqual(s.cores_mw, 4_000)
        self.assertIsNone(s.gfx_mw)
        self.assertEqual(s.npu_busy_pct, 50.0)


class PfmTest(unittest.TestCase):
    def test_round_trip_keeps_top_row_first(self):
        img = np.random.default_rng(1).random((5, 7, 3), dtype=np.float32)
        img[0, 0] = [1.0, 2.0, 3.0]
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "x.pfm")
            write_pfm(path, img)
            back = read_pfm(path)
        np.testing.assert_array_equal(back, img)
        self.assertEqual(back[0, 0].tolist(), [1.0, 2.0, 3.0])


def write_tza(path: Path, tensors: dict[str, np.ndarray]) -> None:
    """Writes a TZA v2 file the way OIDN's training/tza.py Writer does."""
    body = bytearray(struct.pack("<HBBQ", 0x41D7, 2, 0, 0))
    table = []
    for name, t in tensors.items():
        while len(body) % 64:
            body.append(0)
        table.append((name, t.shape, "oihw" if t.ndim == 4 else "x", len(body)))
        body += t.astype(np.float16).tobytes()
    table_offset = len(body)
    body += struct.pack("<I", len(table))
    for name, shape, layout, offset in table:
        body += struct.pack("<H", len(name)) + name.encode()
        body += struct.pack("<B", len(shape)) + struct.pack(f"<{len(shape)}I", *shape)
        body += layout.encode() + b"h" + struct.pack("<Q", offset)
    body[4:12] = struct.pack("<Q", table_offset)
    path.write_bytes(bytes(body))


class TzaTest(unittest.TestCase):
    def test_reader_returns_float32_tensors(self):
        w = np.arange(2 * 3 * 3 * 3, dtype=np.float32).reshape(2, 3, 3, 3) / 8
        b = np.array([0.5, -1.25], dtype=np.float32)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "w.tza"
            write_tza(path, {"conv.weight": w, "conv.bias": b})
            got = oidn_unet.read_tza(str(path))
        self.assertEqual(set(got), {"conv.weight", "conv.bias"})
        self.assertEqual(got["conv.weight"].dtype, np.float32)
        np.testing.assert_array_equal(got["conv.weight"], w)
        np.testing.assert_array_equal(got["conv.bias"], b)


class OidnProcessingTest(unittest.TestCase):
    def test_pu_transfer_round_trip(self):
        y = np.geomspace(1e-6, 6e4, 2000)
        x = oidn_unet._pu_forward(y)
        np.testing.assert_allclose(oidn_unet._pu_inverse(x), y, rtol=1e-6)

    def test_pu_norm_scale_maps_hdr_max_to_one(self):
        top = oidn_unet._pu_forward(np.array(oidn_unet.HDR_Y_MAX)) * oidn_unet.PU_NORM_SCALE
        self.assertAlmostEqual(float(top), 1.0, places=6)

    def test_preprocess_layout_and_postprocess_inverse(self):
        rng = np.random.default_rng(2)
        color = rng.random((32, 48, 3), dtype=np.float32) * 4
        albedo = rng.random((32, 48, 3), dtype=np.float32)
        normal = rng.random((32, 48, 3), dtype=np.float32) * 2 - 1
        x, exposure = oidn_unet.preprocess(color, albedo, normal)
        self.assertEqual(x.shape, (1, 9, 32, 48))
        self.assertEqual(x.dtype, np.float32)
        np.testing.assert_allclose(x[0, 3:6].transpose(1, 2, 0), albedo, rtol=1e-6)
        np.testing.assert_allclose(x[0, 6:9].transpose(1, 2, 0), normal * 0.5 + 0.5, rtol=1e-6)
        # The network's color input, sent straight back, must give the color again.
        back = oidn_unet.postprocess(x[:, 0:3], exposure)
        np.testing.assert_allclose(back, color, rtol=1e-4, atol=1e-6)

    def test_autoexposure_is_key_over_log_average(self):
        flat = np.full((64, 64, 3), 0.5, dtype=np.float32)
        lum = 0.212671 * 0.5 + 0.715160 * 0.5 + 0.072169 * 0.5
        self.assertAlmostEqual(oidn_unet.autoexposure(flat), 0.18 / lum, places=5)


class TilingTest(unittest.TestCase):
    def test_plan_covers_axis_once_with_context(self):
        for total, tile in ((1280, 736), (1920, 1056), (1920, 640), (1000, 400), (512, 1024)):
            plan = oidn_unet.tile_plan(total, tile)
            covered = np.zeros(total, dtype=int)
            for start, lo, hi in plan:
                covered[lo:hi] += 1
                self.assertGreaterEqual(start, 0)
                self.assertLessEqual(start + min(tile, total), total)
                # Interior edges keep at least the overlap as context.
                if lo > 0:
                    self.assertGreaterEqual(lo - start, oidn_unet.TILE_OVERLAP)
                if hi < total:
                    self.assertGreaterEqual(start + tile - hi, oidn_unet.TILE_OVERLAP)
            np.testing.assert_array_equal(covered, 1)

    def test_tiled_run_matches_full_run(self):
        # A local 5x5 box filter stands in for the network: tiles must be exact.
        def box(x: np.ndarray) -> np.ndarray:
            p = np.pad(x, ((0, 0), (0, 0), (2, 2), (2, 2)), mode="edge")
            h, w = x.shape[2:]
            acc = np.zeros_like(x)
            for i in range(5):
                for j in range(5):
                    acc += p[:, :, i : i + h, j : j + w]
            return acc / 25.0

        x = np.random.default_rng(3).random((1, 2, 300, 420), dtype=np.float32)
        tiled = oidn_unet.run_tiled(box, x, 224, 256)
        np.testing.assert_allclose(tiled, box(x), rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
