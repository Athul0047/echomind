"""Pure visualization tests use generated signals and never access a microphone."""
from __future__ import annotations

import unittest

import numpy as np

from echomind.visualization import build_visualization


class VisualizationTests(unittest.TestCase):
    def test_waveform_has_fixed_ordered_bounded_min_max_points(self):
        audio = np.linspace(-1.0, 1.0, 1600, dtype=np.float32)
        result = build_visualization(audio, 16_000)
        points = result["waveform"]["points"]
        self.assertEqual(len(points), 160)
        self.assertTrue(all(-1 <= point["min"] <= point["max"] <= 1 for point in points))
        self.assertLess(points[0]["max"], points[-1]["min"])

    def test_spectrogram_has_bounded_shape_and_dbfs_values(self):
        time_axis = np.arange(16_000, dtype=np.float32) / 16_000
        audio = 0.5 * np.sin(2 * np.pi * 1000 * time_axis)
        result = build_visualization(audio, 16_000)
        frames = result["spectrogram"]["frames"]
        self.assertEqual(len(frames), 64)
        self.assertTrue(all(len(frame) == 64 for frame in frames))
        self.assertTrue(all(-120 <= value <= 0 for frame in frames for value in frame))
        self.assertEqual(result["spectrogram"]["encoding"], "log_power_dbfs")
        self.assertLessEqual(result["spectrogram"]["frequency_max_hz"], 8000)

    def test_silence_has_zero_waveform_and_safe_floor(self):
        result = build_visualization(np.zeros(2048, dtype=np.float32), 16_000)
        self.assertTrue(all(point == {"min": 0.0, "max": 0.0} for point in result["waveform"]["points"]))
        self.assertTrue(all(value == -120.0 for frame in result["spectrogram"]["frames"] for value in frame))
        self.assertEqual(result["rms"], 0.0)
        self.assertEqual(result["rms_dbfs"], -120.0)

    def test_rms_and_dbfs_are_calculated_from_the_window(self):
        result = build_visualization(np.full(1024, 0.5, dtype=np.float32), 16_000)
        self.assertAlmostEqual(result["rms"], 0.5, places=6)
        self.assertAlmostEqual(result["rms_dbfs"], 20 * np.log10(0.5), places=5)
        self.assertIn("not calibrated acoustic SPL", result["rms_note"])

    def test_empty_malformed_nonfinite_and_invalid_rate_are_rejected(self):
        for audio, rate in (
            (np.array([], dtype=np.float32), 16_000),
            (np.zeros((8, 2), dtype=np.float32), 16_000),
            (np.array([np.nan], dtype=np.float32), 16_000),
            (np.ones(8, dtype=np.float32), 0),
        ):
            with self.subTest(audio_shape=np.shape(audio), rate=rate):
                with self.assertRaises(ValueError):
                    build_visualization(audio, rate)

    def test_short_nonempty_window_still_returns_fixed_shapes(self):
        result = build_visualization(np.array([0.25], dtype=np.float32), 8_000)
        self.assertEqual(len(result["waveform"]["points"]), 160)
        self.assertEqual(len(result["spectrogram"]["frames"]), 64)
        self.assertEqual(len(result["spectrogram"]["frames"][0]), 64)
        self.assertEqual(result["waveform"]["points"][0], {"min": 0.25, "max": 0.25})


if __name__ == "__main__":
    unittest.main()
