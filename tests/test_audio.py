"""Small, offline tests for EchoMind audio preprocessing."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from echomind.audio import AudioInputError, load_audio, split_audio


class AudioPreprocessingTests(unittest.TestCase):
    def test_stereo_audio_is_downmixed_and_resampled_to_16khz(self) -> None:
        source_rate = 44_100
        target_rate = 16_000
        frame_count = source_rate * 2
        time_axis = np.arange(frame_count, dtype=np.float32) / source_rate
        left = 0.6 * np.sin(2 * np.pi * 440 * time_axis)
        right = 0.4 * np.sin(2 * np.pi * 880 * time_axis)
        stereo = np.column_stack((left, right)).astype(np.float32)

        with tempfile.TemporaryDirectory() as temporary_directory:
            recording = Path(temporary_directory) / "synthetic-stereo.wav"
            sf.write(recording, stereo, source_rate, subtype="PCM_16")
            actual = load_audio(recording, target_sample_rate=target_rate)

        mono = (stereo[:, 0] + stereo[:, 1]) / 2
        divisor = np.gcd(source_rate, target_rate)
        expected = resample_poly(
            mono,
            up=target_rate // divisor,
            down=source_rate // divisor,
        ).astype(np.float32)
        self.assertEqual(actual.ndim, 1)
        self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(actual.size, expected.size)
        np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-4)

    def test_chunking_keeps_the_short_final_chunk(self) -> None:
        sample_rate = 16_000
        audio = np.zeros(sample_rate * 10 + sample_rate // 2, dtype=np.float32)
        chunks = split_audio(audio, sample_rate=sample_rate, chunk_seconds=10)

        self.assertEqual([chunk.samples.size for chunk in chunks], [160_000, 8_000])
        self.assertEqual(chunks[0].start_seconds, 0)
        self.assertEqual(chunks[0].end_seconds, 10)
        self.assertEqual(chunks[1].start_seconds, 10)
        self.assertEqual(chunks[1].end_seconds, 10.5)

    def test_missing_audio_has_a_clear_input_error(self) -> None:
        with self.assertRaisesRegex(AudioInputError, "was not found"):
            load_audio("a-file-that-does-not-exist.wav")


if __name__ == "__main__":
    unittest.main()
