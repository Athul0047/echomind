"""Bounded derived signal views; this module never returns raw microphone samples."""
from __future__ import annotations

import math

import numpy as np

DEFAULT_WAVEFORM_POINTS = 160
DEFAULT_SPECTROGRAM_FRAMES = 64
DEFAULT_SPECTROGRAM_BINS = 64
_SPECTROGRAM_FFT_SIZE = 1024
_DB_FLOOR = -120.0


def build_visualization(
    samples: np.ndarray,
    sample_rate: int,
    *,
    waveform_points: int = DEFAULT_WAVEFORM_POINTS,
    spectrogram_frames: int = DEFAULT_SPECTROGRAM_FRAMES,
    spectrogram_bins: int = DEFAULT_SPECTROGRAM_BINS,
) -> dict[str, object]:
    """Create min/max waveform buckets, STFT log-power dBFS, and window RMS.

    Spectrogram values are single-sided, Hann-windowed log power in dBFS
    (10*log10(amplitude**2)); they describe energy by time/frequency and do not
    identify sound classes. Silence is represented by the -120 dBFS floor.
    """
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("Visualization requires a non-empty mono audio window.")
    if not np.isfinite(audio).all():
        raise ValueError("Visualization audio contains non-finite samples.")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    if min(waveform_points, spectrogram_frames, spectrogram_bins) <= 0:
        raise ValueError("Visualization dimensions must be positive.")

    buckets = np.array_split(audio, waveform_points)
    waveform = [
        {"min": float(np.min(bucket)), "max": float(np.max(bucket))}
        if bucket.size else {"min": 0.0, "max": 0.0}
        for bucket in buckets
    ]

    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    rms_dbfs = max(_DB_FLOOR, 20.0 * math.log10(max(rms, 10 ** (_DB_FLOOR / 20))))

    fft_size = _SPECTROGRAM_FFT_SIZE
    window = np.hanning(fft_size).astype(np.float32)
    window_sum = float(window.sum())
    max_frequency = min(sample_rate / 2.0, 8_000.0)
    max_bin = min(fft_size // 2, int(max_frequency * fft_size / sample_rate))
    frequency_indices = np.linspace(0, max_bin, spectrogram_bins).round().astype(int)
    centers = np.linspace(0, audio.size - 1, spectrogram_frames).round().astype(int)
    spectrogram: list[list[float]] = []
    for center in centers:
        frame = np.zeros(fft_size, dtype=np.float32)
        start = int(center) - fft_size // 2
        source_start = max(0, start)
        source_end = min(audio.size, start + fft_size)
        if source_end > source_start:
            frame[source_start - start:source_end - start] = audio[source_start:source_end]
        amplitude = np.abs(np.fft.rfft(frame * window)) / max(window_sum, 1.0)
        amplitude[1:-1] *= 2.0
        amplitude[0] *= 0.5
        amplitude[-1] *= 0.5
        power = np.square(amplitude[frequency_indices])
        dbfs = 10.0 * np.log10(np.maximum(power, 10 ** (_DB_FLOOR / 10)))
        spectrogram.append(np.clip(dbfs, _DB_FLOOR, 0.0).astype(float).tolist())

    return {
        "sample_rate": int(sample_rate),
        "sample_count": int(audio.size),
        "duration_seconds": audio.size / sample_rate,
        "waveform": {"encoding": "min_max", "points": waveform},
        "spectrogram": {
            "encoding": "log_power_dbfs",
            "frequency_min_hz": 0.0,
            "frequency_max_hz": float(frequency_indices[-1] * sample_rate / fft_size),
            "frames": spectrogram,
        },
        "rms": rms,
        "rms_dbfs": rms_dbfs,
        "rms_note": "Digital full-scale level; not calibrated acoustic SPL or dB(A).",
    }
