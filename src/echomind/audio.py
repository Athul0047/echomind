"""Audio loading, mono conversion, and sample-rate normalization."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


class AudioInputError(ValueError):
    """Raised when an audio file cannot be read or preprocessed."""


@dataclass(frozen=True)
class AudioChunk:
    """A consecutive audio section and its time boundaries in seconds."""

    start_seconds: float
    end_seconds: float
    samples: np.ndarray


def load_audio(path: str | Path, target_sample_rate: int = 16_000) -> np.ndarray:
    """Read audio as mono float32 samples at ``target_sample_rate`` Hz."""
    audio_path = Path(path)
    if not audio_path.is_file():
        raise AudioInputError(f"Audio file was not found: {audio_path}")
    if target_sample_rate <= 0:
        raise ValueError("target_sample_rate must be a positive integer.")

    try:
        samples, original_sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
    except (OSError, RuntimeError, sf.SoundFileError) as error:
        raise AudioInputError(f"Could not read audio file {audio_path.name}: {error}") from error

    if samples.size == 0:
        raise AudioInputError(f"Audio file is empty: {audio_path.name}")
    if samples.ndim == 2:
        # SoundFile returns frames x channels. Averaging channels produces mono.
        samples = samples.mean(axis=1, dtype=np.float32)
    elif samples.ndim != 1:
        raise AudioInputError(f"Unsupported audio shape in {audio_path.name}: {samples.shape}")

    if not np.isfinite(samples).all():
        raise AudioInputError(f"Audio contains non-finite sample values: {audio_path.name}")
    if original_sample_rate <= 0:
        raise AudioInputError(f"Invalid sample rate in {audio_path.name}: {original_sample_rate}")

    if original_sample_rate != target_sample_rate:
        common_divisor = gcd(original_sample_rate, target_sample_rate)
        samples = resample_poly(
            samples,
            up=target_sample_rate // common_divisor,
            down=original_sample_rate // common_divisor,
        ).astype(np.float32, copy=False)

    return np.ascontiguousarray(samples, dtype=np.float32)


def split_audio(
    samples: np.ndarray,
    sample_rate: int = 16_000,
    chunk_seconds: float = 10.0,
) -> list[AudioChunk]:
    """Split a mono waveform into consecutive chunks, including a short tail."""
    if samples.ndim != 1:
        raise ValueError("split_audio expects a one-dimensional mono waveform.")
    if sample_rate <= 0 or chunk_seconds <= 0:
        raise ValueError("sample_rate and chunk_seconds must be positive.")
    if samples.size == 0:
        raise AudioInputError("Cannot split an empty audio recording.")

    chunk_frames = max(1, round(sample_rate * chunk_seconds))
    chunks: list[AudioChunk] = []
    for start_frame in range(0, samples.size, chunk_frames):
        end_frame = min(start_frame + chunk_frames, samples.size)
        chunks.append(
            AudioChunk(
                start_seconds=start_frame / sample_rate,
                end_seconds=end_frame / sample_rate,
                samples=samples[start_frame:end_frame],
            )
        )
    return chunks
