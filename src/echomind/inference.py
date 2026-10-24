"""Offline Audio Spectrogram Transformer loading and prediction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"


class ModelLoadError(RuntimeError):
    """Raised when the requested model is not available locally."""


@dataclass
class AudioClassifier:
    model: Any
    feature_extractor: Any
    device: torch.device


@dataclass(frozen=True)
class LabelScore:
    label: str
    score: float


def select_device() -> torch.device:
    """Prefer Apple MPS when available, otherwise use the CPU."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_classifier(model_id: str = MODEL_ID) -> AudioClassifier:
    """Load the AST model and feature extractor from the local cache only."""
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_id, local_files_only=True
        )
        model = AutoModelForAudioClassification.from_pretrained(
            model_id, local_files_only=True
        )
    except (OSError, ValueError, EnvironmentError) as error:
        raise ModelLoadError(
            f"Could not load {model_id} from the local Hugging Face cache. "
            "This command is offline and will not download model files. "
            "Check that the model cache contains the model weights and configuration. "
            f"Details: {error}"
        ) from error
    except Exception as error:
        raise ModelLoadError(
            f"The cached model {model_id} could not be initialized: "
            f"{type(error).__name__}: {error}"
        ) from error

    device = select_device()
    try:
        model.to(device)
        model.eval()
    except Exception as error:
        raise ModelLoadError(
            f"Could not prepare the model on {device}: {type(error).__name__}: {error}"
        ) from error
    return AudioClassifier(model=model, feature_extractor=feature_extractor, device=device)


def predict_chunk_scores(
    classifier: AudioClassifier,
    samples: np.ndarray,
    sample_rate: int = 16_000,
) -> torch.Tensor:
    """Return the full per-class probability-like score vector for aggregation."""
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("A prediction chunk must be a non-empty mono waveform.")
    try:
        # max_length padding also handles a final chunk shorter than 10 seconds.
        features = classifier.feature_extractor(
            samples,
            sampling_rate=sample_rate,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        model_inputs = {
            key: value.to(classifier.device) if hasattr(value, "to") else value
            for key, value in features.items()
        }
        with torch.inference_mode():
            logits = classifier.model(**model_inputs).logits
            return torch.softmax(logits[0].float(), dim=-1).detach().cpu()
    except Exception as error:
        raise RuntimeError(
            f"Model inference failed on {classifier.device}: {type(error).__name__}: {error}"
        ) from error


def top_labels(classifier: AudioClassifier, scores: torch.Tensor, top_k: int = 5) -> list[LabelScore]:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    count = min(top_k, scores.numel())
    values, indices = torch.topk(scores, k=count)
    labels = classifier.model.config.id2label
    return [
        LabelScore(str(labels.get(index, labels.get(str(index), f"class {index}"))), float(score))
        for score, index in zip(values.tolist(), indices.tolist())
    ]


def aggregate_score_vectors(
    score_vectors: list[tuple[float, torch.Tensor]],
) -> torch.Tensor:
    """Average complete score vectors using each chunk's duration as weight."""
    if not score_vectors:
        raise ValueError("At least one chunk score vector is required.")
    total_duration = sum(duration for duration, _ in score_vectors)
    if total_duration <= 0:
        raise ValueError("Chunk durations must have a positive total.")
    shape = score_vectors[0][1].shape
    if any(scores.shape != shape for _, scores in score_vectors):
        raise ValueError("All chunk score vectors must have the same shape.")
    weighted = sum(scores * duration for duration, scores in score_vectors)
    return weighted / total_duration
