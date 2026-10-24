"""Production DyMN AudioSet inference using the official EfficientAT implementation."""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from echomind._efficientat.models.dymn.model import get_model
from echomind._efficientat.models.preprocess import AugmentMelSTFT

SAMPLE_RATE = 32_000
MODEL_NAME = "dymn10_as"
DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[2] / "resources" / "dymn10_as.pt"
LABELS_PATH = Path(__file__).resolve().parent / "_efficientat" / "metadata" / "class_labels_indices.csv"


class ProductionModelError(RuntimeError):
    """The official production model could not be loaded or run."""


@dataclass
class DyMNClassifier:
    model: Any
    frontend: Any
    labels: tuple[str, ...]
    device: torch.device


def labels_from_csv(path: Path = LABELS_PATH) -> tuple[str, ...]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream)
        labels = tuple(row["display_name"] for row in rows)
    if len(labels) != 527:
        raise ProductionModelError(f"Expected 527 official AudioSet labels, found {len(labels)} in {path}.")
    return labels


def load_dymn(checkpoint: str | Path = DEFAULT_CHECKPOINT, device: torch.device | None = None) -> DyMNClassifier:
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise ProductionModelError(
            f"Official {MODEL_NAME} checkpoint is missing: {checkpoint}. "
            "Obtain dymn10_as.pt from the EfficientAT v0.0.1 GitHub release."
        )
    selected_device = device or (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
    try:
        # Setting pretrained_name=None is deliberate: the official factory otherwise downloads weights.
        model = get_model(num_classes=527, pretrained_name=None, width_mult=1.0, T_max=1.0)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.to(selected_device).eval()
        frontend = AugmentMelSTFT(n_mels=128, sr=SAMPLE_RATE, win_length=800, hopsize=320, freqm=0, timem=0)
        frontend.to(selected_device).eval()
        return DyMNClassifier(model, frontend, labels_from_csv(), selected_device)
    except Exception as error:
        raise ProductionModelError(f"Could not initialize official {MODEL_NAME}: {type(error).__name__}: {error}") from error


def predict_scores(classifier: DyMNClassifier, mono_32k: np.ndarray) -> tuple[dict[str, float], float]:
    """Return independent AudioSet sigmoid scores and elapsed inference seconds."""
    if mono_32k.ndim != 1 or mono_32k.size == 0:
        raise ValueError("Expected a non-empty, one-dimensional 32 kHz mono window.")
    started = time.perf_counter()
    waveform = torch.from_numpy(np.asarray(mono_32k, dtype=np.float32).copy()[None, :]).to(classifier.device)
    try:
        with torch.inference_mode():
            spec = classifier.frontend(waveform)
            logits, _ = classifier.model(spec.unsqueeze(0))
            scores = torch.sigmoid(logits.float()).squeeze(0).cpu().numpy()
    except Exception as error:
        raise ProductionModelError(f"DyMN inference failed: {type(error).__name__}: {error}") from error
    elapsed = time.perf_counter() - started
    return {label: float(score) for label, score in zip(classifier.labels, scores)}, elapsed
