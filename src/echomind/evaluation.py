"""Small helpers for saving per-recording and per-chunk evaluation scores."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class RecordingSpec:
    filename: str
    condition: str


@dataclass(frozen=True)
class EvaluationLabel:
    label: str
    score: float


@dataclass(frozen=True)
class ChunkEvaluation:
    start_seconds: float
    end_seconds: float
    labels: tuple[EvaluationLabel, ...]


@dataclass(frozen=True)
class RecordingEvaluation:
    filename: str
    condition: str
    duration_seconds: float
    chunks: tuple[ChunkEvaluation, ...]
    overall_labels: tuple[EvaluationLabel, ...]


RECORDINGS = (
    RecordingSpec("Test1 - 23:10:26, 8.25 PM.wav", "speech with ceiling fan"),
    RecordingSpec("fan_only.wav", "fan only"),
    RecordingSpec("speech_only.wav", "speech only"),
)

CSV_FIELDS = (
    "recording",
    "condition",
    "duration_seconds",
    "chunk",
    "label",
    "score",
)


def format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes:02d}:{remainder:05.2f}"


def evaluation_rows(results: Iterable[RecordingEvaluation]) -> list[dict[str, str]]:
    """Flatten evaluation summaries into rows suitable for a CSV file."""
    rows: list[dict[str, str]] = []
    for result in results:
        common = {
            "recording": result.filename,
            "condition": result.condition,
            "duration_seconds": f"{result.duration_seconds:.3f}",
        }
        for label in result.overall_labels:
            rows.append({
                **common,
                "chunk": "overall",
                "label": label.label,
                "score": f"{label.score:.6f}",
            })
        for number, chunk in enumerate(result.chunks, start=1):
            chunk_name = (
                f"chunk {number} [{format_time(chunk.start_seconds)}-"
                f"{format_time(chunk.end_seconds)}]"
            )
            for label in chunk.labels:
                rows.append({
                    **common,
                    "chunk": chunk_name,
                    "label": label.label,
                    "score": f"{label.score:.6f}",
                })
    return rows


def write_evaluation_csv(
    results: Iterable[RecordingEvaluation], reports_directory: str | Path
) -> Path:
    """Write results to a new CSV without overwriting an earlier evaluation."""
    reports_path = Path(reports_directory)
    reports_path.mkdir(parents=True, exist_ok=True)
    rows = evaluation_rows(results)
    candidate = reports_path / "three_recording_evaluation.csv"
    suffix = 2
    while True:
        try:
            with candidate.open("x", newline="", encoding="utf-8") as output_file:
                writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            return candidate
        except FileExistsError:
            candidate = reports_path / f"three_recording_evaluation_{suffix}.csv"
            suffix += 1
