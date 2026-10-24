"""Evaluate the existing three-condition WAV set with the cached AST model."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from echomind.audio import load_audio, split_audio
from echomind.evaluation import (
    RECORDINGS,
    ChunkEvaluation,
    EvaluationLabel,
    RecordingEvaluation,
    format_time,
    write_evaluation_csv,
)
from echomind.inference import (
    MODEL_ID,
    ModelLoadError,
    aggregate_score_vectors,
    load_classifier,
    predict_chunk_scores,
    top_labels,
)


SAMPLE_RATE = 16_000
CHUNK_SECONDS = 10.0
TOP_K = 5


def as_evaluation_labels(predictions) -> tuple[EvaluationLabel, ...]:
    return tuple(EvaluationLabel(item.label, item.score) for item in predictions)


def evaluate_recording(path: Path, condition: str, classifier) -> RecordingEvaluation:
    samples = load_audio(path, target_sample_rate=SAMPLE_RATE)
    duration = len(samples) / SAMPLE_RATE
    chunks = split_audio(samples, sample_rate=SAMPLE_RATE, chunk_seconds=CHUNK_SECONDS)
    chunk_results = []
    score_vectors = []

    print(f"\nRecording: {path.name}")
    print(f"Condition: {condition}")
    print(f"Duration: {duration:.2f} seconds")
    for number, chunk in enumerate(chunks, start=1):
        scores = predict_chunk_scores(classifier, chunk.samples, sample_rate=SAMPLE_RATE)
        chunk_duration = chunk.end_seconds - chunk.start_seconds
        score_vectors.append((chunk_duration, scores))
        labels = as_evaluation_labels(top_labels(classifier, scores, top_k=TOP_K))
        chunk_results.append(
            ChunkEvaluation(chunk.start_seconds, chunk.end_seconds, labels)
        )
        print(
            f"  Chunk {number}/{len(chunks)} "
            f"[{format_time(chunk.start_seconds)}–{format_time(chunk.end_seconds)}] "
            f"({chunk_duration:.2f} s) — top {TOP_K} labels"
        )
        for rank, item in enumerate(labels, start=1):
            print(f"    {rank}. {item.label}: {item.score:.4f}")

    overall_scores = aggregate_score_vectors(score_vectors)
    overall_labels = as_evaluation_labels(
        top_labels(classifier, overall_scores, top_k=TOP_K)
    )
    print(f"  Overall summary — top {TOP_K} labels")
    for rank, item in enumerate(overall_labels, start=1):
        print(f"    {rank}. {item.label}: {item.score:.4f}")

    return RecordingEvaluation(
        filename=path.name,
        condition=condition,
        duration_seconds=duration,
        chunks=tuple(chunk_results),
        overall_labels=overall_labels,
    )


def main() -> int:
    print(f"Model: {MODEL_ID} (local cache only)")
    try:
        classifier = load_classifier(MODEL_ID)
    except ModelLoadError as error:
        print(f"Model error: {error}", file=sys.stderr)
        return 1
    print(f"Device: {classifier.device}")

    results = []
    failed = False
    for recording in RECORDINGS:
        path = PROJECT_ROOT / recording.filename
        try:
            results.append(evaluate_recording(path, recording.condition, classifier))
        except Exception as error:
            failed = True
            print(
                f"Evaluation failed for {recording.filename}: "
                f"{type(error).__name__}: {error}",
                file=sys.stderr,
            )

    if results:
        try:
            report_path = write_evaluation_csv(results, PROJECT_ROOT / "reports")
            print(f"\nCSV results saved to: {report_path}")
        except OSError as error:
            print(f"Could not save evaluation CSV: {error}", file=sys.stderr)
            return 1
    else:
        print("No recordings were evaluated successfully; no CSV was written.", file=sys.stderr)
        return 1

    print(
        "Scores are not calibrated probabilities and do not prove that a sound "
        "is present. This small comparison is not a reliability evaluation; "
        "labelled recordings are needed for that."
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
