"""Run local, chunked sound classification for one WAV recording."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Support direct execution from the repository root before package installation.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from echomind.audio import AudioInputError, load_audio, split_audio
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


def find_default_recording(project_root: Path = PROJECT_ROOT) -> Path:
    """Find the project's Test1*.wav recording without moving it."""
    matches = sorted(
        path
        for path in project_root.iterdir()
        if path.is_file()
        and path.name.lower().startswith("test1")
        and path.suffix.lower() == ".wav"
    )
    if not matches:
        raise AudioInputError(
            f"No WAV recording beginning with 'Test1' was found in {project_root}."
        )
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise AudioInputError(
            f"More than one Test1*.wav recording was found: {names}. "
            "Pass the recording path as an argument to choose one."
        )
    return matches[0]


def format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes:02d}:{remainder:05.2f}"


def print_scores(title: str, predictions) -> None:
    print(title)
    for rank, prediction in enumerate(predictions, start=1):
        print(f"  {rank}. {prediction.label}: {prediction.score:.4f}")


def run_prediction(audio_path: Path) -> None:
    samples = load_audio(audio_path, target_sample_rate=SAMPLE_RATE)
    chunks = split_audio(samples, sample_rate=SAMPLE_RATE, chunk_seconds=CHUNK_SECONDS)
    classifier = load_classifier(MODEL_ID)

    print(f"Audio: {audio_path}")
    print(f"Model: {MODEL_ID} (loaded from local cache only)")
    print(f"Device: {classifier.device}")
    print(f"Duration: {len(samples) / SAMPLE_RATE:.2f} seconds")
    print(f"Chunk size: up to {CHUNK_SECONDS:g} seconds\n")

    all_scores = []
    for number, chunk in enumerate(chunks, start=1):
        scores = predict_chunk_scores(classifier, chunk.samples, sample_rate=SAMPLE_RATE)
        duration = chunk.end_seconds - chunk.start_seconds
        all_scores.append((duration, scores))
        title = (
            f"Chunk {number}/{len(chunks)} "
            f"[{format_time(chunk.start_seconds)}–{format_time(chunk.end_seconds)}] "
            f"({duration:.2f} s) — top {TOP_K} labels"
        )
        print_scores(title, top_labels(classifier, scores, top_k=TOP_K))
        print()

    overall = aggregate_score_vectors(all_scores)
    print_scores(
        f"Overall summary — duration-weighted mean of chunk scores, top {TOP_K}",
        top_labels(classifier, overall, top_k=TOP_K),
    )
    print(
        "\nNote: these model scores are not calibrated probabilities and do not "
        "guarantee that a sound is present. The detector has not been evaluated "
        "against labelled recordings, so its reliability is unknown."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Classify a WAV recording with the locally cached EchoMind AST model."
    )
    parser.add_argument(
        "audio_path",
        nargs="?",
        type=Path,
        help="WAV file to classify (default: the single Test1*.wav in the project root).",
    )
    args = parser.parse_args(argv)

    try:
        audio_path = args.audio_path or find_default_recording()
        run_prediction(audio_path)
    except AudioInputError as error:
        print(f"Audio error: {error}", file=sys.stderr)
        return 1
    except ModelLoadError as error:
        print(f"Model error: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(
            f"Prediction failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
