"""One local microphone session built on EchoMind's existing live worker."""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import sounddevice as sd

from echomind.live import (
    DEFAULT_OFF_THRESHOLD,
    DEFAULT_ON_THRESHOLD,
    LiveInferenceWorker,
    top_score_pairs,
)
from echomind.production import SAMPLE_RATE, predict_scores


class SessionError(RuntimeError):
    """Base class for microphone session lifecycle errors."""


class SessionAlreadyRunning(SessionError):
    pass


class SessionClosed(SessionError):
    pass


class LiveSession:
    """Own exactly one input stream and worker; expose only copied snapshots."""

    def __init__(
        self,
        classifier: Any,
        *,
        window_seconds: float = 4.0,
        hop_seconds: float = 2.0,
        on_threshold: float = DEFAULT_ON_THRESHOLD,
        off_threshold: float = DEFAULT_OFF_THRESHOLD,
        device_query: Callable[..., Any] | None = None,
        stream_factory: Callable[..., Any] | None = None,
        worker_factory: Callable[..., Any] = LiveInferenceWorker,
        infer: Callable[[Any], Any] | None = None,
    ):
        if window_seconds <= 0 or hop_seconds <= 0:
            raise ValueError("Window and hop durations must be positive.")
        if not 0 <= off_threshold < on_threshold <= 1:
            raise ValueError("Require 0 <= release threshold < onset threshold <= 1.")
        self.classifier = classifier
        self.window_seconds = window_seconds
        self.hop_seconds = hop_seconds
        self.on_threshold = on_threshold
        self.off_threshold = off_threshold
        self._device_query = device_query or sd.query_devices
        self._stream_factory = stream_factory or sd.InputStream
        self._worker_factory = worker_factory
        self._infer = infer or (lambda audio: predict_scores(classifier, audio))
        self._lifecycle_lock = threading.RLock()
        self._stream = None
        self._worker = None
        self._device_name: str | None = None
        self._sample_rate: int | None = None
        self._status = "stopped"
        self._error: str | None = None
        self._closed = False

    def start(self) -> dict[str, Any]:
        with self._lifecycle_lock:
            if self._closed:
                raise SessionClosed("This server session has been shut down and cannot restart.")
            if self._status in {"starting", "listening", "stopping"}:
                raise SessionAlreadyRunning("The microphone session is already active.")
            self._status, self._error = "starting", None
            try:
                device = self._device_query(kind="input")
                sample_rate = int(round(device["default_samplerate"]))
                if sample_rate <= 0:
                    raise RuntimeError("The default input device reported an invalid sample rate.")
                worker = self._worker_factory(
                    self._infer,
                    sample_rate=SAMPLE_RATE,
                    source_sample_rate=sample_rate,
                    window_seconds=self.window_seconds,
                    hop_seconds=self.hop_seconds,
                    on_threshold=self.on_threshold,
                    off_threshold=self.off_threshold,
                )
                self._worker = worker
                worker.start()
                stream = self._stream_factory(
                    device=None,
                    channels=1,
                    samplerate=sample_rate,
                    dtype="float32",
                    blocksize=max(1, sample_rate // 100),
                    callback=self._audio_callback,
                )
                self._stream = stream
                stream.start()
                self._device_name = str(device.get("name", "Default input"))
                self._sample_rate = sample_rate
                self._status = "listening"
            except Exception as error:
                self._error = f"{type(error).__name__}: {error}"
                self._stop_resources_locked()
                self._status = "error"
                raise SessionError(self._error) from error
            return self.snapshot()

    def _audio_callback(self, indata, frames, timing, status) -> None:
        worker = self._worker
        if status and worker:
            worker.set_audio_warning(f"Audio input warning: {status}")
        if worker:
            worker.push_audio(indata[:, 0], captured_at=time.monotonic())

    def _stop_resources_locked(self) -> list[str]:
        problems = []
        stream, self._stream = self._stream, None
        if stream is not None:
            for operation in (stream.stop, stream.close):
                try:
                    operation()
                except Exception as error:
                    problems.append(f"microphone cleanup {type(error).__name__}: {error}")
        worker = self._worker
        if worker is not None:
            try:
                worker.stop(timeout=30.0)
                if worker.diagnostics_snapshot()["status"] == "stopping":
                    problems.append("inference worker did not stop within 30 seconds")
            except Exception as error:
                problems.append(f"inference cleanup {type(error).__name__}: {error}")
        return problems

    def stop(self) -> dict[str, Any]:
        with self._lifecycle_lock:
            worker_finished = self._worker is None or self._worker.diagnostics_snapshot()["status"] in {"stopped", "error"}
            if self._status != "listening" and self._stream is None and worker_finished:
                if not self._closed and self._status != "error":
                    self._status = "stopped"
                return self.snapshot()
            worker_diagnostic = self._worker.diagnostics_snapshot() if self._worker else None
            worker_error = (worker_diagnostic.get("error")
                            if worker_diagnostic and worker_diagnostic["status"] == "error" else None)
            prior_error = self._error or worker_error
            self._status = "stopping"
            problems = self._stop_resources_locked()
            if problems:
                self._status = "error"
                self._error = "; ".join(problems)
            else:
                self._status = "error" if prior_error else "stopped"
                self._error = prior_error
            return self.snapshot()

    def close(self) -> None:
        with self._lifecycle_lock:
            self.stop()
            self._closed = True

    def snapshot(self) -> dict[str, Any]:
        """Return JSON-ready copies obtained through the worker's locked APIs."""
        with self._lifecycle_lock:
            worker = self._worker
            diagnostic = worker.diagnostics_snapshot() if worker else None
            if diagnostic and diagnostic["status"] == "error" and self._status == "listening":
                self._status = "error"
                self._error = str(diagnostic["error"] or "Inference worker failed.")
                self._stop_resources_locked()
                diagnostic = worker.diagnostics_snapshot()
            return {
                "status": self._status,
                "error": self._error,
                "device": self._device_name,
                "sample_rate": self._sample_rate,
                "captured_samples": diagnostic["captured_samples"] if diagnostic else 0,
                "retained_samples": diagnostic["retained_samples"] if diagnostic else 0,
                "input_rms": diagnostic["input_rms"] if diagnostic else 0.0,
                "windows_submitted": diagnostic["windows_submitted"] if diagnostic else 0,
                "windows_processed": diagnostic["windows_processed"] if diagnostic else 0,
                "dropped_windows": diagnostic["dropped_windows"] if diagnostic else 0,
                "inference_seconds": diagnostic["inference_seconds"] if diagnostic else None,
                "window_to_result_seconds": diagnostic["window_to_result_seconds"] if diagnostic else None,
                "raw_model_scores": dict(diagnostic["latest_scores"] or {}) if diagnostic else {},
                "active_events": self._active_events(worker),
                "event_history": self._event_history(worker),
                "onset_threshold": self.on_threshold,
                "release_threshold": self.off_threshold,
                "calibrated_probabilities": False,
            }

    def visualization_audio_snapshot(
        self, max_samples: int, *, freshness_seconds: float = 1.5
    ) -> dict[str, Any]:
        """Copy recent audio safely and reject samples from stopped or stalled capture."""
        with self._lifecycle_lock:
            worker = self._worker
            if self._status != "listening" or worker is None:
                return {"available": False, "reason": "microphone_not_running"}
            if not hasattr(worker, "recent_audio_snapshot"):
                return {"available": False, "reason": "audio_samples_unavailable"}
            captured = worker.recent_audio_snapshot(max_samples)
            if captured is None:
                return {"available": False, "reason": "audio_samples_unavailable"}
            age = time.monotonic() - float(captured["captured_at_monotonic"])
            if age > freshness_seconds:
                return {"available": False, "reason": "audio_samples_stale"}
            timestamp = captured.get("captured_at_unix")
            return {
                "available": True,
                "samples": captured["samples"],
                "sample_rate": captured["sample_rate"],
                "sample_sequence_start": max(
                    0, int(captured["captured_samples"]) - len(captured["samples"])
                ),
                "sample_sequence_end": int(captured["captured_samples"]),
                "captured_at": (
                    datetime.fromtimestamp(float(timestamp), tz=timezone.utc).isoformat(timespec="milliseconds")
                    if timestamp is not None else None
                ),
            }

    @staticmethod
    def _active_events(worker) -> list[dict[str, Any]]:
        if worker is None:
            return []
        return [
            {"label": label, "raw_model_score": score, "started_at": started,
             "duration_seconds": duration}
            for label, score, started, duration in worker.events.snapshot()
        ]

    @staticmethod
    def _event_history(worker) -> list[dict[str, Any]]:
        if worker is None:
            return []
        return [
            {"label": event.label, "started_at": event.started_at,
             "duration_seconds": event.duration_seconds, "final_raw_model_score": event.final_score}
            for event in worker.events.history_snapshot()
        ]


def compact_live_update(snapshot: dict[str, Any], top_k: int = 12) -> dict[str, Any]:
    """Build a small WebSocket message without exposing audio or all class scores."""
    scores = snapshot["raw_model_scores"]
    return {
        "type": "update",
        "status": snapshot["status"],
        "error": snapshot["error"],
        "device": snapshot["device"],
        "scores_kind": "raw_model_scores",
        "calibrated_probabilities": False,
        "top_scores": [
            {"label": label, "score": score}
            for label, score in top_score_pairs(scores, top_k)
        ],
        "active_events": snapshot["active_events"],
        "event_history": snapshot["event_history"][-20:],
        "windows_processed": snapshot["windows_processed"],
        "inference_seconds": snapshot["inference_seconds"],
        "window_to_result_seconds": snapshot["window_to_result_seconds"],
    }
