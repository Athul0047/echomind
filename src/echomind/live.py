"""Bounded live-audio buffering, single-worker inference, and persistent events."""
from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import numpy as np
from scipy.signal import resample_poly

DEFAULT_ON_THRESHOLD = 0.35
DEFAULT_OFF_THRESHOLD = 0.30


class AudioRingBuffer:
    """Thread-safe fixed-capacity mono float ring; appends discard oldest samples."""
    def __init__(self, capacity_samples: int):
        if capacity_samples <= 0:
            raise ValueError("capacity_samples must be positive")
        self.capacity_samples = capacity_samples
        self._samples = np.zeros(capacity_samples, dtype=np.float32)
        self._write = 0
        self._size = 0
        self._lock = threading.Lock()

    def append(self, samples: np.ndarray) -> None:
        block = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not block.size:
            return
        with self._lock:
            if block.size >= self.capacity_samples:
                self._samples[:] = block[-self.capacity_samples:]
                self._write = 0
                self._size = self.capacity_samples
                return
            first = min(block.size, self.capacity_samples - self._write)
            self._samples[self._write:self._write + first] = block[:first]
            remaining = block.size - first
            if remaining:
                self._samples[:remaining] = block[first:]
            self._write = (self._write + block.size) % self.capacity_samples
            self._size = min(self.capacity_samples, self._size + block.size)

    def latest(self, count: int) -> np.ndarray | None:
        with self._lock:
            if count <= 0 or self._size < count:
                return None
            start = (self._write - count) % self.capacity_samples
            if start + count <= self.capacity_samples:
                return self._samples[start:start + count].copy()
            first = self.capacity_samples - start
            return np.concatenate((self._samples[start:], self._samples[:count-first]))

    def recent(self, max_count: int) -> np.ndarray:
        """Copy up to ``max_count`` newest samples while holding the ring lock briefly."""
        if max_count <= 0:
            raise ValueError("max_count must be positive")
        with self._lock:
            count = min(max_count, self._size)
            if count == 0:
                return np.empty(0, dtype=np.float32)
            start = (self._write - count) % self.capacity_samples
            if start + count <= self.capacity_samples:
                return self._samples[start:start + count].copy()
            first = self.capacity_samples - start
            return np.concatenate((self._samples[start:], self._samples[:count-first]))

    @property
    def size(self) -> int:
        with self._lock:
            return self._size


def top_score_pairs(scores: dict[str, float], top_k: int = 5) -> list[tuple[str, float]]:
    """Return highest finite independent class scores for transparent diagnostics."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    valid = [(str(label), float(score)) for label, score in scores.items()
             if np.isfinite(score)]
    return sorted(valid, key=lambda pair: pair[1], reverse=True)[:top_k]


@dataclass
class ActiveEvent:
    label: str
    started_monotonic: float
    started_at: str
    score: float
    below_count: int = 0


@dataclass(frozen=True)
class EventRecord:
    label: str
    started_at: str
    duration_seconds: float
    final_score: float


class EventTracker:
    """Two-frame onset and three-frame release hysteresis using raw model scores."""
    def __init__(self, on_threshold: float = DEFAULT_ON_THRESHOLD,
                 off_threshold: float = DEFAULT_OFF_THRESHOLD,
                 onset_frames: int = 2, release_frames: int = 3, history_limit: int = 30):
        if not 0 <= off_threshold < on_threshold <= 1 or min(onset_frames, release_frames, history_limit) < 1:
            raise ValueError("Invalid hysteresis thresholds or frame limits")
        self.on_threshold, self.off_threshold = on_threshold, off_threshold
        self.onset_frames, self.release_frames = onset_frames, release_frames
        self.active: dict[str, ActiveEvent] = {}
        self._above: dict[str, int] = {}
        self.history: deque[EventRecord] = deque(maxlen=history_limit)
        self._lock = threading.Lock()

    def update(self, scores: dict[str, float], now: float | None = None,
               wall_time: datetime | None = None) -> None:
        now = time.monotonic() if now is None else now
        wall_time = datetime.now() if wall_time is None else wall_time
        with self._lock:
            for label, score in scores.items():
                score = float(score)
                if not np.isfinite(score):
                    continue
                if label in self.active:
                    event = self.active[label]
                    event.score = score
                    event.below_count = event.below_count + 1 if score < self.off_threshold else 0
                    if event.below_count >= self.release_frames:
                        self.history.append(EventRecord(label, event.started_at, max(0, now-event.started_monotonic), score))
                        del self.active[label]
                    continue
                self._above[label] = self._above.get(label, 0) + 1 if score >= self.on_threshold else 0
                if self._above[label] >= self.onset_frames:
                    self.active[label] = ActiveEvent(label, now, wall_time.isoformat(timespec="seconds"), score)
                    self._above[label] = 0

    def snapshot(self, now: float | None = None) -> list[tuple[str, float, str, float]]:
        now = time.monotonic() if now is None else now
        with self._lock:
            return [(e.label, e.score, e.started_at, max(0, now-e.started_monotonic)) for e in self.active.values()]

    def history_snapshot(self) -> tuple[EventRecord, ...]:
        """Read event history safely while the inference worker may be appending."""
        with self._lock:
            return tuple(self.history)

    def finish(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        with self._lock:
            for event in list(self.active.values()):
                self.history.append(EventRecord(event.label, event.started_at,
                                                max(0, now-event.started_monotonic), event.score))
            self.active.clear()


class LiveInferenceWorker:
    """A one-item replace-latest queue and one inference thread; overload drops stale windows."""
    def __init__(self, infer: Callable[[np.ndarray], tuple[dict[str, float], float]],
                 sample_rate: int = 32_000, window_seconds: float = 4.0,
                 hop_seconds: float = 2.0, source_sample_rate: int = 32_000,
                 on_threshold: float = DEFAULT_ON_THRESHOLD,
                 off_threshold: float = DEFAULT_OFF_THRESHOLD):
        if min(sample_rate, source_sample_rate) <= 0 or window_seconds <= 0 or hop_seconds <= 0:
            raise ValueError("Rates and window schedule must be positive")
        self.infer = infer
        self.sample_rate = sample_rate
        self.source_sample_rate = source_sample_rate
        self.window_samples = round(window_seconds * source_sample_rate)
        self.hop_seconds = hop_seconds
        self.buffer = AudioRingBuffer(round(source_sample_rate * max(10.0, window_seconds + hop_seconds)))
        self.events = EventTracker(on_threshold=on_threshold, off_threshold=off_threshold)
        self.status = "stopped"
        self.error: str | None = None
        self.inference_seconds: float | None = None
        self.end_to_end_seconds: float | None = None
        self.dropped_windows = 0
        self.captured_samples = 0
        self.last_capture_monotonic: float | None = None
        self.last_capture_unix: float | None = None
        self.input_rms = 0.0
        self.windows_submitted = 0
        self.windows_processed = 0
        self.latest_scores: dict[str, float] | None = None
        self._diagnostics_lock = threading.Lock()
        self._queue: queue.Queue[tuple[np.ndarray, float] | None] = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_offer = 0.0

    def diagnostics_snapshot(self) -> dict[str, object]:
        """Copy capture/inference diagnostics under a lock for safe UI polling."""
        with self._diagnostics_lock:
            result = {
                "status": self.status,
                "error": self.error,
                "captured_samples": self.captured_samples,
                "last_capture_monotonic": self.last_capture_monotonic,
                "last_capture_unix": self.last_capture_unix,
                "retained_samples": self.buffer.size,
                "input_rms": self.input_rms,
                "windows_submitted": self.windows_submitted,
                "windows_processed": self.windows_processed,
                "dropped_windows": self.dropped_windows,
                "inference_seconds": self.inference_seconds,
                "window_to_result_seconds": self.end_to_end_seconds,
                "latest_scores": None if self.latest_scores is None else dict(self.latest_scores),
                "queue_depth": self._queue.qsize(),
            }
        return result

    def set_audio_warning(self, message: str) -> None:
        """Record a PortAudio callback warning without touching UI state."""
        with self._diagnostics_lock:
            self.error = message

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        with self._diagnostics_lock:
            self.status, self.error = "listening", None
        self._thread = threading.Thread(target=self._run, name="echomind-inference", daemon=True)
        self._thread.start()

    def push_audio(self, samples: np.ndarray, captured_at: float | None = None) -> None:
        """Called by microphone callback: bounded copy and non-blocking queue offer only."""
        if self._stop.is_set():
            return
        self.buffer.append(samples)
        block = np.asarray(samples, dtype=np.float32).reshape(-1)
        now = time.monotonic() if captured_at is None else captured_at
        with self._diagnostics_lock:
            self.captured_samples += block.size
            self.input_rms = float(np.sqrt(np.mean(np.square(block)))) if block.size else 0.0
            self.last_capture_monotonic = now
            self.last_capture_unix = time.time()
        if now - self._last_offer < self.hop_seconds:
            return
        window = self.buffer.latest(self.window_samples)
        if window is None:
            return
        self._last_offer = now
        item = (window, now)
        submitted = False
        try:
            self._queue.put_nowait(item)
            submitted = True
        except queue.Full:
            try:
                self._queue.get_nowait()
                with self._diagnostics_lock:
                    self.dropped_windows += 1
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(item)
                submitted = True
            except queue.Full:
                with self._diagnostics_lock:
                    self.dropped_windows += 1
        if submitted:
            with self._diagnostics_lock:
                self.windows_submitted += 1

    def recent_audio_snapshot(self, max_samples: int) -> dict[str, object] | None:
        """Return a bounded sample copy and metadata, never the mutable ring itself."""
        samples = self.buffer.recent(max_samples)
        with self._diagnostics_lock:
            if not samples.size or self.last_capture_monotonic is None:
                return None
            return {
                "samples": samples,
                "sample_rate": self.source_sample_rate,
                "captured_samples": self.captured_samples,
                "captured_at_monotonic": self.last_capture_monotonic,
                "captured_at_unix": self.last_capture_unix,
            }

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    item = self._queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if item is None:
                    break
                samples, captured_at = item
                if self.source_sample_rate != self.sample_rate:
                    from math import gcd
                    divisor = gcd(self.sample_rate, self.source_sample_rate)
                    samples = resample_poly(samples, self.sample_rate // divisor,
                                            self.source_sample_rate // divisor).astype(np.float32)
                scores, elapsed = self.infer(samples)
                finished = time.monotonic()
                window_to_result = finished - (captured_at - self.window_samples / self.source_sample_rate)
                with self._diagnostics_lock:
                    self.inference_seconds = elapsed
                    self.end_to_end_seconds = window_to_result
                    self.latest_scores = {str(label): float(score) for label, score in scores.items()}
                    self.windows_processed += 1
                self.events.update(scores, now=finished)
        except Exception as error:
            with self._diagnostics_lock:
                self.error = f"{type(error).__name__}: {error}"
                self.status = "error"
            self._stop.set()
        finally:
            self.events.finish()
            with self._diagnostics_lock:
                if self.status != "error":
                    self.status = "stopped"

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
        if self._thread:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                with self._diagnostics_lock:
                    self.status = "stopping"  # inference cannot be safely force-killed
                return
        with self._diagnostics_lock:
            if self.status != "error":
                self.status = "stopped"
