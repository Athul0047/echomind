"""Start EchoMind's local microphone and Tk live dashboard."""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import messagebox, ttk

from echomind.live import (
    DEFAULT_OFF_THRESHOLD, DEFAULT_ON_THRESHOLD, LiveInferenceWorker, top_score_pairs,
)
from echomind.production import ProductionModelError, SAMPLE_RATE, load_dymn, predict_scores

WINDOW_SECONDS = 4.0
HOP_SECONDS = 2.0


class EchoMindDashboard:
    def __init__(self, root: tk.Tk, classifier, window_seconds=WINDOW_SECONDS,
                 hop_seconds=HOP_SECONDS, on_threshold=DEFAULT_ON_THRESHOLD,
                 off_threshold=DEFAULT_OFF_THRESHOLD):
        self.root = root
        self.classifier = classifier
        self.window_seconds = window_seconds
        self.hop_seconds = hop_seconds
        self.on_threshold = on_threshold
        self.off_threshold = off_threshold
        self.stream = None
        self.worker = None
        self.capture_rate = 0
        root.title("EchoMind · Live sound events")
        root.geometry("760x560")
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.status = tk.StringVar(value="Stopped")
        self.metrics = tk.StringVar(value=f"Model: dymn10_as · Device: {classifier.device} · Scores are not calibrated probabilities")
        self.capture_diagnostics = tk.StringVar(value="Waiting for microphone audio…")
        self.thresholds_text = tk.StringVar(value=self.thresholds_description())
        self.latest_scores_text = tk.StringVar(value="Latest model scores: no window analyzed yet")
        header = ttk.Frame(root, padding=12)
        header.pack(fill="x")
        ttk.Label(header, text="EchoMind", font=("Helvetica", 20, "bold")).pack(side="left")
        ttk.Label(header, textvariable=self.status).pack(side="left", padx=18)
        self.toggle = ttk.Button(header, text="Start microphone", command=self.start)
        self.toggle.pack(side="right")
        ttk.Label(root, textvariable=self.metrics, padding=(12, 0)).pack(anchor="w")
        ttk.Label(root, textvariable=self.thresholds_text, padding=(12, 4)).pack(anchor="w")
        ttk.Label(root, textvariable=self.capture_diagnostics, padding=(12, 4), wraplength=730).pack(anchor="w")
        ttk.Label(root, textvariable=self.latest_scores_text, padding=(12, 4), wraplength=730).pack(anchor="w")
        ttk.Label(root, text="Active sounds · score · start time · duration", padding=(12, 12, 12, 4)).pack(anchor="w")
        self.active = tk.Listbox(root, height=7, font=("Menlo", 12))
        self.active.pack(fill="both", expand=True, padx=12)
        ttk.Label(root, text="Recent event history", padding=(12, 12, 12, 4)).pack(anchor="w")
        self.history = tk.Listbox(root, height=5)
        self.history.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.root.after(250, self.refresh)

    def start(self):
        if self.worker and self.worker.diagnostics_snapshot()["status"] in {"listening", "stopping"}:
            return
        try:
            device = sd.query_devices(kind="input")
            self.capture_rate = int(round(device["default_samplerate"]))
            if self.capture_rate <= 0:
                raise RuntimeError("The default input device reported an invalid sample rate.")
            self.worker = LiveInferenceWorker(
                lambda window: predict_scores(self.classifier, window),
                sample_rate=SAMPLE_RATE,
                source_sample_rate=self.capture_rate,
                window_seconds=self.window_seconds,
                hop_seconds=self.hop_seconds,
                on_threshold=self.on_threshold,
                off_threshold=self.off_threshold,
            )
            self.worker.start()
            self.stream = sd.InputStream(
                device=None, channels=1, samplerate=self.capture_rate, dtype="float32",
                blocksize=max(1, self.capture_rate // 100), callback=self.audio_callback,
                finished_callback=self.stream_finished,
            )
            self.stream.start()
            self.status.set(f"Listening · {device['name']}")
            self.toggle.configure(text="Stop microphone", command=self.stop)
        except Exception as error:
            self.stop()
            self.status.set("Microphone unavailable")
            messagebox.showerror(
                "Microphone could not start",
                f"Check macOS Privacy & Security → Microphone permission for your terminal/Python app, and confirm an input device is connected.\n\n{type(error).__name__}: {error}",
            )

    def audio_callback(self, indata, frames, timing, status):
        if status and self.worker:
            self.worker.set_audio_warning(f"Audio input warning: {status}")
        if self.worker:
            self.worker.push_audio(indata[:, 0], captured_at=time.monotonic())

    def stream_finished(self):
        pass

    def stop(self):
        stream, self.stream = self.stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as error:
                self.status.set(f"Microphone stop issue: {error}")
        if self.worker:
            self.worker.stop(timeout=8.0)
        if self.worker:
            diagnostic = self.worker.diagnostics_snapshot()
        else:
            diagnostic = None
        if diagnostic and diagnostic["status"] == "error":
            self.status.set(f"Inference error: {diagnostic['error']}")
        elif self.status.get() != "Microphone unavailable":
            self.status.set("Stopped")
        self.toggle.configure(text="Start microphone", command=self.start)

    def refresh(self):
        worker = self.worker
        if worker:
            diagnostic = worker.diagnostics_snapshot()
            if diagnostic["status"] == "error":
                self.status.set(f"Inference error: {diagnostic['error']}")
                self.stop()
            self.capture_diagnostics.set(
                f"Input: {diagnostic['captured_samples']:,} samples captured · "
                f"{diagnostic['retained_samples']:,}/{worker.buffer.capacity_samples:,} retained · "
                f"RMS {diagnostic['input_rms']:.4f} · windows sent/processed/dropped "
                f"{diagnostic['windows_submitted']}/{diagnostic['windows_processed']}/"
                f"{diagnostic['dropped_windows']} · queue {diagnostic['queue_depth']}"
                + (f" · {diagnostic['error']}" if diagnostic["error"] else "")
                + f" · event needs 2 windows ≥ {self.on_threshold:.2f}; "
                  f"3 windows < {self.off_threshold:.2f} to archive"
            )
            self.active.delete(0, tk.END)
            for label, score, started, duration in worker.events.snapshot():
                self.active.insert(tk.END, f"{label:<34} score {score:.3f} · since {started} · {duration:.1f}s")
            self.history.delete(0, tk.END)
            for event in reversed(worker.events.history_snapshot()):
                self.history.insert(tk.END, f"{event.label} · {event.started_at} · {event.duration_seconds:.1f}s · ending score {event.final_score:.3f}")
            scores = diagnostic["latest_scores"]
            if scores is not None:
                top = top_score_pairs(scores, 5)
                above = sum(score >= self.on_threshold for _, score in scores.items())
                rendered = " · ".join(f"{label}: {score:.3f}" for label, score in top)
                self.latest_scores_text.set(
                    f"Latest top scores: {rendered} · {above} classes at/above onset "
                    f"score {self.on_threshold:.2f}; {self.events_required_text()}"
                )
            inference = diagnostic["inference_seconds"]
            latency = diagnostic["window_to_result_seconds"]
            if inference is not None and latency is not None:
                self.metrics.set(
                    f"{classifier_name(self.classifier)} · {self.classifier.device} · "
                    f"inference {inference*1000:.0f} ms · window-to-result "
                    f"{latency*1000:.0f} ms (includes {self.window_seconds:g}s window)"
                )
        self.root.after(250, self.refresh)

    def events_required_text(self):
        return "2 consecutive windows required to start an event"

    def thresholds_description(self):
        return (f"Event thresholds: onset ≥ {self.on_threshold:.2f} for 2 consecutive windows · "
                f"release < {self.off_threshold:.2f} for 3 consecutive windows")

    def close(self):
        self.stop()
        self.root.destroy()


def classifier_name(classifier):
    return "DyMN dymn10_as · AudioSet independent scores"


def main():
    parser = argparse.ArgumentParser(description="Local live multi-label sound-event detection")
    parser.add_argument("--window-seconds", type=float, default=WINDOW_SECONDS,
                        help="Trailing inference window length (default: 4)")
    parser.add_argument("--hop-seconds", type=float, default=HOP_SECONDS,
                        help="Seconds between overlapping windows (default: 2)")
    parser.add_argument("--on-threshold", type=float, default=DEFAULT_ON_THRESHOLD,
                        help="Score needed to start an event (default: 0.35)")
    parser.add_argument("--off-threshold", type=float, default=DEFAULT_OFF_THRESHOLD,
                        help="Score below which an active event may end (default: 0.30)")
    args = parser.parse_args()
    if args.window_seconds <= 0 or args.hop_seconds <= 0 or not 0 <= args.off_threshold < args.on_threshold <= 1:
        parser.error("window/hop must be positive and require 0 <= off-threshold < on-threshold <= 1")
    try:
        classifier = load_dymn()
    except ProductionModelError as error:
        print(f"Model startup failed: {error}", file=sys.stderr)
        return 1
    root = tk.Tk()
    EchoMindDashboard(root, classifier, args.window_seconds, args.hop_seconds,
                      args.on_threshold, args.off_threshold)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
