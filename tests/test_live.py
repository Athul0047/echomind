"""Offline unit tests for bounded live buffers, scores, events and worker lifecycle."""
import time
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from echomind.live import AudioRingBuffer, EventTracker, LiveInferenceWorker, top_score_pairs
from echomind.production import predict_scores


class BufferTests(unittest.TestCase):
    def test_ring_buffer_keeps_latest_samples_and_wraps(self):
        buffer = AudioRingBuffer(5)
        buffer.append(np.array([1, 2, 3], dtype=np.float32))
        buffer.append(np.array([4, 5, 6, 7], dtype=np.float32))
        np.testing.assert_array_equal(buffer.latest(5), [3, 4, 5, 6, 7])
        self.assertEqual(buffer.size, 5)
        self.assertIsNone(buffer.latest(6))

    def test_oversized_append_is_still_bounded(self):
        buffer = AudioRingBuffer(4)
        buffer.append(np.arange(20, dtype=np.float32))
        np.testing.assert_array_equal(buffer.latest(4), [16, 17, 18, 19])


class ScoreAndEventTests(unittest.TestCase):
    def test_prediction_returns_independent_sigmoid_scores(self):
        classifier = SimpleNamespace(
            device=torch.device("cpu"),
            frontend=lambda waveform: torch.zeros((1, 2, 3)),
            model=lambda spec: (torch.tensor([[0.0, 2.0, -2.0]]), None),
            labels=("one", "two", "three"),
        )
        scores, elapsed = predict_scores(classifier, np.zeros(320, dtype=np.float32))
        self.assertAlmostEqual(scores["one"], 0.5, places=5)
        self.assertGreater(scores["two"], scores["one"])
        self.assertLess(scores["three"], scores["one"])
        self.assertGreaterEqual(elapsed, 0)
        self.assertNotAlmostEqual(sum(scores.values()), 1.0, places=2)

    def test_diagnostic_top_scores_show_unfiltered_scores(self):
        pairs = top_score_pairs({"quiet": 0.2, "speech": 0.54, "fan": 0.31,
                                 "invalid": float("nan")}, top_k=3)
        self.assertEqual(pairs, [("speech", 0.54), ("fan", 0.31), ("quiet", 0.2)])

    def test_events_require_onset_and_release_persistence(self):
        tracker = EventTracker(onset_frames=2, release_frames=3)
        tracker.update({"Fan": 0.8}, now=1)
        self.assertEqual(tracker.snapshot(now=1), [])
        tracker.update({"Fan": 0.7}, now=2)
        self.assertEqual(tracker.snapshot(now=3)[0][0], "Fan")
        tracker.update({"Fan": 0.1}, now=4)
        tracker.update({"Fan": 0.1}, now=5)
        self.assertEqual(len(tracker.active), 1)
        tracker.update({"Fan": 0.1}, now=6)
        self.assertEqual(len(tracker.active), 0)
        history = tracker.history_snapshot()
        self.assertIsInstance(history, tuple)
        self.assertEqual(history[-1].duration_seconds, 4)

    def test_default_calibration_accepts_scores_between_new_and_previous_onset(self):
        tracker = EventTracker()
        tracker.update({"Dog barking": 0.37}, now=1)
        self.assertEqual(tracker.snapshot(now=1), [])
        tracker.update({"Dog barking": 0.36}, now=2)
        self.assertEqual([event[0] for event in tracker.snapshot(now=2)], ["Dog barking"])
        self.assertEqual(tracker.on_threshold, 0.35)
        self.assertEqual(tracker.off_threshold, 0.30)

    def test_two_window_onset_resets_when_a_frame_falls_below_onset(self):
        tracker = EventTracker()
        tracker.update({"Fan": 0.42}, now=1)
        tracker.update({"Fan": 0.34}, now=2)
        tracker.update({"Fan": 0.43}, now=3)
        self.assertEqual(tracker.snapshot(now=3), [])
        tracker.update({"Fan": 0.41}, now=4)
        self.assertEqual([event[0] for event in tracker.snapshot(now=4)], ["Fan"])

    def test_release_hysteresis_uses_three_scores_strictly_below_release(self):
        tracker = EventTracker()
        tracker.update({"Fan": 0.5}, now=1)
        tracker.update({"Fan": 0.5}, now=2)
        tracker.update({"Fan": 0.30}, now=3)  # Equality does not count as below.
        tracker.update({"Fan": 0.29}, now=4)
        tracker.update({"Fan": 0.28}, now=5)
        self.assertIn("Fan", tracker.active)
        tracker.update({"Fan": 0.27}, now=6)
        self.assertNotIn("Fan", tracker.active)
        self.assertEqual(len(tracker.history_snapshot()), 1)

    def test_brief_bark_or_sneeze_peak_in_only_one_window_is_not_an_event(self):
        # A one-frame peak is intentionally rejected by the shared persistence rule.
        for label in ("Bark", "Sneeze"):
            with self.subTest(label=label):
                tracker = EventTracker()
                for index, score in enumerate((0.18, 0.48, 0.22, 0.16), start=1):
                    tracker.update({label: score}, now=float(index))
                self.assertEqual(tracker.snapshot(now=5), [])
                self.assertEqual(tracker.history_snapshot(), ())

    def test_brief_sound_spanning_two_windows_qualifies_without_special_case(self):
        for label in ("Bark", "Sneeze"):
            with self.subTest(label=label):
                tracker = EventTracker()
                tracker.update({label: 0.46}, now=1)
                self.assertEqual(tracker.snapshot(now=1), [])
                tracker.update({label: 0.43}, now=2)
                self.assertEqual([event[0] for event in tracker.snapshot(now=2)], [label])

    def test_noisy_background_scores_alternating_around_onset_do_not_start_event(self):
        tracker = EventTracker()
        for index, score in enumerate((0.33, 0.36, 0.34, 0.37, 0.32, 0.36), start=1):
            tracker.update({"Background": score}, now=float(index))
        self.assertEqual(tracker.snapshot(now=7), [])
        self.assertEqual(tracker.history_snapshot(), ())


class WorkerTests(unittest.TestCase):
    def test_worker_processes_audio_and_stops_cleanly(self):
        received = []
        def infer(samples):
            received.append(samples.size)
            return {"Fan": 0.9}, 0.01
        worker = LiveInferenceWorker(infer, source_sample_rate=200, sample_rate=100,
                                     window_seconds=2, hop_seconds=0.01)
        worker.start()
        worker.push_audio(np.ones(400, dtype=np.float32), captured_at=time.monotonic())
        deadline = time.monotonic() + 2
        while worker.inference_seconds is None and time.monotonic() < deadline:
            time.sleep(0.01)
        worker.stop()
        self.assertEqual(worker.status, "stopped")
        self.assertEqual(worker.inference_seconds, 0.01)
        self.assertEqual(received, [200])
        self.assertGreaterEqual(worker.end_to_end_seconds, 2.0)
        diagnostic = worker.diagnostics_snapshot()
        self.assertEqual(diagnostic["captured_samples"], 400)
        self.assertEqual(diagnostic["windows_submitted"], 1)
        self.assertEqual(diagnostic["windows_processed"], 1)
        self.assertEqual(diagnostic["latest_scores"], {"Fan": 0.9})
        self.assertAlmostEqual(diagnostic["input_rms"], 1.0)

    def test_worker_surfaces_inference_errors_and_still_stops(self):
        def fail(_samples):
            raise RuntimeError("synthetic model failure")
        worker = LiveInferenceWorker(fail, source_sample_rate=100, sample_rate=100,
                                     window_seconds=1, hop_seconds=0.01)
        worker.start()
        worker.push_audio(np.ones(100, dtype=np.float32))
        deadline = time.monotonic() + 2
        while worker.status != "error" and time.monotonic() < deadline:
            time.sleep(0.01)
        worker.stop()
        self.assertEqual(worker.status, "error")
        self.assertIn("synthetic model failure", worker.error)
        self.assertEqual(worker.diagnostics_snapshot()["status"], "error")


if __name__ == "__main__":
    unittest.main()
