"""Web API tests use fake microphone streams and workers, never model weights."""
from __future__ import annotations

import unittest
import threading
import time
from datetime import datetime
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

from echomind.live import AudioRingBuffer, EventTracker, LiveInferenceWorker
from echomind.session import LiveSession, SessionClosed
from echomind.web import create_app


class FakeWorker:
    def __init__(self, infer, **kwargs):
        self.infer = infer
        self.kwargs = kwargs
        self.status = "stopped"
        self.error = None
        self.events = EventTracker(
            on_threshold=kwargs["on_threshold"], off_threshold=kwargs["off_threshold"]
        )
        self.latest_scores = {"Dog barking": 0.42, "Speech": 0.18}
        self.stops = 0
        self.source_sample_rate = kwargs["source_sample_rate"]
        self.buffer = AudioRingBuffer(self.source_sample_rate * 10)
        self._capture_lock = threading.Lock()
        self._captured_samples = 0
        self._last_capture_monotonic = None
        self._last_capture_unix = None

    def start(self):
        self.status = "listening"
        self.events.update({"Dog barking": 0.42}, now=1, wall_time=datetime(2026, 1, 1))
        self.events.update({"Dog barking": 0.42}, now=2, wall_time=datetime(2026, 1, 1))

    def push_audio(self, samples, captured_at=None):
        block = np.asarray(samples, dtype=np.float32).reshape(-1)
        self.buffer.append(block)
        with self._capture_lock:
            self._captured_samples += block.size
            self._last_capture_monotonic = time.monotonic() if captured_at is None else captured_at
            self._last_capture_unix = time.time()

    def recent_audio_snapshot(self, max_samples):
        samples = self.buffer.recent(max_samples)
        with self._capture_lock:
            if not samples.size or self._last_capture_monotonic is None:
                return None
            return {
                "samples": samples,
                "sample_rate": self.source_sample_rate,
                "captured_samples": self._captured_samples,
                "captured_at_monotonic": self._last_capture_monotonic,
                "captured_at_unix": self._last_capture_unix,
            }

    def set_audio_warning(self, message):
        pass

    def diagnostics_snapshot(self):
        return {
            "status": self.status,
            "error": self.error,
            "captured_samples": 321,
            "retained_samples": 321,
            "input_rms": 0.1,
            "windows_submitted": 3,
            "windows_processed": 2,
            "dropped_windows": 1,
            "inference_seconds": 0.072,
            "window_to_result_seconds": 4.1,
            "latest_scores": dict(self.latest_scores),
            "queue_depth": 0,
        }

    def stop(self, timeout=30):
        self.stops += 1
        if self.status != "error":
            self.status = "stopped"


class FakeStream:
    def __init__(self, *, fail_start=False, **kwargs):
        self.callback = kwargs["callback"]
        self.fail_start = fail_start
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self):
        if self.fail_start:
            raise RuntimeError("synthetic input error")
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


class WebApiTests(unittest.TestCase):
    def make_session(self, *, device_query=None, stream_factory=None):
        streams = []
        workers = []

        def make_stream(**kwargs):
            stream = stream_factory(**kwargs) if stream_factory else FakeStream(**kwargs)
            streams.append(stream)
            return stream

        def make_worker(infer, **kwargs):
            worker = FakeWorker(infer, **kwargs)
            workers.append(worker)
            return worker

        session = LiveSession(
            SimpleNamespace(device="cpu"),
            device_query=device_query or (lambda kind: {"name": "Test input", "default_samplerate": 44100}),
            stream_factory=make_stream,
            worker_factory=make_worker,
            infer=lambda audio: ({"Dog barking": 0.5}, 0.01),
        )
        return session, streams, workers

    def test_status_scores_events_and_start_stop_endpoints(self):
        session, streams, workers = self.make_session()
        with TestClient(create_app(session)) as client:
            status = client.get("/api/status")
            self.assertEqual(status.status_code, 200)
            self.assertEqual(status.json()["status"], "stopped")
            self.assertEqual(status.json()["onset_threshold"], 0.35)
            self.assertEqual(status.json()["release_threshold"], 0.30)
            self.assertFalse(status.json()["calibrated_probabilities"])

            start = client.post("/api/start")
            self.assertEqual(start.status_code, 200)
            self.assertEqual(start.json()["status"], "started")
            self.assertEqual(start.json()["session"]["device"], "Test input")
            self.assertEqual(len(streams), 1)
            self.assertEqual(len(workers), 1)

            scores = client.get("/api/scores").json()
            self.assertEqual(scores["kind"], "raw_model_scores")
            self.assertFalse(scores["calibrated_probabilities"])
            self.assertEqual(scores["scores"]["Dog barking"], 0.42)

            events = client.get("/api/events").json()
            self.assertEqual(events["active"][0]["label"], "Dog barking")
            self.assertEqual(events["onset_windows_required"], 2)

            stop = client.post("/api/stop")
            self.assertEqual(stop.status_code, 200)
            self.assertEqual(stop.json()["status"], "stopped")
            self.assertTrue(streams[0].stopped and streams[0].closed)
            self.assertEqual(workers[0].stops, 1)

    def test_visualization_returns_bounded_derived_data_only(self):
        session, streams, _ = self.make_session()
        with TestClient(create_app(session)) as client:
            stopped = client.get("/api/visualization").json()
            self.assertEqual(stopped["status"], "unavailable")
            self.assertEqual(stopped["reason"], "microphone_not_running")

            client.post("/api/start")
            samples = np.sin(np.arange(4096, dtype=np.float32) * (2 * np.pi / 64)) * 0.25
            streams[0].callback(samples[:, None], len(samples), None, None)
            result = client.get("/api/visualization")
            self.assertEqual(result.status_code, 200)
            data = result.json()
            self.assertEqual(data["status"], "available")
            self.assertEqual(data["sample_rate"], 44100)
            self.assertEqual(data["sample_count"], len(samples))
            self.assertEqual(len(data["waveform"]["points"]), 160)
            self.assertEqual(len(data["spectrogram"]["frames"]), 64)
            self.assertTrue(all(len(frame) == 64 for frame in data["spectrogram"]["frames"]))
            self.assertEqual(data["spectrogram"]["encoding"], "log_power_dbfs")
            self.assertAlmostEqual(data["rms"], float(np.sqrt(np.mean(samples ** 2))), places=5)
            self.assertEqual(data["sample_sequence_end"] - data["sample_sequence_start"], len(samples))
            self.assertNotIn("samples", data)
            self.assertNotIn("audio", data)

            client.post("/api/stop")
            stopped = client.get("/api/visualization").json()
            self.assertEqual(stopped["status"], "unavailable")
            self.assertEqual(stopped["reason"], "microphone_not_running")

    def test_visualization_requests_are_safe_concurrently(self):
        from concurrent.futures import ThreadPoolExecutor

        session, streams, _ = self.make_session()
        with TestClient(create_app(session)) as client:
            client.post("/api/start")
            samples = np.linspace(-0.5, 0.5, 4410, dtype=np.float32)
            streams[0].callback(samples[:, None], len(samples), None, None)
            with ThreadPoolExecutor(max_workers=6) as pool:
                responses = list(pool.map(lambda _: client.get("/api/visualization"), range(12)))
            self.assertTrue(all(response.status_code == 200 for response in responses))
            self.assertTrue(all(response.json()["status"] == "available" for response in responses))
            self.assertTrue(all("samples" not in response.json() for response in responses))

    def test_visualization_rejects_stale_capture(self):
        session, streams, workers = self.make_session()
        with TestClient(create_app(session)) as client:
            client.post("/api/start")
            streams[0].callback(np.ones((128, 1), dtype=np.float32), 128, None, None)
            workers[0]._last_capture_monotonic -= 2.0
            data = client.get("/api/visualization").json()
            self.assertEqual(data["status"], "unavailable")
            self.assertEqual(data["reason"], "audio_samples_stale")

    def test_duplicate_start_returns_conflict_without_opening_another_stream(self):
        session, streams, _ = self.make_session()
        with TestClient(create_app(session)) as client:
            self.assertEqual(client.post("/api/start").status_code, 200)
            response = client.post("/api/start")
            self.assertEqual(response.status_code, 409)
            self.assertEqual(len(streams), 1)

    def test_device_and_stream_errors_are_reported_and_worker_is_cleaned_up(self):
        broken_device, _, _ = self.make_session(device_query=lambda **_: (_ for _ in ()).throw(RuntimeError("no device")))
        with TestClient(create_app(broken_device)) as client:
            response = client.post("/api/start")
            self.assertEqual(response.status_code, 503)
            self.assertIn("no device", response.json()["detail"])
            self.assertEqual(client.get("/api/status").json()["status"], "error")

        broken_stream, streams, workers = self.make_session(
            stream_factory=lambda **kwargs: FakeStream(fail_start=True, **kwargs)
        )
        with TestClient(create_app(broken_stream)) as client:
            response = client.post("/api/start")
            self.assertEqual(response.status_code, 503)
            self.assertIn("synthetic input error", response.json()["detail"])
            self.assertEqual(workers[0].stops, 1)
            self.assertTrue(streams[0].closed)

    def test_worker_failure_closes_capture_and_is_visible_in_status(self):
        session, streams, workers = self.make_session()
        with TestClient(create_app(session)) as client:
            self.assertEqual(client.post("/api/start").status_code, 200)
            workers[0].status = "error"
            workers[0].error = "synthetic inference failure"
            deadline = time.monotonic() + 1
            while not streams[0].closed and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(streams[0].stopped and streams[0].closed)
            status = client.get("/api/status").json()
            self.assertEqual(status["status"], "error")
            self.assertIn("synthetic inference failure", status["error"])
            self.assertEqual(workers[0].stops, 1)

    def test_closed_session_rejects_start_as_invalid_state(self):
        session, _, _ = self.make_session()
        session.close()
        with TestClient(create_app(session)) as client:
            response = client.post("/api/start")
            self.assertEqual(response.status_code, 409)
            self.assertIn("shut down", response.json()["detail"])

    def test_server_shutdown_stops_stream_and_worker(self):
        session, streams, workers = self.make_session()
        with TestClient(create_app(session)) as client:
            self.assertEqual(client.post("/api/start").status_code, 200)
        self.assertTrue(streams[0].stopped and streams[0].closed)
        self.assertEqual(workers[0].status, "stopped")
        self.assertEqual(workers[0].stops, 1)

    def test_websocket_sends_compact_raw_scores_and_disconnects_cleanly(self):
        session, _, _ = self.make_session()
        with TestClient(create_app(session)) as client:
            self.assertEqual(client.post("/api/start").status_code, 200)
            with client.websocket_connect("/api/ws") as socket:
                update = socket.receive_json()
                self.assertEqual(update["type"], "update")
                self.assertEqual(update["scores_kind"], "raw_model_scores")
                self.assertFalse(update["calibrated_probabilities"])
                self.assertEqual(update["top_scores"][0]["label"], "Dog barking")
                self.assertNotIn("audio", update)
            self.assertEqual(client.get("/api/status").json()["status"], "listening")
            self.assertEqual(client.post("/api/stop").status_code, 200)

    def test_snapshots_are_copies_and_shutdown_prevents_restart(self):
        session, _, workers = self.make_session()
        session.start()
        first = session.snapshot()
        first["raw_model_scores"]["Dog barking"] = 0
        self.assertEqual(session.snapshot()["raw_model_scores"]["Dog barking"], 0.42)
        session.close()
        self.assertEqual(workers[0].status, "stopped")
        with self.assertRaises(SessionClosed):
            session.start()

    def test_session_snapshots_remain_safe_during_concurrent_event_updates(self):
        session = LiveSession(
            SimpleNamespace(device="cpu"),
            device_query=lambda **_: {"name": "Test input", "default_samplerate": 32000},
            stream_factory=FakeStream,
            worker_factory=LiveInferenceWorker,
            infer=lambda audio: ({"Alarm": 0.5}, 0.0),
        )
        session.start()
        worker = session._worker
        failures = []

        def update_events():
            try:
                for index in range(1000):
                    score = 0.5 if index % 5 < 2 else 0.1
                    worker.events.update({"Alarm": score})
            except Exception as error:
                failures.append(error)

        writer = threading.Thread(target=update_events)
        writer.start()
        while writer.is_alive():
            snapshot = session.snapshot()
            self.assertIsInstance(snapshot["active_events"], list)
            self.assertIsInstance(snapshot["event_history"], list)
        writer.join()
        session.close()
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
