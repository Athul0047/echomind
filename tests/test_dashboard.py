"""Dashboard lifecycle tests use fake PortAudio objects and never open a mic."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import live_dashboard


class FakeValue:
    def __init__(self):
        self.value = ""

    def set(self, value):
        self.value = value

    def get(self):
        return self.value


class FakeButton:
    def __init__(self):
        self.options = {}

    def configure(self, **options):
        self.options.update(options)


class DashboardLifecycleTests(unittest.TestCase):
    def make_dashboard(self):
        dashboard = live_dashboard.EchoMindDashboard.__new__(live_dashboard.EchoMindDashboard)
        dashboard.classifier = SimpleNamespace(device="cpu")
        dashboard.window_seconds = 4
        dashboard.hop_seconds = 2
        dashboard.on_threshold = 0.35
        dashboard.off_threshold = 0.30
        dashboard.stream = None
        dashboard.worker = None
        dashboard.capture_rate = 0
        dashboard.status = FakeValue()
        dashboard.toggle = FakeButton()
        return dashboard

    def test_threshold_display_includes_configured_values_and_persistence(self):
        dashboard = self.make_dashboard()
        self.assertEqual(
            dashboard.thresholds_description(),
            "Event thresholds: onset ≥ 0.35 for 2 consecutive windows · "
            "release < 0.30 for 3 consecutive windows",
        )

    def test_start_starts_worker_before_opening_audio_stream(self):
        events = []

        class Worker:
            status = "stopped"

            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                events.append("worker started")
                self.status = "listening"

            def stop(self, timeout=5.0):
                events.append("worker stopped")
                self.status = "stopped"

            def diagnostics_snapshot(self):
                return {"status": self.status, "error": None}

        class Stream:
            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                events.append("stream started")

        with patch.object(live_dashboard.sd, "query_devices", return_value={
            "default_samplerate": 44100, "name": "fake input"
        }), patch.object(live_dashboard.sd, "InputStream", side_effect=lambda **kw: Stream()), \
             patch.object(live_dashboard, "LiveInferenceWorker", Worker):
            dashboard = self.make_dashboard()
            dashboard.start()

        self.assertEqual(events, ["worker started", "stream started"])
        self.assertEqual(dashboard.capture_rate, 44100)
        self.assertEqual(dashboard.status.get(), "Listening · fake input")
        self.assertEqual(dashboard.toggle.options["text"], "Stop microphone")

    def test_stream_permission_error_stops_worker_and_reports_failure(self):
        events = []

        class Worker:
            status = "stopped"

            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                events.append("worker started")
                self.status = "listening"

            def stop(self, timeout=5.0):
                events.append("worker stopped")
                self.status = "stopped"

            def diagnostics_snapshot(self):
                return {"status": self.status, "error": None}

        class Stream:
            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                raise RuntimeError("microphone permission denied")

        with patch.object(live_dashboard.sd, "query_devices", return_value={
            "default_samplerate": 44100, "name": "fake input"
        }), patch.object(live_dashboard.sd, "InputStream", side_effect=lambda **kw: Stream()), \
             patch.object(live_dashboard, "LiveInferenceWorker", Worker), \
             patch.object(live_dashboard.messagebox, "showerror") as showerror:
            dashboard = self.make_dashboard()
            dashboard.start()

        self.assertEqual(events, ["worker started", "worker stopped"])
        self.assertEqual(dashboard.status.get(), "Microphone unavailable")
        showerror.assert_called_once()
        self.assertIn("permission denied", showerror.call_args.args[1])

    def test_missing_or_denied_input_shows_error_and_stays_stopped(self):
        dashboard = self.make_dashboard()
        with patch.object(live_dashboard.sd, "query_devices", side_effect=RuntimeError("permission denied")), \
             patch.object(live_dashboard.messagebox, "showerror") as showerror:
            dashboard.start()
        self.assertEqual(dashboard.status.get(), "Microphone unavailable")
        self.assertIsNone(dashboard.stream)
        showerror.assert_called_once()
        self.assertIn("permission denied", showerror.call_args.args[1])


if __name__ == "__main__":
    unittest.main()
