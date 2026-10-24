"""Local HTTP and WebSocket API for the existing live detector."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from echomind.session import LiveSession, SessionAlreadyRunning, SessionClosed, SessionError, compact_live_update
from echomind.visualization import build_visualization


def create_app(session: LiveSession) -> FastAPI:
    """Create an API around one already-constructed session (and one model)."""

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        monitor = asyncio.create_task(_monitor_session(session), name="echomind-session-monitor")
        try:
            yield
        finally:
            monitor.cancel()
            try:
                await monitor
            except asyncio.CancelledError:
                pass
            # Worker shutdown can wait for an in-flight model call; keep the event loop responsive.
            await asyncio.to_thread(session.close)

    app = FastAPI(
        title="EchoMind local detector API",
        version="0.1.0",
        description="Loopback API for local sound-event detection. No audio upload route is provided.",
        lifespan=lifespan,
    )
    app.state.live_session = session

    @app.get("/api/status")
    def get_status() -> dict[str, Any]:
        snapshot = session.snapshot()
        return {
            key: snapshot[key] for key in (
                "status", "error", "device", "sample_rate", "captured_samples",
                "retained_samples", "input_rms", "windows_submitted", "windows_processed",
                "dropped_windows", "inference_seconds", "window_to_result_seconds",
                "onset_threshold", "release_threshold", "calibrated_probabilities",
            )
        }

    @app.post("/api/start")
    def start() -> dict[str, Any]:
        try:
            return {"status": "started", "session": session.start()}
        except SessionAlreadyRunning as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except SessionClosed as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except SessionError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.post("/api/stop")
    def stop() -> dict[str, Any]:
        snapshot = session.stop()
        if snapshot["status"] == "error":
            raise HTTPException(status_code=503, detail=snapshot["error"] or "Session shutdown failed.")
        return {"status": snapshot["status"]}

    @app.get("/api/events")
    def get_events() -> dict[str, Any]:
        snapshot = session.snapshot()
        return {
            "active": snapshot["active_events"],
            "history": snapshot["event_history"],
            "onset_threshold": snapshot["onset_threshold"],
            "release_threshold": snapshot["release_threshold"],
            "onset_windows_required": 2,
            "release_windows_required": 3,
        }

    @app.get("/api/scores")
    def get_scores() -> dict[str, Any]:
        snapshot = session.snapshot()
        return {
            "kind": "raw_model_scores",
            "calibrated_probabilities": False,
            "scores": snapshot["raw_model_scores"],
            "inference_seconds": snapshot["inference_seconds"],
        }

    @app.get("/api/visualization")
    def get_visualization() -> dict[str, Any]:
        """Return a short, bounded derived view of fresh microphone audio only."""
        session_state = session.snapshot()
        if session_state["status"] != "listening" or not session_state["sample_rate"]:
            return _visualization_unavailable("microphone_not_running")

        # Bound both duration and memory even for an unusual device sample rate.
        max_samples = min(int(session_state["sample_rate"] * 2.0), 96_000)
        audio_state = session.visualization_audio_snapshot(max_samples)
        if not audio_state["available"]:
            return _visualization_unavailable(str(audio_state["reason"]))
        samples = audio_state.pop("samples")
        try:
            data = build_visualization(samples, int(audio_state["sample_rate"]))
        except (TypeError, ValueError) as error:
            return _visualization_unavailable("malformed_audio_window", detail=str(error))
        return {"status": "available", **audio_state, **data}

    @app.websocket("/api/ws")
    async def live_updates(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                snapshot = await asyncio.to_thread(session.snapshot)
                await websocket.send_json(compact_live_update(snapshot))
                await asyncio.sleep(0.5)
        except WebSocketDisconnect:
            return
        except RuntimeError:
            # ASGI servers may report a disconnected socket as a send RuntimeError.
            return

    return app


def _visualization_unavailable(reason: str, *, detail: str | None = None) -> dict[str, Any]:
    """Stable empty schema so clients can distinguish unavailable from silence."""
    response: dict[str, Any] = {
        "status": "unavailable",
        "reason": reason,
        "sample_rate": None,
        "sample_count": 0,
        "duration_seconds": 0.0,
        "sample_sequence_start": None,
        "sample_sequence_end": None,
        "captured_at": None,
        "waveform": {"encoding": "min_max", "points": []},
        "spectrogram": {
            "encoding": "log_power_dbfs",
            "frequency_min_hz": None,
            "frequency_max_hz": None,
            "frames": [],
        },
        "rms": None,
        "rms_dbfs": None,
        "rms_note": "Digital full-scale level; not calibrated acoustic SPL or dB(A).",
    }
    if detail:
        response["detail"] = detail
    return response


async def _monitor_session(session: LiveSession) -> None:
    """Notice inference failures and release the microphone even with no connected clients."""
    while True:
        await asyncio.to_thread(session.snapshot)
        await asyncio.sleep(0.25)
