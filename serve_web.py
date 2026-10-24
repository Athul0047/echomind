"""Run EchoMind's local web API (no audio is uploaded or written to disk)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import uvicorn

from echomind.live import DEFAULT_OFF_THRESHOLD, DEFAULT_ON_THRESHOLD
from echomind.production import ProductionModelError, load_dymn
from echomind.session import LiveSession
from echomind.web import create_app


def main() -> int:
    parser = argparse.ArgumentParser(description="Run EchoMind's local sound detection API")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address; default is localhost. Non-loopback hosts expose the API to other devices and have no authentication.")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port (default: 8765)")
    parser.add_argument("--window-seconds", type=float, default=4.0)
    parser.add_argument("--hop-seconds", type=float, default=2.0)
    parser.add_argument("--on-threshold", type=float, default=DEFAULT_ON_THRESHOLD)
    parser.add_argument("--off-threshold", type=float, default=DEFAULT_OFF_THRESHOLD)
    args = parser.parse_args()
    if (not 1 <= args.port <= 65535 or args.window_seconds <= 0 or args.hop_seconds <= 0
            or not 0 <= args.off_threshold < args.on_threshold <= 1):
        parser.error("port must be 1..65535, durations positive, and require 0 <= off-threshold < on-threshold <= 1")

    try:
        classifier = load_dymn()
    except ProductionModelError as error:
        print(f"Model startup failed: {error}", file=sys.stderr)
        return 1

    session = LiveSession(
        classifier,
        window_seconds=args.window_seconds,
        hop_seconds=args.hop_seconds,
        on_threshold=args.on_threshold,
        off_threshold=args.off_threshold,
    )
    app = create_app(session)
    # Keep ordinary server diagnostics, but never log score arrays or audio payloads.
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", access_log=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
