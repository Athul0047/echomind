# EchoMind

EchoMind is a local-first sound-event detection project. It listens to the default microphone and reports multiple overlapping sound categories, such as speech, a fan, or barking. It is a multi-label detector: it does not separate sound sources, and a model score is not a calibrated probability or proof that a sound is present.

The live production path uses EfficientAT DyMN `dymn10_as`, pretrained on AudioSet. A native Tkinter dashboard and a local React dashboard are available. The web dashboard presents real backend events, raw scores, recent bounded signal visualizations, and session diagnostics.

## Features and architecture

- `sounddevice` captures mono `float32` microphone audio into a fixed-capacity, thread-safe ring buffer.
- A single background `LiveInferenceWorker` uses a bounded replace-latest queue to process overlapping windows. It resamples and prepares audio for the existing EfficientAT scorer; the microphone callback does not run model inference.
- `EventTracker` uses a 0.35 onset threshold for two consecutive windows and a 0.30 release threshold for three consecutive windows below release. These values and persistence rules are configurable in the native dashboard command.
- `live_dashboard.py` is the native Tkinter interface. `serve_web.py` starts a local FastAPI service that reuses the production session and model. Do not run both dashboards at once because each can open the default microphone independently.
- `web/` is a React + TypeScript dashboard. It uses the local REST and WebSocket API. While listening, it polls for a bounded waveform envelope, a compact spectrogram, and digital RMS level. The API returns derived visualizations, never raw audio samples.
- `predict.py` and `evaluate.py` are legacy offline AST utilities retained for earlier recording evaluation; they are separate from the live EfficientAT path.

## Project layout

```text
src/echomind/          Audio pipeline, EfficientAT adapter, live session and API
tests/                 Python regression tests (mock audio/model dependencies)
web/                   React + TypeScript dashboard and frontend tests
audio/                 Placeholder for local audio; recordings are not committed
reports/               Generated evaluation reports; reports are not committed
resources/              Place the local EfficientAT checkpoint here
live_dashboard.py      Native desktop launcher
serve_web.py           Local FastAPI launcher
predict.py             Legacy offline prediction entry point
evaluate.py            Legacy recording evaluation entry point
pyproject.toml          Python package and dependency metadata
```

## Setup

Requirements: Python 3.12, a working microphone, and Node.js/npm for the web interface. On Apple Silicon, PyTorch can use MPS when available; the inference implementation also supports CPU fallback. Install into a project virtual environment:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[web,test]'
```

The EfficientAT checkpoint is not included, automatically downloaded, or tracked by Git. Obtain the `dymn10_as` checkpoint from the [official EfficientAT v0.0.1 release](https://github.com/fschmid56/EfficientAT/releases/tag/v0.0.1) and place it at:

```text
resources/dymn10_as.pt
```

The `.gitignore` excludes model weight files and local recordings. Keep the checkpoint and any personal recordings on your machine; do not add them to a public release.

On macOS, grant microphone access to the terminal or Python launcher in **System Settings → Privacy & Security → Microphone**. EchoMind captures from the default input device. If permission is denied or no input is available, the application reports an error.

## Run the native dashboard

From the project root:

```bash
.venv/bin/python live_dashboard.py
```

The default analysis window is 4 seconds, with a 2-second hop. Example options:

```bash
.venv/bin/python live_dashboard.py --window-seconds 6 --hop-seconds 3 --on-threshold 0.40 --off-threshold 0.30
```

## Run the web dashboard

Start the backend in one terminal from the project root:

```bash
.venv/bin/python serve_web.py
```

It binds to `127.0.0.1:8765` by default. Start the frontend in another terminal:

```bash
cd web
npm ci
npm run dev
```

Open the local address Vite prints (normally `http://127.0.0.1:5173`). Vite proxies API and WebSocket requests to the local Python backend. Keep the backend on its default loopback host: the API has no authentication, and a non-loopback host override can expose microphone controls and scores to other devices.

The web interface visualizes the backend's 160-point min/max waveform and 64×64 time-major log-power spectrogram. Oldest frames appear on the left and low frequencies at the bottom. Spectrogram values are dBFS, not class labels. RMS/dBFS and the meter are digital signal level only; they are not calibrated SPL or dB(A). Visualizations clear when capture stops or fresh data is unavailable.

## Privacy

Microphone processing and inference run locally. EchoMind does not write captured microphone audio to disk or send it to an external service. The web API has no audio-upload route and returns only bounded derived visualization arrays. The model checkpoint must be obtained separately from its official release; it is not fetched by the application at runtime.

WAV recordings, generated reports, model weights, local credentials/configuration, and build/cache files are excluded by `.gitignore`. The ignore rules are safeguards; review the final file list before publishing any repository so private files are not force-added.

## Tests and build

Python tests use generated signals and mocked microphone streams; they do not require the real recordings or model checkpoint:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Frontend tests use mocked API/WebSocket data and a mocked canvas:

```bash
npm test --prefix web
npm run build --prefix web
```

The Python API tests also use mock audio capture. These automated tests do not replace a real microphone smoke test or a labeled sound-event evaluation.

## Model attribution and licensing

EchoMind uses the EfficientAT DyMN architecture and AudioSet label map. Vendored EfficientAT source files and their license are in `src/echomind/_efficientat/`; see that directory's `LICENSE` and `SOURCE.txt` and the [official EfficientAT repository](https://github.com/fschmid56/EfficientAT) for attribution and source details. The pretrained checkpoint is obtained separately and is deliberately not redistributed here. Review the original release's terms before redistributing model weights or derived assets; do not assume that the source-code license automatically covers every checkpoint or dataset.

The included EfficientAT source is covered by its vendored MIT license. No project-level license is included for the rest of EchoMind, so the upstream license should not be interpreted as licensing this repository's original code. Choose and add a project license before inviting reuse or contributions.

## Limitations and evaluation

EfficientAT produces independent raw model scores. Scores are not calibrated probabilities and do not prove a sound is present. Events depend on thresholds and persistence; short sounds can be missed, and overlapping labels do not mean sources have been separated. Real-world performance varies with room acoustics, microphone, background noise, source distance, and sound class. The waveform and spectrogram are visualizations of signal energy, not recognition results.

No sustained real-time performance or general detection-accuracy claim is made. Evaluate using separately labeled positive and negative recordings in the intended environment; do not treat the small existing sample set as proof of general reliability.

The heuristic event thresholds should be calibrated against labeled recordings before relying on detections. The system is not a safety monitor, emergency detector, or source-separation tool.
