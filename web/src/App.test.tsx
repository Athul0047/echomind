import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";

let sessionState = "stopped";
const sampleStatus = () => ({
  status: sessionState, error: null, device: "MacBook Microphone", sample_rate: 44100,
  captured_samples: 44100, retained_samples: 44100, input_rms: 0.012,
  windows_submitted: 7, windows_processed: 6, dropped_windows: 1,
  inference_seconds: 0.072, window_to_result_seconds: 4.1,
  onset_threshold: 0.35, release_threshold: 0.3, calibrated_probabilities: false,
});
const sampleScores = {
  kind: "raw_model_scores", calibrated_probabilities: false,
  scores: { "Dog barking": 0.44, Speech: 0.27, Fan: 0.11 }, inference_seconds: 0.072,
};
const sampleEvents = {
  active: [{ label: "Dog barking", raw_model_score: 0.44, started_at: "2026-10-24T10:00:00", duration_seconds: 2.5 }],
  history: [{ label: "Sneeze", started_at: "2026-10-24T09:59:00", duration_seconds: 0.8, final_raw_model_score: 0.21 }],
  onset_threshold: 0.35, release_threshold: 0.3, onset_windows_required: 2, release_windows_required: 3,
};
const sampleVisualization = () => ({
  status: "available", available: true, sample_rate: 44100, sample_count: 88200,
  duration_seconds: 2, sample_sequence_start: 100, sample_sequence_end: 88300,
  captured_at: "2026-10-24T10:00:00.000+00:00",
  waveform: { encoding: "min_max", points: Array.from({ length: 160 }, (_, index) => ({ min: -0.2, max: 0.2 + index / 1000 })) },
  spectrogram: {
    encoding: "log_power_dbfs", frequency_min_hz: 0, frequency_max_hz: 7971,
    frames: Array.from({ length: 64 }, (_, time) => Array.from({ length: 64 }, (_, frequency) => -100 + time / 10 + frequency / 10)),
  },
  rms: 0.125, rms_dbfs: -18.06, rms_note: "Digital full-scale level; not calibrated acoustic SPL or dB(A).",
});

class MockWebSocket {
  static instances: MockWebSocket[] = [];
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSED = 3;
  onopen: (() => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;
  readyState: number = WebSocket.CONNECTING;
  url: string;
  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
    queueMicrotask(() => {
      this.readyState = WebSocket.OPEN;
      this.onopen?.();
    });
  }
  close() {
    this.readyState = WebSocket.CLOSED;
    this.onclose?.();
  }
  send() {}
  emit(value: unknown) {
    this.onmessage?.({ data: JSON.stringify(value) } as MessageEvent);
  }
}

function liveUpdate() {
  return {
    type: "update", status: sessionState, error: null, device: "MacBook Microphone",
    scores_kind: "raw_model_scores", calibrated_probabilities: false,
    top_scores: [{ label: "Dog barking", score: 0.47 }, { label: "Speech", score: 0.25 }],
    active_events: sampleEvents.active, event_history: sampleEvents.history,
    windows_processed: 6, inference_seconds: 0.072, window_to_result_seconds: 4.1,
  };
}

function mockApi() {
  return vi.fn(async (input: RequestInfo | URL) => {
    const path = String(input);
    if (path === "/api/start") sessionState = "listening";
    if (path === "/api/stop") sessionState = "stopped";
    let body: unknown;
    if (path === "/api/status") body = sampleStatus();
    else if (path === "/api/scores") body = sampleScores;
    else if (path === "/api/events") body = sampleEvents;
    else if (path === "/api/visualization") body = sampleVisualization();
    else body = { status: sessionState };
    return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
  });
}

beforeEach(() => {
  sessionState = "stopped";
  MockWebSocket.instances = [];
  vi.stubGlobal("fetch", mockApi());
  vi.stubGlobal("WebSocket", MockWebSocket);
  const canvasContext = {
    setTransform: vi.fn(), clearRect: vi.fn(), fillRect: vi.fn(), fillStyle: "",
  } as unknown as CanvasRenderingContext2D;
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(canvasContext);
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("EchoMind dashboard", () => {
  it("renders actual backend status, active/history events, and raw score data", async () => {
    render(<App />);
    expect(await screen.findByText("Backend connected")).toBeInTheDocument();
    expect(screen.getAllByText("Dog barking").length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("Sneeze")).toBeInTheDocument();
    expect(screen.getAllByText("0.440").length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText(/not calibrated probabilities/)).toBeInTheDocument();
    expect(screen.getByText("Start the microphone to view signal")).toBeInTheDocument();
  });

  it("renders actual waveform, spectrogram, and digital audio level while listening", async () => {
    sessionState = "listening";
    const fetchMock = mockApi();
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);
    expect(await screen.findByText("LIVE DATA")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /waveform min and max envelope/ })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /microphone spectrogram/i })).toBeInTheDocument();
    expect(screen.getByText("-18.1 dBFS")).toBeInTheDocument();
    expect(screen.getByText(/RMS 0.1250/)).toBeInTheDocument();
    expect(screen.getByText("7.97 kHz")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith("/api/visualization", expect.any(Object));
  });

  it("displays the backend silence floor as digital dBFS", async () => {
    sessionState = "listening";
    const silence = sampleVisualization();
    silence.rms = 0;
    silence.rms_dbfs = -120;
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input);
      const body = path === "/api/status" ? sampleStatus()
        : path === "/api/scores" ? sampleScores
          : path === "/api/events" ? sampleEvents
            : silence;
      return new Response(JSON.stringify(body), { status: 200 });
    }));
    render(<App />);
    expect(await screen.findByText("-120.0 dBFS")).toBeInTheDocument();
    expect(screen.getByText(/RMS 0.0000/)).toBeInTheDocument();
  });

  it("shows a clear unavailable state for stale backend samples and drops the plot", async () => {
    sessionState = "listening";
    let visualizationCalls = 0;
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      if (String(input) === "/api/visualization") {
        visualizationCalls += 1;
        if (visualizationCalls === 1) {
          return new Response(JSON.stringify(sampleVisualization()), { status: 200 });
        }
        return new Response(JSON.stringify({ status: "unavailable", reason: "audio_samples_stale" }), { status: 200 });
      }
      return new Response(JSON.stringify(String(input) === "/api/status" ? sampleStatus() : String(input) === "/api/scores" ? sampleScores : sampleEvents), { status: 200 });
    }));
    render(<App />);
    expect(await screen.findByText("LIVE DATA")).toBeInTheDocument();
    expect(await screen.findByText("Waiting for fresh microphone data")).toBeInTheDocument();
    expect(screen.queryByRole("img", { name: /microphone spectrogram/i })).not.toBeInTheDocument();
  });

  it("does not request or invent visualization while the session is stopped", async () => {
    const fetchMock = mockApi();
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);
    expect(await screen.findByText("Start the microphone to view signal")).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalledWith("/api/visualization", expect.anything());
    expect(screen.queryByRole("img", { name: /microphone spectrogram/i })).not.toBeInTheDocument();
  });

  it("clears visualization data and reports endpoint failures", async () => {
    sessionState = "listening";
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      if (String(input) === "/api/visualization") throw new Error("visual endpoint offline");
      const body = String(input) === "/api/status" ? sampleStatus() : String(input) === "/api/scores" ? sampleScores : sampleEvents;
      return new Response(JSON.stringify(body), { status: 200 });
    }));
    render(<App />);
    expect(await screen.findByText("Visualization temporarily unavailable")).toBeInTheDocument();
    expect(screen.getByText("visual endpoint offline")).toBeInTheDocument();
    expect(screen.queryByRole("img", { name: /microphone spectrogram/ })).not.toBeInTheDocument();
  });

  it("uses the WebSocket payload for live score and active event updates", async () => {
    render(<App />);
    await screen.findByText("Backend connected");
    await waitFor(() => expect(MockWebSocket.instances.length).toBe(1));
    act(() => MockWebSocket.instances[0].emit(liveUpdate()));
    expect(await screen.findByText("0.470")).toBeInTheDocument();
    expect(screen.getByText("Streaming live")).toBeInTheDocument();
    expect(MockWebSocket.instances[0].url).toContain("/api/ws");
  });

  it("calls the actual start and stop endpoints and reflects backend state", async () => {
    const fetchMock = mockApi();
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);
    const start = await screen.findByRole("button", { name: /start listening/i });
    fireEvent.click(start);
    expect(await screen.findByRole("button", { name: /stop session/i })).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith("/api/start", expect.objectContaining({ method: "POST" }));
    fireEvent.click(screen.getByRole("button", { name: /stop session/i }));
    expect(await screen.findByRole("button", { name: /start listening/i })).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith("/api/stop", expect.objectContaining({ method: "POST" }));
  });

  it("starts visualization polling with listening and stops it after the session stops", async () => {
    const fetchMock = mockApi();
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);
    await screen.findByText("Backend connected");
    expect(fetchMock).not.toHaveBeenCalledWith("/api/visualization", expect.anything());

    fireEvent.click(screen.getByRole("button", { name: /start listening/i }));
    await screen.findByText("LIVE DATA");
    expect(fetchMock).toHaveBeenCalledWith("/api/visualization", expect.any(Object));

    fireEvent.click(screen.getByRole("button", { name: /stop session/i }));
    await screen.findByRole("button", { name: /start listening/i });
    const callsAfterStop = fetchMock.mock.calls.filter(([path]) => path === "/api/visualization").length;
    await new Promise((resolve) => window.setTimeout(resolve, 400));
    expect(fetchMock.mock.calls.filter(([path]) => path === "/api/visualization")).toHaveLength(callsAfterStop);
  });

  it("shows backend connection failures without inventing measurements", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline")));
    render(<App />);
    expect(await screen.findByText("Backend offline")).toBeInTheDocument();
    expect(screen.getByText("Could not reach the local EchoMind backend. Check that serve_web.py is running.")).toBeInTheDocument();
    expect(screen.getByText("Visualization backend offline")).toBeInTheDocument();
    expect(screen.getAllByText("—").length).toBeGreaterThan(0);
  });

  it("moves the live feed into reconnecting state when the socket closes", async () => {
    render(<App />);
    await screen.findByText("Backend connected");
    await waitFor(() => expect(MockWebSocket.instances.length).toBe(1));
    act(() => MockWebSocket.instances[0].close());
    expect(await screen.findByText("Live feed reconnecting")).toBeInTheDocument();
  });
});
