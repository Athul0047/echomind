import { describe, expect, it } from "vitest";
import { parseEvents, parseLiveUpdate, parseScores, parseStatus, parseVisualization } from "./api";

function visualizationPayload(overrides: Record<string, unknown> = {}) {
  return {
    status: "available", available: true, sample_rate: 44100, sample_count: 88200,
    duration_seconds: 2, sample_sequence_start: 100, sample_sequence_end: 88300,
    captured_at: "2026-10-24T10:00:00.000+00:00",
    waveform: { encoding: "min_max", points: Array.from({ length: 160 }, (_, index) => ({ min: -index / 160, max: index / 160 })) },
    spectrogram: {
      encoding: "log_power_dbfs", frequency_min_hz: 0, frequency_max_hz: 7971,
      frames: Array.from({ length: 64 }, (_, timeIndex) => Array.from({ length: 64 }, (_, freqIndex) => -120 + timeIndex + freqIndex / 64)),
    },
    rms: 0.25, rms_dbfs: -12.04, rms_note: "Digital full-scale level; not calibrated acoustic SPL or dB(A).",
    ...overrides,
  };
}

describe("backend response parsing", () => {
  it("reads the actual status schema and leaves missing timing unavailable", () => {
    const status = parseStatus({
      status: "listening", error: null, device: "MacBook Microphone", sample_rate: 44100,
      captured_samples: 900, retained_samples: 900, input_rms: 0.02,
      windows_submitted: 4, windows_processed: 3, dropped_windows: 1,
      inference_seconds: null, window_to_result_seconds: 4.1,
      onset_threshold: 0.35, release_threshold: 0.3, calibrated_probabilities: false,
    });
    expect(status.status).toBe("listening");
    expect(status.sample_rate).toBe(44100);
    expect(status.inference_seconds).toBeNull();
    expect(status.calibrated_probabilities).toBe(false);
  });

  it("parses active and archived event field names without inventing data", () => {
    const events = parseEvents({
      active: [{ label: "Dog barking", raw_model_score: 0.44, started_at: "2026-10-24T10:00:00", duration_seconds: 2.5 }],
      history: [{ label: "Sneeze", started_at: "2026-10-24T09:59:00", duration_seconds: 0.8, final_raw_model_score: 0.21 }],
      onset_threshold: 0.35, release_threshold: 0.3,
      onset_windows_required: 2, release_windows_required: 3,
    });
    expect(events.active[0]).toMatchObject({ label: "Dog barking", raw_model_score: 0.44, duration_seconds: 2.5 });
    expect(events.history[0]).toMatchObject({ label: "Sneeze", final_raw_model_score: 0.21 });
  });

  it("accepts only the explicit raw model score schema and filters non-finite values", () => {
    const scores = parseScores({
      kind: "raw_model_scores", calibrated_probabilities: false,
      scores: { Speech: 0.32, Fan: 0.14, Invalid: Number.NaN }, inference_seconds: 0.072,
    });
    expect(scores.scores).toEqual({ Speech: 0.32, Fan: 0.14 });
    expect(scores.inference_seconds).toBe(0.072);
    expect(() => parseScores({ kind: "probabilities", scores: {} })).toThrow(/raw model scores/);
  });

  it("parses compact WebSocket updates and ignores malformed optional score entries", () => {
    const update = parseLiveUpdate({
      type: "update", status: "listening", error: null, device: "Input",
      scores_kind: "raw_model_scores", calibrated_probabilities: false,
      top_scores: [{ label: "Speech", score: 0.6 }, { label: 4, score: 0.2 }],
      active_events: [], event_history: [], windows_processed: 8,
      inference_seconds: 0.07, window_to_result_seconds: 4.2,
    });
    expect(update.top_scores).toEqual([{ label: "Speech", score: 0.6 }]);
    expect(update.calibrated_probabilities).toBe(false);
    expect(() => parseLiveUpdate({ type: "other" })).toThrow(/Unrecognized/);
  });

  it("preserves oldest-first time and low-to-high frequency matrix order", () => {
    const visualization = parseVisualization(visualizationPayload());
    expect(visualization.status).toBe("available");
    if (visualization.status !== "available") throw new Error("expected available visualization");
    expect(visualization.waveform.points).toHaveLength(160);
    expect(visualization.waveform.points[0].max).toBe(0);
    expect(visualization.waveform.points[159].max).toBeCloseTo(159 / 160);
    expect(visualization.spectrogram.frames).toHaveLength(64);
    expect(visualization.spectrogram.frames[0]).toHaveLength(64);
    expect(visualization.spectrogram.frames[0][0]).toBe(-120);
    expect(visualization.spectrogram.frames[63][63]).toBeCloseTo(-56.015625);
    expect(visualization.spectrogram.frequency_max_hz).toBe(7971);
  });

  it("reads the explicit unavailable response without inventing arrays", () => {
    expect(parseVisualization({ status: "unavailable", reason: "audio_samples_stale" })).toEqual({
      status: "unavailable", reason: "audio_samples_stale",
    });
  });

  it("handles digital silence and negative-infinity dBFS safely", () => {
    const visualization = parseVisualization(visualizationPayload({ rms: 0, rms_dbfs: -Infinity }));
    expect(visualization.status).toBe("available");
    if (visualization.status !== "available") throw new Error("expected available visualization");
    expect(visualization.rms).toBe(0);
    expect(visualization.rms_dbfs).toBe(-Infinity);
  });

  it("rejects unexpected matrix dimensions or invalid encodings", () => {
    const invalid = visualizationPayload({
      spectrogram: { encoding: "magnitude", frequency_min_hz: 0, frequency_max_hz: 8000, frames: [] },
    });
    expect(() => parseVisualization(invalid)).toThrow(/spectrogram/);
  });
});
