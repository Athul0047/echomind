export type SessionStatus =
  | "stopped"
  | "starting"
  | "listening"
  | "stopping"
  | "error"
  | string;

export interface StatusSnapshot {
  status: SessionStatus;
  error: string | null;
  device: string | null;
  sample_rate: number | null;
  captured_samples: number | null;
  retained_samples: number | null;
  input_rms: number | null;
  windows_submitted: number | null;
  windows_processed: number | null;
  dropped_windows: number | null;
  inference_seconds: number | null;
  window_to_result_seconds: number | null;
  onset_threshold: number | null;
  release_threshold: number | null;
  calibrated_probabilities: false;
}

export interface ActiveEvent {
  label: string;
  raw_model_score: number | null;
  started_at: string;
  duration_seconds: number | null;
}

export interface HistoryEvent {
  label: string;
  started_at: string;
  duration_seconds: number | null;
  final_raw_model_score: number | null;
}

export interface EventsSnapshot {
  active: ActiveEvent[];
  history: HistoryEvent[];
  onset_threshold: number | null;
  release_threshold: number | null;
  onset_windows_required: number | null;
  release_windows_required: number | null;
}

export interface ScoresSnapshot {
  kind: "raw_model_scores";
  calibrated_probabilities: false;
  scores: Record<string, number>;
  inference_seconds: number | null;
}

export interface RankedScore {
  label: string;
  score: number;
}

export interface LiveUpdate {
  type: "update";
  status: SessionStatus;
  error: string | null;
  device: string | null;
  scores_kind: "raw_model_scores";
  calibrated_probabilities: false;
  top_scores: RankedScore[];
  active_events: ActiveEvent[];
  event_history: HistoryEvent[];
  windows_processed: number;
  inference_seconds: number | null;
  window_to_result_seconds: number | null;
}

export interface VisualizationUnavailable {
  status: "unavailable";
  reason: string;
  detail?: string;
}

export interface VisualizationAvailable {
  status: "available";
  available: true;
  sample_rate: number;
  sample_count: number;
  duration_seconds: number;
  sample_sequence_start: number;
  sample_sequence_end: number;
  captured_at: string | null;
  waveform: { encoding: "min_max"; points: Array<{ min: number; max: number }> };
  spectrogram: {
    encoding: "log_power_dbfs";
    frequency_min_hz: number;
    frequency_max_hz: number;
    /** Backend order is time-major, oldest frame first; each row runs low to high frequency. */
    frames: number[][];
  };
  rms: number | null;
  rms_dbfs: number | null;
  rms_note: string;
}

export type VisualizationSnapshot = VisualizationUnavailable | VisualizationAvailable;

function asObject(value: unknown, context: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`Invalid ${context} response from EchoMind backend.`);
  }
  return value as Record<string, unknown>;
}

function asNumber(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function nullableNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function nullableString(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function finiteNumber(value: unknown, context: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`Invalid ${context} in EchoMind visualization response.`);
  }
  return value;
}

function parseDbfs(value: unknown): number | null {
  if (value === null || value === undefined) return null;
  if (value === Number.NEGATIVE_INFINITY) return Number.NEGATIVE_INFINITY;
  return finiteNumber(value, "dBFS level");
}

export function parseVisualization(value: unknown): VisualizationSnapshot {
  const row = asObject(value, "visualization");
  if (row.status === "unavailable") {
    return {
      status: "unavailable",
      reason: typeof row.reason === "string" ? row.reason : "visualization_unavailable",
      ...(typeof row.detail === "string" ? { detail: row.detail } : {}),
    };
  }
  if (row.status !== "available" || row.available !== true) {
    throw new Error("Visualization response is missing its available state.");
  }

  const sampleRate = finiteNumber(row.sample_rate, "sample rate");
  const sampleCount = finiteNumber(row.sample_count, "sample count");
  const duration = finiteNumber(row.duration_seconds, "duration");
  const startSequence = finiteNumber(row.sample_sequence_start, "start sequence");
  const endSequence = finiteNumber(row.sample_sequence_end, "end sequence");
  if (sampleRate <= 0 || sampleCount <= 0 || sampleCount > 96_000 || duration <= 0 || endSequence < startSequence) {
    throw new Error("Visualization response contains invalid sample bounds.");
  }

  const waveformRow = asObject(row.waveform, "waveform");
  if (waveformRow.encoding !== "min_max" || !Array.isArray(waveformRow.points) || waveformRow.points.length !== 160) {
    throw new Error("Visualization waveform must contain 160 min/max points.");
  }
  const points = waveformRow.points.map((point) => {
    const item = asObject(point, "waveform point");
    const min = finiteNumber(item.min, "waveform minimum");
    const max = finiteNumber(item.max, "waveform maximum");
    if (min > max) throw new Error("Visualization waveform minimum exceeds maximum.");
    return { min, max };
  });

  const spectrumRow = asObject(row.spectrogram, "spectrogram");
  if (spectrumRow.encoding !== "log_power_dbfs" || !Array.isArray(spectrumRow.frames) || spectrumRow.frames.length !== 64) {
    throw new Error("Visualization spectrogram must contain 64 time frames.");
  }
  const frames = spectrumRow.frames.map((frame) => {
    if (!Array.isArray(frame) || frame.length !== 64) throw new Error("Each spectrogram frame must contain 64 frequency bins.");
    return frame.map((bin) => {
      const power = finiteNumber(bin, "spectrogram value");
      if (power < -120 || power > 0) throw new Error("Spectrogram log power is outside its documented dBFS range.");
      return power;
    });
  });
  const frequencyMin = finiteNumber(spectrumRow.frequency_min_hz, "minimum frequency");
  const frequencyMax = finiteNumber(spectrumRow.frequency_max_hz, "maximum frequency");
  if (frequencyMin < 0 || frequencyMax <= frequencyMin || frequencyMax > sampleRate / 2 + 1) {
    throw new Error("Visualization frequency range is invalid for the capture sample rate.");
  }
  const rms = nullableNumber(row.rms);
  if (rms !== null && rms < 0) throw new Error("Visualization RMS cannot be negative.");

  return {
    status: "available",
    available: true,
    sample_rate: sampleRate,
    sample_count: sampleCount,
    duration_seconds: duration,
    sample_sequence_start: startSequence,
    sample_sequence_end: endSequence,
    captured_at: nullableString(row.captured_at),
    waveform: { encoding: "min_max", points },
    spectrogram: {
      encoding: "log_power_dbfs",
      frequency_min_hz: frequencyMin,
      frequency_max_hz: frequencyMax,
      frames,
    },
    rms,
    rms_dbfs: parseDbfs(row.rms_dbfs),
    rms_note: typeof row.rms_note === "string" ? row.rms_note : "Digital signal level; not calibrated SPL or dB(A).",
  };
}

function parseActive(value: unknown): ActiveEvent[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((entry) => {
    const row = asObject(entry, "active event");
    if (typeof row.label !== "string") return [];
    return [{
      label: row.label,
      raw_model_score: nullableNumber(row.raw_model_score),
      started_at: typeof row.started_at === "string" ? row.started_at : "",
      duration_seconds: nullableNumber(row.duration_seconds),
    }];
  });
}

function parseHistory(value: unknown): HistoryEvent[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((entry) => {
    const row = asObject(entry, "history event");
    if (typeof row.label !== "string") return [];
    return [{
      label: row.label,
      started_at: typeof row.started_at === "string" ? row.started_at : "",
      duration_seconds: nullableNumber(row.duration_seconds),
      final_raw_model_score: nullableNumber(row.final_raw_model_score),
    }];
  });
}

export function parseStatus(value: unknown): StatusSnapshot {
  const row = asObject(value, "status");
  if (typeof row.status !== "string") throw new Error("Status response is missing its session state.");
  return {
    status: row.status,
    error: nullableString(row.error),
    device: nullableString(row.device),
    sample_rate: nullableNumber(row.sample_rate),
    captured_samples: nullableNumber(row.captured_samples),
    retained_samples: nullableNumber(row.retained_samples),
    input_rms: nullableNumber(row.input_rms),
    windows_submitted: nullableNumber(row.windows_submitted),
    windows_processed: nullableNumber(row.windows_processed),
    dropped_windows: nullableNumber(row.dropped_windows),
    inference_seconds: nullableNumber(row.inference_seconds),
    window_to_result_seconds: nullableNumber(row.window_to_result_seconds),
    onset_threshold: nullableNumber(row.onset_threshold),
    release_threshold: nullableNumber(row.release_threshold),
    calibrated_probabilities: false,
  };
}

export function parseEvents(value: unknown): EventsSnapshot {
  const row = asObject(value, "events");
  return {
    active: parseActive(row.active),
    history: parseHistory(row.history),
    onset_threshold: nullableNumber(row.onset_threshold),
    release_threshold: nullableNumber(row.release_threshold),
    onset_windows_required: nullableNumber(row.onset_windows_required),
    release_windows_required: nullableNumber(row.release_windows_required),
  };
}

export function parseScores(value: unknown): ScoresSnapshot {
  const row = asObject(value, "scores");
  if (row.kind !== "raw_model_scores") throw new Error("Backend did not identify its scores as raw model scores.");
  const scoreObject = asObject(row.scores, "raw model scores");
  const scores: Record<string, number> = {};
  for (const [label, score] of Object.entries(scoreObject)) {
    if (typeof score === "number" && Number.isFinite(score)) scores[label] = score;
  }
  return {
    kind: "raw_model_scores",
    calibrated_probabilities: false,
    scores,
    inference_seconds: nullableNumber(row.inference_seconds),
  };
}

export function parseLiveUpdate(value: unknown): LiveUpdate {
  const row = asObject(value, "live update");
  if (row.type !== "update" || row.scores_kind !== "raw_model_scores") {
    throw new Error("Unrecognized EchoMind live update.");
  }
  const top_scores = Array.isArray(row.top_scores)
    ? row.top_scores.flatMap((entry) => {
      const score = asObject(entry, "ranked score");
      if (typeof score.label !== "string" || typeof score.score !== "number" || !Number.isFinite(score.score)) return [];
      return [{ label: score.label, score: score.score }];
    })
    : [];
  return {
    type: "update",
    status: typeof row.status === "string" ? row.status : "unknown",
    error: nullableString(row.error),
    device: nullableString(row.device),
    scores_kind: "raw_model_scores",
    calibrated_probabilities: false,
    top_scores,
    active_events: parseActive(row.active_events),
    event_history: parseHistory(row.event_history),
    windows_processed: asNumber(row.windows_processed),
    inference_seconds: nullableNumber(row.inference_seconds),
    window_to_result_seconds: nullableNumber(row.window_to_result_seconds),
  };
}

export async function getJson<T>(path: string, parse: (value: unknown) => T, signal?: AbortSignal): Promise<T> {
  const response = await fetch(path, { signal, headers: { Accept: "application/json" } });
  if (!response.ok) throw new Error(`EchoMind API returned HTTP ${response.status}.`);
  return parse(await response.json());
}

export async function postAction(path: "/api/start" | "/api/stop"): Promise<void> {
  const response = await fetch(path, { method: "POST", headers: { Accept: "application/json" } });
  if (!response.ok) {
    let detail = `EchoMind API returned HTTP ${response.status}.`;
    try {
      const body = asObject(await response.json(), "error");
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      // Keep the useful HTTP status message if the server didn't return JSON.
    }
    throw new Error(detail);
  }
}
