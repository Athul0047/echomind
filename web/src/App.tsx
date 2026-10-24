import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import {
  getJson,
  parseEvents,
  parseLiveUpdate,
  parseScores,
  parseStatus,
  parseVisualization,
  postAction,
  type ActiveEvent,
  type EventsSnapshot,
  type HistoryEvent,
  type LiveUpdate,
  type RankedScore,
  type ScoresSnapshot,
  type StatusSnapshot,
  type VisualizationSnapshot,
} from "./api";
import { AudioVisualization } from "./Visualization";

type BackendConnection = "connecting" | "online" | "offline";
type SocketConnection = "connecting" | "live" | "reconnecting";

const emptyEvents: EventsSnapshot = {
  active: [], history: [], onset_threshold: 0.35, release_threshold: 0.3,
  onset_windows_required: 2, release_windows_required: 3,
};

function Icon({ name, size = 18 }: { name: "wave" | "mic" | "arrow" | "clock" | "signal" | "spark" | "radio"; size?: number }) {
  const common = { width: size, height: size, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 1.7, strokeLinecap: "round" as const, strokeLinejoin: "round" as const, "aria-hidden": true as const };
  const paths: Record<typeof name, React.ReactNode> = {
    wave: <><path d="M2 10v4"/><path d="M6 6v12"/><path d="M10 3v18"/><path d="M14 7v10"/><path d="M18 5v14"/><path d="M22 10v4"/></>,
    mic: <><rect x="9" y="2.5" width="6" height="12" rx="3"/><path d="M5 11.5a7 7 0 0 0 14 0M12 18.5V22M9 22h6"/></>,
    arrow: <><path d="M7 17 17 7M7 7h10v10"/></>,
    clock: <><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></>,
    signal: <><path d="M3 10a13 13 0 0 1 18 0M6 13a9 9 0 0 1 12 0M9 16a5 5 0 0 1 6 0M12 20h.01"/></>,
    spark: <><path d="m12 3 1.5 5.5L19 10l-5.5 1.5L12 17l-1.5-5.5L5 10l5.5-1.5L12 3Z"/><path d="m19 16 .8 2.2L22 19l-2.2.8L19 22l-.8-2.2L16 19l2.2-.8L19 16Z"/></>,
    radio: <><circle cx="12" cy="12" r="2"/><path d="M16.2 7.8a6 6 0 0 1 0 8.4M7.8 16.2a6 6 0 0 1 0-8.4M19 5a10 10 0 0 1 0 14M5 19A10 10 0 0 1 5 5"/></>,
  };
  return <svg {...common}>{paths[name]}</svg>;
}

export default function App() {
  const [status, setStatus] = useState<StatusSnapshot | null>(null);
  const [scores, setScores] = useState<ScoresSnapshot | null>(null);
  const [events, setEvents] = useState<EventsSnapshot>(emptyEvents);
  const [live, setLive] = useState<LiveUpdate | null>(null);
  const [backendConnection, setBackendConnection] = useState<BackendConnection>("connecting");
  const [socketConnection, setSocketConnection] = useState<SocketConnection>("connecting");
  const [lastLiveAt, setLastLiveAt] = useState<number | null>(null);
  const [actionBusy, setActionBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [now, setNow] = useState(Date.now());
  const [visualization, setVisualization] = useState<VisualizationSnapshot | null>(null);
  const [visualizationError, setVisualizationError] = useState<string | null>(null);
  const refreshRef = useRef<() => Promise<void>>(async () => undefined);

  const refresh = useCallback(async (signal?: AbortSignal) => {
    const results = await Promise.allSettled([
      getJson("/api/status", parseStatus, signal),
      getJson("/api/scores", parseScores, signal),
      getJson("/api/events", parseEvents, signal),
    ]);
    const [statusResult, scoresResult, eventsResult] = results;
    if (statusResult.status === "fulfilled") {
      setStatus(statusResult.value);
      setBackendConnection("online");
      setNotice((current) => current?.startsWith("Could not reach") ? null : current);
    } else if (!signal?.aborted) {
      setBackendConnection("offline");
      setNotice("Could not reach the local EchoMind backend. Check that serve_web.py is running.");
    }
    if (scoresResult.status === "fulfilled") setScores(scoresResult.value);
    if (eventsResult.status === "fulfilled") setEvents(eventsResult.value);
    setNow(Date.now());
  }, []);
  refreshRef.current = () => refresh();

  useEffect(() => {
    let mounted = true;
    let socket: WebSocket | null = null;
    let reconnectTimer: number | undefined;
    let pollTimer: number | undefined;
    let retryDelay = 700;
    let requestPending = false;
    const controller = new AbortController();

    const poll = async () => {
      if (!mounted || requestPending) return;
      requestPending = true;
      try {
        await refresh(controller.signal);
      } finally {
        requestPending = false;
      }
    };
    void poll();
    pollTimer = window.setInterval(poll, 5000);

    const connect = () => {
      if (!mounted) return;
      setSocketConnection("connecting");
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      socket = new WebSocket(`${protocol}//${window.location.host}/api/ws`);
      socket.onopen = () => {
        retryDelay = 700;
        if (mounted) setSocketConnection("live");
      };
      socket.onmessage = (message) => {
        try {
          const update = parseLiveUpdate(JSON.parse(String(message.data)));
          if (!mounted) return;
          setLive(update);
          setLastLiveAt(Date.now());
          setNow(Date.now());
          setBackendConnection("online");
          setStatus((current) => current ? {
            ...current,
            status: update.status,
            error: update.error,
            device: update.device,
            windows_processed: update.windows_processed,
            inference_seconds: update.inference_seconds,
            window_to_result_seconds: update.window_to_result_seconds,
          } : current);
          setEvents((current) => ({ ...current, active: update.active_events, history: update.event_history }));
        } catch {
          if (mounted) setNotice("Received an update the dashboard could not read. Reconnecting to live data.");
        }
      };
      socket.onerror = () => socket?.close();
      socket.onclose = () => {
        if (!mounted) return;
        setSocketConnection("reconnecting");
        reconnectTimer = window.setTimeout(() => {
          retryDelay = Math.min(retryDelay * 1.7, 10000);
          connect();
        }, retryDelay);
      };
    };
    connect();

    return () => {
      mounted = false;
      controller.abort();
      if (pollTimer !== undefined) window.clearInterval(pollTimer);
      if (reconnectTimer !== undefined) window.clearTimeout(reconnectTimer);
      if (socket) {
        socket.onclose = null;
        socket.close();
      }
    };
    // One connection, one status poll interval, and one freshness clock per mounted dashboard.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refresh]);

  useEffect(() => {
    if (status?.status !== "listening" || backendConnection !== "online") {
      setVisualization(null);
      setVisualizationError(null);
      return;
    }

    let active = true;
    let timer: number | undefined;
    let requestController: AbortController | null = null;
    setVisualization(null);
    setVisualizationError(null);

    const pollVisualization = async () => {
      const controller = new AbortController();
      requestController = controller;
      let timedOut = false;
      const timeout = window.setTimeout(() => {
        timedOut = true;
        controller.abort();
      }, 2000);
      try {
        const snapshot = await getJson("/api/visualization", parseVisualization, controller.signal);
        if (active) {
          setVisualization(snapshot);
          setVisualizationError(null);
        }
      } catch (error) {
        if (active) {
          setVisualization(null);
          setVisualizationError(timedOut
            ? "Visualization request timed out. Waiting for fresh microphone data."
            : error instanceof Error ? error.message : "Visualization data is temporarily unavailable.");
        }
      } finally {
        window.clearTimeout(timeout);
        if (requestController === controller) requestController = null;
        if (active) timer = window.setTimeout(() => void pollVisualization(), 300);
      }
    };

    void pollVisualization();
    return () => {
      active = false;
      if (timer !== undefined) window.clearTimeout(timer);
      requestController?.abort();
    };
  }, [backendConnection, status?.status]);

  const runControl = async (action: "/api/start" | "/api/stop") => {
    setActionBusy(true);
    setNotice(null);
    try {
      await postAction(action);
      await refreshRef.current();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "The microphone action failed.");
      await refreshRef.current();
    } finally {
      setActionBusy(false);
    }
  };

  const activeEvents = live && socketConnection === "live" ? live.active_events : events.active;
  const historyEvents = live && socketConnection === "live" ? live.event_history : events.history;
  const freshSocket = lastLiveAt !== null && now - lastLiveAt < 5000 && socketConnection === "live";
  const rankedScores: RankedScore[] = useMemo(() => {
    if (freshSocket && live) return live.top_scores.slice(0, 8);
    return Object.entries(scores?.scores ?? {})
      .map(([label, score]) => ({ label, score }))
      .sort((left, right) => right.score - left.score)
      .slice(0, 8);
  }, [freshSocket, live, scores]);

  const timeline = useMemo(() => [
    ...activeEvents.map((event) => ({ ...event, active: true, score: event.raw_model_score })),
    ...historyEvents.map((event) => ({ ...event, active: false, score: event.final_raw_model_score })),
  ].sort((left, right) => dateValue(right.started_at) - dateValue(left.started_at)), [activeEvents, historyEvents]);

  const micListening = status?.status === "listening";
  const lastFeedText = lastLiveAt === null
    ? "Awaiting first update"
    : freshSocket ? "Streaming live" : "Live data delayed";

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="EchoMind home">
          <span className="brand-mark"><Icon name="wave" size={20} /></span>
          <span className="brand-copy"><strong>echo<span>mind</span></strong><small>SOUND INTELLIGENCE</small></span>
        </a>
        <div className="topbar-right">
          <div className="connection-cluster" aria-live="polite">
            <span className={`connection-dot ${backendConnection}`} />
            <div><strong>{backendConnection === "online" ? "Backend connected" : backendConnection === "offline" ? "Backend offline" : "Connecting"}</strong>
              <small>{socketConnection === "live" ? "Live feed · 0.5s updates" : socketConnection === "reconnecting" ? "Live feed reconnecting" : "Opening live feed"}</small></div>
          </div>
          <span className={`session-pill ${micListening ? "is-listening" : ""}`}><span />{status?.status ?? "Checking session"}</span>
          <div className="control-group">
            {!micListening ? <button className="button button-start" onClick={() => void runControl("/api/start")} disabled={actionBusy || status?.status === "starting" || status?.status === "stopping"}>
              <Icon name="mic" size={16} />{actionBusy ? "Starting…" : "Start listening"}
            </button> : <button className="button button-stop" onClick={() => void runControl("/api/stop")} disabled={actionBusy}>
              <span className="stop-square" />{actionBusy ? "Stopping…" : "Stop session"}
            </button>}
          </div>
        </div>
      </header>

      <main id="top" className="main-content">
        <section className="intro-row">
          <div>
            <div className="eyebrow"><span className="eyebrow-line" />LOCAL SOUND MONITOR</div>
            <h1>Hear what’s happening<span>.</span></h1>
            <p className="intro-copy">A live view of the sound events around you, processed privately on this device.</p>
          </div>
          <div className="device-chip"><span className="device-icon"><Icon name="radio" size={17}/></span>
            <div><small>INPUT DEVICE</small><strong>{status?.device ?? "No device selected"}</strong></div>
            <span className="local-lock" title="Audio stays local">LOCAL</span>
          </div>
        </section>

        {notice && <div className="notice" role="alert"><span className="notice-icon">!</span><span>{notice}</span><button aria-label="Dismiss message" onClick={() => setNotice(null)}>×</button></div>}
        {status?.status === "error" && status.error && <div className="error-strip" role="status"><strong>Session issue</strong><span>{status.error}</span></div>}

        <section className="metrics-grid" aria-label="Live diagnostics">
          <MetricCard icon="spark" label="Model inference" value={formatMs(status?.inference_seconds)} detail="per analysis window" />
          <MetricCard icon="clock" label="Window to result" value={formatMs(status?.window_to_result_seconds)} detail="includes audio window" />
          <MetricCard icon="signal" label="Windows processed" value={formatCount(status?.windows_processed)} detail={status ? `${formatCount(status.windows_submitted)} submitted · ${formatCount(status.dropped_windows)} dropped` : "Unavailable"} />
          <MetricCard icon="wave" label="Event thresholds" value={status?.onset_threshold !== null && status?.release_threshold !== null && status ? `${status.onset_threshold.toFixed(2)} / ${status.release_threshold.toFixed(2)}` : "—"} detail="onset / release · 2 / 3 windows" />
        </section>

        <section className="dashboard-grid">
          <div className="panel events-panel">
            <PanelHeading icon="radio" kicker="DETECTION" title="Live sound events" trailing={<span className="count-badge">{activeEvents.length} active</span>} />
            {activeEvents.length ? <div className="active-event-list">
              {activeEvents.map((event) => <ActiveEventCard key={`${event.label}-${event.started_at}`} event={event} />)}
            </div> : <div className="empty-events">
              <div className="empty-orbit"><span><Icon name="wave" size={24} /></span></div>
              <strong>No active events</strong>
              <p>{micListening ? "No sounds currently meet the event threshold and persistence rule." : "Start the microphone to monitor for sound events."}</p>
              <small>Detection uses raw model scores with two-window onset confirmation.</small>
            </div>}
          </div>

          <div className="panel scores-panel">
            <PanelHeading icon="spark" kicker="MODEL OUTPUT" title="Live model scores" trailing={<span className="raw-tag">RAW SCORES</span>} />
            <p className="panel-note">Independent model scores · not calibrated probabilities</p>
            {rankedScores.length ? <div className="score-list">
              {rankedScores.map((item, index) => <ScoreRow key={item.label} item={item} rank={index + 1} />)}
            </div> : <div className="empty-small">{backendConnection === "offline" ? "Waiting for the local backend to return scores." : "No model scores have been returned yet."}</div>}
            <div className="score-footnote">A score is not proof that a sound is present.</div>
          </div>
        </section>

        <section className="lower-grid">
          <div className="panel visualization-panel">
            <PanelHeading icon="wave" kicker="SIGNAL VIEW" title="Audio visualization" trailing={<span className={`visual-state-tag ${visualization?.status === "available" ? "is-live" : ""}`}>{visualization?.status === "available" ? "LIVE DATA" : backendConnection === "offline" ? "OFFLINE" : micListening ? "WAITING FOR DATA" : "IDLE"}</span>} />
            {visualization?.status === "available" ? <AudioVisualization data={visualization} /> : <div className="visual-unavailable">
              <div className="visual-icon"><Icon name="wave" size={22} /></div>
              <div><strong>{visualizationError ? "Visualization temporarily unavailable" : backendConnection === "offline" ? "Visualization backend offline" : visualization?.status === "unavailable" ? unavailableTitle(visualization.reason) : micListening ? "Waiting for microphone samples" : "Start the microphone to view signal"}</strong>
                <p>{visualizationError ?? (backendConnection === "offline" ? "The local backend is unreachable. Reconnect it to resume fresh signal updates." : visualization?.status === "unavailable" ? unavailableDescription(visualization.reason) : micListening ? "The dashboard will show the waveform and spectrogram when fresh captured samples are available." : "No idle or simulated signal is shown. Visualization polling runs only while the session is listening.")}</p></div>
            </div>}
            <div className="visual-privacy"><span className="privacy-check">✓</span> Microphone audio is processed locally and is not sent to this page as a stream.</div>
          </div>

          <div className="panel diagnostics-panel">
            <PanelHeading icon="signal" kicker="SYSTEM" title="Diagnostics" />
            <DiagnosticRow label="Session state" value={status?.status ?? "Unavailable"} state={status?.status === "listening" ? "good" : undefined} />
            <DiagnosticRow label="Backend" value={backendConnection === "online" ? "Connected · localhost" : backendConnection === "offline" ? "Offline" : "Connecting"} state={backendConnection === "online" ? "good" : undefined} />
            <DiagnosticRow label="WebSocket" value={lastFeedText} state={freshSocket ? "good" : undefined} />
            <DiagnosticRow label="Input sample rate" value={status?.sample_rate ? `${(status.sample_rate / 1000).toFixed(1)} kHz` : "Unavailable"} />
            <DiagnosticRow label="Captured samples" value={formatCount(status?.captured_samples, "Unavailable")} />
            <DiagnosticRow label="Input level · RMS" value={status?.input_rms === null || !status ? "Unavailable" : status.input_rms.toFixed(4)} />
            <DiagnosticRow label="Window schedule" value="4 s window · 2 s hop" />
            <DiagnosticRow label="Persistence" value={`${events.onset_windows_required ?? "—"} windows on · ${events.release_windows_required ?? "—"} windows off`} />
          </div>
        </section>

        <section className="panel timeline-panel">
          <PanelHeading icon="clock" kicker="ACTIVITY" title="Event timeline" trailing={<span className="muted-label">Recent events</span>} />
          {timeline.length ? <div className="timeline-list">
            {timeline.slice(0, 12).map((event, index) => <TimelineItem key={`${event.label}-${event.started_at}-${event.active ? "active" : "history"}`} event={event} last={index === Math.min(timeline.length, 12) - 1} />)}
          </div> : <div className="timeline-empty"><span className="timeline-empty-mark"><Icon name="clock" size={18}/></span><div><strong>No event history yet</strong><p>Events appear here after the detector records an onset or completed event.</p></div></div>}
        </section>

        <footer className="page-footer"><span><span className="footer-led"/> LOCAL PROCESSING</span><span>EfficientAT DyMN · AudioSet · 527 sound classes</span><span>Scores are model outputs, not guarantees</span></footer>
      </main>
    </div>
  );
}

function MetricCard({ icon, label, value, detail }: { icon: "spark" | "clock" | "signal" | "wave"; label: string; value: string; detail: string }) {
  return <article className="metric-card"><span className="metric-icon"><Icon name={icon} size={16}/></span><div className="metric-label">{label}</div><strong className="metric-value">{value}</strong><div className="metric-detail">{detail}</div></article>;
}

function PanelHeading({ icon, kicker, title, trailing }: { icon: "radio" | "spark" | "wave" | "signal" | "clock"; kicker: string; title: string; trailing?: ReactNode }) {
  return <div className="panel-heading"><span className="panel-icon"><Icon name={icon} size={17}/></span><div className="panel-title"><small>{kicker}</small><h2>{title}</h2></div>{trailing && <div className="panel-trailing">{trailing}</div>}</div>;
}

function ActiveEventCard({ event }: { event: ActiveEvent }) {
  return <article className="active-event-card"><div className="active-marker"><span/><span/></div><div className="event-copy"><div className="event-label-line"><strong>{event.label}</strong><span className="active-state">ACTIVE</span></div><div className="event-meta"><span>Started {formatDate(event.started_at)}</span><span>{formatDuration(event.duration_seconds)} elapsed</span></div></div><div className="event-score"><small>RAW SCORE</small><strong>{formatScore(event.raw_model_score)}</strong></div></article>;
}

function ScoreRow({ item, rank }: { item: RankedScore; rank: number }) {
  const width = Math.max(0, Math.min(item.score * 100, 100));
  return <div className="score-row"><span className="score-rank">{String(rank).padStart(2, "0")}</span><div className="score-info"><div className="score-label-line"><strong>{item.label}</strong><span>{item.score.toFixed(3)}</span></div><div className="score-track"><span style={{ width: `${width}%` }}/></div></div></div>;
}

function DiagnosticRow({ label, value, state }: { label: string; value: string; state?: "good" }) {
  return <div className="diagnostic-row"><span>{label}</span><strong className={state === "good" ? "diagnostic-good" : ""}>{value}</strong></div>;
}

function TimelineItem({ event, last }: { event: (ActiveEvent | HistoryEvent) & { active: boolean; score: number | null }; last: boolean }) {
  return <div className={`timeline-item ${last ? "is-last" : ""}`}><div className="timeline-rail"><span className={event.active ? "rail-active" : ""}/></div><div className="timeline-content"><div className="timeline-event-title"><strong>{event.label}</strong>{event.active && <span className="active-state">ACTIVE NOW</span>}<span className="timeline-duration"><Icon name="clock" size={13}/>{formatDuration(event.duration_seconds)}</span></div><div className="timeline-details"><time dateTime={event.started_at}>{formatDate(event.started_at)}</time><span>Raw score {formatScore(event.score)}</span></div></div></div>;
}

function dateValue(value: string): number {
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

export function formatDate(value: string): string {
  if (!value) return "time unavailable";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit", second: "2-digit" }).format(parsed);
}

export function formatDuration(seconds: number | null): string {
  if (seconds === null || !Number.isFinite(seconds) || seconds < 0) return "—";
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const minutes = Math.floor(seconds / 60);
  const remaining = Math.floor(seconds % 60);
  return `${minutes}m ${String(remaining).padStart(2, "0")}s`;
}

function formatMs(seconds: number | null | undefined): string {
  return seconds === null || seconds === undefined ? "—" : `${Math.round(seconds * 1000)} ms`;
}

function formatCount(count: number | null | undefined, missing = "—"): string {
  return count === null || count === undefined ? missing : count.toLocaleString();
}

function formatScore(score: number | null): string {
  return score === null ? "Unavailable" : score.toFixed(3);
}

function unavailableTitle(reason: string): string {
  if (reason === "audio_samples_stale") return "Waiting for fresh microphone data";
  if (reason === "microphone_not_running") return "Microphone is not running";
  return "Visualization data unavailable";
}

function unavailableDescription(reason: string): string {
  if (reason === "audio_samples_stale") return "The backend marked the last captured window as stale, so it is not displayed as live.";
  if (reason === "microphone_not_running") return "Start the microphone session to request real waveform and spectrogram data.";
  return "The backend has no fresh audio window to visualize yet.";
}
