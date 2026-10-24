import { useEffect, useRef } from "react";
import type { VisualizationAvailable } from "./api";

export function AudioVisualization({ data }: { data: VisualizationAvailable }) {
  const spectrogramRef = useSpectrogramCanvas(data);
  const levelPercent = data.rms_dbfs === null || !Number.isFinite(data.rms_dbfs)
    ? 0
    : Math.max(0, Math.min(100, ((data.rms_dbfs + 60) / 60) * 100));
  const dbfsText = data.rms_dbfs === null
    ? "Unavailable"
    : data.rms_dbfs === Number.NEGATIVE_INFINITY ? "−∞ dBFS" : `${data.rms_dbfs.toFixed(1)} dBFS`;
  const rmsText = data.rms === null ? "Unavailable" : data.rms.toFixed(4);

  return (
    <div className="audio-visualization">
      <div className="signal-level" aria-label="Digital audio level">
        <div className="level-copy">
          <span className="visual-label">DIGITAL SIGNAL LEVEL <span>· DISPLAY SCALE</span></span>
          <strong>{dbfsText}</strong>
          <small>RMS {rmsText} · not dB(A) or calibrated SPL</small>
        </div>
        <div className="level-track" role="meter" aria-label="Digital signal display scale" aria-valuemin={-60} aria-valuemax={0} aria-valuenow={data.rms_dbfs !== null && Number.isFinite(data.rms_dbfs) ? Math.max(-60, Math.min(0, data.rms_dbfs)) : undefined}>
          <span style={{ width: `${levelPercent}%` }} />
        </div>
      </div>

      <section className="wave-view" aria-labelledby="wave-heading">
        <div className="visual-subheading"><h3 id="wave-heading">Waveform</h3><span>MIN / MAX · ±{waveformScale(data).toFixed(1)} FS</span></div>
        <svg className="wave-canvas" viewBox="0 0 160 80" preserveAspectRatio="none" role="img" aria-label="Recent microphone waveform min and max envelope, oldest samples at left and newest at right">
          <line x1="0" x2="160" y1="40" y2="40" className="wave-zero" />
          <path d={waveformPath(data)} className="wave-envelope" />
        </svg>
        <div className="time-axis"><span>Oldest</span><span>{data.duration_seconds.toFixed(2)} s window</span><span>Newest</span></div>
      </section>

      <section className="spectrum-view" aria-labelledby="spectrum-heading">
        <div className="visual-subheading"><h3 id="spectrum-heading">Spectrogram</h3><span>LOG POWER · dBFS</span></div>
        <div className="spectrum-chart">
          <div className="frequency-axis" aria-hidden="true"><span>{formatFrequency(data.spectrogram.frequency_max_hz)}</span><span>{formatFrequency(data.spectrogram.frequency_min_hz)}</span></div>
          <canvas
            className="spectrum-canvas"
            ref={spectrogramRef}
            role="img"
            aria-label={`Microphone spectrogram, ${data.spectrogram.frames.length} oldest-to-newest time frames and ${data.spectrogram.frames[0]?.length ?? 0} frequency bins from ${formatFrequency(data.spectrogram.frequency_min_hz)} to ${formatFrequency(data.spectrogram.frequency_max_hz)}. Log power in dBFS; this view does not identify sounds.`}
          />
        </div>
        <div className="time-axis spectrum-time-axis"><span>Older</span><span>Time · {data.duration_seconds.toFixed(2)} s</span><span>Newer</span></div>
        <p className="spectrum-note">Energy by time and frequency; it does not identify a sound class.</p>
      </section>
      <div className="capture-meta">{data.sample_rate.toLocaleString()} Hz · {data.sample_count.toLocaleString()} samples · {data.captured_at ? `captured ${new Date(data.captured_at).toLocaleTimeString()}` : "capture time unavailable"}</div>
    </div>
  );
}

function useSpectrogramCanvas(data: VisualizationAvailable) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) return;

    const draw = () => {
      const bounds = canvas.getBoundingClientRect();
      if (bounds.width <= 0 || bounds.height <= 0) return;
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(bounds.width * ratio);
      canvas.height = Math.round(bounds.height * ratio);
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.clearRect(0, 0, bounds.width, bounds.height);
      const frameCount = data.spectrogram.frames.length;
      const binCount = data.spectrogram.frames[0]?.length ?? 0;
      if (!frameCount || !binCount) return;
      const cellWidth = bounds.width / frameCount;
      const cellHeight = bounds.height / binCount;
      // Backend frames are time-major, oldest first; within each frame bins go low-to-high Hz.
      // Invert y only for display, putting low frequencies at the bottom of the canvas.
      data.spectrogram.frames.forEach((frame, timeIndex) => {
        frame.forEach((dbfs, frequencyIndex) => {
          context.fillStyle = spectrumColor(dbfs);
          context.fillRect(
            timeIndex * cellWidth,
            bounds.height - (frequencyIndex + 1) * cellHeight,
            cellWidth + 0.5,
            cellHeight + 0.5,
          );
        });
      });
    };

    draw();
    const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(draw);
    observer?.observe(canvas);
    window.addEventListener("resize", draw);
    return () => {
      observer?.disconnect();
      window.removeEventListener("resize", draw);
    };
  }, [data.spectrogram]);
  return ref;
}

function waveformPath(data: VisualizationAvailable): string {
  const points = data.waveform.points;
  if (!points.length) return "";
  const scale = waveformScale(data);
  const y = (value: number) => 40 - (value / scale) * 37;
  const upper = points.map((point, index) => `${index},${y(point.max)}`);
  const lower = points.map((point, index) => `${index},${y(point.min)}`).reverse();
  return `M${upper.join(" L")} L${lower.join(" L")} Z`;
}

function waveformScale(data: VisualizationAvailable): number {
  return Math.max(1, ...data.waveform.points.map((point) => Math.abs(point.min)),
    ...data.waveform.points.map((point) => Math.abs(point.max)));
}

function spectrumColor(dbfs: number): string {
  const value = Math.max(0, Math.min(1, (dbfs + 120) / 120));
  let start: number[];
  let end: number[];
  let t: number;
  if (value < 0.65) {
    start = [12, 22, 18];
    end = [33, 112, 73];
    t = value / 0.65;
  } else {
    start = [33, 112, 73];
    end = [117, 238, 177];
    t = (value - 0.65) / 0.35;
  }
  const rgb = start.map((channel, index) => Math.round(channel + (end[index] - channel) * t));
  return `rgb(${rgb[0]} ${rgb[1]} ${rgb[2]})`;
}

function formatFrequency(value: number): string {
  return value >= 1000 ? `${(value / 1000).toFixed(value % 1000 === 0 ? 0 : 2)} kHz` : `${Math.round(value)} Hz`;
}
