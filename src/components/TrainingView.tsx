import type { MetricsInfo } from '../types/stan';
import { computeLossDomain, formatTokenCount, formatTrainingStatusLabel } from '../utils/stanState';

interface TrainingViewProps {
  metrics: MetricsInfo | null;
  loading: boolean;
}

function LossChart({
  title,
  metricKey,
  strokeColor,
  metrics,
}: {
  title: string;
  metricKey: 'train_loss' | 'val_loss';
  strokeColor: string;
  metrics: MetricsInfo;
}) {
  const domain = computeLossDomain(metrics.history, metricKey);

  if (domain.points.length === 0) {
    return (
      <div className="rounded-xl border border-slate-200 bg-white p-6">
        <h2 className="font-display text-sm font-semibold text-slate-900">{title}</h2>
        <div className="mt-4 flex h-48 items-center justify-center rounded-lg border border-dashed border-slate-200 bg-slate-50 px-4 text-center">
          <p className="text-xs text-slate-500">
            No {metricKey === 'train_loss' ? 'training' : 'validation'} loss checkpoints recorded yet.
          </p>
        </div>
      </div>
    );
  }

  const width = 560;
  const height = 220;
  const padL = 48;
  const padR = 16;
  const padT = 16;
  const padB = 30;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;

  const toX = (step: number) =>
    padL + ((step - domain.minStep) / (domain.maxStep - domain.minStep)) * plotW;
  const toY = (val: number) =>
    padT + plotH - ((val - domain.minVal) / (domain.maxVal - domain.minVal)) * plotH;

  const polyline = domain.points
    .map((pt) => `${toX(pt.step).toFixed(1)},${toY(pt.value).toFixed(1)}`)
    .join(' ');

  const yTicks = [domain.minVal, (domain.minVal + domain.maxVal) / 2, domain.maxVal];

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-6">
      <div className="flex items-baseline justify-between gap-4">
        <h2 className="font-display text-sm font-semibold text-slate-900">{title}</h2>
        <span className="font-mono text-xs text-slate-600 tabular-nums">
          Latest: {domain.points[domain.points.length - 1].value.toFixed(4)}
        </span>
      </div>

      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="mt-4 w-full h-auto overflow-visible"
        role="img"
        aria-label={`${title} curve across training steps`}
      >
        {yTicks.map((tick, idx) => {
          const y = toY(tick);
          return (
            <g key={idx}>
              <line x1={padL} y1={y} x2={width - padR} y2={y} stroke="#e2e8f0" strokeWidth="1" />
              <text
                x={padL - 8}
                y={y + 4}
                textAnchor="end"
                className="fill-slate-500 text-[10px] font-mono"
              >
                {tick.toFixed(2)}
              </text>
            </g>
          );
        })}

        <polyline
          fill="none"
          stroke={strokeColor}
          strokeWidth="2.2"
          strokeLinejoin="round"
          strokeLinecap="round"
          points={polyline}
        />

        {domain.points.length <= 40 &&
          domain.points.map((pt) => (
            <circle
              key={pt.step}
              cx={toX(pt.step)}
              cy={toY(pt.value)}
              r="3"
              fill={strokeColor}
            />
          ))}

        <text x={padL} y={height - 6} className="fill-slate-500 text-[10px] font-mono">
          step {domain.minStep}
        </text>
        <text
          x={width - padR}
          y={height - 6}
          textAnchor="end"
          className="fill-slate-500 text-[10px] font-mono"
        >
          step {domain.maxStep}
        </text>
      </svg>
    </div>
  );
}

export function TrainingView({ metrics, loading }: TrainingViewProps) {
  if (loading && !metrics) {
    return (
      <section className="space-y-4">
        <div className="h-8 w-64 animate-pulse rounded bg-slate-200" />
        <div className="h-64 animate-pulse rounded-xl border border-slate-200 bg-white" />
      </section>
    );
  }

  const m: MetricsInfo = metrics ?? {
    training_status: 'untrained',
    global_step: 0,
    max_steps: 0,
    progress_percent: 0,
    dataset_tokens: 0,
    latest_train_loss: null,
    latest_val_loss: null,
    checkpoint_version: null,
    checkpoint_path: null,
    saved_at: null,
    history: [],
    warning: null,
  };

  return (
    <section aria-labelledby="training-heading" className="space-y-8">
      <div className="border-b border-slate-200 pb-5">
        <h1 id="training-heading" className="font-display text-2xl font-bold tracking-tight text-slate-900">
          Training Dashboard
        </h1>
        <p className="mt-1.5 text-sm text-slate-600">
          Real training telemetry read from <span className="font-mono">GET /metrics</span> and checkpoint history. Zero simulated metrics.
        </p>
      </div>

      {/* Top KPI Row */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-5">
        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <p className="text-xs font-medium text-slate-500">Training Status</p>
          <p className="mt-1.5 font-display text-base font-bold text-slate-900">
            {formatTrainingStatusLabel(m.training_status)}
          </p>
        </div>
        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <p className="text-xs font-medium text-slate-500">Training Steps</p>
          <p className="mt-1.5 font-mono text-base font-semibold text-slate-900 tabular-nums">
            {m.global_step.toLocaleString('en-US')} / {m.max_steps.toLocaleString('en-US')}
          </p>
        </div>
        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <p className="text-xs font-medium text-slate-500">Dataset Token Count</p>
          <p className="mt-1.5 font-mono text-base font-semibold text-slate-900 tabular-nums">
            {formatTokenCount(m.dataset_tokens)}
          </p>
        </div>
        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <p className="text-xs font-medium text-slate-500">Latest Train Loss</p>
          <p className="mt-1.5 font-mono text-base font-semibold text-slate-900 tabular-nums">
            {m.latest_train_loss !== null ? m.latest_train_loss.toFixed(4) : '—'}
          </p>
        </div>
        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <p className="text-xs font-medium text-slate-500">Latest Val Loss</p>
          <p className="mt-1.5 font-mono text-base font-semibold text-slate-900 tabular-nums">
            {m.latest_val_loss !== null ? m.latest_val_loss.toFixed(4) : '—'}
          </p>
        </div>
      </div>

      {/* Progress & Latest Checkpoint */}
      <div className="rounded-xl border border-slate-200 bg-white p-6 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
          <span className="font-semibold text-slate-800">Training Progress</span>
          <span className="font-mono text-slate-700 tabular-nums">{m.progress_percent}%</span>
        </div>
        <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-100">
          <div
            className="h-full bg-blue-700 transition-all"
            style={{ width: `${Math.min(100, Math.max(0, m.progress_percent))}%` }}
          />
        </div>
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 pt-1 text-xs text-slate-600">
          <span>
            Latest Checkpoint:{' '}
            <span className="font-mono text-slate-900">{m.checkpoint_version ?? 'None'}</span>
          </span>
          {m.saved_at && (
            <>
              <span aria-hidden="true">·</span>
              <span className="font-mono">Saved {new Date(m.saved_at).toLocaleString()}</span>
            </>
          )}
        </div>
      </div>

      {/* Loss Graphs */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <LossChart title="Training Loss" metricKey="train_loss" strokeColor="#1d4ed8" metrics={m} />
        <LossChart title="Validation Loss" metricKey="val_loss" strokeColor="#0f766e" metrics={m} />
      </div>

      {/* CLI Guide when Untrained */}
      {m.history.length === 0 && (
        <div className="rounded-xl border border-slate-200 bg-white p-6 space-y-3">
          <h2 className="font-display text-sm font-semibold text-slate-900">
            How to Train STAN AI from Random Initialization
          </h2>
          <p className="text-xs text-slate-600 leading-relaxed">
            Run the corpus preparation and Kaggle/GPU training scripts inside <span className="font-mono">STAN-AI/</span>. Checkpoints save automatically and resume seamlessly after session restarts.
          </p>
          <pre className="overflow-x-auto rounded-lg border border-slate-200 bg-slate-50 p-4 text-xs font-mono text-slate-800 leading-relaxed">
{`# 1. Download verified public-domain corpora, clean, deduplicate, train BPE tokenizer, and pack tokens
python STAN-AI/scripts/prepare_corpus.py --sources-limit 5 --vocab-size 8192

# 2. Train STAN AI (~51.66M parameters) with automatic checkpoint resume
python STAN-AI/scripts/kaggle_train.py --resume auto`}
          </pre>
        </div>
      )}
    </section>
  );
}
