import { useCallback, useEffect, useState } from 'react';
import {
  fetchEvaluationReport,
  importCheckpointFile,
  runEvaluationSuite,
  startTrainingJob,
  stopTrainingJob,
} from '../api/stanClient';
import type { EvaluationReport, MetricsInfo } from '../types/stan';
import {
  computeLossDomain,
  formatLearningRate,
  formatTokenCount,
  formatTrainingStageLabel,
  formatTrainingStatusLabel,
} from '../utils/stanState';

interface TrainingViewProps {
  metrics: MetricsInfo | null;
  loading: boolean;
  apiBaseUrl: string;
  onRefresh: () => Promise<void>;
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
        className="mt-4 h-auto w-full overflow-visible"
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
                className="fill-slate-500 font-mono text-[10px]"
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

        <text x={padL} y={height - 6} className="fill-slate-500 font-mono text-[10px]">
          step {domain.minStep}
        </text>
        <text
          x={width - padR}
          y={height - 6}
          textAnchor="end"
          className="fill-slate-500 font-mono text-[10px]"
        >
          step {domain.maxStep}
        </text>
      </svg>
    </div>
  );
}

export function TrainingView({ metrics, loading, apiBaseUrl, onRefresh }: TrainingViewProps) {
  const [actionBusy, setActionBusy] = useState(false);
  const [actionMessage, setActionMessage] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [overwriteExisting, setOverwriteExisting] = useState(false);

  const [sourceCheckpointPath, setSourceCheckpointPath] = useState('');
  const [tokenizerSourcePath, setTokenizerSourcePath] = useState('');
  const [importOverwrite, setImportOverwrite] = useState(false);
  const [importBusy, setImportBusy] = useState(false);

  const [evalReport, setEvalReport] = useState<EvaluationReport | null>(null);
  const [evalBusy, setEvalBusy] = useState(false);
  const [evalError, setEvalError] = useState<string | null>(null);

  const loadEvaluation = useCallback(async () => {
    try {
      const report = await fetchEvaluationReport(apiBaseUrl);
      setEvalReport(report);
    } catch {
      // ignore initial fetch error; explicit run will surface details
    }
  }, [apiBaseUrl]);

  useEffect(() => {
    void loadEvaluation();
  }, [loadEvaluation]);

  const activeJobStatus = metrics?.active_job?.status ?? 'idle';
  const jobIsRunning = activeJobStatus === 'running' || activeJobStatus === 'stopping';

  useEffect(() => {
    if (!jobIsRunning) return;
    const timer = setInterval(() => {
      void onRefresh();
    }, 1800);
    return () => clearInterval(timer);
  }, [jobIsRunning, onRefresh]);

  useEffect(() => {
    if (activeJobStatus === 'completed' || activeJobStatus === 'stopped') {
      setActionMessage(null);
    }
  }, [activeJobStatus]);

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
    training_stage: 'untrained',
    current_learning_rate: null,
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
    gpu_info: undefined,
    checkpoint_history: [],
    active_job: { status: 'idle' },
  };

  const hardware = m.gpu_info;
  const checkpoints = m.checkpoint_history ?? [];
  const stageLabel = formatTrainingStageLabel(m.training_stage);

  async function handleStartJob(
    stage: 'smoke_test' | 'pretraining' | 'instruction_tuning',
    resume: boolean,
  ) {
    setActionBusy(true);
    setActionError(null);
    setActionMessage(null);
    try {
      const job = await startTrainingJob(apiBaseUrl, {
        stage,
        resume,
        overwrite_existing: overwriteExisting,
      });
      setActionMessage(`Training job started (${stage}) · Status: ${job.status}`);
      await onRefresh();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Failed to launch training job.');
    } finally {
      setActionBusy(false);
    }
  }

  async function handleStopJob() {
    setActionBusy(true);
    setActionError(null);
    try {
      const job = await stopTrainingJob(apiBaseUrl);
      setActionMessage(`Stop signal dispatched · Status: ${job.status}`);
      await onRefresh();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Failed to stop training job.');
    } finally {
      setActionBusy(false);
    }
  }

  async function handleImportCheckpoint(e: React.FormEvent) {
    e.preventDefault();
    if (!sourceCheckpointPath.trim()) {
      setActionError('Enter a checkpoint file path (.pt) to validate and import.');
      return;
    }
    setImportBusy(true);
    setActionError(null);
    setActionMessage(null);
    try {
      const res = await importCheckpointFile(apiBaseUrl, {
        source_checkpoint: sourceCheckpointPath.trim(),
        tokenizer_source: tokenizerSourcePath.trim() ? tokenizerSourcePath.trim() : null,
        confirm_overwrite: importOverwrite,
      });
      setActionMessage(
        `Checkpoint imported to ${res.target_checkpoint} (step ${res.global_step ?? 0}, stage ${res.training_stage ?? 'unknown'}).`,
      );
      setSourceCheckpointPath('');
      setTokenizerSourcePath('');
      setImportOverwrite(false);
      await onRefresh();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Checkpoint import failed.');
    } finally {
      setImportBusy(false);
    }
  }

  async function handleRunEvaluation() {
    setEvalBusy(true);
    setEvalError(null);
    try {
      const report = await runEvaluationSuite(apiBaseUrl);
      setEvalReport(report);
    } catch (err) {
      setEvalError(err instanceof Error ? err.message : 'Failed to run evaluation suite.');
    } finally {
      setEvalBusy(false);
    }
  }

  return (
    <section aria-labelledby="training-heading" className="space-y-8">
      <div className="border-b border-slate-200 pb-5">
        <h1
          id="training-heading"
          className="font-display text-2xl font-bold tracking-tight text-slate-900"
        >
          Training, Compute & Evaluation Dashboard
        </h1>
        <p className="mt-1.5 text-sm text-slate-600">
          Live telemetry from <span className="font-mono">/metrics</span> and{' '}
          <span className="font-mono">/evaluation</span>. Clearly separates synthetic smoke tests,
          Stage 1 English pretraining, and Stage 2 instruction tuning.
        </p>
      </div>

      {/* Explicit Stage Separation Overview */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <div className="rounded-xl border border-slate-200 bg-white p-5">
          <p className="text-xs font-medium text-slate-500">Verification Only</p>
          <h2 className="mt-1 font-display text-sm font-semibold text-slate-900">
            Smoke Test (Synthetic)
          </h2>
          <p className="mt-1.5 text-xs text-slate-600 leading-relaxed">
            Runs an isolated synthetic loop in temporary storage to verify tensor math and optimizer
            steps. Never enables production chat or overwrites English checkpoints.
          </p>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-5">
          <p className="text-xs font-medium text-slate-500">Stage 1 · Random Initialization</p>
          <h2 className="mt-1 font-display text-sm font-semibold text-slate-900">
            English Pretraining
          </h2>
          <p className="mt-1.5 text-xs text-slate-600 leading-relaxed">
            Next-token prediction over deduplicated, license-verified English corpora using STAN AI&apos;s
            8,192-vocab SentencePiece BPE tokenizer with strict train/validation split.
          </p>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-5">
          <p className="text-xs font-medium text-slate-500">Stage 2 · Requires Stage 1 Checkpoint</p>
          <h2 className="mt-1 font-display text-sm font-semibold text-slate-900">
            Instruction Tuning (SFT)
          </h2>
          <p className="mt-1.5 text-xs text-slate-600 leading-relaxed">
            Supervised dialogue tuning on open-license conversations with assistant-only loss
            masking and personality anchors. Cannot start from random or smoke-test weights.
          </p>
        </div>
      </div>

      {/* Top KPI Row */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-5">
        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <p className="text-xs font-medium text-slate-500">Model & Stage State</p>
          <p className="mt-1.5 font-display text-sm font-bold text-slate-900">
            {formatTrainingStatusLabel(m.training_status)}
          </p>
          <p className="mt-1 text-xs text-slate-600">{stageLabel}</p>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <p className="text-xs font-medium text-slate-500">Training Steps</p>
          <p className="mt-1.5 font-mono text-base font-semibold text-slate-900 tabular-nums">
            {m.global_step.toLocaleString('en-US')} / {m.max_steps.toLocaleString('en-US')}
          </p>
          <p className="mt-1 font-mono text-xs text-slate-500 tabular-nums">
            Progress {m.progress_percent}%
          </p>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <p className="text-xs font-medium text-slate-500">Dataset Token Count</p>
          <p className="mt-1.5 font-mono text-base font-semibold text-slate-900 tabular-nums">
            {formatTokenCount(m.dataset_tokens)}
          </p>
          <p className="mt-1 text-xs text-slate-500">Packed Corpus</p>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <p className="text-xs font-medium text-slate-500">Current Learning Rate</p>
          <p className="mt-1.5 font-mono text-base font-semibold text-slate-900 tabular-nums">
            {formatLearningRate(m.current_learning_rate)}
          </p>
          <p className="mt-1 text-xs text-slate-500">Warmup + Cosine</p>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <p className="text-xs font-medium text-slate-500">Train / Validation Loss</p>
          <p className="mt-1.5 font-mono text-base font-semibold text-slate-900 tabular-nums">
            {m.latest_train_loss !== null ? m.latest_train_loss.toFixed(4) : '—'}
            {' / '}
            {m.latest_val_loss !== null ? m.latest_val_loss.toFixed(4) : '—'}
          </p>
          <p className="mt-1 text-xs text-slate-500">Cross-Entropy</p>
        </div>
      </div>

      {/* Hardware & CUDA Environment */}
      <div className="rounded-xl border border-slate-200 bg-white p-6 space-y-4">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h2 className="font-display text-sm font-semibold text-slate-900">
            Host Hardware & CUDA Compute Inspection
          </h2>
          <span className="font-mono text-xs text-slate-600 tabular-nums">
            PyTorch {hardware?.pytorch_version ?? 'unknown'} · Python {hardware?.python_version ?? 'unknown'}
          </span>
        </div>

        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4 text-xs">
          <div>
            <p className="text-slate-500">CUDA Available</p>
            <p className="mt-0.5 font-mono font-semibold text-slate-900">
              {hardware?.cuda_available ? `Yes (${hardware.gpu_count} GPU)` : 'No (0 GPUs)'}
            </p>
          </div>
          <div>
            <p className="text-slate-500">Active Device</p>
            <p className="mt-0.5 font-mono font-semibold text-slate-900">
              {hardware?.active_device ?? 'cpu'}
            </p>
          </div>
          <div>
            <p className="text-slate-500">CPU Cores</p>
            <p className="mt-0.5 font-mono font-semibold text-slate-900 tabular-nums">
              {hardware?.cpu_cores ?? '—'}
            </p>
          </div>
          <div>
            <p className="text-slate-500">System RAM</p>
            <p className="mt-0.5 font-mono font-semibold text-slate-900 tabular-nums">
              {hardware?.ram_gb ? `${hardware.ram_gb} GiB` : '—'}
            </p>
          </div>
        </div>

        {hardware && hardware.gpus.length > 0 && (
          <div className="border-t border-slate-100 pt-3 text-xs text-slate-700">
            {hardware.gpus.map((gpu) => (
              <div key={gpu.index} className="font-mono">
                GPU {gpu.index}: {gpu.name} ({gpu.vram_gb} GiB VRAM)
              </div>
            ))}
          </div>
        )}

        <p className="text-xs text-slate-600 leading-relaxed border-t border-slate-100 pt-3">
          {hardware?.recommended_training_target ??
            'Use GPU hardware (Google Colab / Kaggle) for full 51.66M English pretraining and import the resulting checkpoint below.'}
        </p>
      </div>

      {/* Training Controls & Progress */}
      <div className="rounded-xl border border-slate-200 bg-white p-6 space-y-5">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h2 className="font-display text-sm font-semibold text-slate-900">
              Training Pipeline Controls
            </h2>
            <p className="mt-1 text-xs text-slate-600">
              Background Worker Status:{' '}
              <span className="font-mono font-semibold text-slate-900">{activeJobStatus}</span>
              {m.active_job?.stage ? ` · Stage: ${m.active_job.stage}` : ''}
              {m.checkpoint_version ? ` · Checkpoint: ${m.checkpoint_version}` : ''}
            </p>
          </div>

          <label className="inline-flex items-center gap-2 text-xs text-slate-700 cursor-pointer">
            <input
              type="checkbox"
              checked={overwriteExisting}
              onChange={(e) => setOverwriteExisting(e.target.checked)}
              className="h-4 w-4 rounded border-slate-300 text-blue-700 focus:ring-blue-700"
            />
            <span>Allow overwriting existing checkpoint</span>
          </label>
        </div>

        <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-100">
          <div
            className="h-full bg-blue-700 transition-all"
            style={{ width: `${Math.min(100, Math.max(0, m.progress_percent))}%` }}
          />
        </div>

        <div className="flex flex-wrap items-center gap-2.5">
          <button
            type="button"
            disabled={actionBusy || jobIsRunning}
            onClick={() => void handleStartJob('smoke_test', false)}
            className="min-h-[40px] rounded-lg border border-slate-300 bg-white px-4 py-2 text-xs font-semibold text-slate-900 transition-colors hover:bg-slate-50 disabled:opacity-50 whitespace-nowrap shrink-0"
          >
            Run Smoke Verification
          </button>

          <button
            type="button"
            disabled={actionBusy || jobIsRunning}
            onClick={() => void handleStartJob('pretraining', false)}
            className="min-h-[40px] rounded-lg bg-blue-700 px-4 py-2 text-xs font-semibold text-white transition-colors hover:bg-blue-800 disabled:opacity-50 whitespace-nowrap shrink-0"
          >
            Start Stage 1 Pretraining
          </button>

          <button
            type="button"
            disabled={actionBusy || jobIsRunning}
            onClick={() => void handleStartJob('pretraining', true)}
            className="min-h-[40px] rounded-lg border border-slate-300 bg-white px-4 py-2 text-xs font-semibold text-slate-900 transition-colors hover:bg-slate-50 disabled:opacity-50 whitespace-nowrap shrink-0"
          >
            Resume Stage 1
          </button>

          <button
            type="button"
            disabled={actionBusy || jobIsRunning}
            onClick={() => void handleStartJob('instruction_tuning', false)}
            className="min-h-[40px] rounded-lg border border-slate-300 bg-white px-4 py-2 text-xs font-semibold text-slate-900 transition-colors hover:bg-slate-50 disabled:opacity-50 whitespace-nowrap shrink-0"
          >
            Start Stage 2 SFT
          </button>

          <button
            type="button"
            disabled={!jobIsRunning || actionBusy}
            onClick={() => void handleStopJob()}
            className="min-h-[40px] rounded-lg border border-red-300 bg-white px-4 py-2 text-xs font-semibold text-red-700 transition-colors hover:bg-red-50 disabled:opacity-40 whitespace-nowrap shrink-0"
          >
            Stop & Save Checkpoint
          </button>
        </div>

        {(actionMessage || actionError || m.active_job?.error) && (
          <div className="space-y-2">
            {actionMessage && (
              <p className="rounded-lg border border-slate-200 bg-slate-50 px-3.5 py-2.5 text-xs text-slate-800">
                {actionMessage}
              </p>
            )}
            {(actionError || m.active_job?.error) && (
              <p
                role="alert"
                className="rounded-lg border border-red-200 bg-red-50 px-3.5 py-2.5 text-xs text-red-800"
              >
                {actionError || m.active_job?.error}
              </p>
            )}
          </div>
        )}
      </div>

      {/* Loss Graphs */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <LossChart title="Training Loss" metricKey="train_loss" strokeColor="#1d4ed8" metrics={m} />
        <LossChart title="Validation Loss" metricKey="val_loss" strokeColor="#0f766e" metrics={m} />
      </div>

      {/* Checkpoint History & External GPU Checkpoint Import */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div className="rounded-xl border border-slate-200 bg-white p-6 space-y-4">
          <div className="flex items-baseline justify-between gap-2">
            <h2 className="font-display text-sm font-semibold text-slate-900">
              Checkpoint History
            </h2>
            <span className="font-mono text-xs text-slate-500 tabular-nums">
              {checkpoints.length} file{checkpoints.length === 1 ? '' : 's'}
            </span>
          </div>

          {checkpoints.length === 0 ? (
            <p className="text-xs text-slate-500 py-4">
              No saved checkpoints in <span className="font-mono">STAN-AI/checkpoints/</span> yet.
            </p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-200 text-slate-500">
                    <th className="pb-2 font-medium">File</th>
                    <th className="pb-2 font-medium">Stage</th>
                    <th className="pb-2 font-medium text-right">Step</th>
                    <th className="pb-2 font-medium text-right">Size</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {checkpoints.map((ckpt) => (
                    <tr key={ckpt.filename} className="text-slate-800">
                      <td className="py-2 font-mono">{ckpt.filename}</td>
                      <td className="py-2 text-slate-600">{ckpt.training_stage ?? 'unknown'}</td>
                      <td className="py-2 text-right font-mono tabular-nums">
                        {ckpt.global_step ?? '—'}
                      </td>
                      <td className="py-2 text-right font-mono tabular-nums">
                        {ckpt.size_mb} MB
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        <form
          onSubmit={(e) => void handleImportCheckpoint(e)}
          className="rounded-xl border border-slate-200 bg-white p-6 space-y-4"
        >
          <div>
            <h2 className="font-display text-sm font-semibold text-slate-900">
              Import GPU-Trained Checkpoint
            </h2>
            <p className="mt-1 text-xs text-slate-600">
              Validate architecture compatibility and import <span className="font-mono">.pt</span>{' '}
              weights trained in Colab or Kaggle into runtime storage.
            </p>
          </div>

          <div className="space-y-3">
            <div>
              <label
                htmlFor="source-ckpt-path"
                className="block text-xs font-medium text-slate-700"
              >
                Source Checkpoint Path (.pt)
              </label>
              <input
                id="source-ckpt-path"
                type="text"
                value={sourceCheckpointPath}
                onChange={(e) => setSourceCheckpointPath(e.target.value)}
                placeholder="/path/to/downloaded/latest.pt"
                className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs font-mono text-slate-900 focus:border-blue-700 focus:outline-none"
              />
            </div>

            <div>
              <label
                htmlFor="source-tok-path"
                className="block text-xs font-medium text-slate-700"
              >
                Tokenizer Model Path (optional tokenizer.model)
              </label>
              <input
                id="source-tok-path"
                type="text"
                value={tokenizerSourcePath}
                onChange={(e) => setTokenizerSourcePath(e.target.value)}
                placeholder="/path/to/downloaded/tokenizer.model"
                className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs font-mono text-slate-900 focus:border-blue-700 focus:outline-none"
              />
            </div>

            <label className="inline-flex items-center gap-2 text-xs text-slate-700 cursor-pointer">
              <input
                type="checkbox"
                checked={importOverwrite}
                onChange={(e) => setImportOverwrite(e.target.checked)}
                className="h-4 w-4 rounded border-slate-300 text-blue-700 focus:ring-blue-700"
              />
              <span>Confirm overwrite of existing active checkpoint</span>
            </label>
          </div>

          <button
            type="submit"
            disabled={importBusy}
            className="min-h-[40px] rounded-lg bg-slate-900 px-4 py-2 text-xs font-semibold text-white transition-colors hover:bg-slate-800 disabled:opacity-50 whitespace-nowrap shrink-0"
          >
            {importBusy ? 'Validating & Importing...' : 'Validate & Import Checkpoint'}
          </button>
        </form>
      </div>

      {/* 9-Dimension Evaluation Suite */}
      <div className="rounded-xl border border-slate-200 bg-white p-6 space-y-5">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h2 className="font-display text-sm font-semibold text-slate-900">
              9-Dimension Measured Evaluation Suite
            </h2>
            <p className="mt-1 text-xs text-slate-600">
              Evaluates real Transformer outputs alongside memory, contradiction, personality, and
              uncertainty judgment. Never fabricates neural scores for untrained weights.
            </p>
          </div>

          <button
            type="button"
            disabled={evalBusy}
            onClick={() => void handleRunEvaluation()}
            className="min-h-[40px] rounded-lg bg-blue-700 px-4 py-2 text-xs font-semibold text-white transition-colors hover:bg-blue-800 disabled:opacity-50 whitespace-nowrap shrink-0"
          >
            {evalBusy ? 'Evaluating...' : 'Run Evaluation'}
          </button>
        </div>

        {evalError && (
          <p
            role="alert"
            className="rounded-lg border border-red-200 bg-red-50 px-3.5 py-2.5 text-xs text-red-800"
          >
            {evalError}
          </p>
        )}

        {!evalReport || evalReport.categories.length === 0 ? (
          <div className="rounded-lg border border-dashed border-slate-200 bg-slate-50 px-4 py-6 text-center text-xs text-slate-500">
            No evaluation report saved yet. Click Run Evaluation to measure current cognitive and
            neural capabilities.
          </div>
        ) : (
          <div className="space-y-4">
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-600">
              <span>
                Evaluated:{' '}
                <strong className="font-mono text-slate-900 tabular-nums">
                  {evalReport.evaluated_category_count}/{evalReport.total_category_count}
                </strong>{' '}
                dimensions
              </span>
              <span aria-hidden="true">·</span>
              <span>
                Average Score:{' '}
                <strong className="font-mono text-slate-900 tabular-nums">
                  {evalReport.average_evaluated_score !== null
                    ? `${(evalReport.average_evaluated_score * 100).toFixed(1)}%`
                    : '—'}
                </strong>
              </span>
              <span aria-hidden="true">·</span>
              <span>
                Neural Weights Evaluated:{' '}
                <strong className="font-mono text-slate-900">
                  {evalReport.neural_weights_evaluated ? 'Yes' : 'No (Untrained)'}
                </strong>
              </span>
              {evalReport.evaluated_at && (
                <>
                  <span aria-hidden="true">·</span>
                  <span className="font-mono">
                    {new Date(evalReport.evaluated_at).toLocaleString()}
                  </span>
                </>
              )}
            </div>

            <div className="divide-y divide-slate-100 border-t border-slate-200">
              <div className="hidden md:grid md:grid-cols-[minmax(170px,1.1fr)_minmax(170px,1.1fr)_72px_2fr] md:gap-4 py-2.5 text-xs font-medium text-slate-500">
                <div>Dimension</div>
                <div>Measurement State</div>
                <div className="text-right">Score</div>
                <div>Measured Evidence</div>
              </div>
              {evalReport.categories.map((cat) => (
                <div
                  key={cat.id}
                  className="py-3 space-y-1.5 md:space-y-0 md:grid md:grid-cols-[minmax(170px,1.1fr)_minmax(170px,1.1fr)_72px_2fr] md:items-start md:gap-4 text-xs"
                >
                  <div className="flex items-baseline justify-between gap-2 md:block">
                    <span className="font-semibold text-slate-900">{cat.label}</span>
                    <span className="font-mono font-semibold text-slate-900 tabular-nums md:hidden">
                      {cat.score !== null ? `${(cat.score * 100).toFixed(1)}%` : '—'}
                    </span>
                  </div>
                  <div
                    className={
                      cat.status === 'evaluated'
                        ? 'font-mono text-emerald-700'
                        : 'font-mono text-amber-700'
                    }
                  >
                    {cat.status === 'evaluated' ? 'Evaluated' : 'Not Evaluated (Untrained)'}
                  </div>
                  <div className="hidden md:block text-right font-mono text-slate-900 tabular-nums">
                    {cat.score !== null ? `${(cat.score * 100).toFixed(1)}%` : '—'}
                  </div>
                  <div className="text-slate-600 leading-relaxed break-words">
                    {cat.measured_details}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Colab & Kaggle Full GPU Training Instructions */}
      <div className="rounded-xl border border-slate-200 bg-white p-6 space-y-3">
        <h2 className="font-display text-sm font-semibold text-slate-900">
          Reproducible GPU Training Workflow (Colab & Kaggle)
        </h2>
        <p className="text-xs text-slate-600 leading-relaxed">
          Open{' '}
          <span className="font-mono">STAN-AI/notebooks/STAN_AI_Colab_Kaggle_Training.ipynb</span>{' '}
          on a CUDA GPU runtime or execute these scripts in sequence:
        </p>
        <pre className="overflow-x-auto rounded-lg border border-slate-200 bg-slate-50 p-4 font-mono text-xs text-slate-800 leading-relaxed">
{`# Stage 1: Prepare English corpus (verified public-domain sources), train 8192 BPE tokenizer & pack tokens
python STAN-AI/scripts/prepare_corpus.py --sources-limit 10 --vocab-size 8192
python STAN-AI/scripts/kaggle_train.py --resume auto

# Stage 2: Prepare OpenAssistant/oasst1 instruction data & run SFT from Stage 1 checkpoint
python STAN-AI/scripts/prepare_sft_corpus.py --max-examples 8000
python STAN-AI/scripts/kaggle_train.py --stage instruction_tuning --resume auto

# Validate & import trained checkpoint cleanly back into STAN-AI/checkpoints/
python STAN-AI/scripts/import_checkpoint.py --source /path/to/latest.pt --tokenizer /path/to/tokenizer.model`}
        </pre>
      </div>
    </section>
  );
}
