import { useState, type FormEvent } from 'react';
import { tokenizeText } from '../api/stanClient';
import type { ModelInfo, TokenizeResult } from '../types/stan';
import { formatParamCount, formatTrainingStatusLabel } from '../utils/stanState';

interface ModelInfoViewProps {
  modelInfo: ModelInfo | null;
  loading: boolean;
  apiBaseUrl: string;
}

export function ModelInfoView({ modelInfo, loading, apiBaseUrl }: ModelInfoViewProps) {
  const [sampleText, setSampleText] = useState(
    'STAN AI is an independent 50M-parameter decoder-only Transformer.',
  );
  const [tokResult, setTokResult] = useState<TokenizeResult | null>(null);
  const [tokError, setTokError] = useState<string | null>(null);
  const [tokBusy, setTokBusy] = useState(false);

  async function handleTokenize(e: FormEvent) {
    e.preventDefault();
    if (!sampleText.trim()) return;
    setTokBusy(true);
    setTokError(null);
    try {
      const res = await tokenizeText(apiBaseUrl, sampleText);
      setTokResult(res);
    } catch (err) {
      setTokResult(null);
      setTokError(err instanceof Error ? err.message : 'Tokenization request failed.');
    } finally {
      setTokBusy(false);
    }
  }

  if (loading && !modelInfo) {
    return (
      <section className="space-y-4">
        <div className="h-8 w-64 animate-pulse rounded bg-slate-200" />
        <div className="h-64 animate-pulse rounded-xl border border-slate-200 bg-white" />
      </section>
    );
  }

  if (!modelInfo) {
    return (
      <section className="rounded-xl border border-slate-200 bg-white p-8 text-center">
        <h1 className="font-display text-xl font-bold text-slate-900">Model Information Unavailable</h1>
        <p className="mt-2 text-sm text-slate-600">
          Connect to the Python FastAPI inference service to inspect STAN AI&apos;s architecture and parameter count.
        </p>
      </section>
    );
  }

  const specRows: Array<{ label: string; value: string }> = [
    { label: 'Model Identity', value: `${modelInfo.model_name} v${modelInfo.version}` },
    { label: 'Architecture', value: 'Decoder-only Transformer (Pre-Norm Causal)' },
    { label: 'Trainable Parameters', value: formatParamCount(modelInfo.parameter_count) },
    { label: 'Transformer Blocks', value: String(modelInfo.n_layers) },
    { label: 'Hidden Dimension', value: String(modelInfo.d_model) },
    { label: 'Attention Heads', value: `${modelInfo.n_heads} (head_dim = ${modelInfo.d_head})` },
    { label: 'Feed-Forward Dimension', value: String(modelInfo.d_ff) },
    { label: 'Activation & Normalization', value: `${modelInfo.activation.toUpperCase()} / ${modelInfo.norm_type.toUpperCase()}` },
    { label: 'Context Length', value: `${modelInfo.max_seq_len} tokens` },
    { label: 'Vocabulary Size', value: `${modelInfo.vocab_size.toLocaleString('en-US')} (Custom SentencePiece BPE)` },
    { label: 'Weight Tying', value: modelInfo.tie_embeddings ? 'Token embedding tied to LM head' : 'Untied' },
    { label: 'Training Status', value: formatTrainingStatusLabel(modelInfo.training_status) },
    { label: 'Checkpoint Version', value: modelInfo.checkpoint_version ?? 'None loaded' },
    { label: 'Tokenizer Loaded', value: modelInfo.tokenizer_loaded ? 'Yes (tokenizer.model)' : 'Not trained yet' },
  ];

  const breakdown = modelInfo.parameter_breakdown;

  return (
    <section aria-labelledby="model-info-heading" className="space-y-8">
      <div className="border-b border-slate-200 pb-5">
        <h1 id="model-info-heading" className="font-display text-2xl font-bold tracking-tight text-slate-900">
          Model Architecture &amp; Parameters
        </h1>
        <p className="mt-1.5 text-sm text-slate-600">
          Live metadata reported directly by STAN AI&apos;s PyTorch runtime on{' '}
          <span className="font-mono text-slate-900">{modelInfo.device}</span>.
        </p>
      </div>

      {/* Specification & Parameter Breakdown Grid */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2 rounded-xl border border-slate-200 bg-white overflow-hidden">
          <div className="border-b border-slate-200 px-5 py-3.5">
            <h2 className="font-display text-sm font-semibold text-slate-900">
              Architecture Specification
            </h2>
          </div>
          <dl className="divide-y divide-slate-100">
            {specRows.map((row) => (
              <div
                key={row.label}
                className="flex flex-col gap-1 px-5 py-3 sm:flex-row sm:items-center sm:justify-between"
              >
                <dt className="text-xs font-medium text-slate-500">{row.label}</dt>
                <dd className="text-sm font-medium text-slate-900 font-mono tabular-nums">
                  {row.value}
                </dd>
              </div>
            ))}
          </dl>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white overflow-hidden self-start">
          <div className="border-b border-slate-200 px-5 py-3.5">
            <h2 className="font-display text-sm font-semibold text-slate-900">
              Parameter Breakdown
            </h2>
          </div>
          <dl className="divide-y divide-slate-100">
            <div className="flex items-center justify-between px-5 py-3">
              <dt className="text-xs text-slate-500">Token Embeddings</dt>
              <dd className="text-xs font-mono text-slate-900 tabular-nums">
                {breakdown.token_embedding_parameters.toLocaleString('en-US')}
              </dd>
            </div>
            <div className="flex items-center justify-between px-5 py-3">
              <dt className="text-xs text-slate-500">Positional Embeddings</dt>
              <dd className="text-xs font-mono text-slate-900 tabular-nums">
                {breakdown.positional_embedding_parameters.toLocaleString('en-US')}
              </dd>
            </div>
            <div className="flex items-center justify-between px-5 py-3">
              <dt className="text-xs text-slate-500">{modelInfo.n_layers} Transformer Blocks</dt>
              <dd className="text-xs font-mono text-slate-900 tabular-nums">
                {breakdown.transformer_blocks_parameters.toLocaleString('en-US')}
              </dd>
            </div>
            <div className="flex items-center justify-between px-5 py-3">
              <dt className="text-xs text-slate-500">Final RMSNorm</dt>
              <dd className="text-xs font-mono text-slate-900 tabular-nums">
                {breakdown.final_norm_parameters.toLocaleString('en-US')}
              </dd>
            </div>
            <div className="flex items-center justify-between px-5 py-3 bg-slate-50">
              <dt className="text-xs font-semibold text-slate-900">Total Trainable</dt>
              <dd className="text-xs font-mono font-semibold text-slate-900 tabular-nums">
                {breakdown.trainable_parameters.toLocaleString('en-US')}
              </dd>
            </div>
          </dl>
        </div>
      </div>

      {/* Live Tokenizer Diagnostic */}
      <div className="rounded-xl border border-slate-200 bg-white p-6 space-y-4">
        <div>
          <h2 className="font-display text-base font-semibold text-slate-900">
            Tokenizer Round-Trip Diagnostic
          </h2>
          <p className="mt-1 text-xs text-slate-600">
            Sends text to <span className="font-mono">POST /tokenize</span> on the Python FastAPI service to inspect SentencePiece BPE pieces and token IDs.
          </p>
        </div>

        <form onSubmit={handleTokenize} className="flex flex-col gap-3 sm:flex-row">
          <label htmlFor="tokenizer-test-input" className="sr-only">
            Text to tokenize
          </label>
          <input
            id="tokenizer-test-input"
            type="text"
            value={sampleText}
            onChange={(e) => setSampleText(e.target.value)}
            className="w-full rounded-lg border border-slate-300 px-3.5 py-2 text-sm text-slate-900 focus:border-blue-700 focus:outline-none"
          />
          <button
            type="submit"
            disabled={tokBusy || !sampleText.trim()}
            className="min-h-[40px] rounded-lg bg-slate-900 px-5 py-2 text-xs font-semibold text-white hover:bg-slate-800 disabled:opacity-50 whitespace-nowrap shrink-0"
          >
            {tokBusy ? 'Tokenizing...' : 'Run /tokenize'}
          </button>
        </form>

        {tokError && (
          <p role="alert" className="text-xs font-medium text-amber-800 bg-amber-50 border border-amber-200 rounded-lg px-4 py-3">
            {tokError}
          </p>
        )}

        {tokResult && (
          <div className="space-y-3 border-t border-slate-100 pt-4 text-xs">
            <div className="flex flex-wrap gap-x-4 gap-y-1 font-mono text-slate-700 tabular-nums">
              <span>tokens={tokResult.token_count}</span>
              <span>vocab_size={tokResult.vocab_size}</span>
              <span>roundtrip_exact={String(tokResult.roundtrip_exact)}</span>
            </div>
            <div>
              <p className="font-semibold text-slate-700 mb-1">Token IDs</p>
              <p className="font-mono text-slate-800 bg-slate-50 border border-slate-200 rounded-lg p-3 break-all">
                [{tokResult.token_ids.join(', ')}]
              </p>
            </div>
            <div>
              <p className="font-semibold text-slate-700 mb-1">BPE Pieces</p>
              <p className="font-mono text-slate-800 bg-slate-50 border border-slate-200 rounded-lg p-3 break-all">
                {tokResult.pieces.join(' | ')}
              </p>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
