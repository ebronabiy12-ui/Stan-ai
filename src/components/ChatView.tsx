import { useState, type FormEvent } from 'react';
import { AlertTriangle, Send, SlidersHorizontal, Square, Trash2 } from 'lucide-react';
import { streamGenerate } from '../api/stanClient';
import type { ChatMessage, GenerationSettings, HealthInfo, ModelInfo } from '../types/stan';
import { formatParamCount, formatTrainingStatusLabel, isProductionChatReady } from '../utils/stanState';

interface ChatViewProps {
  health: HealthInfo | null;
  modelInfo: ModelInfo | null;
  settings: GenerationSettings;
  onUpdateSettings: (patch: Partial<GenerationSettings>) => void;
  onNavigateTab: (tab: 'model' | 'training' | 'settings') => void;
  backendError: string | null;
}

export function ChatView({
  health,
  modelInfo,
  settings,
  onUpdateSettings,
  onNavigateTab,
  backendError,
}: ChatViewProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [prompt, setPrompt] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [showQuickSettings, setShowQuickSettings] = useState(false);
  const [abortCtrl, setAbortCtrl] = useState<AbortController | null>(null);

  const chatReady = health ? isProductionChatReady(health) : false;
  const statusLabel = formatTrainingStatusLabel(health?.training_status || 'untrained');

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const trimmed = prompt.trim();
    if (!trimmed || !chatReady || isGenerating) return;

    setErrorMsg(null);
    const userMsg: ChatMessage = {
      id: `u_${Date.now()}`,
      role: 'user',
      content: trimmed,
      createdAt: new Date().toLocaleTimeString(),
    };
    const assistantId = `a_${Date.now()}`;
    const placeholderAssistant: ChatMessage = {
      id: assistantId,
      role: 'assistant',
      content: '',
      createdAt: new Date().toLocaleTimeString(),
    };

    setMessages((prev) => [...prev, userMsg, placeholderAssistant]);
    setPrompt('');
    setIsGenerating(true);

    const controller = new AbortController();
    setAbortCtrl(controller);

    try {
      await streamGenerate(
        settings.apiBaseUrl,
        trimmed,
        settings,
        (chunk) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? {
                    ...m,
                    content: chunk.generated_text,
                    meta:
                      chunk.event === 'done'
                        ? {
                            prompt_tokens: chunk.prompt_tokens,
                            generated_tokens: chunk.generated_tokens,
                            elapsed_ms: chunk.elapsed_ms,
                            tokens_per_second: chunk.tokens_per_second,
                            stopped_reason: chunk.stopped_reason,
                            checkpoint_version: modelInfo?.checkpoint_version ?? null,
                          }
                        : {
                            generated_tokens: chunk.generated_tokens,
                            elapsed_ms: chunk.elapsed_ms,
                          },
                  }
                : m,
            ),
          );
        },
        controller.signal,
      );
    } catch (err) {
      const detail = err instanceof Error ? err.message : 'Generation request failed.';
      if ((err as { name?: string })?.name !== 'AbortError') {
        setErrorMsg(detail);
        setMessages((prev) => prev.filter((m) => m.id !== assistantId || m.content.length > 0));
      }
    } finally {
      setIsGenerating(false);
      setAbortCtrl(null);
    }
  }

  function handleStop() {
    abortCtrl?.abort();
    setIsGenerating(false);
    setAbortCtrl(null);
  }

  function handleClear() {
    setMessages([]);
    setErrorMsg(null);
  }

  return (
    <section aria-labelledby="chat-heading" className="space-y-6">
      {/* Page Header & Unboxed Metadata */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between border-b border-slate-200 pb-5">
        <div>
          <h1 id="chat-heading" className="font-display text-2xl font-bold tracking-tight text-slate-900">
            STAN AI Inference Workspace
          </h1>
          <div className="mt-1.5 flex flex-wrap items-center gap-x-2.5 gap-y-1 text-xs text-slate-600 tabular-nums">
            <span className="font-medium text-slate-800">{statusLabel}</span>
            <span aria-hidden="true">·</span>
            <span>
              {modelInfo ? formatParamCount(modelInfo.parameter_count) : 'Architecture pending'}
            </span>
            <span aria-hidden="true">·</span>
            <span>Device: {health?.device ?? 'unknown'}</span>
            {modelInfo?.checkpoint_version && (
              <>
                <span aria-hidden="true">·</span>
                <span className="font-mono">{modelInfo.checkpoint_version}</span>
              </>
            )}
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <button
            type="button"
            onClick={() => setShowQuickSettings((v) => !v)}
            aria-expanded={showQuickSettings}
            className="inline-flex min-h-[40px] items-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-2 text-xs font-semibold text-slate-800 hover:bg-slate-50 transition-colors whitespace-nowrap shrink-0"
          >
            <SlidersHorizontal className="h-3.5 w-3.5" aria-hidden="true" />
            <span>Generation Settings</span>
          </button>
          <button
            type="button"
            onClick={handleClear}
            disabled={messages.length === 0 || isGenerating}
            className="inline-flex min-h-[40px] items-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-2 text-xs font-semibold text-slate-800 hover:bg-slate-50 disabled:opacity-50 transition-colors whitespace-nowrap shrink-0"
          >
            <Trash2 className="h-3.5 w-3.5" aria-hidden="true" />
            <span>Clear Conversation</span>
          </button>
        </div>
      </div>

      {/* Quick Generation Settings Drawer */}
      {showQuickSettings && (
        <div className="rounded-xl border border-slate-200 bg-white p-5">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <div>
              <label htmlFor="quick-temp" className="block text-xs font-semibold text-slate-700">
                Temperature ({settings.temperature.toFixed(2)})
              </label>
              <input
                id="quick-temp"
                type="range"
                min={0}
                max={2}
                step={0.05}
                value={settings.temperature}
                onChange={(e) => onUpdateSettings({ temperature: Number(e.target.value) })}
                className="mt-2 w-full accent-blue-700"
              />
            </div>
            <div>
              <label htmlFor="quick-topk" className="block text-xs font-semibold text-slate-700">
                Top-k ({settings.top_k})
              </label>
              <input
                id="quick-topk"
                type="number"
                min={0}
                max={8192}
                value={settings.top_k}
                onChange={(e) => onUpdateSettings({ top_k: Number(e.target.value) })}
                className="mt-1.5 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm font-mono text-slate-900"
              />
            </div>
            <div>
              <label htmlFor="quick-topp" className="block text-xs font-semibold text-slate-700">
                Top-p ({settings.top_p.toFixed(2)})
              </label>
              <input
                id="quick-topp"
                type="range"
                min={0.05}
                max={1}
                step={0.05}
                value={settings.top_p}
                onChange={(e) => onUpdateSettings({ top_p: Number(e.target.value) })}
                className="mt-2 w-full accent-blue-700"
              />
            </div>
            <div>
              <label htmlFor="quick-maxtok" className="block text-xs font-semibold text-slate-700">
                Max New Tokens ({settings.max_new_tokens})
              </label>
              <input
                id="quick-maxtok"
                type="number"
                min={1}
                max={512}
                value={settings.max_new_tokens}
                onChange={(e) => onUpdateSettings({ max_new_tokens: Number(e.target.value) })}
                className="mt-1.5 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm font-mono text-slate-900"
              />
            </div>
          </div>
        </div>
      )}

      {/* Backend connection error or Untrained checkpoint warning */}
      {backendError && (
        <div
          role="alert"
          className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-900 flex items-start gap-3"
        >
          <AlertTriangle className="h-5 w-5 text-red-700 shrink-0 mt-0.5" aria-hidden="true" />
          <div>
            <p className="font-semibold">Python Inference Service Unreachable</p>
            <p className="mt-1 text-red-800">{backendError}</p>
          </div>
        </div>
      )}

      {!backendError && health && !chatReady && (
        <div
          role="status"
          data-testid="untrained-warning-banner"
          className="rounded-xl border border-amber-300 bg-amber-50 p-5 text-sm text-amber-950"
        >
          <div className="flex items-start gap-3">
            <AlertTriangle className="h-5 w-5 text-amber-700 shrink-0 mt-0.5" aria-hidden="true" />
            <div className="space-y-2">
              <p className="font-semibold text-amber-950">
                Production Chat Disabled — Model State: {statusLabel}
              </p>
              <p className="text-amber-900 leading-relaxed">
                {health.warning ||
                  'No trained STAN AI checkpoint is currently loaded. STAN AI never substitutes external LLM APIs or simulates responses from random weights.'}
              </p>
              <div className="pt-1 flex flex-wrap gap-3">
                <button
                  type="button"
                  onClick={() => onNavigateTab('model')}
                  className="text-xs font-semibold text-amber-950 underline underline-offset-4 hover:text-amber-800"
                >
                  Inspect Architecture &amp; Tokenizer
                </button>
                <button
                  type="button"
                  onClick={() => onNavigateTab('training')}
                  className="text-xs font-semibold text-amber-950 underline underline-offset-4 hover:text-amber-800"
                >
                  View Training Setup &amp; Metrics
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Conversation Transcript */}
      <div className="rounded-xl border border-slate-200 bg-white">
        <div className="min-h-[340px] max-h-[520px] overflow-y-auto p-6 space-y-6">
          {messages.length === 0 ? (
            <div className="flex h-[290px] flex-col items-center justify-center text-center px-4">
              <p className="font-display text-base font-semibold text-slate-900">
                {chatReady
                  ? 'STAN AI Checkpoint Ready'
                  : 'Waiting for Trained STAN AI Checkpoint'}
              </p>
              <p className="mt-2 max-w-md text-sm text-slate-600 leading-relaxed">
                {chatReady
                  ? 'Enter a prompt below to generate text directly from STAN AI’s local PyTorch weights and custom BPE tokenizer.'
                  : 'Once STAN AI has been trained on the verified English corpus and a checkpoint is placed in STAN-AI/checkpoints/latest.pt, interactive generation unlocks automatically.'}
              </p>
            </div>
          ) : (
            messages.map((msg) => (
              <article
                key={msg.id}
                className={`space-y-1.5 ${
                  msg.role === 'user' ? 'border-b border-slate-100 pb-5' : ''
                }`}
              >
                <div className="flex items-center gap-2 text-xs text-slate-500">
                  <span className="font-semibold text-slate-900">
                    {msg.role === 'user' ? 'Prompt' : 'STAN AI'}
                  </span>
                  <span aria-hidden="true">·</span>
                  <span className="tabular-nums">{msg.createdAt}</span>
                  {msg.meta?.checkpoint_version && (
                    <>
                      <span aria-hidden="true">·</span>
                      <span className="font-mono">{msg.meta.checkpoint_version}</span>
                    </>
                  )}
                </div>

                <div className="whitespace-pre-wrap text-sm leading-relaxed text-slate-900">
                  {msg.content || (isGenerating && msg.role === 'assistant' ? 'Generating tokens...' : '')}
                </div>

                {msg.meta && (msg.meta.generated_tokens !== undefined || msg.meta.elapsed_ms !== undefined) && (
                  <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1 pt-1 text-xs font-mono text-slate-500 tabular-nums">
                    {msg.meta.prompt_tokens !== undefined && (
                      <span>prompt_tokens={msg.meta.prompt_tokens}</span>
                    )}
                    {msg.meta.generated_tokens !== undefined && (
                      <span>generated={msg.meta.generated_tokens}</span>
                    )}
                    {msg.meta.tokens_per_second !== undefined && (
                      <span>{msg.meta.tokens_per_second} tok/s</span>
                    )}
                    {msg.meta.elapsed_ms !== undefined && <span>{msg.meta.elapsed_ms} ms</span>}
                    {msg.meta.stopped_reason && <span>stop={msg.meta.stopped_reason}</span>}
                  </div>
                )}
              </article>
            ))
          )}
        </div>

        {errorMsg && (
          <div
            role="alert"
            className="border-t border-red-200 bg-red-50 px-6 py-3 text-xs font-medium text-red-800"
          >
            {errorMsg}
          </div>
        )}

        {/* Prompt Input Bar */}
        <form onSubmit={handleSubmit} className="border-t border-slate-200 p-4 bg-slate-50/70 rounded-b-xl">
          <label htmlFor="chat-prompt-input" className="sr-only">
            Prompt for STAN AI
          </label>
          <div className="flex flex-col gap-3 sm:flex-row sm:items-end">
            <textarea
              id="chat-prompt-input"
              rows={2}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              disabled={!chatReady || isGenerating}
              placeholder={
                chatReady
                  ? 'Enter an English prompt for STAN AI...'
                  : 'Production chat disabled until a trained STAN AI checkpoint is loaded.'
              }
              className="w-full resize-y rounded-lg border border-slate-300 bg-white px-3.5 py-2.5 text-sm text-slate-900 placeholder:text-slate-400 focus:border-blue-700 focus:outline-none disabled:bg-slate-100 disabled:text-slate-500"
            />
            {isGenerating ? (
              <button
                type="button"
                onClick={handleStop}
                className="inline-flex min-h-[44px] items-center justify-center gap-2 rounded-lg bg-red-700 px-5 py-2.5 text-sm font-semibold text-white hover:bg-red-800 transition-colors whitespace-nowrap shrink-0"
              >
                <Square className="h-4 w-4" aria-hidden="true" />
                <span>Stop</span>
              </button>
            ) : (
              <button
                type="submit"
                disabled={!chatReady || !prompt.trim()}
                className="inline-flex min-h-[44px] items-center justify-center gap-2 rounded-lg bg-blue-700 px-5 py-2.5 text-sm font-semibold text-white hover:bg-blue-800 disabled:bg-slate-300 disabled:text-slate-600 transition-colors whitespace-nowrap shrink-0"
              >
                <Send className="h-4 w-4" aria-hidden="true" />
                <span>Send</span>
              </button>
            )}
          </div>
        </form>
      </div>
    </section>
  );
}
