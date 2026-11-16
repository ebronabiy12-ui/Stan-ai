import { useEffect, useState, type FormEvent } from 'react';
import {
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  Send,
  SlidersHorizontal,
  Square,
  Trash2,
} from 'lucide-react';
import { fetchMemories, runReasoning, streamGenerate } from '../api/stanClient';
import type {
  ChatMessage,
  GenerationSettings,
  HealthInfo,
  IdentityProfile,
  ModelInfo,
} from '../types/stan';
import {
  formatParamCount,
  formatRegisterLabel,
  formatStanceLabel,
  formatTrainingStatusLabel,
  isProductionChatReady,
} from '../utils/stanState';

interface ChatViewProps {
  health: HealthInfo | null;
  modelInfo: ModelInfo | null;
  identity: IdentityProfile | null;
  settings: GenerationSettings;
  onUpdateSettings: (patch: Partial<GenerationSettings>) => void;
  onNavigateTab: (tab: 'cognitive' | 'model' | 'training' | 'settings') => void;
  backendError: string | null;
}

type InferenceMode = 'cognitive' | 'neural';

const QUICK_PROMPTS = [
  'Who am I?',
  'Who created you and what is Kalda?',
  'Are you conscious or powered by Gemini?',
  'I think we should just fabricate training metrics so the UI looks finished.',
];

export function ChatView({
  health,
  modelInfo,
  identity,
  settings,
  onUpdateSettings,
  onNavigateTab,
  backendError,
}: ChatViewProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [historyLoaded, setHistoryLoaded] = useState(false);
  const [prompt, setPrompt] = useState('');
  const [mode, setMode] = useState<InferenceMode>('cognitive');
  const [isGenerating, setIsGenerating] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [showQuickSettings, setShowQuickSettings] = useState(false);
  const [expandedTraceIds, setExpandedTraceIds] = useState<Record<string, boolean>>({});
  const [abortCtrl, setAbortCtrl] = useState<AbortController | null>(null);

  const chatReady = health ? isProductionChatReady(health) : false;
  const statusLabel = formatTrainingStatusLabel(health?.training_status || 'untrained');
  const canSubmitPrompt =
    !backendError &&
    !isGenerating &&
    (mode === 'cognitive' ? Boolean(health) : chatReady);

  useEffect(() => {
    if (historyLoaded || backendError || !health) return;
    let active = true;
    (async () => {
      try {
        const res = await fetchMemories(settings.apiBaseUrl, 'conversation_history', 16);
        if (!active) return;
        const chronological = [...res.records].reverse();
        const restored: ChatMessage[] = [];
        for (const rec of chronological) {
          const marker = '\nSTAN AI: ';
          const idx = rec.content.indexOf(marker);
          if (rec.content.startsWith('User: ') && idx !== -1) {
            const userText = rec.content.slice(6, idx).trim();
            const assistantText = rec.content.slice(idx + marker.length).trim();
            const timeLabel = new Date(rec.created_at).toLocaleTimeString();
            const metaObj = (rec.metadata || {}) as {
              stance?: string;
              register?: string;
              signals?: string[];
            };
            restored.push({
              id: `${rec.id}_u`,
              role: 'user',
              content: userText,
              createdAt: timeLabel,
            });
            restored.push({
              id: `${rec.id}_a`,
              role: 'assistant',
              content: assistantText,
              createdAt: timeLabel,
              mode: 'cognitive',
              reasoning: {
                user_message: userText,
                response: assistantText,
                stance: metaObj.stance || 'answer',
                register: metaObj.register || 'direct',
                detected_signals: Array.isArray(metaObj.signals) ? metaObj.signals : [],
                reasoning_steps: ['Restored from persistent SQLite conversation_history store.'],
                facts: [],
                opinions: [],
                uncertainties: [],
                contradictions_found: [],
                retrieved_memories: [],
                memory_updates: [],
                provenance: {
                  model_weights_status: health.training_status,
                  neural_generation_used: false,
                  identity_active: true,
                  memory_store_active: true,
                  reasoning_engine_active: true,
                  conscious: false,
                },
              },
            });
          }
        }
        if (restored.length > 0) {
          setMessages(restored);
        }
      } catch {
        // non-blocking history restore
      } finally {
        if (active) setHistoryLoaded(true);
      }
    })();
    return () => {
      active = false;
    };
  }, [backendError, health, historyLoaded, settings.apiBaseUrl]);

  function toggleTrace(msgId: string) {
    setExpandedTraceIds((prev) => ({ ...prev, [msgId]: !prev[msgId] }));
  }

  async function submitPromptText(rawPrompt: string) {
    const trimmed = rawPrompt.trim();
    if (!trimmed || !canSubmitPrompt) return;

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
      mode,
    };

    setMessages((prev) => [...prev, userMsg, placeholderAssistant]);
    setPrompt('');
    setIsGenerating(true);

    const controller = new AbortController();
    setAbortCtrl(controller);

    try {
      if (mode === 'cognitive') {
        const result = await runReasoning(
          settings.apiBaseUrl,
          trimmed,
          settings,
          true,
          controller.signal,
        );
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? {
                  ...m,
                  content: result.response,
                  mode: 'cognitive',
                  reasoning: result,
                  meta: result.provenance.neural_meta
                    ? {
                        prompt_tokens: result.provenance.neural_meta.prompt_tokens,
                        generated_tokens: result.provenance.neural_meta.generated_tokens,
                        elapsed_ms: result.provenance.neural_meta.elapsed_ms,
                        tokens_per_second: result.provenance.neural_meta.tokens_per_second,
                        stopped_reason: result.provenance.neural_meta.stopped_reason,
                        checkpoint_version: result.provenance.neural_meta.checkpoint_version,
                      }
                    : undefined,
                }
              : m,
          ),
        );
        setExpandedTraceIds((prev) => ({ ...prev, [assistantId]: true }));
      } else {
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
                      mode: 'neural',
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
      }
    } catch (err) {
      const detail = err instanceof Error ? err.message : 'Request failed.';
      if ((err as { name?: string })?.name !== 'AbortError') {
        setErrorMsg(detail);
        setMessages((prev) => prev.filter((m) => m.id !== assistantId || m.content.length > 0));
      }
    } finally {
      setIsGenerating(false);
      setAbortCtrl(null);
    }
  }

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    void submitPromptText(prompt);
  }

  function handleStop() {
    abortCtrl?.abort();
    setIsGenerating(false);
    setAbortCtrl(null);
  }

  function handleClearViewport() {
    setMessages([]);
    setErrorMsg(null);
  }

  return (
    <section aria-labelledby="chat-heading" className="space-y-6">
      {/* Page Header & 4-Layer Provenance Metadata */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between border-b border-slate-200 pb-5">
        <div>
          <h1 id="chat-heading" className="font-display text-2xl font-bold tracking-tight text-slate-900">
            STAN AI Workspace
          </h1>
          <div className="mt-1.5 flex flex-wrap items-center gap-x-2.5 gap-y-1 text-xs text-slate-600 tabular-nums">
            <span className="font-medium text-slate-900">
              Identity: {identity?.name ?? 'STAN AI'} ({identity?.owner_name ?? 'Stan'})
            </span>
            <span aria-hidden="true">·</span>
            <span>Memory: 7-Store SQLite</span>
            <span aria-hidden="true">·</span>
            <span>Model Weights: {statusLabel}</span>
            <span aria-hidden="true">·</span>
            <span>
              {modelInfo ? formatParamCount(modelInfo.parameter_count) : '51.66M Architecture'}
            </span>
            {modelInfo?.checkpoint_version && (
              <>
                <span aria-hidden="true">·</span>
                <span className="font-mono">{modelInfo.checkpoint_version}</span>
              </>
            )}
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2 shrink-0">
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
            onClick={handleClearViewport}
            disabled={messages.length === 0 || isGenerating}
            className="inline-flex min-h-[40px] items-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-2 text-xs font-semibold text-slate-800 hover:bg-slate-50 disabled:opacity-50 transition-colors whitespace-nowrap shrink-0"
          >
            <Trash2 className="h-3.5 w-3.5" aria-hidden="true" />
            <span>Clear View</span>
          </button>
        </div>
      </div>

      {/* Mode Switcher */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between rounded-xl border border-slate-200 bg-white px-4 py-3">
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => setMode('cognitive')}
            aria-pressed={mode === 'cognitive'}
            className={`min-h-[36px] rounded-lg px-3.5 py-1.5 text-xs font-semibold transition-colors whitespace-nowrap shrink-0 ${
              mode === 'cognitive'
                ? 'bg-blue-700 text-white'
                : 'border border-slate-300 bg-white text-slate-700 hover:bg-slate-50'
            }`}
          >
            Cognitive Reasoning
          </button>
          <button
            type="button"
            onClick={() => chatReady && setMode('neural')}
            aria-pressed={mode === 'neural'}
            disabled={!chatReady}
            title={
              chatReady
                ? 'Stream tokens directly from STAN AI PyTorch Transformer weights'
                : 'Disabled until a trained STAN AI checkpoint is loaded'
            }
            className={`min-h-[36px] rounded-lg px-3.5 py-1.5 text-xs font-semibold transition-colors whitespace-nowrap shrink-0 ${
              mode === 'neural'
                ? 'bg-blue-700 text-white'
                : 'border border-slate-300 bg-white text-slate-700 hover:bg-slate-50 disabled:bg-slate-100 disabled:text-slate-400 disabled:cursor-not-allowed'
            }`}
          >
            Raw Transformer Stream
          </button>
        </div>

        <button
          type="button"
          onClick={() => onNavigateTab('cognitive')}
          className="text-xs font-semibold text-blue-700 hover:text-blue-800 underline underline-offset-4 text-left sm:text-right"
        >
          Manage 7-Store Memory &amp; Personality →
        </button>
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

      {/* Backend connection error or Untrained checkpoint transparency notice */}
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
                Transformer Weights Untrained — Cognitive Reasoning Active
              </p>
              <p className="text-amber-900 leading-relaxed">
                Raw neural token streaming remains disabled until a trained STAN AI checkpoint is loaded.
                You can interact right now with STAN AI&apos;s Identity, Persistent SQLite Memory, and
                Reasoning Engine below; every turn reports explicit provenance (
                <code className="font-mono text-xs">neural_generation_used: false</code>).
              </p>
              <div className="pt-1 flex flex-wrap gap-4">
                <button
                  type="button"
                  onClick={() => onNavigateTab('model')}
                  className="text-xs font-semibold text-amber-950 underline underline-offset-4 hover:text-amber-800"
                >
                  Inspect 51.66M Architecture &amp; Tokenizer
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
        <div className="min-h-[340px] max-h-[560px] overflow-y-auto p-6 space-y-6">
          {messages.length === 0 ? (
            <div className="flex min-h-[280px] flex-col items-center justify-center text-center px-4">
              <p className="font-display text-base font-semibold text-slate-900">
                STAN AI Cognitive Reasoning &amp; Persistent Identity Ready
              </p>
              <p className="mt-2 max-w-lg text-sm text-slate-600 leading-relaxed">
                Ask about identity, test contradiction and honesty checks, record facts or preferences,
                or inspect how STAN AI separates facts, opinions, uncertainties, and neural weights.
              </p>
              <div className="mt-5 flex flex-wrap items-center justify-center gap-2 max-w-2xl">
                {QUICK_PROMPTS.map((q) => (
                  <button
                    key={q}
                    type="button"
                    disabled={!canSubmitPrompt}
                    onClick={() => void submitPromptText(q)}
                    className="rounded-lg border border-slate-300 bg-slate-50 px-3 py-2 text-xs font-medium text-slate-800 hover:bg-slate-100 disabled:opacity-50 transition-colors text-left"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((msg) => {
              const r = msg.reasoning;
              const traceOpen = Boolean(expandedTraceIds[msg.id]);

              return (
                <article
                  key={msg.id}
                  className={`space-y-2 ${
                    msg.role === 'user' ? 'border-b border-slate-100 pb-5' : ''
                  }`}
                >
                  <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-slate-500">
                    <span className="font-semibold text-slate-900">
                      {msg.role === 'user' ? identity?.owner_name || 'User' : 'STAN AI'}
                    </span>
                    <span aria-hidden="true">·</span>
                    <span className="tabular-nums">{msg.createdAt}</span>
                    {r && (
                      <>
                        <span aria-hidden="true">·</span>
                        <span className="font-medium text-slate-700">
                          {formatStanceLabel(r.stance)} ({formatRegisterLabel(r.register)})
                        </span>
                        <span aria-hidden="true">·</span>
                        <span className="font-mono text-slate-600">
                          weights={r.provenance.model_weights_status || 'untrained'} (neural=
                          {r.provenance.neural_generation_used ? 'yes' : 'no'})
                        </span>
                      </>
                    )}
                    {msg.meta?.checkpoint_version && (
                      <>
                        <span aria-hidden="true">·</span>
                        <span className="font-mono">{msg.meta.checkpoint_version}</span>
                      </>
                    )}
                  </div>

                  <div className="whitespace-pre-wrap text-sm leading-relaxed text-slate-900">
                    {msg.content || (isGenerating && msg.role === 'assistant' ? 'Reasoning...' : '')}
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

                  {r && (
                    <div className="pt-1">
                      <button
                        type="button"
                        onClick={() => toggleTrace(msg.id)}
                        aria-expanded={traceOpen}
                        className="inline-flex items-center gap-1.5 text-xs font-semibold text-blue-700 hover:text-blue-800"
                      >
                        {traceOpen ? (
                          <ChevronDown className="h-3.5 w-3.5" aria-hidden="true" />
                        ) : (
                          <ChevronRight className="h-3.5 w-3.5" aria-hidden="true" />
                        )}
                        <span>Reasoning &amp; Provenance Trace</span>
                      </button>

                      {traceOpen && (
                        <div className="mt-2.5 rounded-lg border border-slate-200 bg-slate-50/70 p-4 space-y-3 text-xs text-slate-700">
                          {r.detected_signals.length > 0 && (
                            <div>
                              <span className="font-semibold text-slate-900">Detected Signals: </span>
                              <span className="font-mono">{r.detected_signals.join(', ')}</span>
                            </div>
                          )}

                          {r.contradictions_found.length > 0 && (
                            <div className="rounded-md border border-amber-300 bg-amber-50 p-2.5 text-amber-950">
                              <p className="font-semibold">Contradictions Identified</p>
                              <ul className="mt-1 list-disc pl-4 space-y-0.5">
                                {r.contradictions_found.map((c, idx) => (
                                  <li key={idx}>{c}</li>
                                ))}
                              </ul>
                            </div>
                          )}

                          <div>
                            <p className="font-semibold text-slate-900">Reasoning Steps</p>
                            <ul className="mt-1 space-y-0.5 font-mono text-[11px] text-slate-600">
                              {r.reasoning_steps.map((step, idx) => (
                                <li key={idx}>{step}</li>
                              ))}
                            </ul>
                          </div>

                          {(r.facts.length > 0 ||
                            r.opinions.length > 0 ||
                            r.uncertainties.length > 0) && (
                            <div className="grid grid-cols-1 gap-3 md:grid-cols-3 border-t border-slate-200 pt-2.5">
                              <div>
                                <p className="font-semibold text-slate-900">Verified Facts</p>
                                {r.facts.length > 0 ? (
                                  <ul className="mt-1 list-disc pl-4 space-y-0.5">
                                    {r.facts.map((f, idx) => (
                                      <li key={idx}>{f}</li>
                                    ))}
                                  </ul>
                                ) : (
                                  <p className="mt-1 text-slate-500">None cited</p>
                                )}
                              </div>
                              <div>
                                <p className="font-semibold text-slate-900">Reasoned Opinions</p>
                                {r.opinions.length > 0 ? (
                                  <ul className="mt-1 list-disc pl-4 space-y-0.5">
                                    {r.opinions.map((o, idx) => (
                                      <li key={idx}>{o}</li>
                                    ))}
                                  </ul>
                                ) : (
                                  <p className="mt-1 text-slate-500">None formed</p>
                                )}
                              </div>
                              <div>
                                <p className="font-semibold text-slate-900">Uncertainties</p>
                                {r.uncertainties.length > 0 ? (
                                  <ul className="mt-1 list-disc pl-4 space-y-0.5">
                                    {r.uncertainties.map((u, idx) => (
                                      <li key={idx}>{u}</li>
                                    ))}
                                  </ul>
                                ) : (
                                  <p className="mt-1 text-slate-500">None flagged</p>
                                )}
                              </div>
                            </div>
                          )}

                          {(r.retrieved_memories.length > 0 || r.memory_updates.length > 0) && (
                            <div className="border-t border-slate-200 pt-2.5 space-y-1.5">
                              {r.retrieved_memories.length > 0 && (
                                <p>
                                  <span className="font-semibold text-slate-900">
                                    Retrieved Memories ({r.retrieved_memories.length}):{' '}
                                  </span>
                                  <span>
                                    {r.retrieved_memories
                                      .map((m) => `${m.subject} [${m.category}]`)
                                      .join(' · ')}
                                  </span>
                                </p>
                              )}
                              {r.memory_updates.length > 0 && (
                                <p>
                                  <span className="font-semibold text-slate-900">
                                    Memory Updates ({r.memory_updates.length}):{' '}
                                  </span>
                                  <span className="font-mono">
                                    {r.memory_updates
                                      .map((u) => `${u.action || 'updated'}(${u.record?.category || u.category || 'history'})`)
                                      .join(', ')}
                                  </span>
                                </p>
                              )}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </article>
              );
            })
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
              disabled={!health || isGenerating || (mode === 'neural' && !chatReady)}
              placeholder={
                mode === 'cognitive'
                  ? 'Ask STAN AI something, share a fact, or challenge its judgment...'
                  : chatReady
                    ? 'Enter an English prompt for raw Transformer generation...'
                    : 'Raw Transformer stream disabled until a trained STAN AI checkpoint is loaded.'
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
                disabled={!canSubmitPrompt || !prompt.trim()}
                className="inline-flex min-h-[44px] items-center justify-center gap-2 rounded-lg bg-blue-700 px-5 py-2.5 text-sm font-semibold text-white hover:bg-blue-800 disabled:bg-slate-300 disabled:text-slate-600 transition-colors whitespace-nowrap shrink-0"
              >
                <Send className="h-4 w-4" aria-hidden="true" />
                <span>{mode === 'cognitive' ? 'Ask STAN AI' : 'Generate'}</span>
              </button>
            )}
          </div>
        </form>
      </div>
    </section>
  );
}
