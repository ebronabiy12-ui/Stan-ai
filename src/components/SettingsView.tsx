import { useState } from 'react';
import { fetchHealth } from '../api/stanClient';
import type { GenerationSettings } from '../types/stan';
import { DEFAULT_GENERATION_SETTINGS } from '../utils/stanState';

interface SettingsViewProps {
  settings: GenerationSettings;
  onUpdateSettings: (patch: Partial<GenerationSettings>) => void;
  onResetDefaults: () => void;
}

export function SettingsView({ settings, onUpdateSettings, onResetDefaults }: SettingsViewProps) {
  const [testResult, setTestResult] = useState<string | null>(null);
  const [testing, setTesting] = useState(false);

  async function handleTestConnection() {
    setTesting(true);
    setTestResult(null);
    try {
      const h = await fetchHealth(settings.apiBaseUrl);
      setTestResult(
        `Connected (${h.service}) · status=${h.training_status} · device=${h.device}`,
      );
    } catch (err) {
      setTestResult(err instanceof Error ? err.message : 'Connection failed.');
    } finally {
      setTesting(false);
    }
  }

  return (
    <section aria-labelledby="settings-heading" className="space-y-8">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between border-b border-slate-200 pb-5">
        <div>
          <h1 id="settings-heading" className="font-display text-2xl font-bold tracking-tight text-slate-900">
            Generation &amp; Inference Settings
          </h1>
          <p className="mt-1.5 text-sm text-slate-600">
            Configure decoding parameters sent directly to STAN AI&apos;s PyTorch generation engine.
          </p>
        </div>
        <button
          type="button"
          onClick={onResetDefaults}
          className="min-h-[40px] rounded-lg border border-slate-300 bg-white px-4 py-2 text-xs font-semibold text-slate-800 hover:bg-slate-50 whitespace-nowrap shrink-0"
        >
          Reset to Defaults
        </button>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Sampling Controls */}
        <div className="rounded-xl border border-slate-200 bg-white p-6 space-y-5">
          <h2 className="font-display text-sm font-semibold text-slate-900">Sampling Parameters</h2>

          <div>
            <div className="flex items-center justify-between text-xs font-semibold text-slate-700">
              <label htmlFor="setting-temp">Temperature</label>
              <span className="font-mono tabular-nums">{settings.temperature.toFixed(2)}</span>
            </div>
            <input
              id="setting-temp"
              type="range"
              min={0}
              max={2}
              step={0.05}
              value={settings.temperature}
              onChange={(e) => onUpdateSettings({ temperature: Number(e.target.value) })}
              className="mt-2 w-full accent-blue-700"
            />
            <p className="mt-1 text-xs text-slate-500">
              0.00 performs greedy decoding; higher values increase token diversity.
            </p>
          </div>

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div>
              <label htmlFor="setting-topk" className="block text-xs font-semibold text-slate-700">
                Top-k
              </label>
              <input
                id="setting-topk"
                type="number"
                min={0}
                max={8192}
                value={settings.top_k}
                onChange={(e) => onUpdateSettings({ top_k: Number(e.target.value) })}
                className="mt-1.5 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm font-mono text-slate-900"
              />
            </div>

            <div>
              <label htmlFor="setting-topp" className="block text-xs font-semibold text-slate-700">
                Top-p (Nucleus)
              </label>
              <input
                id="setting-topp"
                type="number"
                min={0.05}
                max={1}
                step={0.05}
                value={settings.top_p}
                onChange={(e) => onUpdateSettings({ top_p: Number(e.target.value) })}
                className="mt-1.5 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm font-mono text-slate-900"
              />
            </div>
          </div>

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div>
              <label htmlFor="setting-maxtok" className="block text-xs font-semibold text-slate-700">
                Maximum Tokens
              </label>
              <input
                id="setting-maxtok"
                type="number"
                min={1}
                max={512}
                value={settings.max_new_tokens}
                onChange={(e) => onUpdateSettings({ max_new_tokens: Number(e.target.value) })}
                className="mt-1.5 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm font-mono text-slate-900"
              />
            </div>

            <div>
              <label htmlFor="setting-reppen" className="block text-xs font-semibold text-slate-700">
                Repetition Penalty
              </label>
              <input
                id="setting-reppen"
                type="number"
                min={1}
                max={2}
                step={0.05}
                value={settings.repetition_penalty}
                onChange={(e) => onUpdateSettings({ repetition_penalty: Number(e.target.value) })}
                className="mt-1.5 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm font-mono text-slate-900"
              />
            </div>
          </div>

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 items-end">
            <div>
              <label htmlFor="setting-seed" className="block text-xs font-semibold text-slate-700">
                Random Seed (Optional)
              </label>
              <input
                id="setting-seed"
                type="number"
                min={0}
                placeholder="Random"
                value={settings.seed ?? ''}
                onChange={(e) =>
                  onUpdateSettings({
                    seed: e.target.value === '' ? null : Number(e.target.value),
                  })
                }
                className="mt-1.5 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm font-mono text-slate-900"
              />
            </div>

            <label className="flex min-h-[40px] items-center gap-2.5 text-xs font-semibold text-slate-800 cursor-pointer">
              <input
                type="checkbox"
                checked={settings.stop_on_eos}
                onChange={(e) => onUpdateSettings({ stop_on_eos: e.target.checked })}
                className="h-4 w-4 accent-blue-700 rounded border-slate-300"
              />
              <span>Stop on EOS Token (&lt;/s&gt;)</span>
            </label>
          </div>
        </div>

        {/* Inference Endpoint Configuration */}
        <div className="rounded-xl border border-slate-200 bg-white p-6 space-y-5 self-start">
          <h2 className="font-display text-sm font-semibold text-slate-900">
            Python FastAPI Connection
          </h2>
          <div>
            <label htmlFor="setting-api-url" className="block text-xs font-semibold text-slate-700">
              Configurable Inference API URL
            </label>
            <input
              id="setting-api-url"
              type="text"
              value={settings.apiBaseUrl}
              onChange={(e) => onUpdateSettings({ apiBaseUrl: e.target.value })}
              placeholder={DEFAULT_GENERATION_SETTINGS.apiBaseUrl}
              className="mt-1.5 w-full rounded-lg border border-slate-300 px-3.5 py-2 text-sm font-mono text-slate-900"
            />
            <p className="mt-1.5 text-xs text-slate-500">
              Defaults to <span className="font-mono">/api/stan</span> (proxied by Express to Python FastAPI).
            </p>
          </div>

          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={handleTestConnection}
              disabled={testing}
              className="min-h-[40px] rounded-lg bg-slate-900 px-4 py-2 text-xs font-semibold text-white hover:bg-slate-800 disabled:opacity-60"
            >
              {testing ? 'Testing...' : 'Test Connection'}
            </button>
          </div>

          {testResult && (
            <p className="rounded-lg border border-slate-200 bg-slate-50 px-3.5 py-2.5 text-xs font-mono text-slate-800">
              {testResult}
            </p>
          )}
        </div>
      </div>
    </section>
  );
}
