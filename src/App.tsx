import { useCallback, useEffect, useState } from 'react';
import { fetchHealth, fetchMetrics, fetchModelInfo } from './api/stanClient';
import { ChatView } from './components/ChatView';
import { ModelInfoView } from './components/ModelInfoView';
import { SettingsView } from './components/SettingsView';
import { TopNav, type NavTab } from './components/TopNav';
import { TrainingView } from './components/TrainingView';
import type { GenerationSettings, HealthInfo, MetricsInfo, ModelInfo } from './types/stan';
import { clampGenerationSettings, DEFAULT_GENERATION_SETTINGS } from './utils/stanState';

const SETTINGS_STORAGE_KEY = 'stan_ai_settings_v1';

function loadInitialSettings(): GenerationSettings {
  try {
    const raw = localStorage.getItem(SETTINGS_STORAGE_KEY);
    if (!raw) return DEFAULT_GENERATION_SETTINGS;
    return clampGenerationSettings(JSON.parse(raw) as Partial<GenerationSettings>);
  } catch {
    return DEFAULT_GENERATION_SETTINGS;
  }
}

export default function App() {
  const [activeTab, setActiveTab] = useState<NavTab>('chat');
  const [settings, setSettings] = useState<GenerationSettings>(loadInitialSettings);
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [metrics, setMetrics] = useState<MetricsInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [backendError, setBackendError] = useState<string | null>(null);

  const syncRuntime = useCallback(async (allowRetry = true) => {
    setLoading(true);
    setBackendError(null);
    try {
      const [h, info, met] = await Promise.all([
        fetchHealth(settings.apiBaseUrl),
        fetchModelInfo(settings.apiBaseUrl),
        fetchMetrics(settings.apiBaseUrl),
      ]);
      setHealth(h);
      setModelInfo(info);
      setMetrics(met);
      setLoading(false);
    } catch (err) {
      if (allowRetry) {
        setTimeout(() => {
          void syncRuntime(false);
        }, 1200);
        return;
      }
      setBackendError(
        err instanceof Error
          ? err.message
          : 'Failed to reach STAN AI Python inference service.',
      );
      setLoading(false);
    }
  }, [settings.apiBaseUrl]);

  useEffect(() => {
    void syncRuntime(true);
  }, [syncRuntime]);

  const handleUpdateSettings = useCallback((patch: Partial<GenerationSettings>) => {
    setSettings((prev) => {
      const next = clampGenerationSettings({ ...prev, ...patch });
      try {
        localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(next));
      } catch {
        // ignore storage quota failures
      }
      return next;
    });
  }, []);

  const handleResetDefaults = useCallback(() => {
    setSettings(DEFAULT_GENERATION_SETTINGS);
    try {
      localStorage.removeItem(SETTINGS_STORAGE_KEY);
    } catch {
      // ignore
    }
  }, []);

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-900">
      <TopNav
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        onRefresh={() => void syncRuntime()}
        refreshing={loading}
      />

      <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-8 sm:px-6">
        {activeTab === 'chat' && (
          <ChatView
            health={health}
            modelInfo={modelInfo}
            settings={settings}
            onUpdateSettings={handleUpdateSettings}
            onNavigateTab={setActiveTab}
            backendError={backendError}
          />
        )}
        {activeTab === 'model' && (
          <ModelInfoView
            modelInfo={modelInfo}
            loading={loading}
            apiBaseUrl={settings.apiBaseUrl}
          />
        )}
        {activeTab === 'training' && (
          <TrainingView metrics={metrics} loading={loading} />
        )}
        {activeTab === 'settings' && (
          <SettingsView
            settings={settings}
            onUpdateSettings={handleUpdateSettings}
            onResetDefaults={handleResetDefaults}
          />
        )}
      </main>

      <footer className="border-t border-slate-200 bg-white py-4">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-2 px-4 sm:px-6 text-xs text-slate-500">
          <span>STAN AI · Independent Decoder-Only Transformer</span>
          <span>PyTorch + SentencePiece BPE + FastAPI</span>
        </div>
      </footer>
    </div>
  );
}
