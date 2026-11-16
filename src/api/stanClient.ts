import type {
  EvaluationReport,
  GenerateResult,
  GenerationSettings,
  HealthInfo,
  IdentityProfile,
  MemoryListResponse,
  MemoryRecord,
  MetricsInfo,
  ModelInfo,
  PersonalityConfig,
  ReasonResponse,
  TokenizeResult,
  TrainingJobStatus,
} from '../types/stan';

async function parseErrorDetail(res: Response): Promise<string> {
  try {
    const data = (await res.json()) as { detail?: unknown; message?: string };
    if (typeof data.detail === 'string') return data.detail;
    if (Array.isArray(data.detail)) {
      return data.detail
        .map((item: { msg?: string }) => item.msg || JSON.stringify(item))
        .join('; ');
    }
    if (typeof data.message === 'string') return data.message;
  } catch {
    // fallback to status text
  }
  return `Request failed with HTTP ${res.status} (${res.statusText || 'Error'})`;
}

export async function fetchHealth(baseUrl: string): Promise<HealthInfo> {
  const res = await fetch(`${baseUrl}/health`);
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return (await res.json()) as HealthInfo;
}

export async function fetchModelInfo(baseUrl: string): Promise<ModelInfo> {
  const res = await fetch(`${baseUrl}/model/info`);
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return (await res.json()) as ModelInfo;
}

export async function fetchMetrics(baseUrl: string): Promise<MetricsInfo> {
  const res = await fetch(`${baseUrl}/metrics`);
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return (await res.json()) as MetricsInfo;
}

export async function tokenizeText(
  baseUrl: string,
  text: string,
  addBos = false,
  addEos = false,
): Promise<TokenizeResult> {
  const res = await fetch(`${baseUrl}/tokenize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, add_bos: addBos, add_eos: addEos }),
  });
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return (await res.json()) as TokenizeResult;
}

export async function fetchIdentity(baseUrl: string): Promise<IdentityProfile> {
  const res = await fetch(`${baseUrl}/identity`);
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return (await res.json()) as IdentityProfile;
}

export async function updateIdentity(
  baseUrl: string,
  payload: { personality?: PersonalityConfig; core_principles?: string[] },
): Promise<IdentityProfile> {
  const res = await fetch(`${baseUrl}/identity`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return (await res.json()) as IdentityProfile;
}

export async function fetchMemories(
  baseUrl: string,
  category?: string,
  limit = 100,
): Promise<MemoryListResponse> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (category && category !== 'all') {
    params.set('category', category);
  }
  const res = await fetch(`${baseUrl}/memory?${params.toString()}`);
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return (await res.json()) as MemoryListResponse;
}

export async function createMemory(
  baseUrl: string,
  payload: {
    category: string;
    subject: string;
    content: string;
    source?: string;
    confidence?: number;
    pinned?: boolean;
  },
): Promise<MemoryRecord> {
  const res = await fetch(`${baseUrl}/memory`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return (await res.json()) as MemoryRecord;
}

export async function updateMemory(
  baseUrl: string,
  recordId: string,
  payload: {
    category?: string;
    subject?: string;
    content?: string;
    confidence?: number;
    pinned?: boolean;
  },
): Promise<MemoryRecord> {
  const res = await fetch(`${baseUrl}/memory/${encodeURIComponent(recordId)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return (await res.json()) as MemoryRecord;
}

export async function deleteMemory(
  baseUrl: string,
  recordId: string,
): Promise<{ deleted: boolean; id: string }> {
  const res = await fetch(`${baseUrl}/memory/${encodeURIComponent(recordId)}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return (await res.json()) as { deleted: boolean; id: string };
}

export async function searchMemories(
  baseUrl: string,
  query: string,
  categories?: string[],
  topK = 12,
): Promise<{ query: string; count: number; records: MemoryRecord[] }> {
  const res = await fetch(`${baseUrl}/memory/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      categories: categories && categories.length > 0 ? categories : null,
      top_k: topK,
    }),
  });
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return (await res.json()) as { query: string; count: number; records: MemoryRecord[] };
}

export async function runReasoning(
  baseUrl: string,
  message: string,
  settings: GenerationSettings,
  persistTurn = true,
  signal?: AbortSignal,
): Promise<ReasonResponse> {
  const res = await fetch(`${baseUrl}/reason`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message,
      temperature: settings.temperature,
      top_k: settings.top_k,
      top_p: settings.top_p,
      max_new_tokens: settings.max_new_tokens,
      repetition_penalty: settings.repetition_penalty,
      seed: settings.seed,
      persist_turn: persistTurn,
    }),
    signal,
  });
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return (await res.json()) as ReasonResponse;
}

export interface StreamEventChunk {
  event: 'token' | 'done';
  generated_text: string;
  generated_tokens: number;
  elapsed_ms: number;
  prompt_tokens?: number;
  tokens_per_second?: number;
  stopped_reason?: string;
}

export async function streamGenerate(
  baseUrl: string,
  prompt: string,
  settings: GenerationSettings,
  onChunk: (chunk: StreamEventChunk) => void,
  signal?: AbortSignal,
): Promise<void> {
  const body = {
    prompt,
    temperature: settings.temperature,
    top_k: settings.top_k,
    top_p: settings.top_p,
    max_new_tokens: settings.max_new_tokens,
    repetition_penalty: settings.repetition_penalty,
    seed: settings.seed,
    stop_on_eos: settings.stop_on_eos,
  };

  const res = await fetch(`${baseUrl}/generate/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  });

  if (!res.ok) {
    throw new Error(await parseErrorDetail(res));
  }
  if (!res.body) {
    const fallback = await fetch(`${baseUrl}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal,
    });
    if (!fallback.ok) throw new Error(await parseErrorDetail(fallback));
    const json = (await fallback.json()) as GenerateResult;
    onChunk({
      event: 'done',
      generated_text: json.generated_text,
      generated_tokens: json.generated_tokens,
      elapsed_ms: json.elapsed_ms,
      prompt_tokens: json.prompt_tokens,
      tokens_per_second: json.tokens_per_second,
      stopped_reason: json.stopped_reason,
    });
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split('\n\n');
    buffer = lines.pop() ?? '';
    for (const block of lines) {
      const dataLine = block
        .split('\n')
        .find((line) => line.startsWith('data: '));
      if (!dataLine) continue;
      const jsonStr = dataLine.slice(6).trim();
      if (!jsonStr) continue;
      const parsed = JSON.parse(jsonStr) as StreamEventChunk;
      onChunk(parsed);
    }
  }
}

export async function fetchTrainingStatus(baseUrl: string): Promise<{
  training_status: string;
  training_stage: string;
  hardware: import('../types/stan').HardwareInfo;
  job: TrainingJobStatus;
  checkpoint_history: import('../types/stan').CheckpointRecord[];
}> {
  const res = await fetch(`${baseUrl}/training/status`);
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return (await res.json()) as {
    training_status: string;
    training_stage: string;
    hardware: import('../types/stan').HardwareInfo;
    job: TrainingJobStatus;
    checkpoint_history: import('../types/stan').CheckpointRecord[];
  };
}

export async function startTrainingJob(
  baseUrl: string,
  payload: {
    stage: 'smoke_test' | 'pretraining' | 'instruction_tuning';
    resume?: boolean;
    overwrite_existing?: boolean;
    max_steps?: number;
  },
): Promise<TrainingJobStatus> {
  const res = await fetch(`${baseUrl}/training/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return (await res.json()) as TrainingJobStatus;
}

export async function stopTrainingJob(baseUrl: string): Promise<TrainingJobStatus> {
  const res = await fetch(`${baseUrl}/training/stop`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return (await res.json()) as TrainingJobStatus;
}

export async function importCheckpointFile(
  baseUrl: string,
  payload: {
    source_checkpoint: string;
    tokenizer_source?: string | null;
    confirm_overwrite: boolean;
  },
): Promise<{
  imported: boolean;
  target_checkpoint: string;
  global_step?: number | null;
  training_status?: string | null;
  training_stage?: string | null;
  checkpoint_version?: string | null;
}> {
  const res = await fetch(`${baseUrl}/training/import`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return (await res.json()) as {
    imported: boolean;
    target_checkpoint: string;
    global_step?: number | null;
    training_status?: string | null;
    training_stage?: string | null;
    checkpoint_version?: string | null;
  };
}

export async function fetchEvaluationReport(baseUrl: string): Promise<EvaluationReport> {
  const res = await fetch(`${baseUrl}/evaluation`);
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return (await res.json()) as EvaluationReport;
}

export async function runEvaluationSuite(baseUrl: string): Promise<EvaluationReport> {
  const res = await fetch(`${baseUrl}/evaluation/run`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return (await res.json()) as EvaluationReport;
}

