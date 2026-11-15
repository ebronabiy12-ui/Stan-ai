import type {
  GenerateResult,
  GenerationSettings,
  HealthInfo,
  MetricsInfo,
  ModelInfo,
  TokenizeResult,
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
