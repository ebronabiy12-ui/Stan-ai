import type {
  GenerationSettings,
  MemoryCategory,
  PersonalityConfig,
  TrainingMetricPoint,
} from '../types/stan';

export const DEFAULT_GENERATION_SETTINGS: GenerationSettings = {
  temperature: 0.8,
  top_k: 40,
  top_p: 0.9,
  max_new_tokens: 128,
  repetition_penalty: 1.1,
  seed: null,
  stop_on_eos: true,
  apiBaseUrl:
    (import.meta as unknown as { env?: Record<string, string | undefined> }).env
      ?.VITE_STAN_API_URL || '/api/stan',
};

export const MEMORY_CATEGORIES: MemoryCategory[] = [
  'identity_personality',
  'people_relationships',
  'user_facts',
  'conversation_history',
  'project_knowledge',
  'learned_preferences',
  'decisions_experiences',
];

export const MEMORY_CATEGORY_META: Record<
  MemoryCategory,
  { label: string; description: string }
> = {
  identity_personality: {
    label: 'Identity & Personality',
    description: 'Core system identity, non-consciousness boundaries, and temperament principles.',
  },
  people_relationships: {
    label: 'People & Relationships',
    description: 'Owner, family context (Stan, Kalda), and verified relational facts.',
  },
  user_facts: {
    label: 'User-Provided Facts',
    description: 'Explicit facts shared by Stan or corrected across conversations.',
  },
  conversation_history: {
    label: 'Conversation History',
    description: 'Durable episodic turns with stances, registers, and signals.',
  },
  project_knowledge: {
    label: 'Project Knowledge',
    description: 'STAN AI architecture, Kalda project details, and engineering constraints.',
  },
  learned_preferences: {
    label: 'Learned Preferences',
    description: 'Communication, formatting, and collaboration preferences over time.',
  },
  decisions_experiences: {
    label: 'Decisions & Experiences',
    description: 'Recorded conclusions, reassessments, and verified reasoning outcomes.',
  },
};

export const DEFAULT_PERSONALITY_CONFIG: PersonalityConfig = {
  directness: 0.8,
  independence: 0.9,
  skepticism: 0.75,
  humor: 0.45,
  warmth: 0.55,
  verbosity: 0.45,
  allow_disagreement: true,
  allow_bluntness: true,
  challenge_bad_assumptions: true,
};

const PRODUCTION_READY_STATES = new Set(['trained', 'instruction-tuned']);

export function isProductionChatReady(params: {
  checkpoint_loaded: boolean;
  tokenizer_loaded: boolean;
  training_status: string;
  production_chat_enabled?: boolean;
}): boolean {
  if (typeof params.production_chat_enabled === 'boolean') {
    return params.production_chat_enabled;
  }
  return (
    params.checkpoint_loaded &&
    params.tokenizer_loaded &&
    PRODUCTION_READY_STATES.has(params.training_status)
  );
}

export function formatTrainingStatusLabel(status: string): string {
  switch (status) {
    case 'untrained':
      return 'Untrained (No Checkpoint)';
    case 'training':
      return 'Training In Progress';
    case 'trained':
      return 'Trained';
    case 'instruction-tuned':
      return 'Instruction-Tuned';
    case 'smoke_test':
      return 'Smoke Test Only';
    default:
      return status || 'Unknown';
  }
}

export function formatStanceLabel(stance: string): string {
  switch (stance) {
    case 'answer':
      return 'Direct Answer';
    case 'agree':
      return 'Reasoned Agreement';
    case 'disagree':
      return 'Disagreement';
    case 'challenge':
      return 'Challenge Assumption';
    case 'uncertain':
      return 'Epistemic Uncertainty';
    case 'reassess':
      return 'Self-Correction';
    case 'boundary':
      return 'Boundary / Pushback';
    default:
      return stance || 'Unknown';
  }
}

export function formatTrainingStageLabel(stage?: string | null): string {
  switch (stage) {
    case 'pretraining':
      return 'Stage 1: English Pretraining';
    case 'instruction_tuning':
      return 'Stage 2: Instruction Tuning (SFT)';
    case 'smoke_test':
      return 'Smoke Test (Synthetic Verification)';
    case 'untrained':
    case null:
    case undefined:
      return 'Untrained (Random / Uninitialized)';
    default:
      return stage;
  }
}

export function formatLearningRate(lr?: number | null): string {
  if (typeof lr !== 'number' || !Number.isFinite(lr) || lr <= 0) return '—';
  return lr.toExponential(2);
}

export function formatRegisterLabel(register: string): string {
  switch (register) {
    case 'direct':
      return 'Direct';
    case 'serious':
      return 'Serious';
    case 'gentle':
      return 'Gentle';
    case 'blunt':
      return 'Blunt';
    case 'humorous':
      return 'Dry Humor';
    default:
      return register || 'Direct';
  }
}

export function formatParamCount(count: number): string {
  if (!Number.isFinite(count) || count < 0) return '0';
  const millions = (count / 1_000_000).toFixed(2);
  return `${count.toLocaleString('en-US')} (${millions}M)`;
}

export function formatTokenCount(count: number): string {
  if (!Number.isFinite(count) || count <= 0) return '0';
  if (count >= 1_000_000) {
    return `${(count / 1_000_000).toFixed(2)}M`;
  }
  if (count >= 1_000) {
    return `${(count / 1_000).toFixed(1)}K`;
  }
  return count.toLocaleString('en-US');
}

export function clampGenerationSettings(input: Partial<GenerationSettings>): GenerationSettings {
  const clamp = (val: number, min: number, max: number) => Math.min(max, Math.max(min, val));
  const num = (val: unknown, fallback: number) =>
    typeof val === 'number' && Number.isFinite(val) ? val : fallback;

  const rawSeed = input.seed;
  const validSeed =
    typeof rawSeed === 'number' && Number.isInteger(rawSeed) && rawSeed >= 0 ? rawSeed : null;

  const baseUrl =
    typeof input.apiBaseUrl === 'string' && input.apiBaseUrl.trim().length > 0
      ? input.apiBaseUrl.trim().replace(/\/+$/, '')
      : DEFAULT_GENERATION_SETTINGS.apiBaseUrl;

  return {
    temperature: Number(
      clamp(num(input.temperature, DEFAULT_GENERATION_SETTINGS.temperature), 0, 2).toFixed(2),
    ),
    top_k: Math.round(clamp(num(input.top_k, DEFAULT_GENERATION_SETTINGS.top_k), 0, 8192)),
    top_p: Number(clamp(num(input.top_p, DEFAULT_GENERATION_SETTINGS.top_p), 0.05, 1).toFixed(2)),
    max_new_tokens: Math.round(
      clamp(num(input.max_new_tokens, DEFAULT_GENERATION_SETTINGS.max_new_tokens), 1, 512),
    ),
    repetition_penalty: Number(
      clamp(
        num(input.repetition_penalty, DEFAULT_GENERATION_SETTINGS.repetition_penalty),
        1,
        2,
      ).toFixed(2),
    ),
    seed: validSeed,
    stop_on_eos:
      typeof input.stop_on_eos === 'boolean'
        ? input.stop_on_eos
        : DEFAULT_GENERATION_SETTINGS.stop_on_eos,
    apiBaseUrl: baseUrl,
  };
}

export function clampPersonalityConfig(input: Partial<PersonalityConfig>): PersonalityConfig {
  const clamp01 = (val: unknown, fallback: number) => {
    const n = typeof val === 'number' && Number.isFinite(val) ? val : fallback;
    return Number(Math.min(1, Math.max(0, n)).toFixed(2));
  };
  const boolVal = (val: unknown, fallback: boolean) =>
    typeof val === 'boolean' ? val : fallback;

  return {
    directness: clamp01(input.directness, DEFAULT_PERSONALITY_CONFIG.directness),
    independence: clamp01(input.independence, DEFAULT_PERSONALITY_CONFIG.independence),
    skepticism: clamp01(input.skepticism, DEFAULT_PERSONALITY_CONFIG.skepticism),
    humor: clamp01(input.humor, DEFAULT_PERSONALITY_CONFIG.humor),
    warmth: clamp01(input.warmth, DEFAULT_PERSONALITY_CONFIG.warmth),
    verbosity: clamp01(input.verbosity, DEFAULT_PERSONALITY_CONFIG.verbosity),
    allow_disagreement: boolVal(
      input.allow_disagreement,
      DEFAULT_PERSONALITY_CONFIG.allow_disagreement,
    ),
    allow_bluntness: boolVal(input.allow_bluntness, DEFAULT_PERSONALITY_CONFIG.allow_bluntness),
    challenge_bad_assumptions: boolVal(
      input.challenge_bad_assumptions,
      DEFAULT_PERSONALITY_CONFIG.challenge_bad_assumptions,
    ),
  };
}

export function computeLossDomain(
  history: TrainingMetricPoint[],
  metric: 'train_loss' | 'val_loss',
): {
  points: Array<{ step: number; value: number }>;
  minVal: number;
  maxVal: number;
  minStep: number;
  maxStep: number;
} {
  const points = history
    .filter((pt) => typeof pt[metric] === 'number' && Number.isFinite(pt[metric] as number))
    .map((pt) => ({ step: pt.step, value: pt[metric] as number }));

  if (points.length === 0) {
    return { points: [], minVal: 0, maxVal: 1, minStep: 0, maxStep: 1 };
  }

  let minVal = points[0].value;
  let maxVal = points[0].value;
  let minStep = points[0].step;
  let maxStep = points[0].step;

  for (const p of points) {
    if (p.value < minVal) minVal = p.value;
    if (p.value > maxVal) maxVal = p.value;
    if (p.step < minStep) minStep = p.step;
    if (p.step > maxStep) maxStep = p.step;
  }

  if (Math.abs(maxVal - minVal) < 1e-6) {
    minVal = Math.max(0, minVal - 0.5);
    maxVal = maxVal + 0.5;
  } else {
    const pad = (maxVal - minVal) * 0.1;
    minVal = Math.max(0, minVal - pad);
    maxVal = maxVal + pad;
  }

  if (minStep === maxStep) {
    maxStep = minStep + 1;
  }

  return { points, minVal, maxVal, minStep, maxStep };
}
