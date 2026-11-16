export type TrainingStatus =
  | 'untrained'
  | 'training'
  | 'trained'
  | 'instruction-tuned'
  | 'smoke_test';

export type MemoryCategory =
  | 'identity_personality'
  | 'people_relationships'
  | 'user_facts'
  | 'conversation_history'
  | 'project_knowledge'
  | 'learned_preferences'
  | 'decisions_experiences';

export interface HealthInfo {
  status: string;
  service: string;
  device: string;
  cuda_available: boolean;
  checkpoint_loaded: boolean;
  tokenizer_loaded: boolean;
  training_status: TrainingStatus | string;
  production_chat_enabled: boolean;
  warning: string | null;
}

export interface ParameterBreakdown {
  total_parameters: number;
  trainable_parameters: number;
  token_embedding_parameters: number;
  positional_embedding_parameters: number;
  transformer_blocks_parameters: number;
  final_norm_parameters: number;
  lm_head_unique_parameters: number;
}

export interface ModelInfo {
  model_name: string;
  version: string;
  architecture: string;
  training_status: TrainingStatus | string;
  production_chat_enabled: boolean;
  checkpoint_loaded: boolean;
  checkpoint_version: string | null;
  tokenizer_loaded: boolean;
  device: string;
  parameter_count: number;
  parameter_breakdown: ParameterBreakdown;
  n_layers: number;
  d_model: number;
  n_heads: number;
  d_head: number;
  d_ff: number;
  max_seq_len: number;
  vocab_size: number;
  activation: string;
  norm_type: string;
  tie_embeddings: boolean;
  warning: string | null;
}

export interface TrainingMetricPoint {
  step: number;
  train_loss: number | null;
  val_loss: number | null;
  learning_rate: number | null;
}

export interface GpuDeviceInfo {
  index: number;
  name: string;
  vram_gb: number;
}

export interface HardwareInfo {
  cuda_available: boolean;
  gpu_count: number;
  gpus: GpuDeviceInfo[];
  active_device: string;
  cpu_cores: number;
  ram_gb: number | null;
  pytorch_version: string;
  python_version: string;
  recommended_training_target: string;
}

export interface CheckpointRecord {
  filename: string;
  path: string;
  size_mb: number;
  global_step: number | null;
  training_status: string | null;
  training_stage: string | null;
  checkpoint_version: string;
  saved_at: string;
  error?: string;
}

export interface TrainingJobStatus {
  status: 'idle' | 'running' | 'stopping' | 'completed' | 'stopped' | 'failed' | string;
  stage?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
  error?: string | null;
  summary?: Record<string, unknown> | null;
}

export interface EvaluationCategoryResult {
  id: string;
  label: string;
  status: 'evaluated' | 'not_evaluated_untrained' | string;
  score: number | null;
  measured_details: string;
  samples: Array<Record<string, unknown>>;
}

export interface EvaluationReport {
  evaluated_at: string | null;
  model_weights_status: string;
  training_stage: string;
  checkpoint_version: string | null;
  neural_weights_evaluated: boolean;
  evaluated_category_count: number;
  total_category_count: number;
  average_evaluated_score: number | null;
  categories: EvaluationCategoryResult[];
}

export interface MetricsInfo {
  training_status: TrainingStatus | string;
  training_stage?: string;
  current_learning_rate?: number | null;
  global_step: number;
  max_steps: number;
  progress_percent: number;
  dataset_tokens: number;
  latest_train_loss: number | null;
  latest_val_loss: number | null;
  checkpoint_version: string | null;
  checkpoint_path: string | null;
  saved_at: string | null;
  history: TrainingMetricPoint[];
  warning: string | null;
  gpu_info?: HardwareInfo;
  checkpoint_history?: CheckpointRecord[];
  active_job?: TrainingJobStatus;
}

export interface TokenizeResult {
  text: string;
  token_ids: number[];
  pieces: string[];
  token_count: number;
  vocab_size: number;
  roundtrip_text: string;
  roundtrip_exact: boolean;
}

export interface GenerateResult {
  prompt: string;
  generated_text: string;
  full_text: string;
  prompt_tokens: number;
  generated_tokens: number;
  total_tokens: number;
  stopped_reason: string;
  elapsed_ms: number;
  tokens_per_second: number;
  model_name: string;
  model_version: string;
  training_status: string;
  checkpoint_version: string | null;
}

export interface GenerationSettings {
  temperature: number;
  top_k: number;
  top_p: number;
  max_new_tokens: number;
  repetition_penalty: number;
  seed: number | null;
  stop_on_eos: boolean;
  apiBaseUrl: string;
}

export interface PersonalityConfig {
  directness: number;
  independence: number;
  skepticism: number;
  humor: number;
  warmth: number;
  verbosity: number;
  allow_disagreement: boolean;
  allow_bluntness: boolean;
  challenge_bad_assumptions: boolean;
}

export interface IdentityProfile {
  name: string;
  owner_name: string;
  owner_relationship: string;
  project_name: string;
  system_nature: string;
  core_principles: string[];
  personality: PersonalityConfig;
  updated_at: string;
}

export interface MemoryRecord {
  id: string;
  category: MemoryCategory | string;
  subject: string;
  content: string;
  source: string;
  confidence: number;
  pinned: boolean;
  created_at: string;
  updated_at: string;
  metadata: Record<string, unknown>;
}

export interface MemoryListResponse {
  categories: MemoryCategory[];
  counts: Record<string, number>;
  records: MemoryRecord[];
}

export interface ReasonProvenance {
  identity_active?: boolean;
  memory_store_active?: boolean;
  reasoning_engine_active?: boolean;
  model_weights_status?: string;
  neural_generation_used?: boolean;
  neural_meta?: {
    prompt_tokens?: number;
    generated_tokens?: number;
    elapsed_ms?: number;
    tokens_per_second?: number;
    stopped_reason?: string;
    checkpoint_version?: string | null;
  } | null;
  conscious?: boolean;
}

export interface ReasonResponse {
  user_message: string;
  response: string;
  stance: string;
  register: string;
  detected_signals: string[];
  reasoning_steps: string[];
  facts: string[];
  opinions: string[];
  uncertainties: string[];
  contradictions_found: string[];
  retrieved_memories: Array<{
    id: string;
    category: string;
    subject: string;
    content: string;
    pinned?: boolean;
  }>;
  memory_updates: Array<{
    action?: string;
    id?: string;
    category?: string;
    subject?: string;
    record?: MemoryRecord;
  }>;
  provenance: ReasonProvenance;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  createdAt: string;
  mode?: 'cognitive' | 'neural';
  reasoning?: ReasonResponse;
  meta?: {
    prompt_tokens?: number;
    generated_tokens?: number;
    elapsed_ms?: number;
    tokens_per_second?: number;
    stopped_reason?: string;
    checkpoint_version?: string | null;
  };
}
