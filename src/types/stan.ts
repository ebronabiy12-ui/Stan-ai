export type TrainingStatus =
  | 'untrained'
  | 'training'
  | 'trained'
  | 'instruction-tuned'
  | 'smoke_test';

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

export interface MetricsInfo {
  training_status: TrainingStatus | string;
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

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  createdAt: string;
  meta?: {
    prompt_tokens?: number;
    generated_tokens?: number;
    elapsed_ms?: number;
    tokens_per_second?: number;
    stopped_reason?: string;
    checkpoint_version?: string | null;
  };
}
