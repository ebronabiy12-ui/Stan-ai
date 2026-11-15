from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8192, description="Input prompt text.")
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_k: int = Field(40, ge=0, le=8192)
    top_p: float = Field(0.9, gt=0.0, le=1.0)
    max_new_tokens: int = Field(128, ge=1, le=512)
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0)
    seed: Optional[int] = Field(None, ge=0)
    stop_on_eos: bool = True

    @field_validator("prompt")
    @classmethod
    def prompt_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Prompt must not be empty or whitespace only.")
        return v


class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str
    full_text: str
    prompt_tokens: int
    generated_tokens: int
    total_tokens: int
    stopped_reason: str
    elapsed_ms: float
    tokens_per_second: float
    model_name: str
    model_version: str
    training_status: str
    checkpoint_version: Optional[str]


class TokenizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=16384)
    add_bos: bool = False
    add_eos: bool = False


class TokenizeResponse(BaseModel):
    text: str
    token_ids: list[int]
    pieces: list[str]
    token_count: int
    vocab_size: int
    roundtrip_text: str
    roundtrip_exact: bool


class HealthResponse(BaseModel):
    status: str
    service: str
    device: str
    cuda_available: bool
    checkpoint_loaded: bool
    tokenizer_loaded: bool
    training_status: str
    production_chat_enabled: bool
    warning: Optional[str]


class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    architecture: str
    training_status: str
    production_chat_enabled: bool
    checkpoint_loaded: bool
    checkpoint_version: Optional[str]
    tokenizer_loaded: bool
    device: str
    parameter_count: int
    parameter_breakdown: dict[str, int]
    n_layers: int
    d_model: int
    n_heads: int
    d_head: int
    d_ff: int
    max_seq_len: int
    vocab_size: int
    activation: str
    norm_type: str
    tie_embeddings: bool
    warning: Optional[str]


class TrainingMetricPoint(BaseModel):
    step: int
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    learning_rate: Optional[float] = None


class MetricsResponse(BaseModel):
    training_status: str
    global_step: int
    max_steps: int
    progress_percent: float
    dataset_tokens: int
    latest_train_loss: Optional[float]
    latest_val_loss: Optional[float]
    checkpoint_version: Optional[str]
    checkpoint_path: Optional[str]
    saved_at: Optional[str]
    history: list[TrainingMetricPoint]
    warning: Optional[str]
    extra: dict[str, Any] = Field(default_factory=dict)
