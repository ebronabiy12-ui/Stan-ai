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
    training_stage: str = "untrained"
    current_learning_rate: Optional[float] = None
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
    gpu_info: dict[str, Any] = Field(default_factory=dict)
    checkpoint_history: list[dict[str, Any]] = Field(default_factory=list)
    active_job: dict[str, Any] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)


class PersonalityConfigSchema(BaseModel):
    directness: float = Field(0.80, ge=0.0, le=1.0)
    independence: float = Field(0.90, ge=0.0, le=1.0)
    skepticism: float = Field(0.75, ge=0.0, le=1.0)
    humor: float = Field(0.45, ge=0.0, le=1.0)
    warmth: float = Field(0.55, ge=0.0, le=1.0)
    verbosity: float = Field(0.45, ge=0.0, le=1.0)
    allow_disagreement: bool = True
    allow_bluntness: bool = True
    challenge_bad_assumptions: bool = True


class IdentityResponse(BaseModel):
    name: str
    owner_name: str
    owner_relationship: str
    project_name: str
    system_nature: str
    core_principles: list[str]
    personality: PersonalityConfigSchema
    updated_at: str


class IdentityUpdateRequest(BaseModel):
    personality: Optional[PersonalityConfigSchema] = None
    core_principles: Optional[list[str]] = None


class MemoryRecordSchema(BaseModel):
    id: str
    category: str
    subject: str
    content: str
    source: str
    confidence: float
    pinned: bool
    created_at: str
    updated_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryCreateRequest(BaseModel):
    category: str
    subject: str = Field(..., min_length=1, max_length=256)
    content: str = Field(..., min_length=1, max_length=8192)
    source: str = "owner_entry"
    confidence: float = Field(0.95, ge=0.0, le=1.0)
    pinned: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryUpdateRequest(BaseModel):
    category: Optional[str] = None
    subject: Optional[str] = Field(None, min_length=1, max_length=256)
    content: Optional[str] = Field(None, min_length=1, max_length=8192)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    pinned: Optional[bool] = None
    metadata: Optional[dict[str, Any]] = None


class MemoryListResponse(BaseModel):
    categories: list[str]
    counts: dict[str, int]
    records: list[MemoryRecordSchema]


class MemorySearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1024)
    categories: Optional[list[str]] = None
    top_k: int = Field(8, ge=1, le=50)


class ReasonRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8192)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_k: int = Field(40, ge=0, le=8192)
    top_p: float = Field(0.9, gt=0.0, le=1.0)
    max_new_tokens: int = Field(128, ge=1, le=512)
    repetition_penalty: float = Field(1.15, ge=1.0, le=2.0)
    seed: Optional[int] = Field(None, ge=0)
    persist_turn: bool = True

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Message must not be empty or whitespace only.")
        return v


class ReasonResponse(BaseModel):
    user_message: str
    response: str
    stance: str
    register: str
    detected_signals: list[str]
    reasoning_steps: list[str]
    facts: list[str]
    opinions: list[str]
    uncertainties: list[str]
    contradictions_found: list[str]
    retrieved_memories: list[dict[str, Any]]
    memory_updates: list[dict[str, Any]]
    provenance: dict[str, Any]


class TrainingStartRequest(BaseModel):
    stage: str = Field(..., pattern="^(smoke_test|pretraining|instruction_tuning)$")
    resume: bool = False
    overwrite_existing: bool = False
    max_steps: Optional[int] = Field(None, ge=1, le=500000)


class CheckpointImportRequest(BaseModel):
    source_checkpoint: str = Field(..., min_length=1, max_length=1024)
    tokenizer_source: Optional[str] = Field(None, max_length=1024)
    confirm_overwrite: bool = False
