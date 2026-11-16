from __future__ import annotations

import json
from typing import Iterator, Optional

import torch
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from backend.schemas import (
    CheckpointImportRequest,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    IdentityResponse,
    IdentityUpdateRequest,
    MemoryCreateRequest,
    MemoryListResponse,
    MemoryRecordSchema,
    MemorySearchRequest,
    MemoryUpdateRequest,
    MetricsResponse,
    ModelInfoResponse,
    PersonalityConfigSchema,
    ReasonRequest,
    ReasonResponse,
    TokenizeRequest,
    TokenizeResponse,
    TrainingMetricPoint,
    TrainingStartRequest,
)
from evaluation.suite import load_latest_evaluation_report, run_evaluation_suite
from identity.profile import PersonalityConfig, save_identity
from inference.generate import GenerationConfig, generate_text, stream_generate_text
from inference.load_model import STANModelRegistry
from memory.store import MEMORY_CATEGORIES
from training.checkpoint import import_checkpoint, list_checkpoints
from training.manager import get_hardware_info

router = APIRouter()


def _get_registry(request: Request) -> STANModelRegistry:
    return request.app.state.registry


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    state = _get_registry(request).get_state()
    return HealthResponse(
        status="ok",
        service="STAN AI Inference Service",
        device=str(state.device),
        cuda_available=torch.cuda.is_available(),
        checkpoint_loaded=state.checkpoint_loaded,
        tokenizer_loaded=state.tokenizer is not None,
        training_status=state.training_status,
        production_chat_enabled=state.production_chat_enabled,
        warning=state.warning,
    )


@router.get("/model/info", response_model=ModelInfoResponse)
def model_info(request: Request) -> ModelInfoResponse:
    state = _get_registry(request).get_state()
    cfg = state.model_config
    return ModelInfoResponse(
        model_name=cfg.model_name,
        version=cfg.version,
        architecture=cfg.architecture,
        training_status=state.training_status,
        production_chat_enabled=state.production_chat_enabled,
        checkpoint_loaded=state.checkpoint_loaded,
        checkpoint_version=state.checkpoint_version,
        tokenizer_loaded=state.tokenizer is not None,
        device=str(state.device),
        parameter_count=state.parameter_breakdown["trainable_parameters"],
        parameter_breakdown=state.parameter_breakdown,
        n_layers=cfg.n_layers,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_head=cfg.d_head,
        d_ff=cfg.d_ff,
        max_seq_len=cfg.max_seq_len,
        vocab_size=cfg.vocab_size,
        activation=cfg.activation,
        norm_type=cfg.norm_type,
        tie_embeddings=cfg.tie_embeddings,
        warning=state.warning,
    )


@router.get("/metrics", response_model=MetricsResponse)
def metrics(request: Request) -> MetricsResponse:
    registry = _get_registry(request)
    state = registry.get_state()
    manager = getattr(request.app.state, "training_manager", None)
    hist_points = [TrainingMetricPoint(**pt) for pt in state.training_history]
    latest_train = next(
        (pt.train_loss for pt in reversed(hist_points) if pt.train_loss is not None),
        None,
    )
    latest_val = next(
        (pt.val_loss for pt in reversed(hist_points) if pt.val_loss is not None),
        None,
    )
    current_lr = state.current_learning_rate
    if current_lr is None:
        current_lr = next(
            (pt.learning_rate for pt in reversed(hist_points) if pt.learning_rate is not None),
            None,
        )
    progress = (
        min(100.0, round((state.global_step / state.max_steps) * 100.0, 2))
        if state.max_steps > 0
        else 0.0
    )
    return MetricsResponse(
        training_status=state.training_status,
        training_stage=state.training_stage,
        current_learning_rate=current_lr,
        global_step=state.global_step,
        max_steps=state.max_steps,
        progress_percent=progress,
        dataset_tokens=state.dataset_tokens,
        latest_train_loss=latest_train,
        latest_val_loss=latest_val,
        checkpoint_version=state.checkpoint_version,
        checkpoint_path=state.checkpoint_path,
        saved_at=state.saved_at,
        history=hist_points,
        warning=state.warning,
        gpu_info=get_hardware_info(),
        checkpoint_history=list_checkpoints(registry.checkpoint_dir),
        active_job=manager.get_status() if manager is not None else {"status": "idle"},
    )


@router.post("/tokenize", response_model=TokenizeResponse)
def tokenize(payload: TokenizeRequest, request: Request) -> TokenizeResponse:
    state = _get_registry(request).get_state()
    if state.tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Custom SentencePiece tokenizer.model is not loaded. "
                "Train the tokenizer on the project corpus first using STAN-AI/tokenizer/train_tokenizer.py."
            ),
        )
    ids = state.tokenizer.encode(
        payload.text, add_bos=payload.add_bos, add_eos=payload.add_eos
    )
    pieces = state.tokenizer.encode_pieces(payload.text)
    roundtrip = state.tokenizer.decode(ids, skip_special_tokens=True)
    return TokenizeResponse(
        text=payload.text,
        token_ids=ids,
        pieces=pieces,
        token_count=len(ids),
        vocab_size=state.tokenizer.vocab_size,
        roundtrip_text=roundtrip,
        roundtrip_exact=(roundtrip == payload.text),
    )


def _validate_generation_ready(request: Request):
    state = _get_registry(request).get_state()
    if not state.checkpoint_loaded or state.model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No trained STAN AI checkpoint is loaded. "
                "Production text generation is disabled until trained weights are saved in checkpoints/."
            ),
        )
    if not state.production_chat_enabled:
        raise HTTPException(
            status_code=503,
            detail=state.warning
            or f"Model state '{state.training_status}' is not ready for production chat.",
        )
    if state.tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="STAN AI tokenizer.model is missing; cannot encode prompt.",
        )
    return state


@router.post("/generate", response_model=GenerateResponse)
def generate(payload: GenerateRequest, request: Request) -> GenerateResponse:
    state = _validate_generation_ready(request)
    assert state.model is not None and state.tokenizer is not None

    gen_cfg = GenerationConfig(
        temperature=payload.temperature,
        top_k=payload.top_k,
        top_p=payload.top_p,
        max_new_tokens=payload.max_new_tokens,
        repetition_penalty=payload.repetition_penalty,
        seed=payload.seed,
        stop_on_eos=payload.stop_on_eos,
    )
    res = generate_text(
        model=state.model,
        tokenizer=state.tokenizer,
        prompt=payload.prompt,
        config=gen_cfg,
        device=state.device,
    )
    return GenerateResponse(
        prompt=res.prompt,
        generated_text=res.generated_text,
        full_text=res.full_text,
        prompt_tokens=res.prompt_tokens,
        generated_tokens=res.generated_tokens,
        total_tokens=res.total_tokens,
        stopped_reason=res.stopped_reason,
        elapsed_ms=res.elapsed_ms,
        tokens_per_second=res.tokens_per_second,
        model_name=state.model_config.model_name,
        model_version=state.model_config.version,
        training_status=state.training_status,
        checkpoint_version=state.checkpoint_version,
    )


@router.post("/generate/stream")
def generate_stream(payload: GenerateRequest, request: Request) -> StreamingResponse:
    state = _validate_generation_ready(request)
    assert state.model is not None and state.tokenizer is not None

    gen_cfg = GenerationConfig(
        temperature=payload.temperature,
        top_k=payload.top_k,
        top_p=payload.top_p,
        max_new_tokens=payload.max_new_tokens,
        repetition_penalty=payload.repetition_penalty,
        seed=payload.seed,
        stop_on_eos=payload.stop_on_eos,
    )

    def event_stream() -> Iterator[str]:
        for chunk in stream_generate_text(
            model=state.model,
            tokenizer=state.tokenizer,
            prompt=payload.prompt,
            config=gen_cfg,
            device=state.device,
        ):
            yield f"data: {json.dumps(chunk)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Identity, Personality, Persistent Memory & Reasoning Endpoints
# ---------------------------------------------------------------------------


@router.get("/identity", response_model=IdentityResponse)
def get_identity(request: Request) -> IdentityResponse:
    profile = request.app.state.identity
    return IdentityResponse(
        name=profile.name,
        owner_name=profile.owner_name,
        owner_relationship=profile.owner_relationship,
        project_name=profile.project_name,
        system_nature=profile.system_nature,
        core_principles=profile.core_principles,
        personality=PersonalityConfigSchema(**profile.personality.__dict__),
        updated_at=profile.updated_at,
    )


@router.put("/identity", response_model=IdentityResponse)
def update_identity(payload: IdentityUpdateRequest, request: Request) -> IdentityResponse:
    profile = request.app.state.identity
    if payload.personality is not None:
        profile.personality = PersonalityConfig(**payload.personality.model_dump()).clamp()
    if payload.core_principles is not None:
        cleaned = [p.strip() for p in payload.core_principles if p.strip()]
        if cleaned:
            profile.core_principles = cleaned
    save_identity(profile, request.app.state.identity_path)
    return get_identity(request)


@router.get("/memory", response_model=MemoryListResponse)
def list_memories(
    request: Request,
    category: Optional[str] = Query(None),
    limit: int = Query(200, ge=1, le=500),
) -> MemoryListResponse:
    store = request.app.state.memory_store
    try:
        recs = store.list_memories(category=category, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return MemoryListResponse(
        categories=list(MEMORY_CATEGORIES),
        counts=store.category_counts(),
        records=[MemoryRecordSchema(**r.to_dict()) for r in recs],
    )


@router.post("/memory", response_model=MemoryRecordSchema)
def create_memory(payload: MemoryCreateRequest, request: Request) -> MemoryRecordSchema:
    store = request.app.state.memory_store
    try:
        rec = store.add_memory(
            category=payload.category,
            subject=payload.subject,
            content=payload.content,
            source=payload.source,
            confidence=payload.confidence,
            pinned=payload.pinned,
            metadata=payload.metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return MemoryRecordSchema(**rec.to_dict())


@router.patch("/memory/{record_id}", response_model=MemoryRecordSchema)
def patch_memory(
    record_id: str,
    payload: MemoryUpdateRequest,
    request: Request,
) -> MemoryRecordSchema:
    store = request.app.state.memory_store
    try:
        rec = store.update_memory(
            record_id=record_id,
            subject=payload.subject,
            content=payload.content,
            category=payload.category,
            confidence=payload.confidence,
            pinned=payload.pinned,
            metadata=payload.metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Memory record '{record_id}' not found.")
    return MemoryRecordSchema(**rec.to_dict())


@router.delete("/memory/{record_id}")
def delete_memory(record_id: str, request: Request) -> dict[str, object]:
    store = request.app.state.memory_store
    deleted = store.delete_memory(record_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Memory record '{record_id}' not found.")
    return {"deleted": True, "id": record_id}


@router.post("/memory/search", response_model=list[MemoryRecordSchema])
def search_memories(payload: MemorySearchRequest, request: Request) -> list[MemoryRecordSchema]:
    store = request.app.state.memory_store
    try:
        recs = store.search_memories(
            query=payload.query,
            categories=payload.categories,
            top_k=payload.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return [MemoryRecordSchema(**r.to_dict()) for r in recs]


@router.post("/reason", response_model=ReasonResponse)
def reason(payload: ReasonRequest, request: Request) -> ReasonResponse:
    state = _get_registry(request).get_state()
    engine = request.app.state.reasoning_engine
    gen_cfg = GenerationConfig(
        temperature=payload.temperature,
        top_k=payload.top_k,
        top_p=payload.top_p,
        max_new_tokens=payload.max_new_tokens,
        repetition_penalty=payload.repetition_penalty,
        seed=payload.seed,
    )
    result = engine.process(
        user_message=payload.message,
        model_state=state,
        generation_config=gen_cfg,
        persist_turn=payload.persist_turn,
    )
    return ReasonResponse(**result.to_dict())


# ---------------------------------------------------------------------------
# Phase 2: Training Infrastructure, Checkpoint Import & 9-Dimension Evaluation
# ---------------------------------------------------------------------------


@router.get("/training/status")
def training_status(request: Request) -> dict[str, object]:
    registry = _get_registry(request)
    state = registry.get_state()
    manager = getattr(request.app.state, "training_manager", None)
    return {
        "training_status": state.training_status,
        "training_stage": state.training_stage,
        "hardware": get_hardware_info(),
        "job": manager.get_status() if manager is not None else {"status": "idle"},
        "checkpoint_history": list_checkpoints(registry.checkpoint_dir),
    }


@router.post("/training/start")
def training_start(payload: TrainingStartRequest, request: Request) -> dict[str, object]:
    manager = getattr(request.app.state, "training_manager", None)
    if manager is None:
        raise HTTPException(status_code=503, detail="Training manager is not initialized.")
    try:
        return manager.start_job(
            stage=payload.stage,
            resume=payload.resume,
            overwrite_existing=payload.overwrite_existing,
            max_steps=payload.max_steps,
        )
    except (ValueError, RuntimeError, FileExistsError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/training/stop")
def training_stop(request: Request) -> dict[str, object]:
    manager = getattr(request.app.state, "training_manager", None)
    if manager is None:
        raise HTTPException(status_code=503, detail="Training manager is not initialized.")
    return manager.stop_job()


@router.post("/training/import")
def training_import_checkpoint(
    payload: CheckpointImportRequest,
    request: Request,
) -> dict[str, object]:
    registry = _get_registry(request)
    try:
        res = import_checkpoint(
            source_checkpoint=payload.source_checkpoint,
            target_checkpoint_dir=registry.checkpoint_dir,
            tokenizer_source=payload.tokenizer_source,
            confirm_overwrite=payload.confirm_overwrite,
        )
        registry.reload()
        return res
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/evaluation")
def get_evaluation_report(request: Request) -> dict[str, object]:
    registry = _get_registry(request)
    report = load_latest_evaluation_report(registry.checkpoint_dir)
    if report is not None:
        return report
    state = registry.get_state()
    return {
        "evaluated_at": None,
        "model_weights_status": state.training_status,
        "training_stage": state.training_stage,
        "checkpoint_version": state.checkpoint_version,
        "neural_weights_evaluated": False,
        "evaluated_category_count": 0,
        "total_category_count": 9,
        "average_evaluated_score": None,
        "categories": [],
    }


@router.post("/evaluation/run")
def execute_evaluation_suite(request: Request) -> dict[str, object]:
    registry = _get_registry(request)
    state = registry.get_state()
    return run_evaluation_suite(
        model_state=state,
        report_path=registry.checkpoint_dir / "eval_report.json",
    )
