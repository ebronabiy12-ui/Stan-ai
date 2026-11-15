from __future__ import annotations

import json
from typing import Iterator

import torch
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from backend.schemas import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    MetricsResponse,
    ModelInfoResponse,
    TokenizeRequest,
    TokenizeResponse,
    TrainingMetricPoint,
)
from inference.generate import GenerationConfig, generate_text, stream_generate_text
from inference.load_model import STANModelRegistry

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
    state = _get_registry(request).get_state()
    hist_points = [TrainingMetricPoint(**pt) for pt in state.training_history]
    latest_train = next(
        (pt.train_loss for pt in reversed(hist_points) if pt.train_loss is not None),
        None,
    )
    latest_val = next(
        (pt.val_loss for pt in reversed(hist_points) if pt.val_loss is not None),
        None,
    )
    progress = (
        min(100.0, round((state.global_step / state.max_steps) * 100.0, 2))
        if state.max_steps > 0
        else 0.0
    )
    return MetricsResponse(
        training_status=state.training_status,
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
