from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

STAN_ROOT = Path(__file__).resolve().parents[1]
if str(STAN_ROOT) not in sys.path:
    sys.path.insert(0, str(STAN_ROOT))

from backend.routes import router
from identity.profile import load_or_create_identity
from inference.load_model import STANModelRegistry
from memory.store import MemoryStore
from reasoning.engine import STANReasoningEngine
from training.manager import TrainingJobManager


def create_app(
    model_config_path: Optional[str | Path] = None,
    checkpoint_dir: Optional[str | Path] = None,
    tokenizer_path: Optional[str | Path] = None,
    device_override: Optional[str] = None,
    identity_path: Optional[str | Path] = None,
    memory_db_path: Optional[str | Path] = None,
) -> FastAPI:
    registry = STANModelRegistry(
        model_config_path=model_config_path,
        checkpoint_dir=checkpoint_dir,
        tokenizer_path=tokenizer_path,
        device_override=device_override,
    )

    resolved_identity_path = Path(
        identity_path
        or os.environ.get("STAN_IDENTITY_PATH", str(STAN_ROOT / "memory_store" / "identity.json"))
    )
    resolved_memory_db = Path(
        memory_db_path
        or os.environ.get("STAN_MEMORY_DB", str(STAN_ROOT / "memory_store" / "stan_memory.db"))
    )

    identity = load_or_create_identity(resolved_identity_path)
    memory_store = MemoryStore(resolved_memory_db)
    reasoning_engine = STANReasoningEngine(identity=identity, memory=memory_store)
    training_manager = TrainingJobManager(registry=registry)

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI):
        app_instance.state.registry = registry
        app_instance.state.identity = identity
        app_instance.state.identity_path = resolved_identity_path
        app_instance.state.memory_store = memory_store
        app_instance.state.reasoning_engine = reasoning_engine
        app_instance.state.training_manager = training_manager
        # Pre-load STAN AI weights & tokenizer once on startup
        registry.get_state()
        yield

    application = FastAPI(
        title="STAN AI Inference API",
        description="Independent PyTorch Transformer, persistent memory, identity, and reasoning service for STAN AI.",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.state.registry = registry
    application.state.identity = identity
    application.state.identity_path = resolved_identity_path
    application.state.memory_store = memory_store
    application.state.reasoning_engine = reasoning_engine
    application.state.training_manager = training_manager

    allowed_origins = os.environ.get("STAN_CORS_ORIGINS", "*").split(",")
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in allowed_origins if o.strip()],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(router)
    return application


app = create_app()
