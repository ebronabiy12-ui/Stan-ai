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
from inference.load_model import STANModelRegistry


def create_app(
    model_config_path: Optional[str | Path] = None,
    checkpoint_dir: Optional[str | Path] = None,
    tokenizer_path: Optional[str | Path] = None,
    device_override: Optional[str] = None,
) -> FastAPI:
    registry = STANModelRegistry(
        model_config_path=model_config_path,
        checkpoint_dir=checkpoint_dir,
        tokenizer_path=tokenizer_path,
        device_override=device_override,
    )

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI):
        app_instance.state.registry = registry
        # Pre-load STAN AI weights & tokenizer once on startup
        registry.get_state()
        yield

    application = FastAPI(
        title="STAN AI Inference API",
        description="Independent PyTorch Transformer inference service for STAN AI.",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.state.registry = registry

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
