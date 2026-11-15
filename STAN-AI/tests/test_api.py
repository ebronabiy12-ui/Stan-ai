from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.main import create_app
from model import STANConfig, STANTransformer
from tokenizer import STANTokenizer, train_bpe_tokenizer
from training.checkpoint import save_checkpoint


def test_api_untrained_blocks_production_chat(tmp_path: Path) -> None:
    cfg = STANConfig(n_layers=2, d_model=64, n_heads=4, d_head=16, d_ff=128, max_seq_len=32, vocab_size=512)
    cfg_path = tmp_path / "model.yaml"
    cfg.to_yaml(cfg_path)

    app = create_app(model_config_path=cfg_path, checkpoint_dir=tmp_path / "no_ckpts")
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        hdata = health.json()
        assert hdata["status"] == "ok"
        assert hdata["checkpoint_loaded"] is False
        assert hdata["training_status"] == "untrained"
        assert hdata["production_chat_enabled"] is False

        info = client.get("/model/info")
        assert info.status_code == 200
        assert info.json()["parameter_count"] > 0

        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        assert metrics.json()["history"] == []

        gen = client.post("/generate", json={"prompt": "Hello STAN"})
        assert gen.status_code == 503


def test_api_trained_endpoints_and_validation(tmp_path: Path) -> None:
    corpus = tmp_path / "train.txt"
    corpus.write_text(
        "\n".join(
            [
                "STAN AI is an independent transformer trained from scratch in PyTorch with custom BPE.",
                "FastAPI serves health, model metadata, tokenization, metrics, and text generation endpoints.",
                "Validation loss is measured on held-out documents strictly excluded from tokenizer training.",
                "Resumable training checkpoints enable recovery after session restarts on cloud GPUs.",
                "Deterministic random seeds allow reproducible autoregressive sampling across runs.",
                "Every transformer block applies pre-norm residual connections and causal attention masking.",
            ]
            * 40
        ),
        encoding="utf-8",
    )
    ckpt_dir = tmp_path / "ckpts"
    tok_model, _ = train_bpe_tokenizer(
        train_files=[corpus],
        output_dir=ckpt_dir,
        vocab_size=512,
    )
    cfg = STANConfig(n_layers=2, d_model=64, n_heads=4, d_head=16, d_ff=128, max_seq_len=64, vocab_size=512)
    model = STANTransformer(cfg, log_params=False)
    save_checkpoint(
        checkpoint_dir=ckpt_dir,
        model=model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        global_step=25,
        model_config=cfg,
        training_config={"max_steps": 25},
        training_history=[
            {"step": 10, "train_loss": 4.5, "val_loss": 4.6, "learning_rate": 3e-4},
            {"step": 25, "train_loss": 2.1, "val_loss": 2.3, "learning_rate": 1e-4},
        ],
        training_status="trained",
        tokenizer_path=tok_model,
        dataset_tokens=12000,
    )

    app = create_app(
        model_config_path=ckpt_dir / "model_config.yaml",
        checkpoint_dir=ckpt_dir,
    )
    with TestClient(app) as client:
        tok_resp = client.post("/tokenize", json={"text": "STAN AI transformer"})
        assert tok_resp.status_code == 200
        assert tok_resp.json()["roundtrip_exact"] is True

        # Invalid request: blank prompt
        bad_req = client.post("/generate", json={"prompt": "   "})
        assert bad_req.status_code == 422

        # Invalid request: temperature out of range
        bad_temp = client.post("/generate", json={"prompt": "STAN", "temperature": 9.9})
        assert bad_temp.status_code == 422

        gen_resp = client.post(
            "/generate",
            json={
                "prompt": "STAN AI",
                "temperature": 0.7,
                "top_k": 20,
                "top_p": 0.9,
                "max_new_tokens": 12,
                "seed": 42,
            },
        )
        assert gen_resp.status_code == 200
        gdata = gen_resp.json()
        assert gdata["training_status"] == "trained"
        assert gdata["checkpoint_version"] == "step-0000025"
        assert gdata["generated_tokens"] <= 12

        metrics_resp = client.get("/metrics")
        assert metrics_resp.status_code == 200
        mdata = metrics_resp.json()
        assert mdata["global_step"] == 25
        assert mdata["progress_percent"] == 100.0
        assert len(mdata["history"]) == 2
