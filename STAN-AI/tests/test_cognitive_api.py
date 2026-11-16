from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.main import create_app
from model import STANConfig, STANTransformer
from tokenizer import train_bpe_tokenizer
from training.checkpoint import save_checkpoint


def test_cognitive_endpoints_and_persistence_across_restart(tmp_path: Path) -> None:
    cfg = STANConfig(n_layers=2, d_model=64, n_heads=4, d_head=16, d_ff=128, max_seq_len=32, vocab_size=512)
    cfg_path = tmp_path / "model.yaml"
    cfg.to_yaml(cfg_path)
    id_path = tmp_path / "identity.json"
    db_path = tmp_path / "stan_memory.db"

    app1 = create_app(
        model_config_path=cfg_path,
        checkpoint_dir=tmp_path / "empty_ckpts",
        identity_path=id_path,
        memory_db_path=db_path,
    )
    with TestClient(app1) as client:
        # 1. Inspect identity & update owner personality
        ident = client.get("/identity")
        assert ident.status_code == 200
        assert ident.json()["name"] == "STAN AI"
        assert ident.json()["owner_name"] == "Stan"

        upd_ident = client.put(
            "/identity",
            json={
                "personality": {
                    "directness": 0.95,
                    "independence": 0.95,
                    "skepticism": 0.85,
                    "humor": 0.3,
                    "warmth": 0.4,
                    "verbosity": 0.4,
                    "allow_disagreement": True,
                    "allow_bluntness": True,
                    "challenge_bad_assumptions": True,
                }
            },
        )
        assert upd_ident.status_code == 200
        assert upd_ident.json()["personality"]["directness"] == 0.95

        # 2. Memory CRUD
        mem_create = client.post(
            "/memory",
            json={
                "category": "decisions_experiences",
                "subject": "Tokenizer target",
                "content": "Chose 8192 BPE vocabulary for balanced sequence efficiency.",
            },
        )
        assert mem_create.status_code == 200
        created_id = mem_create.json()["id"]

        # 3. Cognitive reasoning endpoint (untrained weights honestly marked)
        reason_res = client.post("/reason", json={"message": "Who am I?"})
        assert reason_res.status_code == 200
        rdata = reason_res.json()
        assert rdata["response"] == "You are Stan, Kalda's son."
        assert rdata["provenance"]["model_weights_status"] == "untrained"
        assert rdata["provenance"]["neural_generation_used"] is False

    # Simulate application restart with fresh FastAPI app pointing to same identity & DB
    app2 = create_app(
        model_config_path=cfg_path,
        checkpoint_dir=tmp_path / "empty_ckpts",
        identity_path=id_path,
        memory_db_path=db_path,
    )
    with TestClient(app2) as client2:
        ident2 = client2.get("/identity")
        assert ident2.json()["personality"]["directness"] == 0.95

        mem_list = client2.get("/memory", params={"category": "decisions_experiences"})
        assert mem_list.status_code == 200
        assert any(r["id"] == created_id for r in mem_list.json()["records"])

        del_res = client2.delete(f"/memory/{created_id}")
        assert del_res.status_code == 200


def test_reason_connects_to_trained_pytorch_weights(tmp_path: Path) -> None:
    corpus = tmp_path / "train.txt"
    corpus.write_text(
        "\n".join(
            [
                "STAN AI connects persistent memory and structured reasoning to PyTorch weights.",
                "Independent judgment prioritizes accuracy and checks contradictions carefully.",
                "Kalda project builds custom transformers from random initialization.",
                "Direct natural language skips robotic templates and empty apologies.",
            ]
            * 60
        ),
        encoding="utf-8",
    )
    ckpt_dir = tmp_path / "ckpts"
    tok_model, _ = train_bpe_tokenizer(train_files=[corpus], output_dir=ckpt_dir, vocab_size=512)
    cfg = STANConfig(n_layers=2, d_model=64, n_heads=4, d_head=16, d_ff=128, max_seq_len=256, vocab_size=512)
    model = STANTransformer(cfg, log_params=False)
    save_checkpoint(
        checkpoint_dir=ckpt_dir,
        model=model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        global_step=5,
        model_config=cfg,
        training_config={"max_steps": 5},
        training_history=[{"step": 5, "train_loss": 3.0, "val_loss": 3.1, "learning_rate": 1e-4}],
        training_status="trained",
        tokenizer_path=tok_model,
        dataset_tokens=2000,
    )

    app = create_app(
        model_config_path=ckpt_dir / "model_config.yaml",
        checkpoint_dir=ckpt_dir,
        identity_path=tmp_path / "identity.json",
        memory_db_path=tmp_path / "stan_memory.db",
    )
    with TestClient(app) as client:
        res = client.post("/reason", json={"message": "Should I pick option A or option B?", "max_new_tokens": 12})
        assert res.status_code == 200
        data = res.json()
        assert data["provenance"]["model_weights_status"] == "trained"
        assert data["provenance"]["neural_generation_used"] is True
        assert data["provenance"]["neural_meta"]["checkpoint_version"] == "step-0000005"
