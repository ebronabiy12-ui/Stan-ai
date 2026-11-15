from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference import GenerationConfig, generate_text, load_stan_runtime
from model import STANConfig, STANTransformer
from tokenizer import STANTokenizer, train_bpe_tokenizer
from training.checkpoint import save_checkpoint


def test_untrained_state_when_no_checkpoint(tmp_path: Path) -> None:
    cfg = STANConfig(n_layers=2, d_model=64, n_heads=4, d_head=16, d_ff=128, max_seq_len=32, vocab_size=512)
    cfg_path = tmp_path / "model.yaml"
    cfg.to_yaml(cfg_path)

    state = load_stan_runtime(
        model_config_path=cfg_path,
        checkpoint_dir=tmp_path / "empty_ckpts",
    )
    assert state.checkpoint_loaded is False
    assert state.training_status == "untrained"
    assert state.production_chat_enabled is False
    assert state.warning is not None


def test_independent_generation_determinism_and_eos(tmp_path: Path) -> None:
    corpus = tmp_path / "train.txt"
    corpus.write_text(
        "\n".join(
            [
                "The independent language model processes subword tokens carefully across causal layers.",
                "Transformer blocks combine RMSNorm, multi-head self-attention, and residual projections.",
                "Public-domain literature, educational articles, and verified dialogue form the training corpus.",
                "Temperature, top-k filtering, and nucleus top-p sampling shape autoregressive token selection.",
                "Checkpoints store model weights, optimizer state, scheduler progress, and evaluation loss.",
                "Dedicated special tokens mark padding, unknown bytes, sequence start, and end of sequence.",
            ]
            * 40
        ),
        encoding="utf-8",
    )
    tok_model, _ = train_bpe_tokenizer(
        train_files=[corpus],
        output_dir=tmp_path / "ckpts",
        vocab_size=512,
    )
    tokenizer = STANTokenizer(tok_model)

    cfg = STANConfig(n_layers=2, d_model=64, n_heads=4, d_head=16, d_ff=128, max_seq_len=64, vocab_size=512)
    model = STANTransformer(cfg, log_params=False)

    ckpt_dir = tmp_path / "ckpts"
    save_checkpoint(
        checkpoint_dir=ckpt_dir,
        model=model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        global_step=10,
        model_config=cfg,
        training_config={"max_steps": 10},
        training_history=[{"step": 10, "train_loss": 2.1, "val_loss": 2.2, "learning_rate": 1e-4}],
        training_status="trained",
        tokenizer_path=tok_model,
        dataset_tokens=5000,
    )

    state = load_stan_runtime(
        model_config_path=ckpt_dir / "model_config.yaml",
        checkpoint_dir=ckpt_dir,
    )
    assert state.checkpoint_loaded is True
    assert state.production_chat_enabled is True
    assert state.model is not None and state.tokenizer is not None

    gen_cfg = GenerationConfig(
        temperature=0.8,
        top_k=20,
        top_p=0.9,
        max_new_tokens=16,
        repetition_penalty=1.15,
        seed=777,
    )
    res1 = generate_text(state.model, state.tokenizer, "The independent", gen_cfg, state.device)
    res2 = generate_text(state.model, state.tokenizer, "The independent", gen_cfg, state.device)

    assert res1.generated_tokens <= 16
    assert res1.generated_text == res2.generated_text
    assert res1.prompt_tokens > 0
