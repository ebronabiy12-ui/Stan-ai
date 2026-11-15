from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import STANConfig, STANTransformer


def test_default_50m_parameter_count() -> None:
    cfg = STANConfig.from_yaml(ROOT / "configs" / "model.yaml")
    model = STANTransformer(cfg, log_params=True)
    trainable = model.count_parameters(trainable_only=True)
    # 15L / 512D / 8H / 2048FF / 8192V / 512C GELU tied = 51,658,240 params (~51.66M)
    assert 48_000_000 <= trainable <= 55_000_000
    assert trainable == 51_658_240
    assert model.lm_head.weight is model.embeddings.token_embedding.weight


def test_configurable_dimensions_and_swiglu() -> None:
    cfg = STANConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        d_head=16,
        d_ff=128,
        max_seq_len=32,
        vocab_size=256,
        activation="swiglu",
    )
    model = STANTransformer(cfg, log_params=False)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(x, targets=x)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert loss is not None and torch.isfinite(loss)


def test_causal_attention_invariance() -> None:
    cfg = STANConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        d_head=16,
        d_ff=128,
        max_seq_len=32,
        vocab_size=128,
        dropout=0.0,
    )
    torch.manual_seed(123)
    model = STANTransformer(cfg, log_params=False)
    model.eval()

    seq_a = torch.tensor([[2, 10, 20, 30, 40, 50]], dtype=torch.long)
    seq_b = torch.tensor([[2, 10, 20, 99, 88, 77]], dtype=torch.long)

    with torch.no_grad():
        logits_a, _ = model(seq_a)
        logits_b, _ = model(seq_b)

    # First 3 positions (0, 1, 2) must produce identical logits because future tokens at 3..5 are masked
    assert torch.allclose(logits_a[:, :3, :], logits_b[:, :3, :], atol=1e-6)
    # Position 3 has different input token, so logits must differ
    assert not torch.allclose(logits_a[:, 3, :], logits_b[:, 3, :], atol=1e-5)


def test_cross_entropy_loss_and_gradient_propagation() -> None:
    cfg = STANConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        d_head=16,
        d_ff=128,
        max_seq_len=32,
        vocab_size=128,
    )
    model = STANTransformer(cfg, log_params=False)
    input_ids = torch.randint(0, cfg.vocab_size, (3, 16))
    targets = torch.randint(0, cfg.vocab_size, (3, 16))

    logits, loss = model(input_ids, targets=targets)
    assert logits.shape == (3, 16, cfg.vocab_size)
    assert loss is not None
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Missing gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"


def test_sequence_length_bounds() -> None:
    cfg = STANConfig(n_layers=1, d_model=32, n_heads=2, d_head=16, d_ff=64, max_seq_len=8, vocab_size=64)
    model = STANTransformer(cfg, log_params=False)
    too_long = torch.randint(0, 64, (1, 9))
    with pytest.raises(ValueError, match="exceeds model max_seq_len"):
        model(too_long)
