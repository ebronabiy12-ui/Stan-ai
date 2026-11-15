from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import torch
import torch.nn.functional as F

from model.transformer import STANTransformer
from tokenizer.tokenizer import STANTokenizer


@dataclass
class GenerationConfig:
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9
    max_new_tokens: int = 128
    repetition_penalty: float = 1.1
    seed: Optional[int] = None
    stop_on_eos: bool = True


@dataclass
class GenerationResult:
    prompt: str
    generated_text: str
    full_text: str
    prompt_tokens: int
    generated_tokens: int
    total_tokens: int
    stopped_reason: str
    elapsed_ms: float
    tokens_per_second: float


def _apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: list[int],
    penalty: float,
) -> torch.Tensor:
    if penalty == 1.0 or not generated_ids:
        return logits
    unique_ids = set(generated_ids)
    for token_id in unique_ids:
        val = logits[token_id]
        if val < 0:
            logits[token_id] = val * penalty
        else:
            logits[token_id] = val / penalty
    return logits


def _sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    generator: Optional[torch.Generator],
) -> int:
    if temperature <= 0.0:
        return int(torch.argmax(logits, dim=-1).item())

    scaled = logits / float(temperature)

    if top_k > 0 and top_k < scaled.size(-1):
        top_vals, _ = torch.topk(scaled, top_k)
        cutoff = top_vals[-1]
        scaled = torch.where(scaled < cutoff, torch.tensor(float("-inf"), device=scaled.device), scaled)

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative_probs > top_p
        # Keep at least the highest probability token
        sorted_mask[1:] = sorted_mask[:-1].clone()
        sorted_mask[0] = False
        indices_to_remove = sorted_indices[sorted_mask]
        scaled[indices_to_remove] = float("-inf")

    probs = F.softmax(scaled.float(), dim=-1)
    if torch.isnan(probs).any() or probs.sum() <= 0:
        return int(torch.argmax(logits, dim=-1).item())

    next_id = torch.multinomial(probs, num_samples=1, generator=generator)
    return int(next_id.item())


@torch.no_grad()
def stream_generate_text(
    model: STANTransformer,
    tokenizer: STANTokenizer,
    prompt: str,
    config: GenerationConfig,
    device: torch.device,
) -> Iterator[dict[str, Any]]:
    """
    Yield progressive tokens from STAN AI's own PyTorch Transformer weights.
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt must be a non-empty string.")

    model.eval()
    start_time = time.perf_counter()

    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    if not prompt_ids:
        prompt_ids = [tokenizer.bos_id]

    max_ctx = model.config.max_seq_len
    if len(prompt_ids) >= max_ctx:
        prompt_ids = prompt_ids[-(max_ctx - 1) :]

    generator: Optional[torch.Generator] = None
    if config.seed is not None:
        generator = torch.Generator(device=device.type if device.type == "cuda" else "cpu")
        generator.manual_seed(int(config.seed))

    context_ids = list(prompt_ids)
    generated_ids: list[int] = []
    stopped_reason = "max_new_tokens"

    for _ in range(config.max_new_tokens):
        window = context_ids[-max_ctx:]
        input_tensor = torch.tensor([window], dtype=torch.long, device=device)
        logits, _ = model(input_tensor)
        next_logits = logits[0, -1, :].clone()

        next_logits = _apply_repetition_penalty(
            next_logits, context_ids, penalty=config.repetition_penalty
        )
        next_id = _sample_next_token(
            next_logits,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            generator=generator,
        )

        if config.stop_on_eos and next_id == tokenizer.eos_id:
            stopped_reason = "eos_token"
            break

        generated_ids.append(next_id)
        context_ids.append(next_id)

        decoded_so_far = tokenizer.decode(generated_ids, skip_special_tokens=True)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        yield {
            "event": "token",
            "token_id": next_id,
            "generated_text": decoded_so_far,
            "generated_tokens": len(generated_ids),
            "elapsed_ms": round(elapsed_ms, 2),
        }

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    final_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    tps = (len(generated_ids) / max(elapsed_ms / 1000.0, 1e-6)) if generated_ids else 0.0
    yield {
        "event": "done",
        "prompt": prompt,
        "generated_text": final_text,
        "full_text": prompt + final_text,
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": len(generated_ids),
        "total_tokens": len(prompt_ids) + len(generated_ids),
        "stopped_reason": stopped_reason,
        "elapsed_ms": round(elapsed_ms, 2),
        "tokens_per_second": round(tps, 2),
    }


def generate_text(
    model: STANTransformer,
    tokenizer: STANTokenizer,
    prompt: str,
    config: GenerationConfig,
    device: torch.device,
) -> GenerationResult:
    final_payload: Optional[dict[str, Any]] = None
    for chunk in stream_generate_text(model, tokenizer, prompt, config, device):
        if chunk["event"] == "done":
            final_payload = chunk
    assert final_payload is not None
    return GenerationResult(
        prompt=final_payload["prompt"],
        generated_text=final_payload["generated_text"],
        full_text=final_payload["full_text"],
        prompt_tokens=final_payload["prompt_tokens"],
        generated_tokens=final_payload["generated_tokens"],
        total_tokens=final_payload["total_tokens"],
        stopped_reason=final_payload["stopped_reason"],
        elapsed_ms=final_payload["elapsed_ms"],
        tokens_per_second=final_payload["tokens_per_second"],
    )
