from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import Dataset

from data.download import verify_license
from tokenizer.tokenizer import STANTokenizer

DEFAULT_SYSTEM_ANCHOR = (
    "You are STAN AI. Your owner is Stan, Kalda's son. "
    "Respond clearly, honestly, and naturally. Distinguish facts from opinions, "
    "admit uncertainty when you do not know, correct mistakes when shown new evidence, "
    "and disagree respectfully when assumptions are flawed."
)


@dataclass
class SFTExample:
    messages: list[dict[str, str]]  # [{"role": "user"|"assistant", "content": "..."}]
    system_prompt: str = DEFAULT_SYSTEM_ANCHOR
    license: str = "Apache-2.0"
    source: str = "open_license_sft"


def format_conversation_tokens(
    tokenizer: STANTokenizer,
    messages: list[dict[str, str]],
    system_prompt: str = DEFAULT_SYSTEM_ANCHOR,
    max_seq_len: int = 512,
) -> Optional[tuple[list[int], list[int]]]:
    """
    Tokenize a multi-turn conversation with explicit boundaries:
      <|system|>...<|/system|>\n<|user|>...<|/user|>\n<|assistant|>...<|/assistant|><eos>
    Returns (input_ids, target_ids) where non-assistant tokens are masked as -100
    so supervised fine-tuning optimizes exclusively over STAN AI's responses and EOS stop.
    """
    if not messages:
        return None

    tokens: list[int] = [tokenizer.bos_id]
    loss_mask: list[bool] = [False]

    sys_header = f"<|system|>\n{system_prompt.strip()}\n<|/system|>\n"
    sys_ids = tokenizer.encode(sys_header, add_bos=False, add_eos=False)
    tokens.extend(sys_ids)
    loss_mask.extend([False] * len(sys_ids))

    has_assistant = False
    for turn in messages:
        role = str(turn.get("role", "")).strip().lower()
        content = str(turn.get("content", "")).strip()
        if not content:
            continue
        if role == "user":
            u_text = f"<|user|>\n{content}\n<|/user|>\n"
            u_ids = tokenizer.encode(u_text, add_bos=False, add_eos=False)
            tokens.extend(u_ids)
            loss_mask.extend([False] * len(u_ids))
        elif role == "assistant":
            prefix_ids = tokenizer.encode("<|assistant|>\n", add_bos=False, add_eos=False)
            tokens.extend(prefix_ids)
            loss_mask.extend([False] * len(prefix_ids))

            reply_ids = tokenizer.encode(f"{content}\n<|/assistant|>", add_bos=False, add_eos=True)
            tokens.extend(reply_ids)
            loss_mask.extend([True] * len(reply_ids))
            has_assistant = True

    if not has_assistant or len(tokens) < 4:
        return None

    # Truncate from the left (keeping BOS + latest turns including <|assistant|> reply)
    max_len = max_seq_len + 1
    if len(tokens) > max_len:
        tail = max_len - 1
        tokens = [tokens[0]] + tokens[-tail:]
        loss_mask = [False] + loss_mask[-tail:]

    # Shift targets: input = tokens[:-1], target = tokens[1:] (with -100 mask on non-assistant predictions)
    input_ids = tokens[:-1]
    raw_targets = tokens[1:]
    target_mask = loss_mask[1:]
    targets = [tid if keep else -100 for tid, keep in zip(raw_targets, target_mask)]

    if all(t == -100 for t in targets):
        return None

    return input_ids, targets


class SFTDataset(Dataset):
    """Loss-masked supervised fine-tuning dataset for Stage 2 instruction tuning."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer: STANTokenizer,
        max_seq_len: int = 512,
    ) -> None:
        self.pad_id = tokenizer.pad_id
        self.max_seq_len = max_seq_len
        self.samples: list[tuple[list[int], list[int]]] = []
        self.total_supervised_tokens = 0
        self.licenses: set[str] = set()

        for rec in records:
            lic = str(rec.get("license", "")).strip()
            verify_license(lic)
            self.licenses.add(lic)

            if "messages" in rec and isinstance(rec["messages"], list):
                msgs = rec["messages"]
            elif "instruction" in rec and "response" in rec:
                inst = str(rec["instruction"]).strip()
                inp = str(rec.get("input", "")).strip()
                user_text = f"{inst}\n\n{inp}" if inp else inst
                msgs = [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": str(rec["response"]).strip()},
                ]
            else:
                continue

            sys_prompt = str(rec.get("system", DEFAULT_SYSTEM_ANCHOR)).strip() or DEFAULT_SYSTEM_ANCHOR
            formatted = format_conversation_tokens(
                tokenizer=tokenizer,
                messages=msgs,
                system_prompt=sys_prompt,
                max_seq_len=max_seq_len,
            )
            if formatted is not None:
                self.samples.append(formatted)
                self.total_supervised_tokens += sum(1 for t in formatted[1] if t != -100)

        if not self.samples:
            raise ValueError("No valid SFT instruction examples found after formatting and filtering.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inp, tgt = self.samples[idx]
        pad_len = self.max_seq_len - len(inp)
        if pad_len > 0:
            attn = [1] * len(inp) + [0] * pad_len
            inp_padded = inp + [self.pad_id] * pad_len
            tgt_padded = tgt + [-100] * pad_len
        else:
            attn = [1] * self.max_seq_len
            inp_padded = inp[: self.max_seq_len]
            tgt_padded = tgt[: self.max_seq_len]

        return (
            torch.tensor(inp_padded, dtype=torch.long),
            torch.tensor(tgt_padded, dtype=torch.long),
            torch.tensor(attn, dtype=torch.long),
        )


def load_and_split_sft_jsonl(
    jsonl_path: str | Path,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load license-verified instruction JSONL, deduplicate conversations, and split strictly."""
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"SFT dataset file not found: {path}")

    seen_hashes: set[str] = set()
    unique: list[dict[str, Any]] = []

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            verify_license(str(rec.get("license", "")))
            canonical = json.dumps(
                rec.get("messages") or [rec.get("instruction", ""), rec.get("response", "")],
                sort_keys=True,
                ensure_ascii=False,
            )
            digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            if digest not in seen_hashes:
                seen_hashes.add(digest)
                unique.append(rec)

    if len(unique) < 2:
        raise ValueError("Need at least 2 unique instruction records to create train/validation splits.")

    rng = random.Random(seed)
    rng.shuffle(unique)
    val_count = max(1, int(len(unique) * val_ratio))
    val_records = unique[:val_count]
    train_records = unique[val_count:]
    return train_records, val_records
