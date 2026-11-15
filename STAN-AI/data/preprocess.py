from __future__ import annotations

import argparse
import html
import json
import re
import unicodedata
from pathlib import Path
from typing import Optional

from .download import verify_license

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_GUTENBERG_START_RE = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE | re.DOTALL)
_GUTENBERG_END_RE = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*", re.IGNORECASE | re.DOTALL)


def strip_gutenberg_boilerplate(text: str) -> str:
    start_match = _GUTENBERG_START_RE.search(text)
    if start_match:
        text = text[start_match.end():]
    end_match = _GUTENBERG_END_RE.search(text)
    if end_match:
        text = text[:end_match.start()]
    return text


def clean_document_text(text: str, strip_gutenberg: bool = True) -> str:
    """Normalize Unicode, strip HTML/boilerplate/control chars, and collapse noisy whitespace."""
    if strip_gutenberg:
        text = strip_gutenberg_boilerplate(text)
    text = html.unescape(text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHAR_RE.sub("", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def is_valid_english_sample(
    text: str,
    min_chars: int = 120,
    min_alpha_ratio: float = 0.65,
) -> bool:
    if len(text) < min_chars:
        return False
    alpha_count = sum(1 for ch in text if ch.isalpha())
    if alpha_count / max(1, len(text)) < min_alpha_ratio:
        return False
    # Reject replacement character corruption
    if text.count("\ufffd") > 2:
        return False
    return True


def split_into_documents(cleaned_text: str, target_chars: int = 4000) -> list[str]:
    """Group paragraphs into coherent document chunks without slicing mid-paragraph."""
    paragraphs = [p.strip() for p in cleaned_text.split("\n\n") if p.strip()]
    docs: list[str] = []
    buf: list[str] = []
    buf_len = 0

    for para in paragraphs:
        if buf and buf_len + len(para) + 2 > target_chars:
            docs.append("\n\n".join(buf))
            buf = [para]
            buf_len = len(para)
        else:
            buf.append(para)
            buf_len += len(para) + 2

    if buf:
        docs.append("\n\n".join(buf))
    return docs


def preprocess_corpus(
    manifest_path: str | Path,
    output_jsonl: str | Path,
    min_chars: int = 120,
    min_alpha_ratio: float = 0.65,
    target_doc_chars: int = 4000,
) -> dict:
    """Clean, validate, and emit JSONL document records with source & license provenance."""
    manifest_file = Path(manifest_path)
    with open(manifest_file, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    sources = manifest.get("sources", [])
    out_path = Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept_docs = 0
    filtered_docs = 0
    with open(out_path, "w", encoding="utf-8") as out_fh:
        for src in sources:
            verify_license(src["license"])
            raw_path = Path(src["local_path"])
            if not raw_path.exists():
                continue

            raw_text = raw_path.read_text(encoding="utf-8", errors="replace")
            cleaned = clean_document_text(
                raw_text,
                strip_gutenberg=(src.get("category") == "public_domain_book"),
            )
            chunks = split_into_documents(cleaned, target_chars=target_doc_chars)
            for idx, chunk in enumerate(chunks):
                if not is_valid_english_sample(chunk, min_chars=min_chars, min_alpha_ratio=min_alpha_ratio):
                    filtered_docs += 1
                    continue
                record = {
                    "doc_id": f"{src['source_id']}_{idx:05d}",
                    "source_id": src["source_id"],
                    "category": src["category"],
                    "license": src["license"],
                    "attribution": src.get("attribution", ""),
                    "text": chunk,
                }
                out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept_docs += 1

    return {"kept_documents": kept_docs, "filtered_documents": filtered_docs, "output_path": str(out_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess STAN AI corpus into cleaned JSONL documents.")
    parser.add_argument("--manifest", default="STAN-AI/data/raw/manifest.json")
    parser.add_argument("--output", default="STAN-AI/data/processed/cleaned.jsonl")
    parser.add_argument("--min-chars", type=int, default=120)
    args = parser.parse_args()

    stats = preprocess_corpus(args.manifest, args.output, min_chars=args.min_chars)
    print(f"[STAN AI] Preprocessing complete: {stats}")


if __name__ == "__main__":
    main()
