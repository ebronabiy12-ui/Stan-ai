from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ALLOWED_LICENSES = {
    "Public Domain",
    "CC0-1.0",
    "CC-BY-4.0",
    "CC-BY-SA-4.0",
    "Apache-2.0",
    "MIT",
}


@dataclass(frozen=True)
class CorpusSource:
    """Verified open-license English corpus source descriptor."""

    source_id: str
    title: str
    category: str  # public_domain_book | educational | encyclopedia | dialogue
    url: str
    license: str
    attribution: str


# Curated verified public-domain & open-license English sources (downloaded directly by user/training pipeline)
VERIFIED_PUBLIC_DOMAIN_BOOKS: list[CorpusSource] = [
    CorpusSource(
        source_id="gutenberg_pride_prejudice",
        title="Pride and Prejudice by Jane Austen",
        category="public_domain_book",
        url="https://www.gutenberg.org/files/1342/1342-0.txt",
        license="Public Domain",
        attribution="Project Gutenberg (Public Domain in the USA)",
    ),
    CorpusSource(
        source_id="gutenberg_frankenstein",
        title="Frankenstein by Mary Shelley",
        category="public_domain_book",
        url="https://www.gutenberg.org/files/84/84-0.txt",
        license="Public Domain",
        attribution="Project Gutenberg (Public Domain in the USA)",
    ),
    CorpusSource(
        source_id="gutenberg_sherlock_holmes",
        title="The Adventures of Sherlock Holmes by Arthur Conan Doyle",
        category="public_domain_book",
        url="https://www.gutenberg.org/files/1661/1661-0.txt",
        license="Public Domain",
        attribution="Project Gutenberg (Public Domain in the USA)",
    ),
    CorpusSource(
        source_id="gutenberg_moby_dick",
        title="Moby-Dick by Herman Melville",
        category="public_domain_book",
        url="https://www.gutenberg.org/files/2701/2701-0.txt",
        license="Public Domain",
        attribution="Project Gutenberg (Public Domain in the USA)",
    ),
    CorpusSource(
        source_id="gutenberg_alice_wonderland",
        title="Alice's Adventures in Wonderland by Lewis Carroll",
        category="public_domain_book",
        url="https://www.gutenberg.org/files/11/11-0.txt",
        license="Public Domain",
        attribution="Project Gutenberg (Public Domain in the USA)",
    ),
]


def verify_license(license_name: str) -> None:
    if license_name not in ALLOWED_LICENSES:
        raise ValueError(
            f"Unverified or disallowed license '{license_name}'. "
            f"Allowed licenses: {sorted(ALLOWED_LICENSES)}"
        )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def download_corpus_sources(
    sources: Iterable[CorpusSource],
    output_dir: str | Path = "STAN-AI/data/raw",
    timeout_sec: int = 30,
) -> Path:
    """Download verified open-license sources and write a source/license manifest."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_records: list[dict] = []

    for src in sources:
        verify_license(src.license)
        req = urllib.request.Request(
            src.url,
            headers={"User-Agent": "STAN-AI-Dataset-Builder/0.1 (research; open-license-verified)"},
        )
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw_bytes = resp.read()

        target_file = out_dir / f"{src.source_id}.txt"
        target_file.write_bytes(raw_bytes)

        record = asdict(src)
        record.update(
            {
                "local_path": str(target_file),
                "bytes": len(raw_bytes),
                "sha256": _sha256_bytes(raw_bytes),
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        manifest_records.append(record)

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump({"sources": manifest_records}, fh, indent=2)
    return manifest_path


def register_local_source(
    file_path: str | Path,
    source_id: str,
    title: str,
    category: str,
    license_name: str,
    attribution: str,
    output_dir: str | Path = "STAN-AI/data/raw",
) -> dict:
    """Register a locally placed dataset file into manifest.json after validating license metadata."""
    verify_license(license_name)
    src_path = Path(file_path)
    if not src_path.exists():
        raise FileNotFoundError(f"Local corpus file not found: {src_path}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"

    existing: list[dict] = []
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as fh:
            existing = json.load(fh).get("sources", [])

    raw_bytes = src_path.read_bytes()
    record = {
        "source_id": source_id,
        "title": title,
        "category": category,
        "url": f"local://{src_path.name}",
        "license": license_name,
        "attribution": attribution,
        "local_path": str(src_path),
        "bytes": len(raw_bytes),
        "sha256": _sha256_bytes(raw_bytes),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }
    existing = [r for r in existing if r.get("source_id") != source_id]
    existing.append(record)

    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump({"sources": existing}, fh, indent=2)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Download license-verified English corpora for STAN AI.")
    parser.add_argument("--output-dir", default="STAN-AI/data/raw", help="Destination directory.")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of curated sources to fetch.")
    args = parser.parse_args()

    selected = VERIFIED_PUBLIC_DOMAIN_BOOKS[: max(1, args.limit)]
    manifest = download_corpus_sources(selected, output_dir=args.output_dir)
    print(f"[STAN AI] Downloaded {len(selected)} verified sources -> {manifest}")


if __name__ == "__main__":
    main()
