#!/usr/bin/env python3
"""Import a Colab/Kaggle-trained STAN AI checkpoint cleanly after verifying STANTransformer compatibility."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.checkpoint import import_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Import validated STAN AI checkpoint into checkpoints/.")
    parser.add_argument("--source-checkpoint", required=True, help="Path to downloaded .pt checkpoint file.")
    parser.add_argument("--tokenizer", default=None, help="Path to matching tokenizer.model.")
    parser.add_argument("--checkpoint-dir", default=str(ROOT / "checkpoints"))
    parser.add_argument("--confirm-overwrite", action="store_true", help="Confirm replacing existing latest.pt.")
    args = parser.parse_args()

    result = import_checkpoint(
        source_checkpoint=args.source_checkpoint,
        target_checkpoint_dir=args.checkpoint_dir,
        tokenizer_source=args.tokenizer,
        confirm_overwrite=args.confirm_overwrite,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
