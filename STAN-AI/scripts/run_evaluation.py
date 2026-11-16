#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.suite import run_evaluation_suite
from inference.load_model import load_stan_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 9-category STAN AI evaluation suite.")
    parser.add_argument("--checkpoint-dir", default=str(ROOT / "checkpoints"))
    parser.add_argument("--model-config", default=str(ROOT / "configs" / "model.yaml"))
    args = parser.parse_args()

    state = load_stan_runtime(
        model_config_path=args.model_config,
        checkpoint_dir=args.checkpoint_dir,
    )
    report = run_evaluation_suite(
        state,
        report_path=Path(args.checkpoint_dir) / "eval_report.json",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
