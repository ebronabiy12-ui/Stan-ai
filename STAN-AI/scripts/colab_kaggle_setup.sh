#!/usr/bin/env bash
set -euo pipefail

# Reproducible environment setup for STAN AI training on Google Colab / Kaggle GPU runtimes.
echo "[STAN AI] Setting up reproducible Python training environment..."
python -m pip install --upgrade pip
pip install -r STAN-AI/requirements.txt

echo "[STAN AI] Verifying GPU / CUDA hardware..."
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
else:
    print('WARNING: No CUDA GPU detected. Training 51.66M parameters on CPU will be slow.')
"

echo "[STAN AI] Environment ready. Next steps:"
echo "  1. Stage 1 Corpus: python STAN-AI/scripts/prepare_corpus.py --sources-limit 12 --vocab-size 8192"
echo "  2. Stage 1 Train:  python STAN-AI/scripts/kaggle_train.py --resume auto"
echo "  3. Stage 2 SFT:    python STAN-AI/training/sft.py --pretrained-checkpoint STAN-AI/checkpoints/latest.pt --overwrite"
echo "  4. Evaluate:       python STAN-AI/scripts/run_evaluation.py"
