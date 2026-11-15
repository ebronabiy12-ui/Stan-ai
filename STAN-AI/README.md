# STAN AI — Independent ~50M Language Model

STAN AI is an independent decoder-only Transformer language model (~51.66M trainable parameters) built from scratch in Python and PyTorch, paired with a custom SentencePiece BPE tokenizer, open-license dataset preparation pipeline, resumable training engine, FastAPI inference service, Node.js proxy runtime, and React + TypeScript web application.

No external LLM APIs or pretrained model weights are used anywhere in STAN AI.

---

## 1. Architecture Summary

Configured in `STAN-AI/configs/model.yaml`:

- **Architecture**: Decoder-only Transformer (Pre-Norm Causal Self-Attention)
- **Transformer Blocks**: 15
- **Hidden Dimension (`d_model`)**: 512
- **Attention Heads (`n_heads`)**: 8 (`d_head = 64`)
- **Feed-Forward Dimension (`d_ff`)**: 2048 (`activation: gelu`, `swiglu` also supported)
- **Context Length**: 512 tokens
- **Vocabulary Size**: 8,192 tokens (SentencePiece BPE, `byte_fallback=True`)
- **Positional Encoding**: Learned absolute positional embeddings
- **Normalization**: `RMSNorm` (`eps = 1e-6`)
- **Output Projection**: Weight-tied to token embeddings
- **Initialization**: Random Gaussian initialization (`std = 0.02`, residual projections scaled by `1 / sqrt(2 * n_layers)`)
- **Verified Parameter Count**: **51,658,240** trainable parameters (`51.66M`)

---

## 2. Repository Layout

```text
STAN-AI/
  model/            # STANConfig, RMSNorm, Embeddings, CausalSelfAttention, STANTransformer
  tokenizer/        # Custom SentencePiece BPE trainer & STANTokenizer wrapper
  data/             # License-verified downloader, cleaner, deduplicator, binary dataset packer
  training/         # AdamW, cosine warmup scheduler, atomic checkpoints, eval, resumable trainer
  inference/        # Checkpoint loader, temperature/top-k/top-p/repetition-penalty/streaming engine
  backend/          # FastAPI app (/health, /model/info, /generate, /generate/stream, /tokenize, /metrics)
  configs/          # model.yaml and training.yaml
  scripts/          # prepare_corpus.py, kaggle_train.py, run_smoke_train.py, start_inference.sh
  checkpoints/      # Runtime model checkpoints, tokenizer.model, training_history.json
  tests/            # Pytest suite covering model, tokenizer, data, training, inference, and API
  requirements.txt
  LICENSE
src/                # React + TypeScript web application (Chat, Model Info, Training, Settings)
server.ts           # Node.js Express runtime serving Vite/React and proxying /api/stan/* to FastAPI
```

---

## 3. Dataset Pipeline & Licensing

STAN AI enforces explicit license verification (`Public Domain`, `CC0-1.0`, `CC-BY-4.0`, `CC-BY-SA-4.0`, `Apache-2.0`, `MIT`) and separates train/validation documents before tokenizer training to prevent data leakage.

```bash
# Download verified public-domain sources, clean, deduplicate, train BPE tokenizer (train split only), and pack binary tokens
python STAN-AI/scripts/prepare_corpus.py --sources-limit 5 --vocab-size 8192
```

For synthetic numerical mechanics verification (not an English language corpus):

```bash
python STAN-AI/scripts/run_smoke_train.py --steps 40
```

---

## 4. Kaggle & GPU Training Instructions

1. Upload or clone the repository into a Kaggle Notebook with GPU enabled (e.g., T4 / P100).
2. Install Python dependencies:
   ```bash
   pip install -r STAN-AI/requirements.txt
   ```
3. Prepare the English corpus and train the 8192-vocab BPE tokenizer:
   ```bash
   python STAN-AI/scripts/prepare_corpus.py --sources-limit 5 --vocab-size 8192
   ```
4. Start training with automatic checkpoint resume (safe across Kaggle 9–12 hour session limits):
   ```bash
   python STAN-AI/scripts/kaggle_train.py --resume auto
   ```
5. Save `STAN-AI/checkpoints/latest.pt`, `tokenizer.model`, `tokenizer_config.json`, `model_config.yaml`, and `training_history.json` into Kaggle Output/Dataset storage, then pass `--resume auto` on the next session to restore model weights, AdamW state, LR scheduler state, AMP scaler state, and `global_step`.

---

## 5. Inference & Deployment

### FastAPI Service (Python)

```bash
./STAN-AI/scripts/start_inference.sh
```

Required endpoints:
- `GET /health`
- `GET /model/info`
- `POST /generate` (and `POST /generate/stream` for SSE streaming)
- `POST /tokenize`
- `GET /metrics`

### Full-Stack Web App (Node.js + React)

```bash
npm install
npm run build
STAN_API_URL="http://127.0.0.1:8000" npm start
```

In development, `npm run dev` runs `server.ts` on port `3000`, mounts Vite middleware, and proxies `/api/stan/*` to the Python FastAPI service (auto-spawning local `uvicorn` if `.venv` is present).

When no trained English checkpoint exists in `STAN-AI/checkpoints/`, STAN AI honestly reports `training_status: "untrained"` and disables production chat.

---

## 6. Automated Verification

```bash
# Python unit & integration tests (model, tokenizer, data, training, inference, FastAPI)
.venv/bin/pytest -v STAN-AI/tests/

# Frontend/Node logic tests
npx --yes tsx --test src/utils/stanState.test.ts

# TypeScript & production build check
npm run lint && npm run build
```
