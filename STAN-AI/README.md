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
  backend/          # FastAPI app (/health, /model/info, /generate, /tokenize, /metrics, /identity, /memory, /reason)
  identity/         # Permanent IdentityProfile (STAN AI, Stan, Kalda) & owner-configurable PersonalityConfig
  memory/           # SQLite 7-category Persistent MemoryStore (CRUD + search + contradiction anchors)
  reasoning/        # STANReasoningEngine (input signals, contradiction checks, epistemic separation, judgment)
  configs/          # model.yaml and training.yaml
  scripts/          # prepare_corpus.py, kaggle_train.py, run_smoke_train.py, start_inference.sh
  checkpoints/      # Runtime model checkpoints, tokenizer.model, training_history.json
  memory_store/     # Persistent identity.json and stan_memory.db
  tests/            # Pytest suite covering model, tokenizer, data, training, inference, memory, reasoning & API
  requirements.txt
  LICENSE
src/                # React + TypeScript web application (Chat, Memory & Identity, Model Info, Training, Settings)
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

## 4. Phase 2 Training & Notebook Workflow (Colab / Kaggle GPU)

A turnkey Jupyter notebook is provided at `STAN-AI/notebooks/STAN_AI_Colab_Kaggle_Training.ipynb` alongside `STAN-AI/scripts/colab_kaggle_setup.sh`.

1. Open the notebook or upload the repository into Google Colab / Kaggle with a CUDA GPU enabled (e.g., T4 / L4 / P100).
2. Install Python dependencies:
   ```bash
   bash STAN-AI/scripts/colab_kaggle_setup.sh
   ```
3. **Stage 1 — English Pretraining** (random initialization, next-token prediction):
   ```bash
   python STAN-AI/scripts/prepare_corpus.py --sources-limit 10 --vocab-size 8192
   python STAN-AI/scripts/kaggle_train.py --stage pretraining --resume auto
   ```
4. **Stage 2 — Instruction Tuning (SFT)** (requires completed Stage 1 checkpoint; uses Apache-2.0 `OpenAssistant/oasst1`):
   ```bash
   python STAN-AI/scripts/prepare_sft_corpus.py --max-examples 8000
   python STAN-AI/scripts/kaggle_train.py --stage instruction_tuning --resume auto
   ```
5. **Validate & Import GPU Checkpoints Back Locally**:
   ```bash
   python STAN-AI/scripts/import_checkpoint.py --source /path/to/latest.pt --tokenizer /path/to/tokenizer.model --confirm-overwrite
   ```

---

## 5. Inference, Identity, Memory, Evaluation & Deployment

### FastAPI Service (Python)

```bash
./STAN-AI/scripts/start_inference.sh
```

Core, Cognitive, Training & Evaluation Endpoints:
- `GET /health`, `GET /model/info`, `POST /tokenize`, `GET /metrics`
- `POST /generate` & `POST /generate/stream` (gated when model weights are `untrained` or `smoke_test`)
- `GET /identity` & `PUT /identity` (permanent identity anchor & owner-configurable personality)
- `GET /memory`, `POST /memory`, `PATCH /memory/{id}`, `DELETE /memory/{id}`, `POST /memory/search` (7-category SQLite persistent memory)
- `POST /reason` (Stage 3 personality-conditioned neural generation + cognitive reasoning engine)
- `GET /training/status`, `POST /training/start`, `POST /training/stop`, `POST /training/import`
- `GET /evaluation`, `POST /evaluation/run` (9-dimension measured evaluation suite)

### Full-Stack Web App (Node.js + React)

```bash
npm install
npm run build
STAN_API_URL="http://127.0.0.1:8001" npm start
```

In development, `npm run dev` runs `server.ts` on port `3000`, mounts Vite middleware, and proxies `/api/stan/*` to the Python FastAPI service (auto-spawning local `uvicorn` on `127.0.0.1:8001` if `.venv` is present).

When no trained English checkpoint exists in `STAN-AI/checkpoints/`, STAN AI honestly reports `training_status: "untrained"`, keeps raw neural generation disabled, and transparently reports `neural_generation_used: false` in reasoning and `not_evaluated_untrained` for neural evaluation categories.

---

## 6. Automated Verification

```bash
# Python unit & integration tests (model, tokenizer, data, Stage 1/2/3 training, evaluation, memory, reasoning & API)
.venv/bin/pytest -v STAN-AI/tests/

# 9-dimension evaluation CLI check
.venv/bin/python STAN-AI/scripts/run_evaluation.py

# Frontend/Node logic tests
npx --yes tsx --test src/utils/stanState.test.ts

# TypeScript & production build check
npm run lint && npm run build
```

