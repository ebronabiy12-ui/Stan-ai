# STAN AI Checkpoints Directory

No pretrained weights are bundled with this repository.

When STAN AI is trained from random initialization using `training/train.py` or `scripts/kaggle_train.py`, the following artifacts are written here:

- `latest.pt` — latest resumable checkpoint (`model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `scaler_state_dict`, `global_step`, `training_status`, `training_history`)
- `step_XXXXXXX.pt` — periodic step snapshots
- `tokenizer.model` / `tokenizer.vocab` / `tokenizer_config.json` — STAN AI's custom SentencePiece BPE tokenizer trained exclusively on the project's training split
- `model_config.yaml` — exact model hyperparameters matching the checkpoint
- `training_history.json` — recorded training & validation loss history
