#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_ROOT="$(cd "$ROOT_DIR/.." && pwd)"

if [ -x "$APP_ROOT/.venv/bin/uvicorn" ]; then
  UVICORN_BIN="$APP_ROOT/.venv/bin/uvicorn"
else
  UVICORN_BIN="uvicorn"
fi

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
exec "$UVICORN_BIN" backend.main:app --host "${STAN_API_HOST:-127.0.0.1}" --port "${STAN_API_PORT:-8001}"
