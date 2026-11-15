# STAN AI Web Application

The interactive React + TypeScript frontend (`src/`) and Node.js Express runtime (`server.ts`) are mounted at the repository root so Vite builds and Express serves seamlessly on port `3000`.

- Frontend views: `Chat`, `Model Information`, `Training Dashboard`, `Settings`
- Proxy route: Express forwards `/api/stan/*` to the Python FastAPI service (`STAN_API_URL`, default `http://127.0.0.1:8000`).
- No external LLM APIs are called. All inference requests go exclusively to STAN AI's Python FastAPI service.
