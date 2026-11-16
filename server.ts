import { spawn, type ChildProcess } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import dotenv from 'dotenv';
import express, { type Request, type Response } from 'express';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT || 3000);
const STAN_API_URL = (process.env.STAN_API_URL || 'http://127.0.0.1:8001').replace(/\/+$/, '');
const STAN_AUTO_SPAWN = process.env.STAN_AUTO_SPAWN !== 'false';
const IS_PROD = process.env.NODE_ENV === 'production';

let pythonProc: ChildProcess | null = null;
let shuttingDown = false;

async function isFastApiReachable(): Promise<boolean> {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 1000);
    const res = await fetch(`${STAN_API_URL}/health`, { signal: controller.signal });
    clearTimeout(timer);
    if (!res.ok) return false;
    const body = (await res.json()) as { service?: string };
    return body?.service === 'STAN AI Inference Service';
  } catch {
    return false;
  }
}

function startPythonInferenceIfLocal(): void {
  if (!STAN_AUTO_SPAWN) return;
  const isLocal = STAN_API_URL.includes('127.0.0.1') || STAN_API_URL.includes('localhost');
  if (!isLocal) return;

  const venvPython = path.join(__dirname, '.venv', 'bin', 'python');
  const pythonBin = fs.existsSync(venvPython) ? venvPython : 'python3';
  const stanRoot = path.join(__dirname, 'STAN-AI');
  if (!fs.existsSync(path.join(stanRoot, 'backend', 'main.py'))) {
    console.warn('[STAN Node] STAN-AI/backend/main.py not found; skipping Python auto-spawn.');
    return;
  }

  let parsedPort = '8001';
  try {
    const u = new URL(STAN_API_URL);
    if (u.port) parsedPort = u.port;
  } catch {
    // use default 8001
  }

  const child = spawn(
    pythonBin,
    ['-m', 'uvicorn', 'backend.main:app', '--host', '127.0.0.1', '--port', parsedPort],
    {
      cwd: __dirname,
      env: {
        ...process.env,
        PYTHONPATH: stanRoot + (process.env.PYTHONPATH ? `:${process.env.PYTHONPATH}` : ''),
      },
      stdio: ['ignore', 'pipe', 'pipe'],
    },
  );

  pythonProc = child;
  child.stdout?.on('data', (chunk: Buffer) => {
    process.stdout.write(`[STAN FastAPI] ${chunk.toString()}`);
  });
  child.stderr?.on('data', (chunk: Buffer) => {
    process.stderr.write(`[STAN FastAPI] ${chunk.toString()}`);
  });
  child.on('exit', (code) => {
    pythonProc = null;
    if (!shuttingDown) {
      console.warn(`[STAN FastAPI] Exited with code ${code}`);
    }
  });
}

async function ensurePythonService(): Promise<void> {
  if (await isFastApiReachable()) return;
  if (!pythonProc) {
    startPythonInferenceIfLocal();
  }
}

async function proxyToStan(req: Request, res: Response, targetPath: string): Promise<void> {
  const qsIndex = req.originalUrl.indexOf('?');
  const queryString = qsIndex >= 0 ? req.originalUrl.slice(qsIndex) : '';
  const url = `${STAN_API_URL}${targetPath}${queryString}`;
  const isStream = targetPath === '/generate/stream';

  try {
    const headers: Record<string, string> = {};
    if (req.headers['content-type']) {
      headers['content-type'] = String(req.headers['content-type']);
    } else if (req.method !== 'GET' && req.method !== 'HEAD') {
      headers['content-type'] = 'application/json';
    }

    const fetchOptions: RequestInit = {
      method: req.method,
      headers,
    };
    if (req.method !== 'GET' && req.method !== 'HEAD') {
      fetchOptions.body = JSON.stringify(req.body ?? {});
    }

    const upstream = await fetch(url, fetchOptions);

    if (isStream && upstream.ok && upstream.body) {
      res.status(upstream.status);
      res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      const reader = upstream.body.getReader();
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          if (value) {
            res.write(Buffer.from(value));
          }
        }
      } finally {
        reader.releaseLock();
      }
      res.end();
      return;
    }

    const rawText = await upstream.text();
    const contentType = upstream.headers.get('content-type') || 'application/json';
    res.status(upstream.status).setHeader('Content-Type', contentType).send(rawText);
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unable to reach STAN AI inference service.';
    res.status(503).json({
      detail: `STAN AI Python FastAPI service at ${STAN_API_URL} is unavailable (${message}).`,
    });
  }
}

async function createServer(): Promise<void> {
  const app = express();
  app.use(express.json({ limit: '1mb' }));

  // Ensure Python FastAPI inference service is started locally if configured
  void ensurePythonService();

  app.get('/api/stan/health', (req, res) => void proxyToStan(req, res, '/health'));
  app.get('/api/stan/model/info', (req, res) => void proxyToStan(req, res, '/model/info'));
  app.get('/api/stan/metrics', (req, res) => void proxyToStan(req, res, '/metrics'));
  app.post('/api/stan/tokenize', (req, res) => void proxyToStan(req, res, '/tokenize'));
  app.post('/api/stan/generate', (req, res) => void proxyToStan(req, res, '/generate'));
  app.post('/api/stan/generate/stream', (req, res) => void proxyToStan(req, res, '/generate/stream'));

  // Identity, Persistent Memory & Reasoning routes
  app.get('/api/stan/identity', (req, res) => void proxyToStan(req, res, '/identity'));
  app.put('/api/stan/identity', (req, res) => void proxyToStan(req, res, '/identity'));
  app.get('/api/stan/memory', (req, res) => void proxyToStan(req, res, '/memory'));
  app.post('/api/stan/memory', (req, res) => void proxyToStan(req, res, '/memory'));
  app.post('/api/stan/memory/search', (req, res) => void proxyToStan(req, res, '/memory/search'));
  app.patch('/api/stan/memory/:recordId', (req, res) =>
    void proxyToStan(req, res, `/memory/${encodeURIComponent(req.params.recordId)}`),
  );
  app.delete('/api/stan/memory/:recordId', (req, res) =>
    void proxyToStan(req, res, `/memory/${encodeURIComponent(req.params.recordId)}`),
  );
  app.post('/api/stan/reason', (req, res) => void proxyToStan(req, res, '/reason'));

  // Phase 2 Training Infrastructure & 9-Dimension Evaluation routes
  app.get('/api/stan/training/status', (req, res) => void proxyToStan(req, res, '/training/status'));
  app.post('/api/stan/training/start', (req, res) => void proxyToStan(req, res, '/training/start'));
  app.post('/api/stan/training/stop', (req, res) => void proxyToStan(req, res, '/training/stop'));
  app.post('/api/stan/training/import', (req, res) => void proxyToStan(req, res, '/training/import'));
  app.get('/api/stan/evaluation', (req, res) => void proxyToStan(req, res, '/evaluation'));
  app.post('/api/stan/evaluation/run', (req, res) => void proxyToStan(req, res, '/evaluation/run'));

  if (!IS_PROD) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.join(__dirname, 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  const server = app.listen(PORT, '0.0.0.0', () => {
    console.log(`[STAN Web] Listening on http://0.0.0.0:${PORT} -> FastAPI ${STAN_API_URL}`);
  });

  const cleanup = () => {
    shuttingDown = true;
    if (pythonProc && !pythonProc.killed) {
      pythonProc.kill('SIGTERM');
    }
    server.close(() => process.exit(0));
  };
  process.on('SIGINT', cleanup);
  process.on('SIGTERM', cleanup);
}

void createServer();
