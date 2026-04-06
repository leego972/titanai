# TitanAI API — Archibald Integration Guide

## Overview

The TitanAI API is OpenAI-compatible. Archibald can route requests to TitanAI
by adding it as a provider in `server/_core/llm.ts` and setting the
`TITAN_API_URL` environment variable.

---

## Step 1: Add env variable to Archibald

In your Railway / `.env` file, add:

```env
TITAN_API_URL=http://<vast_ai_ip>:8000
TITAN_API_KEY=<your_titan_api_key>   # optional, leave empty for dev
```

---

## Step 2: Patch `server/_core/llm.ts`

Add TitanAI as a provider in the `invokeLLM` function. Find the section where
providers are dispatched (around the `VENICE_API_KEY` check) and add:

```typescript
// ── TitanAI provider ──────────────────────────────────────────────────────
// Routes model: "titan-*" requests to the self-hosted TitanAI API server.
const TITAN_API_URL = process.env.TITAN_API_URL || "";
const TITAN_API_KEY = process.env.TITAN_API_KEY || "";

if (model.startsWith("titan-") && TITAN_API_URL) {
  const titanResponse = await fetch(`${TITAN_API_URL}/v1/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(TITAN_API_KEY ? { "Authorization": `Bearer ${TITAN_API_KEY}` } : {}),
    },
    body: JSON.stringify({
      model,
      messages,
      max_tokens: maxTokens,
      temperature,
      stream,
    }),
  });

  if (!titanResponse.ok) {
    const err = await titanResponse.text();
    throw new Error(`TitanAI API error: ${titanResponse.status} — ${err}`);
  }

  if (stream) {
    return titanResponse; // SSE stream passthrough
  }

  const data = await titanResponse.json();
  return data.choices[0].message.content;
}
```

---

## Step 3: Update `server/_core/env.ts`

Add to the ENV object:

```typescript
titanApiUrl: process.env.TITAN_API_URL ?? "",
titanApiKey: process.env.TITAN_API_KEY ?? "",
```

---

## Step 4: Add TitanAI to the model selector

In `shared/` or wherever models are listed, add:

```typescript
{
  id: "titan-probe-v0.1.5",
  name: "Titan Probe v0.1.5",
  provider: "titanai",
  description: "TitanAI base model — 45M params, trained from scratch",
  tier: "pro",  // or "admin" until fully trained
}
```

---

## Step 5: Start the TitanAI API on the GPU server

After training completes on Vast.ai:

```bash
ssh -i ~/.ssh/vastai_key -p 11328 root@ssh3.vast.ai

cd /workspace/titanai
TITAN_CHECKPOINT_PATH=checkpoints/probe_v015/final.pt \
TITAN_REQUIRE_AUTH=true \
TITAN_API_KEY=your_secret_key \
./scripts/start_api.sh
```

Or in a persistent tmux session:

```bash
tmux new-session -d -s api
tmux send-keys -t api 'cd /workspace/titanai && ./scripts/start_api.sh' Enter
```

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/v1/chat/completions` | POST | OpenAI-compatible chat (Archibald drop-in) |
| `/v1/completions` | POST | Raw text completion |
| `/v1/generate` | POST | Native TitanAI generation |
| `/v1/models` | GET | List checkpoints |
| `/v1/models/{id}` | GET | Model info |
| `/v1/models/load` | POST | Hot-load a checkpoint |
| `/health` | GET | Liveness check |
| `/health/gpu` | GET | GPU memory stats |
| `/health/model` | GET | Model status |
| `/metrics` | GET | Training run metrics |
| `/docs` | GET | Swagger UI |

---

## Example: Test the API

```bash
# Health check
curl http://localhost:8000/health

# Chat completion (same format as OpenAI)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "titan-probe-v0.1.5",
    "messages": [
      {"role": "user", "content": "Explain SQL injection in one sentence."}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'

# Streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "titan-probe-v0.1.5", "messages": [{"role": "user", "content": "Hello"}], "stream": true}'
```
