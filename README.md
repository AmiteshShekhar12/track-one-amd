# Smart General-Purpose Agent — AMD Hackathon Track 1

A token-efficient general-purpose AI agent. For every prompt it first makes a
tiny classification call to the **smallest** allowed model, which labels the
task with one of the 8 capability categories and a difficulty level. It then
routes the prompt to the model best suited for that category/difficulty, so
easy tasks never burn tokens on a large model.

## How it works

```
/input/tasks.json
      │
      ▼
┌─────────────────────┐   category (1-8)    ┌──────────────────────┐
│ Stage 1: CLASSIFY   │ ──────────────────▶ │ Stage 2: SOLVE       │
│ smallest model,     │   + difficulty      │ routed model,        │
│ ~30 output tokens   │   (easy/med/hard)   │ per-category token   │
└─────────────────────┘                     │ cap + system prompt  │
                                            └──────────────────────┘
      │
      ▼
/output/results.json
```

### Categories and routing

Models are ranked smallest → largest at runtime from `ALLOWED_MODELS` using
name heuristics (e.g. `Qwen3.6 Plus` < `Qwen3.7 Plus` < `MiniMax-M3` <
`Kimi K2.7 Code`), and a code-specialised model is detected by name
(`code`/`coder`/`kimi`). Nothing is hardcoded — the roles adapt to whatever
model list is published on launch day.

| Category | easy | medium | hard |
|---|---|---|---|
| Factual knowledge | small | small | medium |
| Mathematical reasoning | small | medium | large |
| Sentiment classification | small | small | small |
| Text summarisation | small | small | medium |
| Named entity recognition | small | small | medium |
| Code debugging | code | code | code |
| Logical reasoning | medium | large | large |
| Code generation | code | code | code |

Token efficiency measures (scoring ranks by total tokens):

- Classification uses a truncated prompt (first 1500 chars) and a compact JSON reply.
- Each category has a `max_tokens` cap and a system prompt that demands concise answers.
- `temperature=0` for deterministic, non-rambling output.
- 8 tasks run concurrently; per-request timeout 25s; a global 9-minute budget
  guarantees `/output/results.json` is written before the 10-minute limit.
- Any failure falls back to the medium model; a task that still fails gets an
  empty answer rather than corrupting the output JSON.

## Project layout

```
main.py            # the whole agent (classify → route → solve → write results)
Dockerfile         # python:3.12-slim, runs main.py
requirements.txt   # openai (async client, OpenAI-compatible endpoints)
.env.example       # template for local development
input/tasks.json   # sample tasks covering all 8 categories, for local testing
```

## Environment variables

Injected by the judging harness (read at runtime, never hardcoded):

| Variable | Description |
|---|---|
| `FIREWORKS_API_KEY` | API key (harness-provided) |
| `FIREWORKS_BASE_URL` | Base URL — **all** calls go through it |
| `ALLOWED_MODELS` | Comma-separated permitted model IDs |

Optional (local development): `OPENAI_API_KEY`/`OPENAI_BASE_URL` are accepted
as fallbacks, plus `INPUT_PATH` / `OUTPUT_PATH` to avoid needing `/input` and
`/output` on your machine.

## Run locally

```bash
pip install -r requirements.txt

cp .env.example .env        # fill in your key, base URL and model list

INPUT_PATH=./input/tasks.json OUTPUT_PATH=./output/results.json python main.py

cat output/results.json
```

## Run with Docker

```bash
docker build --platform linux/amd64 -t smart-agent .

docker run --rm \
  -e FIREWORKS_API_KEY=your-key \
  -e FIREWORKS_BASE_URL=https://your-endpoint/v1 \
  -e ALLOWED_MODELS="qwen3.6-plus,qwen3.7-plus,minimax-m3,kimi-k2.7-code" \
  -v "$(pwd)/input:/input:ro" \
  -v "$(pwd)/output:/output" \
  smart-agent
```

## Submit

Build for `linux/amd64` (required — the judging VM rejects other
architectures) and push to a public registry:

```bash
docker buildx build --platform linux/amd64 \
  -t docker.io/<your-user>/smart-agent:latest --push .
```

The image contains no `.env` and no credentials; the harness injects the real
values at evaluation time.

## Input / output format

`/input/tasks.json`:

```json
[ { "task_id": "t1", "prompt": "..." } ]
```

`/output/results.json`:

```json
[ { "task_id": "t1", "answer": "..." } ]
```

The agent exits `0` on success and non-zero on failure, and always emits valid
JSON with one entry per input task (empty string for any task that could not
be answered).
