# Quick Reference — Track 1 Smart Agent

One-page cheat sheet. Details: [README.md](README.md) · project state: [HANDOFF.md](HANDOFF.md).

## What's what

| File | Purpose |
|---|---|
| `main.py` | The agent: classify (local model, free) → route → solve → `/output/results.json` |
| `streamlit_app.py` | Live demo UI over the same pipeline (judges / testing) |
| `evaluate.py` | Dev-only LLM-judge scoring (accuracy, tokens, latency) |
| `Dockerfile` | Submission image: python-slim + llama-cpp-python (source-built) + bundled 3B GGUF |
| `demo_app/sample_tasks_and_solutions/` | 28 labelled demo tasks + ideal answers (used by the Streamlit app) |
| `.github/workflows/docker-build.yml` | CI: builds → pushes to GHCR → verifies linux/amd64 → smoke-tests |

## Setup (once)

```bash
pip install -r requirements.txt llama-cpp-python   # llama-cpp only needed for USE_LOCAL

mkdir -p models        # local weights, ~1.9 GB
curl -fL --http1.1 --retry 5 -o models/model.gguf \
  https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf

cp .env.example .env   # put your FIREWORKS_API_KEY in it
```

Key `.env` entries: `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`
(`https://api.fireworks.ai/inference/v1`), `ALLOWED_MODELS` (full
`accounts/fireworks/models/<name>` IDs — short names 404),
`USE_GEMMA=false` (until you deploy Gemma at <https://app.fireworks.ai/models>),
`USE_LOCAL=true`, `LOCAL_MODEL_PATH=./models/model.gguf`.

## Run the agent + evaluate

```bash
INPUT_PATH=./input/tasks.json OUTPUT_PATH=./output/results.json python main.py
INPUT_PATH=./input/tasks.json OUTPUT_PATH=./output/results.json python evaluate.py
```

Outputs: `output/results.json` (answers), `output/metrics.json` (tokens +
latency per task), `output/evaluation.json` (judge report).

## Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

- Credentials/models prefill from `.env`; you can override them in the sidebar.
- **Single prompt** tab: pick one of the 28 labelled samples (shows routing
  vs ground truth + ideal answer) or write your own.
- **Batch** tab: demo set (with routing-accuracy score), bundled
  `input/tasks.json`, or upload — totals + `results.json` download.
- Leave **USE_LOCAL** unchecked unless `models/model.gguf` exists and
  llama-cpp-python is installed on this machine.

Deploy to <https://share.streamlit.io>: New app → this repo → main file
`streamlit_app.py` → add `FIREWORKS_API_KEY` etc. under app **Secrets**
(full steps in README → "Deploying to Streamlit Community Cloud").

## Build the Docker image

**Easiest — CI does it**: every push to `main` touching
`main.py`/`Dockerfile`/`requirements.txt` builds and pushes
`ghcr.io/<owner>/track-one-amd`, verifies the `linux/amd64` manifest, and
smoke-tests the container. One-time: flip the GHCR package to **Public**
(github.com → your profile → Packages → package settings → visibility).

**Manual, from an Apple Silicon Mac** (judging VM is linux/amd64 — a plain
`docker build` on M1 produces arm64 and scores zero):

```bash
docker buildx create --use 2>/dev/null || true

docker buildx build --platform linux/amd64 \
  -t docker.io/<user>/smart-agent:latest --push .

# must show linux/amd64
docker buildx imagetools inspect docker.io/<user>/smart-agent:latest
```

Run it like the judging harness does:

```bash
docker run --rm --platform linux/amd64 \
  -e FIREWORKS_API_KEY=... \
  -e FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1 \
  -e ALLOWED_MODELS="accounts/fireworks/models/minimax-m3,accounts/fireworks/models/kimi-k2p7-code" \
  -e USE_GEMMA=false \
  -v "$(pwd)/input:/input:ro" -v "$(pwd)/output:/output" \
  <image>
```

Don't touch the Dockerfile's `--no-binary llama-cpp-python` (the wheel
alternative is musl-linked and silently kills the local model in the
container) or its `GGML_NATIVE=OFF` (portable amd64 binary).

## Submitting

- Ship with `USE_GEMMA` **unset** (defaults true) — the judges serve all models.
- Image must be publicly pullable, `linux/amd64`, under 10 GB compressed.
- Limits: 10 submissions/hour, 10-minute runtime, exit code 0, valid JSON output.
