# Quick Reference — Track 1

Two submission images, both CI-built for `linux/amd64`, both public on GHCR.
Details in [README.md](README.md); project state in [HANDOFF.md](HANDOFF.md).

## Submission links

| Variant | Image | Validation score | Billable tokens |
|---|---|---|---|
| **Local-only (primary)** | `ghcr.io/amiteshshekhar12/track-one-local-only:latest` | 10/10 | 0 |
| Hybrid (fallback) | `ghcr.io/amiteshshekhar12/track-one-amd:latest` | 8/10 | ~3,760 |

Submit under **Track 1** only — the Track 2 slot injects no env vars and
judges video captions; these agents will always fail there.

## How the images get (re)built

- Hybrid: push to `main` touching `main.py` / `Dockerfile` /
  `requirements.txt` → `docker-build.yml` rebuilds `track-one-amd`.
- Local-only: push to branch `local-only` touching `only_local_approach/**`
  → `docker-build-local-only.yml` rebuilds `track-one-local-only`.
- Check progress: repo → **Actions** tab. Green check = image is on GHCR
  (push happens before the smoke-test steps). Both workflows verify the
  amd64 manifest and smoke-test the container like the harness does.

## Local dev loop

```bash
pip install -r requirements.txt llama-cpp-python
cp .env.example .env          # Fireworks key; keep USE_GEMMA=false unless deployed
mkdir -p models && curl -fL --retry 5 -o models/model-3b.gguf \
  https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf

# run hybrid → results + metrics, then judge-score the run
INPUT_PATH=./input/tasks.json OUTPUT_PATH=./output/results.json python main.py
INPUT_PATH=./input/tasks.json OUTPUT_PATH=./output/results.json python evaluate.py

# demo UI (same Agent, live)
streamlit run streamlit_app.py
```

Local-only agent: `only_local_approach/main.py` (on the `local-only`
branch) — needs no env vars at all; `LOCAL_MODEL_PATH` points at a GGUF,
`MAX_RUNTIME_S` tunes the time budget (default 570 s).

## Key facts the judges' harness relies on

- Reads `/input/tasks.json`, writes valid `/output/results.json` (one entry
  per task, IDs preserved) — the local-only agent writes incrementally, so
  even a timeout leaves valid output.
- Hybrid reads `FIREWORKS_API_KEY` / `FIREWORKS_BASE_URL` /
  `ALLOWED_MODELS` from the environment; nothing hardcoded. Local-only
  needs nothing.
- Images: hybrid ~2.3 GB, local-only ~5.2 GB — both far under the 10 GB cap.
- Limits: 10-minute runtime, 10 submissions/hour, exit code 0.
