# Smart General-Purpose Agent — Track 1

Two Track 1 submissions built from one codebase, scored on the organisers'
retired validation set (their pass criteria, LLM-judge confirmed):

| Variant | Image | Score | Billable tokens |
|---|---|---|---|
| **Local-only** (primary) | `ghcr.io/amiteshshekhar12/track-one-local-only:latest` | **10/10** | **0** |
| **Hybrid** (fallback) | `ghcr.io/amiteshshekhar12/track-one-amd:latest` | 8/10 | ~3,760 |

Both read `/input/tasks.json`, write `/output/results.json`, exit 0, and are
built for `linux/amd64` by CI. Scoring ranks accuracy-gate passers by total
billable tokens, ascending — which is why the local-only variant exists.

## Local-only variant (branch `local-only`, folder `only_local_approach/`)

All inference — classification and answering — runs on a **Qwen2.5-7B-Instruct
Q4 GGUF bundled in the image** (llama-cpp-python, in-process, no Ollama). The
container needs **no env vars and no network**: the judging proxy records
**zero tokens**.

Accuracy hardening (all generic, nothing keyed to known answers):

- **Coverage-first prompts** per category — answer every sub-question, both
  sides of any comparison; mixed-tone sentiment is never labelled Negative.
- **Deterministic format verification** — "exactly N sentences / N bullets,
  each ≤ M words" constraints are parsed from the task, checked
  programmatically, and violations trigger one corrective retry (free —
  local tokens cost nothing).
- **Extract-then-write summarisation scaffold** — list the passage's items
  first, then write the summary naming every item, one theme per
  bullet/sentence.
- **NER output discipline** — verbatim entities, exact label names, full
  dates as one entity, no invented generics.
- **Crash-safe output** — `results.json` is rewritten after every task; a
  timeout still leaves valid, complete-so-far JSON. Budget via
  `MAX_RUNTIME_S` (default 570 s).

Sequential 7B inference on CPU is the trade-off: ~10 tasks fit the limit on
decent hardware, and the incremental writer degrades gracefully if not.

## Hybrid variant (branch `main`, `main.py`)

Two-stage routing that halves billable tokens while staying accurate:

1. **Classify** (free): a bundled Qwen2.5-3B labels each prompt with one of
   8 categories + difficulty via a few-shot prompt; falls back to the
   smallest Fireworks model if the local model is unavailable.
2. **Route & solve**: easy sentiment/factual/summarisation/NER → answered
   locally (0 tokens); code → the code-specialised model; hard math/logic →
   the largest general model; middle tiers → cheapest capable model. Tiers
   are inferred at runtime from `ALLOWED_MODELS` by name heuristics —
   nothing hardcoded. Remote failures fall back to local and vice versa, so
   the judge never sees an empty answer.

Env vars (harness-injected; never hardcoded): `FIREWORKS_API_KEY`,
`FIREWORKS_BASE_URL`, `ALLOWED_MODELS`. Optional flags: `USE_GEMMA=false`
drops `gemma-*` models when testing with a key that hasn't deployed them
(Gemma is *on-demand* on Fireworks — deploy at
<https://app.fireworks.ai/models>; a 404 means "not deployed", not
"banned"); `USE_LOCAL=false` disables the local model;
`INPUT_PATH`/`OUTPUT_PATH` override paths for local runs.

## Repository layout

```
main.py                  # hybrid agent (classify → route → solve)
only_local_approach/     # local-only agent + its Dockerfile (local-only branch)
evaluate.py              # dev-only LLM-judge eval (accuracy, tokens, latency)
streamlit_app.py         # live demo UI over the hybrid pipeline
Dockerfile               # hybrid image: python-slim + llama-cpp + 3B GGUF
requirements.txt         # openai, streamlit, pandas (pinned for Streamlit Cloud)
input/tasks.json         # 8 sample tasks, one per category
demo_app/sample_tasks_and_solutions/  # 28 labelled demo tasks for the UI
.github/workflows/       # CI: one workflow per image (see below)
```

## CI/CD — how the images get built

No manual Docker builds needed; GitHub Actions builds natively on amd64:

- **`docker-build.yml`** — push to `main` touching
  `main.py`/`Dockerfile`/`requirements.txt` → builds, pushes
  `ghcr.io/amiteshshekhar12/track-one-amd` (`:latest` + commit SHA),
  verifies the `linux/amd64` manifest, smoke-tests the container.
- **`docker-build-local-only.yml`** — push to the `local-only` branch
  touching `only_local_approach/**` → same pipeline for
  `ghcr.io/amiteshshekhar12/track-one-local-only`, including a **no-env**
  container smoke test (that's how the harness runs it).

Green check in the Actions tab = image is already on GHCR. Both packages
are public (anonymous pull verified).

## Run and evaluate locally

```bash
pip install -r requirements.txt llama-cpp-python

mkdir -p models   # hybrid uses the 3B; local-only variant uses the 7B
curl -fL --retry 5 -o models/model-3b.gguf https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf

cp .env.example .env    # your Fireworks key; USE_GEMMA=false unless deployed

# hybrid
INPUT_PATH=./input/tasks.json OUTPUT_PATH=./output/results.json \
  LOCAL_MODEL_PATH=./models/model-3b.gguf python main.py

# local-only (from the local-only branch)
cd only_local_approach && INPUT_PATH=../input/tasks.json \
  OUTPUT_PATH=../output/results.json LOCAL_MODEL_PATH=../models/model-3b.gguf python main.py

# LLM-judge evaluation of any run (accuracy, tokens, per-task latency)
INPUT_PATH=./input/tasks.json OUTPUT_PATH=./output/results.json python evaluate.py
```

The agents write `metrics.json` next to the results (per-task tokens —
remote vs free local — model used, latency); `evaluate.py` consumes it and
writes `evaluation.json`. This all runs the same way in a hosted Jupyter
instance — clone, `%pip install`, set the env in a cell, `!python main.py`.

## Streamlit demo

```bash
streamlit run streamlit_app.py
```

Live wrapper around the real hybrid `Agent`: pick one of 28 labelled sample
tasks (shows routing vs ground truth + ideal answer), write your own prompt,
or run batches with token totals and a `results.json` download. Deployed on
Streamlit Community Cloud (main file `streamlit_app.py`; credentials via
app **Secrets** — never prefilled into widgets).

## Input / output contract

```json
[ { "task_id": "t1", "prompt": "..." } ]        // /input/tasks.json
[ { "task_id": "t1", "answer": "..." } ]        // /output/results.json
```

One entry per input task, task IDs preserved, valid JSON always — even on
partial failure. Exit code 0 on success.
