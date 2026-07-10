# Smart General-Purpose Agent — AMD Hackathon Track 1

A token-efficient hybrid agent. Every prompt is first **classified** into one
of the 8 capability categories plus a difficulty level — by a small **local
model bundled in the image** (its tokens are never recorded by the judging
proxy, so classification is free). The prompt is then **routed**: easy tasks
are answered by the local model at zero token cost, everything else goes to
the Fireworks model best suited for that category/difficulty.

## How it works

```
/input/tasks.json
      │
      ▼
┌──────────────────────────┐  category (1-8)   ┌───────────────────────────┐
│ Stage 1: CLASSIFY        │ ────────────────▶ │ Stage 2: SOLVE            │
│ local GGUF model         │  + difficulty     │ easy → local model (free) │
│ (0 recorded tokens);     │  (easy/med/hard)  │ rest → routed Fireworks   │
│ falls back to smallest   │                   │ model, per-category token │
│ Fireworks model          │                   │ cap + system prompt       │
└──────────────────────────┘                   └───────────────────────────┘
      │
      ▼
/output/results.json  (+ /output/metrics.json)
```

### Models and routing

Fireworks models are ranked smallest → largest at runtime from
`ALLOWED_MODELS` using name heuristics — for the published list that gives:

`gemma-4-26b-a4b-it` (MoE, few active params) < `gemma-4-31b-it-nvfp4`
(quantised) < `gemma-4-31b-it` < `minimax-m3` < `kimi-k2p7-code`
(code-specialised, detected by name). Nothing is hardcoded — roles adapt to
whatever list the harness injects.

| Category | easy | medium | hard |
|---|---|---|---|
| Factual knowledge | **local** | small | medium |
| Mathematical reasoning | small | medium | large |
| Sentiment classification | **local** | **local** | small |
| Text summarisation | **local** | small | medium |
| Named entity recognition | **local** | small | medium |
| Code debugging | code | code | code |
| Logical reasoning | medium | large | large |
| Code generation | code | code | code |

If the local model is unavailable, "local" degrades to "small" (smallest
Fireworks model). If a remote call fails after retries, the local model
answers as a last resort so the accuracy gate never sees an empty answer.

### The local model

The judging VM has **no Ollama or model runtime pre-installed**, so the GGUF
weights are baked into the Docker image at build time and served with
`llama-cpp-python` (pure in-process, CPU). Default weights:
Qwen2.5-3B-Instruct Q4_K_M (~2 GB) — swap via the `MODEL_URL` build arg.
Local tokens count as **zero** for the final score, which is why the agent
pushes classification and easy categories onto it.

### Token-efficiency measures

- Classification runs locally (free) with a truncated prompt and compact JSON reply.
- Easy/low-risk categories are answered locally — zero recorded tokens.
- Each category has a `max_tokens` cap and a system prompt demanding concise answers.
- `temperature=0`; 8 tasks run concurrently; per-request timeout 25 s; a
  global 9-minute budget guarantees `/output/results.json` is written before
  the 10-minute limit.

## Flags and environment variables

Injected by the judging harness (read at runtime, never hardcoded):

| Variable | Description |
|---|---|
| `FIREWORKS_API_KEY` | API key (harness-provided) |
| `FIREWORKS_BASE_URL` | Base URL — **all** remote calls go through it |
| `ALLOWED_MODELS` | Comma-separated permitted model IDs |

Agent flags (optional):

| Variable | Default | Description |
|---|---|---|
| `USE_GEMMA` | `true` | **Set to `false` when testing with your own Fireworks key if you have not deployed the Gemma models.** Gemma is allowed but *on-demand*: deploy it at <https://app.fireworks.ai/models> first — a 404 means "not deployed", not "banned". With `USE_GEMMA=false` all `gemma-*` entries are dropped from `ALLOWED_MODELS` and routing uses the remaining models. |
| `USE_LOCAL` | `true` | Set to `false` to disable the bundled local model (everything then goes to Fireworks). |
| `LOCAL_MODEL_PATH` | `/app/models/model.gguf` | Path to the GGUF weights. |
| `INPUT_PATH` / `OUTPUT_PATH` / `METRICS_PATH` | `/input/tasks.json`, `/output/results.json`, `<output dir>/metrics.json` | Handy for local runs. |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` | — | Accepted as fallbacks for the Fireworks vars during local development. |

### Using the Gemma models (deploying them on Fireworks)

The Gemma entries in `ALLOWED_MODELS` are **on-demand** models: they are
allowed, but nothing is serving them until *you* deploy them on your
Fireworks account. Until then every call returns **HTTP 404 — that means
"not deployed", not "banned"**.

Where the flag lives:

- `main.py` → `get_config()` reads `env_flag("USE_GEMMA", True)` and, when
  false, drops every model whose ID contains `gemma` from `ALLOWED_MODELS`
  before routing roles are assigned.
- `.env` / `.env.example` → `USE_GEMMA=false` is pre-set for local testing.

To actually use Gemma:

1. Log in to <https://app.fireworks.ai/models> with the account that owns
   your API key.
2. Search for the model (e.g. `gemma-4-26b-a4b-it`) and click **Deploy** to
   create an on-demand deployment. Wait until its status is *Ready*
   (billed per GPU-second while deployed — undeploy when done testing).
3. Verify it responds — a deployed model returns 200:

   ```bash
   curl -s -o /dev/null -w "%{http_code}\n" \
     https://api.fireworks.ai/inference/v1/chat/completions \
     -H "Authorization: Bearer $FIREWORKS_API_KEY" -H "Content-Type: application/json" \
     -d '{"model":"accounts/fireworks/models/gemma-4-26b-a4b-it","max_tokens":5,"messages":[{"role":"user","content":"hi"}]}'
   ```

4. Set `USE_GEMMA=true` (or just delete the line — `true` is the default)
   and the router will start using the Gemma tiers again.

During judging the organisers' harness serves all allowed models, so the
submitted container should run with the default `USE_GEMMA=true`.

## Project layout

```
main.py              # the agent (classify → route → solve → write results + metrics)
evaluate.py          # local evaluation pipeline (LLM judge, tokens, latency)
streamlit_app.py     # live-demo UI wrapping the same pipeline (see below)
Dockerfile           # python:3.12-slim + llama-cpp-python + bundled GGUF weights
requirements.txt     # openai, streamlit, pandas
.env.example         # template for local development
input/tasks.json     # sample tasks covering all 8 categories, for local testing
models/model.gguf    # local weights (downloaded; baked into the image at build)
.github/workflows/   # CI: builds/pushes/smoke-tests the linux/amd64 image on every push
```

## Run locally

```bash
pip install -r requirements.txt llama-cpp-python

# download the local model weights once (skip and set USE_LOCAL=false to go remote-only)
mkdir -p models
curl -fL -o models/model.gguf \
  https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf

cp .env.example .env        # fill in your key; keep USE_GEMMA=false unless deployed

INPUT_PATH=./input/tasks.json OUTPUT_PATH=./output/results.json python main.py
cat output/results.json
```

The agent also writes `output/metrics.json`: per-task token usage (remote and
local counted separately), the model used, per-response elapsed time, and
totals. The judging harness ignores it; the evaluation pipeline consumes it.

## Run in a Jupyter instance at notebooks.amd.com (testing)

1. Log in at <https://notebooks.amd.com> and start a Jupyter instance.
2. Upload the project (or clone it) and install dependencies — in a notebook
   cell:

   ```python
   !git clone https://github.com/<your-user>/<your-repo>.git agent && cd agent
   %cd agent
   %pip install -r requirements.txt llama-cpp-python
   ```

3. Download the local model weights once:

   ```python
   !mkdir -p models
   !curl -fL -o models/model.gguf \
     https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf
   ```

4. Set the environment (cell-level env is inherited by `!` subprocesses):

   ```python
   import os
   os.environ.update({
       "FIREWORKS_API_KEY": "<your-fireworks-key>",
       "FIREWORKS_BASE_URL": "https://api.fireworks.ai/inference/v1",
       "ALLOWED_MODELS": "accounts/fireworks/models/minimax-m3,accounts/fireworks/models/kimi-k2p7-code,accounts/fireworks/models/gemma-4-31b-it,accounts/fireworks/models/gemma-4-26b-a4b-it,accounts/fireworks/models/gemma-4-31b-it-nvfp4",
       "USE_GEMMA": "false",          # flip to "true" once you deploy Gemma
       "LOCAL_MODEL_PATH": "models/model.gguf",
       "INPUT_PATH": "input/tasks.json",
       "OUTPUT_PATH": "output/results.json",
   })
   ```

5. Run the agent, then (optionally) the evaluation pipeline:

   ```python
   !python main.py
   !python evaluate.py
   import json, pathlib
   print(json.dumps(json.loads(pathlib.Path("output/results.json").read_text()), indent=2)[:2000])
   ```

Notes: llama-cpp-python runs the 3B model on CPU, which is plenty for
classification and easy tasks even without touching the instance's GPUs. If
you want AMD-GPU acceleration for local inference, build it with ROCm/hipBLAS
(`CMAKE_ARGS="-DGGML_HIP=on" pip install llama-cpp-python --no-binary :all:`)
— optional, not required for testing.

## Evaluation pipeline

`evaluate.py` scores a finished agent run so you can iterate on accuracy and
token efficiency locally — mirroring how the hackathon judges (LLM-judge
accuracy gate, then token ranking).

What it measures:

- **Accuracy** — for every task the pipeline first calls a *bigger* model
  (`JUDGE_MODEL`, defaulting to the largest model in `ALLOWED_MODELS` after
  the `USE_GEMMA` filter) to generate an ideal reference answer, then asks
  the same model to grade the agent's answer against that reference on a
  0.0–1.0 scale (semantic correctness, not wording). Reported as mean score
  plus a pass rate at the 0.7 threshold.
- **Tokens used** — total and per task, read from the agent's
  `output/metrics.json`; remote (billable) and local (free) tokens are
  tracked separately, and judge tokens never count against the agent.
- **Time elapsed per response** — per-task wall time from `metrics.json`.
- **Total time elapsed** — the agent's full run time.

### How to run

```bash
# 1. run the agent first so results.json + metrics.json exist
INPUT_PATH=./input/tasks.json OUTPUT_PATH=./output/results.json python main.py

# 2. evaluate the run (same env vars / .env as the agent)
INPUT_PATH=./input/tasks.json OUTPUT_PATH=./output/results.json python evaluate.py

# optional: pin a specific reference/judge model
JUDGE_MODEL=accounts/fireworks/models/minimax-m3 INPUT_PATH=./input/tasks.json \
  OUTPUT_PATH=./output/results.json python evaluate.py
```

The console prints a per-task table (score, tokens, latency, category, judge
reason) and an aggregate summary. A full report — including each generated
ideal answer — is written to `output/evaluation.json`. Evaluation env vars
(all optional): `JUDGE_MODEL`, `METRICS_PATH`, `EVAL_OUTPUT_PATH`.
`evaluate.py` is a local dev tool only — it is not copied into the container.

## Streamlit demo UI

`streamlit_app.py` is a thin visual wrapper around the exact same `Agent`
class, `ROUTING` table and model-tier heuristics in `main.py` — it is not a
second implementation. It's the live-demo artifact for judges: paste a
prompt or upload a `tasks.json` and watch the classification, routing
decision, answer, and token/latency cost happen in real time against your
real Fireworks account. It is **not** copied into the submitted Docker
image (the Dockerfile only `COPY`s `main.py`); it's a separate deployment.

Run it locally:

```bash
pip install -r requirements.txt   # now includes streamlit + pandas
streamlit run streamlit_app.py
```

Fill in the Fireworks credentials and `ALLOWED_MODELS` in the sidebar (they
default to whatever is already in your environment/`.env`), then use the
"Single prompt" tab for a one-off demo or "Batch (tasks.json)" to run the
full sample set and see the aggregate token/time metrics. Leave `USE_LOCAL`
unchecked unless you've downloaded `models/model.gguf` and installed
`llama-cpp-python` on the machine running Streamlit — there are no bundled
weights outside the Docker image.

### Deploying to Streamlit Community Cloud

1. Go to <https://share.streamlit.io>, sign in, and pick **New app** from
   this GitHub repo.
2. Main file path: `streamlit_app.py`. Branch: `main`.
3. Streamlit Cloud auto-installs from the repo-root `requirements.txt`
   (already includes `streamlit`/`pandas` alongside the agent's `openai`
   dependency) — no extra config needed.
4. Optional: under **Settings → Secrets**, add `FIREWORKS_API_KEY`,
   `FIREWORKS_BASE_URL` and `ALLOWED_MODELS` so the deployed app comes up
   pre-filled for a live demo instead of requiring the presenter to paste a
   key into the sidebar each time. The app reads `st.secrets` as a fallback
   whenever those keys aren't already in the environment.
5. Leave `USE_LOCAL` off in the deployed app — Community Cloud has no
   bundled GGUF weights and limited RAM/CPU; the demo is meant to show the
   real Fireworks-routing behavior, not the local-model path (that's what
   the Docker image + CI smoke test verify instead).

## Build the Docker image

The image bundles the GGUF weights (no runtime is pre-installed on the
judging VM) and must target `linux/amd64`:

```bash
# plain build (Intel/AMD host)
docker build -t smart-agent .

# optional: bundle different weights
docker build --build-arg MODEL_URL=https://huggingface.co/.../other.gguf -t smart-agent .
```

### Building on an Apple Silicon Mac (M1/M2/M3/M4)

The judging VM runs `linux/amd64`; an image built with a plain
`docker build` on an M-series Mac gets a `linux/arm64` manifest and will
**fail to pull and score zero**. Always pass `--platform linux/amd64`:

```bash
# one-time: make sure a buildx builder exists (ships with Docker Desktop)
docker buildx create --use 2>/dev/null || true

# build for linux/amd64 and push to a public registry in one step
docker buildx build --platform linux/amd64 \
  -t docker.io/<your-user>/smart-agent:latest --push .

# verify the pushed image really has a linux/amd64 manifest
docker buildx imagetools inspect docker.io/<your-user>/smart-agent:latest
```

Notes:

- Building works fine on M1 — the Python deps install as amd64 under
  emulation and the GGUF weights are architecture-independent. Expect the
  llama-cpp-python step to take several minutes under QEMU.
- *Running* the amd64 container locally on the Mac also works
  (`docker run --platform linux/amd64 ...`) but is slow under emulation —
  use native Python (above) for your test loop and Docker only for the
  final check + push.

Run it like the harness does:

```bash
docker run --rm \
  -e FIREWORKS_API_KEY=your-key \
  -e FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1 \
  -e ALLOWED_MODELS="accounts/fireworks/models/minimax-m3,accounts/fireworks/models/kimi-k2p7-code,accounts/fireworks/models/gemma-4-31b-it,accounts/fireworks/models/gemma-4-26b-a4b-it,accounts/fireworks/models/gemma-4-31b-it-nvfp4" \
  -e USE_GEMMA=false \
  -v "$(pwd)/input:/input:ro" \
  -v "$(pwd)/output:/output" \
  smart-agent
```

For the actual submission leave `USE_GEMMA` unset (defaults to `true`) —
the organisers' harness runs against deployed models. The image contains no
`.env` and no credentials; the harness injects the real values at evaluation
time. Compressed image size stays far under the 10 GB limit (~2.5 GB with the
default 3B Q4 weights).

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
