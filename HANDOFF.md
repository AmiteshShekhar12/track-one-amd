# Handoff — AMD Hackathon Track 1: Smart General-Purpose Agent

Last updated: 2026-07-10. This document is the single place to understand
where the project stands, how to run everything, and what's left before
submission. Full usage detail lives in [README.md](README.md).

## What this is

A Track 1 (General-Purpose AI Agent) submission. Reads `/input/tasks.json`,
answers each prompt, writes `/output/results.json`. Scored by an LLM-judge
accuracy gate, then ranked by **fewest billable tokens**. The core idea:

1. A small **local model bundled in the Docker image** (Qwen2.5-3B-Instruct
   Q4_K_M GGUF via llama-cpp-python — no Ollama, per the rules) classifies
   every prompt into one of 8 categories + difficulty using a few-shot
   prompt. Local tokens are not recorded by the judging proxy → free.
2. Easy/low-risk tasks (sentiment, easy factual/summarisation/NER) are
   answered locally at zero token cost; everything else routes to the
   cheapest suitable Fireworks model. Code → Kimi-K2p7-Code, hard
   reasoning → MiniMax-M3, mid tiers → Gemma (when deployed).
3. Model tiers are inferred from `ALLOWED_MODELS` at runtime by name
   heuristics (`main.py` → `SIZE_PATTERNS`, `build_roles()`) — nothing is
   hardcoded, per the rules.

## Current status — reverified on 2026-07-10

All testing ran natively (macOS, real Fireworks API, real local inference):

| Run | Remote tokens | Local (free) tokens | Judge accuracy | Pass ≥0.7 | Wall time |
|---|---|---|---|---|---|
| Remote-only (no local model) | 7,351 | 0 | 98.1% | 8/8 | 4.7 s |
| Hybrid (3B local classifier + answers) | **2,856** | 3,429 | 98.75% | 8/8 | 27.1 s |

- The hybrid roughly halves billable tokens — the leaderboard metric — while
  still passing the accuracy gate on all 8 sample tasks (`input/tasks.json`).
  Exact token counts vary run-to-run (LLM outputs, classifier misses on
  1-2 tasks route them differently) but both configurations stay well above
  the 0.7 pass threshold on every task.
- **Bug fixed 2026-07-10**: `.env.example` and three snippets in README.md
  (Jupyter env block, `JUDGE_MODEL` example, `docker run` example) used
  short Fireworks model IDs (e.g. `minimax-m3`), which return **HTTP 404**
  against the real API — confirmed by curl. Fireworks model IDs must be the
  full form `accounts/fireworks/models/<name>`; all four spots are now
  fixed to match what HANDOFF's own docker-run example already had right.
  If you pulled `.env.example` before this fix, regenerate your `.env`.
- MiniMax-M3 and Kimi-K2p7-Code respond serverless (HTTP 200, confirmed via
  curl). All three Gemma models return **404 until you deploy them** on
  your account (see README → "Using the Gemma models"). Testing runs use
  `USE_GEMMA=false`.

## Repository map

```
main.py            # the agent: classify → route → solve → results + metrics
evaluate.py        # dev-only LLM-judge eval (accuracy, tokens, latency)
Dockerfile         # python:3.12-slim + llama-cpp-python + bundled GGUF
requirements.txt   # openai
.env.example       # local-dev template (copy to .env; .env is gitignored)
input/tasks.json   # 8 sample tasks, one per category
HANDOFF.md         # this file
```

Not in git (`.gitignore`): `.env` (holds the real API key), `models/`
(GGUF weights, up to 1.8 GB), `output/`, `__pycache__/`, the guide PDFs.

## Run it locally (native Python — the fast test loop)

```bash
pip install -r requirements.txt llama-cpp-python

mkdir -p models   # ~1.9 GB, one-time
curl -fL --http1.1 --retry 5 -o models/model.gguf \
  https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf

cp .env.example .env   # put the team's Fireworks key in FIREWORKS_API_KEY
                       # keep USE_GEMMA=false unless Gemma is deployed

INPUT_PATH=./input/tasks.json OUTPUT_PATH=./output/results.json python main.py
INPUT_PATH=./input/tasks.json OUTPUT_PATH=./output/results.json python evaluate.py
```

Key flags: `USE_GEMMA` (false = drop gemma-* from routing), `USE_LOCAL`
(false = remote-only), `LOCAL_MODEL_PATH`, `JUDGE_MODEL` (evaluate.py).
The README also has a step-by-step for Jupyter at notebooks.amd.com.

## Build and push the Docker image

**CI now does this automatically.** `.github/workflows/docker-build.yml`
builds the image natively for `linux/amd64` on GitHub's amd64 runners (no
QEMU needed there), pushes it to `ghcr.io/<owner>/<repo>`, verifies the
pushed manifest really is `linux/amd64`, then smoke-tests the container
with deliberately-fake Fireworks credentials — this exercises the bundled
local model, classification, routing, and the remote-failure→local
fallback path without needing a real API key in CI, and fails the job if
any of the 8 sample tasks come back empty. It runs on every push to `main`
that touches `main.py`/`Dockerfile`/`requirements.txt`, or on-demand via
"Run workflow". **One manual step after the first successful run**: the
GHCR package is created private by default — go to
`github.com/<owner>?tab=packages` → the package → Package settings →
change visibility to Public, since the judging harness pulls without auth.
(Note: this repo itself is currently private on GitHub; that does not need
to change — only the pushed *image* needs to be public.)

Local/manual build is still useful for fast iteration. The judging VM is
`linux/amd64`; the image bundles the model weights (downloaded at build
time via `MODEL_URL` build arg — the build needs internet). From an Apple
Silicon Mac:

```bash
# one-time
docker buildx create --use 2>/dev/null || true

# build for linux/amd64 and push to a public registry
docker buildx build --platform linux/amd64 \
  -t docker.io/<user>/smart-agent:latest --push .

# MUST show linux/amd64 before submitting
docker buildx imagetools inspect docker.io/<user>/smart-agent:latest
```

On an Intel/AMD host or CI, a plain `docker build -t ... . && docker push`
works. Smoke-test the pushed image the way the harness runs it:

```bash
docker run --rm --platform linux/amd64 \
  -e FIREWORKS_API_KEY=... \
  -e FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1 \
  -e ALLOWED_MODELS="accounts/fireworks/models/minimax-m3,accounts/fireworks/models/kimi-k2p7-code" \
  -e USE_GEMMA=false \
  -v "$(pwd)/input:/input:ro" -v "$(pwd)/output:/output" \
  docker.io/<user>/smart-agent:latest
```

(Emulated amd64 on a Mac is slow — fine for a smoke test, not for iterating.)

Cross-build hardening already in the Dockerfile — don't remove:
`build-essential cmake git` (source-build fallback), `GGML_NATIVE=OFF`
(portable binary; native tuning under QEMU crashes on the judging VM),
`llama-cpp-python==0.3.19` pin (prebuilt cp312 wheel — avoids an hours-long
QEMU compile), `CMAKE_BUILD_PARALLEL_LEVEL=4` (QEMU OOM), curl retries with
resume (HF drops long downloads; hit this in practice at 92%).

## Submission checklist

- [ ] Deploy the three Gemma models on the team Fireworks account
      (app.fireworks.ai/models → Deploy; verify 200 via curl in README)
      or accept minimax/kimi-only routing.
- [ ] Ship with `USE_GEMMA` **unset** (defaults true) — the judges' harness
      serves all allowed models.
- [x] `docker buildx build --platform linux/amd64 --push`, verify manifest —
      automated by `.github/workflows/docker-build.yml` on every push to
      `main`. Check the Actions tab for a green run before submitting.
- [ ] GHCR package visibility flipped to Public (one-time, see above) — the
      workflow's last step prints the exact settings link as a reminder.
- [ ] Remember: max 10 submissions/hour, 10-minute runtime cap, exit 0.

## Known issues / next steps

1. **Local answer latency**: locally-answered tasks took up to ~13 s wall
   time each on an M-series CPU in the 2026-07-10 rerun (up to ~35 s in
   earlier testing). The 30 s/request rule is about API requests, but if
   the judging VM's CPU is slow, consider routing fewer categories to
   "local" (edit `ROUTING` in `main.py`) or shrinking `MAX_TOKENS`.
2. **Classifier quality**: the few-shot prompt doesn't always land the exact
   category/difficulty with the 3B model (e.g. one sample task classified
   as `code_debug` instead of `code_gen` in the 2026-07-10 rerun); misses so
   far were harmless (adjacent categories routed to the same model, or a
   "medium" instead of "easy" difficulty just meaning one more task goes
   remote instead of local). More few-shot examples or a slightly larger
   local model are the levers if this needs tightening further.
3. ~~Untested: an actual `docker buildx --platform linux/amd64` build~~ —
   now covered by CI (see above); confirm a green Actions run before
   submitting, since that's the actual artifact judges pull.
4. `evaluate.py` is dev-only and never runs in the container; its judge
   defaults to the largest allowed model (Kimi) — override with
   `JUDGE_MODEL` if grading feels off.
5. **Repo is currently private on GitHub.** The GHCR image can be made
   public independently of the repo (see above), so this is only a problem
   if the hackathon rules require the *source* to be publicly visible —
   double-check the rules and flip repo visibility if so.
