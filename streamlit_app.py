"""
Streamlit demo UI for the Track 1 hybrid agent.

This is a thin visual wrapper around the exact classify -> route -> solve
pipeline in main.py (same Agent class, same ROUTING table, same model-tier
heuristics) — it is not a separate implementation. Useful for a live judge
demo: pick one of the 28 labelled sample tasks (with ground-truth routing
comparison and ideal answers), paste your own prompt, or upload a
tasks.json, and watch the routing decision, answer, and token/latency cost
in real time.

Deploy on Streamlit Community Cloud: main file path = streamlit_app.py.
Nothing is hardcoded — everything below is either typed into the sidebar
at runtime or read from the environment / Streamlit secrets, matching how
main.py itself is configured in the submitted container.
"""

import asyncio
import json
import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from openai import AsyncOpenAI

import main as agent_mod
from main import Agent, build_roles, load_local_model

ROOT = Path(__file__).resolve().parent
# make config independent of the launch directory
agent_mod.load_dotenv(str(ROOT / ".env"))

SAMPLES_DIR = ROOT / "demo_app" / "sample_tasks_and_solutions"

# ground-truth label -> agent-internal category name
GT_CATEGORY = {
    "factual_knowledge": "factual",
    "mathematical_reasoning": "math",
    "sentiment_classification": "sentiment",
    "text_summarisation": "summarization",
    "named_entity_recognition": "ner",
    "code_debugging": "code_debug",
    "logical_reasoning": "logic",
    "code_generation": "code_gen",
}

st.set_page_config(page_title="Smart Agent — Track 1 demo", page_icon="🤖", layout="wide")


def secret(name, default=""):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def short(model_id):
    return model_id.replace("accounts/fireworks/models/", "")


@st.cache_data
def load_samples():
    """28 labelled demo tasks + ideal answers; empty if not present."""
    try:
        tasks = json.loads((SAMPLES_DIR / "tasks.json").read_text())
        ideal = {x["task_id"]: x for x in
                 json.loads((SAMPLES_DIR / "ideal_answers.json").read_text())}
        return tasks, ideal
    except OSError:
        return [], {}


st.title("🤖 Smart General-Purpose Agent — live demo")
st.caption(
    "Same classify → route → solve pipeline as the submitted agent "
    "(`main.py`). Calls your real Fireworks account."
)

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input(
        "FIREWORKS_API_KEY", type="password",
        value=os.environ.get("FIREWORKS_API_KEY", secret("FIREWORKS_API_KEY")),
    )
    base_url = st.text_input(
        "FIREWORKS_BASE_URL",
        value=os.environ.get(
            "FIREWORKS_BASE_URL", secret("FIREWORKS_BASE_URL",
                                          "https://api.fireworks.ai/inference/v1")),
    )
    models_raw = st.text_area(
        "ALLOWED_MODELS (comma-separated, full accounts/fireworks/models/<name> form)",
        value=os.environ.get(
            "ALLOWED_MODELS",
            secret("ALLOWED_MODELS",
                   "accounts/fireworks/models/minimax-m3,"
                   "accounts/fireworks/models/kimi-k2p7-code"),
        ),
        height=90,
    )
    use_gemma = st.checkbox(
        "USE_GEMMA", value=False,
        help="Only enable if the Gemma models are deployed on this Fireworks account.",
    )
    use_local = st.checkbox(
        "USE_LOCAL (bundled GGUF classifier)", value=False,
        help="Requires llama-cpp-python and a downloaded GGUF file on this host. "
             "Leave off on Streamlit Community Cloud — there is no bundled model there.",
    )
    local_model_path = None
    if use_local:
        local_model_path = st.text_input(
            "LOCAL_MODEL_PATH", value=os.environ.get("LOCAL_MODEL_PATH", "models/model.gguf"))

    st.divider()
    st.caption(
        "This mirrors exactly how main.py reads FIREWORKS_API_KEY / "
        "FIREWORKS_BASE_URL / ALLOWED_MODELS / USE_GEMMA / USE_LOCAL from "
        "the environment inside the submitted container."
    )

models = [m.strip() for m in models_raw.split(",") if m.strip()]
if not use_gemma:
    models = [m for m in models if "gemma" not in m.lower()]

ready = bool(api_key and base_url and models)
if not ready:
    st.info("Fill in a Fireworks API key, base URL and at least one allowed model in the sidebar to start.")


@st.cache_resource(show_spinner="Loading local model…")
def get_local_model(enabled, path):
    if not enabled:
        return None
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved  # relative paths are repo-root relative
    os.environ["USE_LOCAL"] = "true"
    os.environ["LOCAL_MODEL_PATH"] = str(resolved)
    return load_local_model()


local_model = get_local_model(use_local, local_model_path or "models/model.gguf")
if use_local and local_model is None:
    st.sidebar.warning("Local model failed to load — running remote-only. "
                       "Check LOCAL_MODEL_PATH and that llama-cpp-python is installed.")

with st.sidebar:
    if models:
        roles_preview = build_roles(models)
        st.subheader("Role assignment")
        st.table({"role": list(roles_preview),
                  "model": [short(m) for m in roles_preview.values()]})
        st.markdown("**Local model:** " +
                    (f"`{local_model.name}` ✅" if local_model else "off"))


def make_agent():
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0)
    roles = build_roles(models)
    if local_model is not None:
        # the cached model's Lock may be bound to a previous asyncio.run() loop
        local_model.lock = asyncio.Lock()
    return Agent(client, roles, local=local_model)


def run_batch(agent, tasks):
    """Solve tasks with the same bounded concurrency the container uses."""

    async def go():
        semaphore = asyncio.Semaphore(agent_mod.CONCURRENCY)

        async def worker(task):
            async with semaphore:
                return await agent.solve(task)

        return await asyncio.gather(*(worker(t) for t in tasks))

    return asyncio.run(go())


sample_tasks, sample_ideal = load_samples()

tab1, tab2, tab3 = st.tabs(["Single prompt", "Batch", "How it works"])

# ---------------------------------------------------------------------------
# Single prompt — free-form or one of the labelled demo tasks
# ---------------------------------------------------------------------------

with tab1:
    options = ["✍️ Write my own prompt"] + [
        f"{t['task_id']} — {t['prompt'][:80]}" for t in sample_tasks
    ]
    choice = st.selectbox("Prompt source", options) if sample_tasks else options[0]
    if choice.startswith("✍️"):
        sample_id = None
        prompt = st.text_area(
            "Prompt", height=140,
            placeholder="e.g. Write a Python function that reverses a linked list.",
        )
    else:
        sample_id = choice.split(" — ")[0]
        prompt = next(t["prompt"] for t in sample_tasks if t["task_id"] == sample_id)
        st.text_area("Prompt", value=prompt, height=140, disabled=True)

    if st.button("Run", disabled=not (ready and prompt.strip())):
        agent = make_agent()
        try:
            with st.spinner("Classifying → routing → answering…"):
                answer, stats = asyncio.run(
                    agent.solve({"task_id": sample_id or "demo", "prompt": prompt}))
        except Exception as e:  # noqa: BLE001 — surface to the demo UI, don't crash the app
            st.error(f"Run failed: {e}")
        else:
            where = "🖥️ local — 0 billable tokens" if stats["model"].startswith("local:") \
                else "☁️ Fireworks"
            st.success(
                f"Category: **{stats['category']}** · Difficulty: **{stats['difficulty']}** "
                f"· Model: `{short(stats['model'])}` ({where}) · {stats['elapsed_s']}s"
            )
            st.markdown(answer if answer else "_empty answer_")
            c1, c2 = st.columns(2)
            c1.metric("Remote tokens (billable)", stats["tokens"]["total"])
            c2.metric("Local tokens (free)", stats["local_tokens"]["total"])

            if sample_id and sample_id in sample_ideal:
                truth = sample_ideal[sample_id]
                expected = GT_CATEGORY.get(truth["category"], truth["category"])
                ok_cat = "✅" if stats["category"] == expected else "❌"
                ok_dif = "✅" if stats["difficulty"] == truth["difficulty"] else "≈"
                st.caption(
                    f"Ground truth: category **{expected}** {ok_cat} · "
                    f"difficulty **{truth['difficulty']}** {ok_dif}"
                )
                with st.expander("Show ideal answer"):
                    st.markdown(truth["ideal_answer"])

# ---------------------------------------------------------------------------
# Batch — demo set with ground truth, bundled sample, or uploaded tasks.json
# ---------------------------------------------------------------------------

with tab2:
    sources = []
    if sample_tasks:
        sources.append(f"Demo set with ground truth ({len(sample_tasks)} tasks)")
    if (ROOT / "input" / "tasks.json").exists():
        sources.append("Bundled input/tasks.json")
    sources.append("Upload a tasks.json")
    source = st.radio("Task source", sources, horizontal=True)

    tasks = None
    if source.startswith("Demo set"):
        n = st.slider("Number of tasks", 1, len(sample_tasks), min(8, len(sample_tasks)))
        tasks = sample_tasks[:n]
    elif source.startswith("Bundled"):
        tasks = json.loads((ROOT / "input" / "tasks.json").read_text())
    else:
        uploaded = st.file_uploader("tasks.json", type="json")
        if uploaded is not None:
            tasks = json.load(uploaded)

    if tasks:
        st.write(f"{len(tasks)} task(s) loaded.")

    if st.button("Run batch", disabled=not (ready and tasks)):
        agent = make_agent()
        try:
            with st.spinner(f"Running {len(tasks)} task(s) "
                            f"({agent_mod.CONCURRENCY} in parallel)…"):
                t0 = time.monotonic()
                results = run_batch(agent, tasks)
                elapsed = time.monotonic() - t0
        except Exception as e:  # noqa: BLE001
            st.error(f"Batch run failed: {e}")
        else:
            rows, answers = [], []
            total_remote = total_local = cat_hits = truth_known = 0
            for answer, stats in results:
                truth = sample_ideal.get(stats["task_id"])
                row = {
                    "task_id": stats["task_id"],
                    "category": stats["category"],
                    "difficulty": stats["difficulty"],
                    "model": short(stats["model"]),
                    "remote_tokens": stats["tokens"]["total"],
                    "local_tokens": stats["local_tokens"]["total"],
                    "elapsed_s": stats["elapsed_s"],
                }
                if truth:
                    expected = GT_CATEGORY.get(truth["category"], truth["category"])
                    row["truth"] = f'{expected}/{truth["difficulty"]}'
                    row["cat ✓"] = "✅" if stats["category"] == expected else "❌"
                    truth_known += 1
                    cat_hits += stats["category"] == expected
                row["answer"] = answer
                rows.append(row)
                answers.append({"task_id": stats["task_id"], "answer": answer})
                total_remote += stats["tokens"]["total"]
                total_local += stats["local_tokens"]["total"]

            cols = st.columns(4)
            cols[0].metric("Total remote tokens (billable)", total_remote)
            cols[1].metric("Total local tokens (free)", total_local)
            if truth_known:
                cols[2].metric("Routing accuracy", f"{cat_hits}/{truth_known}")
            cols[3].metric("Wall time", f"{elapsed:.1f}s")

            st.dataframe(pd.DataFrame(rows), width="stretch")

            st.download_button(
                "Download results.json",
                json.dumps(answers, ensure_ascii=False, indent=2),
                file_name="results.json", mime="application/json",
            )

# ---------------------------------------------------------------------------
# How it works
# ---------------------------------------------------------------------------

with tab3:
    st.markdown(
        """
### Two-stage routing

1. **Classify** — every prompt is labelled with one of 8 capability
   categories and a difficulty (`easy` / `medium` / `hard`) using a few-shot
   prompt. Inside the submitted container this runs on the **bundled local
   model**, so it costs **zero billable tokens**; otherwise the smallest
   Fireworks model.
2. **Solve** — the prompt is routed by a category × difficulty table
   (see `ROUTING` in `main.py`): easy tasks stay **local** (free), code goes
   to the code-specialised model, hard reasoning to the largest general
   model, and everything else to the cheapest tier that can handle it.

### Why

The hackathon ranks submissions that pass the LLM-judge accuracy gate by
**total billable tokens, ascending**. Local tokens are recorded as zero, so
every task the local model absorbs — and every classification call — is a
direct leaderboard gain. In native testing the hybrid roughly halved
billable tokens versus remote-only while keeping an 8/8 judge pass rate.

### Model tiers

Tiers are inferred from `ALLOWED_MODELS` at runtime by name heuristics —
nothing is hardcoded. Gemma models are *on-demand*: they 404 until deployed
at [app.fireworks.ai/models](https://app.fireworks.ai/models), which is what
the **USE_GEMMA** checkbox is for.
        """
    )
