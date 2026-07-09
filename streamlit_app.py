"""
Streamlit demo UI for the Track 1 hybrid agent.

This is a thin visual wrapper around the exact classify -> route -> solve
pipeline in main.py (same Agent class, same ROUTING table, same model-tier
heuristics) — it is not a separate implementation. Useful for a live judge
demo: paste a prompt or upload a tasks.json and watch the routing decision,
answer, and token/latency cost in real time.

Deploy on Streamlit Community Cloud: main file path = streamlit_app.py.
Nothing is hardcoded — everything below is either typed into the sidebar
at runtime or read from the environment / Streamlit secrets, matching how
main.py itself is configured in the submitted container.
"""

import asyncio
import json
import os
import time

import pandas as pd
import streamlit as st
from openai import AsyncOpenAI

from main import Agent, build_roles, load_local_model

st.set_page_config(page_title="Smart Agent — Track 1 demo", page_icon="🤖", layout="wide")


def secret(name, default=""):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


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
    os.environ["USE_LOCAL"] = "true"
    os.environ["LOCAL_MODEL_PATH"] = path
    return load_local_model()


local_model = get_local_model(use_local, local_model_path or "models/model.gguf")


def make_agent():
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0)
    roles = build_roles(models)
    return Agent(client, roles, local=local_model), roles


tab1, tab2 = st.tabs(["Single prompt", "Batch (tasks.json)"])

with tab1:
    prompt = st.text_area(
        "Prompt", height=140,
        placeholder="e.g. Write a Python function that reverses a linked list.",
    )
    if st.button("Run", disabled=not (ready and prompt.strip())):
        agent, roles = make_agent()
        try:
            with st.spinner("Classifying → routing → answering…"):
                answer, stats = asyncio.run(
                    agent.solve({"task_id": "demo", "prompt": prompt}))
        except Exception as e:  # noqa: BLE001 — surface to the demo UI, don't crash the app
            st.error(f"Run failed: {e}")
        else:
            st.success(
                f"Category: **{stats['category']}** · Difficulty: **{stats['difficulty']}** "
                f"· Model: `{stats['model']}` · {stats['elapsed_s']}s"
            )
            st.markdown(answer)
            c1, c2 = st.columns(2)
            c1.metric("Remote tokens (billable)", stats["tokens"]["total"])
            c2.metric("Local tokens (free)", stats["local_tokens"]["total"])

with tab2:
    uploaded = st.file_uploader("tasks.json", type="json")
    use_sample = st.checkbox("Use bundled input/tasks.json sample", value=uploaded is None)
    tasks = None
    if uploaded is not None:
        tasks = json.load(uploaded)
    elif use_sample and os.path.exists("input/tasks.json"):
        with open("input/tasks.json") as f:
            tasks = json.load(f)

    if tasks:
        st.write(f"{len(tasks)} task(s) loaded.")

    if st.button("Run batch", disabled=not (ready and tasks)):
        agent, roles = make_agent()

        async def run_all():
            return await asyncio.gather(*(agent.solve(t) for t in tasks))

        try:
            with st.spinner(f"Running {len(tasks)} task(s)…"):
                t0 = time.monotonic()
                results = asyncio.run(run_all())
                elapsed = time.monotonic() - t0
        except Exception as e:  # noqa: BLE001
            st.error(f"Batch run failed: {e}")
        else:
            rows, total_remote, total_local = [], 0, 0
            for answer, stats in results:
                rows.append({
                    "task_id": stats["task_id"],
                    "category": stats["category"],
                    "difficulty": stats["difficulty"],
                    "model": stats["model"],
                    "remote_tokens": stats["tokens"]["total"],
                    "local_tokens": stats["local_tokens"]["total"],
                    "elapsed_s": stats["elapsed_s"],
                    "answer": answer,
                })
                total_remote += stats["tokens"]["total"]
                total_local += stats["local_tokens"]["total"]

            st.dataframe(pd.DataFrame(rows), width="stretch")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total remote tokens (billable)", total_remote)
            c2.metric("Total local tokens (free)", total_local)
            c3.metric("Wall time", f"{elapsed:.1f}s")
