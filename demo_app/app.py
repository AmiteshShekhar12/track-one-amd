"""
Streamlit demo for the Track 1 routing agent.

Shows the two-stage pipeline live: a prompt is classified (locally when
possible), routed to the cheapest suitable model, and answered — with the
routing decision, token spend, and latency made visible. Sample tasks come
with ground-truth labels and ideal answers for comparison.

Run from the repository root:
    streamlit run demo_app/app.py
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import main as agent_mod  # noqa: E402  (the routing agent)
from openai import AsyncOpenAI  # noqa: E402

# make config independent of the launch directory
agent_mod.load_dotenv(str(ROOT / ".env"))

SAMPLES_DIR = Path(__file__).resolve().parent / "sample_tasks_and_solutions"

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

st.set_page_config(page_title="Routing Agent Demo", page_icon="🔀", layout="wide")


def short(model_id):
    return model_id.replace("accounts/fireworks/models/", "")


@st.cache_data
def load_samples():
    tasks = json.loads((SAMPLES_DIR / "tasks.json").read_text())
    ideal = {x["task_id"]: x for x in
             json.loads((SAMPLES_DIR / "ideal_answers.json").read_text())}
    return tasks, ideal


@st.cache_resource
def load_local_model(path, enabled):
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved  # relative paths are repo-root relative
    os.environ["USE_LOCAL"] = "true" if enabled else "false"
    os.environ["LOCAL_MODEL_PATH"] = str(resolved)
    return agent_mod.load_local_model()


def build_agent(use_gemma, use_local, local_path):
    os.environ["USE_GEMMA"] = "true" if use_gemma else "false"
    try:
        api_key, base_url, models = agent_mod.get_config()
    except SystemExit:
        st.error(
            "Missing configuration. Put FIREWORKS_API_KEY, FIREWORKS_BASE_URL "
            "and ALLOWED_MODELS in the environment or in `.env` at the repo root."
        )
        st.stop()
    roles = agent_mod.build_roles(models)
    local = load_local_model(local_path, use_local)
    if local is not None:
        # a cached Lock may be bound to a previous asyncio.run() loop
        local.lock = asyncio.Lock()
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0)
    return agent_mod.Agent(client, roles, local=local), base_url, models, roles


def run_tasks(agent, tasks):
    """Solve tasks concurrently; returns {task_id: (answer, stats) | Exception}."""

    async def go():
        semaphore = asyncio.Semaphore(agent_mod.CONCURRENCY)
        results = {}

        async def worker(task):
            async with semaphore:
                try:
                    results[task["task_id"]] = await agent.solve(task)
                except Exception as e:  # noqa: BLE001 — shown per-task in the UI
                    results[task["task_id"]] = e

        await asyncio.gather(*(worker(t) for t in tasks))
        return results

    return asyncio.run(go())


def routing_badges(stats):
    where = "🖥️ local (0 billable tokens)" if stats["model"].startswith("local:") \
        else "☁️ Fireworks"
    cols = st.columns(4)
    cols[0].metric("Category", stats["category"])
    cols[1].metric("Difficulty", stats["difficulty"])
    cols[2].metric("Model", short(stats["model"]), delta=where, delta_color="off")
    cols[3].metric("Time", f"{stats['elapsed_s']} s")
    cols = st.columns(4)
    cols[0].metric("Billable tokens", stats["tokens"]["total"])
    cols[1].metric("Free local tokens", stats["local_tokens"]["total"])


# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🔀 Routing Agent")
    st.caption("AMD Hackathon Track 1 — classify locally, route to the "
               "cheapest capable model, spend as few billable tokens as possible.")

    use_gemma = st.toggle(
        "Use Gemma models", value=agent_mod.env_flag("USE_GEMMA", True),
        help="Gemma is on-demand on Fireworks: deploy it at "
             "app.fireworks.ai/models first, otherwise calls return 404. "
             "Keep off unless deployed on your account.",
    )
    use_local = st.toggle(
        "Use local model", value=agent_mod.env_flag("USE_LOCAL", True),
        help="Bundled GGUF model via llama-cpp-python. Classifies every "
             "prompt and answers easy tasks for free.",
    )
    local_path = st.text_input(
        "Local model path",
        value=os.environ.get("LOCAL_MODEL_PATH", "./models/model-3b.gguf"),
    )

    agent, base_url, models, roles = build_agent(use_gemma, use_local, local_path)

    st.divider()
    st.subheader("Resolved configuration")
    st.markdown(f"**Endpoint:** `{base_url}`")
    st.markdown("**Local model:** " +
                (f"`{agent.local.name}` ✅" if agent.local else "not loaded ⚠️"))
    st.markdown("**Role assignment:**")
    st.table({"role": list(roles), "model": [short(m) for m in roles.values()]})

tasks, ideal = load_samples()

tab_play, tab_batch, tab_how = st.tabs(["🎯 Playground", "📊 Batch demo", "ℹ️ How it works"])

# ---------------------------------------------------------------------------
# Playground — one prompt at a time
# ---------------------------------------------------------------------------

with tab_play:
    options = ["✍️ Write my own prompt"] + [
        f"{t['task_id']} — {t['prompt'][:80]}" for t in tasks
    ]
    choice = st.selectbox("Pick a sample task or write your own", options)
    if choice.startswith("✍️"):
        task_id = None
        prompt = st.text_area("Prompt", height=140,
                              placeholder="e.g. A laptop costs $1200, gets 25% off…")
    else:
        task_id = choice.split(" — ")[0]
        prompt = next(t["prompt"] for t in tasks if t["task_id"] == task_id)
        st.text_area("Prompt", value=prompt, height=140, disabled=True)

    if st.button("Route & answer", type="primary", disabled=not prompt):
        with st.spinner("Classifying and routing…"):
            answer, stats = asyncio.run(
                agent.solve({"task_id": task_id or "custom", "prompt": prompt}))

        routing_badges(stats)

        st.subheader("Answer")
        st.markdown(answer if answer else "_empty answer_")

        if task_id and task_id in ideal:
            truth = ideal[task_id]
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
# Batch demo — run many tasks, show the routing table and token totals
# ---------------------------------------------------------------------------

with tab_batch:
    n = st.slider("Number of sample tasks", 1, len(tasks), min(8, len(tasks)))
    subset = tasks[:n]

    if st.button(f"Run {n} tasks", type="primary"):
        t0 = time.monotonic()
        with st.spinner(f"Running {n} tasks ({agent_mod.CONCURRENCY} in parallel)…"):
            results = run_tasks(agent, subset)
        wall = time.monotonic() - t0

        rows, remote_total, local_total, cat_hits, answers = [], 0, 0, 0, []
        for t in subset:
            r = results.get(t["task_id"])
            if isinstance(r, Exception) or r is None:
                rows.append({"task": t["task_id"], "error": str(r)})
                continue
            answer, s = r
            truth = ideal.get(t["task_id"], {})
            expected = GT_CATEGORY.get(truth.get("category"), "?")
            hit = s["category"] == expected
            cat_hits += hit
            remote_total += s["tokens"]["total"]
            local_total += s["local_tokens"]["total"]
            answers.append({"task_id": t["task_id"], "answer": answer})
            rows.append({
                "task": t["task_id"],
                "routed": f'{s["category"]}/{s["difficulty"]}',
                "truth": f'{expected}/{truth.get("difficulty", "?")}',
                "cat ✓": "✅" if hit else "❌",
                "model": short(s["model"]),
                "billable tok": s["tokens"]["total"],
                "local tok": s["local_tokens"]["total"],
                "time (s)": s["elapsed_s"],
            })

        cols = st.columns(4)
        cols[0].metric("Billable tokens", remote_total)
        cols[1].metric("Free local tokens", local_total)
        cols[2].metric("Routing accuracy", f"{cat_hits}/{n}")
        cols[3].metric("Wall time", f"{wall:.1f} s")

        st.dataframe(rows, use_container_width=True)

        with st.expander("Answers"):
            for a in answers:
                st.markdown(f"**{a['task_id']}**")
                st.markdown(a["answer"])
                st.divider()

        st.download_button(
            "Download results.json",
            json.dumps(answers, ensure_ascii=False, indent=2),
            file_name="results.json", mime="application/json",
        )

# ---------------------------------------------------------------------------
# How it works
# ---------------------------------------------------------------------------

with tab_how:
    st.markdown(
        """
### Two-stage routing

1. **Classify** — every prompt is labelled with one of 8 capability
   categories and a difficulty (`easy` / `medium` / `hard`) using a few-shot
   prompt. This runs on the **bundled local model** when available, so it
   costs **zero billable tokens**; otherwise the smallest Fireworks model.
2. **Solve** — the prompt is routed by a category × difficulty table
   (see `ROUTING` in `main.py`): easy tasks stay **local** (free), code goes
   to the code-specialised model, hard reasoning to the largest general
   model, and everything else to the cheapest tier that can handle it.

### Why

The hackathon ranks submissions that pass the LLM-judge accuracy gate by
**total billable tokens, ascending**. Local tokens are recorded as zero, so
every task the local model absorbs — and every classification call — is a
direct leaderboard gain. In native testing this halved billable tokens
(5,394 → 2,686 on the 8-task sample) while keeping an 8/8 pass rate.

### Model tiers

Tiers are inferred from `ALLOWED_MODELS` at runtime by name heuristics —
nothing is hardcoded. Gemma models are *on-demand*: they 404 until deployed
at [app.fireworks.ai/models](https://app.fireworks.ai/models), which is what
the **Use Gemma models** toggle is for.
        """
    )
