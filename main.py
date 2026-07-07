"""
AMD Hackathon Track 1 — General-Purpose AI Agent.

Pipeline per task:
  1. Classify the prompt with the SMALLEST allowed model into one of 8
     categories + a difficulty level (cheap, tiny token budget).
  2. Route the prompt to the model best suited for that category/difficulty.
  3. Write all answers to /output/results.json.

All model IDs come from ALLOWED_MODELS at runtime; nothing is hardcoded.
"""

import asyncio
import json
import os
import re
import sys
import time

from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")
METRICS_PATH = os.environ.get(
    "METRICS_PATH",
    os.path.join(os.path.dirname(OUTPUT_PATH) or ".", "metrics.json"),
)

MAX_RUNTIME_S = 540          # write results well before the 10-minute kill
PER_REQUEST_TIMEOUT_S = 25   # harness requires <30s per request
CONCURRENCY = 8
CLASSIFY_MAX_PROMPT_CHARS = 1500


def load_dotenv(path=".env"):
    """Minimal .env loader for local development only."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def get_config():
    load_dotenv()
    api_key = os.environ.get("FIREWORKS_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("FIREWORKS_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    models_raw = os.environ.get("ALLOWED_MODELS") or os.environ.get("MODELS", "")
    models = [m.strip() for m in models_raw.split(",") if m.strip()]
    if not api_key or not base_url or not models:
        print(
            "Missing configuration: need FIREWORKS_API_KEY, FIREWORKS_BASE_URL "
            "and ALLOWED_MODELS in the environment.",
            file=sys.stderr,
        )
        sys.exit(1)
    return api_key, base_url, models


# ---------------------------------------------------------------------------
# Model tiering — rank ALLOWED_MODELS by (heuristic) size/cost
# ---------------------------------------------------------------------------

# Lower rank = smaller/cheaper. Patterns are matched case-insensitively
# against the model ID. Unknown models land mid-pack.
SIZE_PATTERNS = [
    (r"qwen.*3[.\-_]?6", 0),      # Qwen3.6 Plus — previous gen, cheapest
    (r"qwen.*(plus|turbo|flash)", 1),
    (r"qwen", 1),
    (r"minimax", 2),              # MiniMax-M3 — large reasoning model
    (r"kimi|k2", 3),              # Kimi K2.7 Code — largest
]
DEFAULT_RANK = 1

CODE_PATTERN = re.compile(r"code|coder|kimi|k2", re.IGNORECASE)


def rank_model(model_id):
    for pattern, rank in SIZE_PATTERNS:
        if re.search(pattern, model_id, re.IGNORECASE):
            return rank
    return DEFAULT_RANK


def build_roles(models):
    """Map roles -> model IDs from whatever ALLOWED_MODELS contains."""
    ordered = sorted(models, key=rank_model)
    code_models = [m for m in ordered if CODE_PATTERN.search(m)]
    roles = {
        "classifier": ordered[0],                 # smallest model
        "small": ordered[0],
        "medium": ordered[(len(ordered) - 1) // 2],
        "large": ordered[-1],
        "code": code_models[-1] if code_models else ordered[-1],
    }
    # Prefer a large *general* model for reasoning if the top model is
    # code-specialised and a non-code alternative exists.
    non_code = [m for m in ordered if not CODE_PATTERN.search(m)]
    if non_code and CODE_PATTERN.search(roles["large"]):
        roles["large"] = non_code[-1]
    return roles


# ---------------------------------------------------------------------------
# Categories and routing policy
# ---------------------------------------------------------------------------

CATEGORIES = {
    1: "factual",
    2: "math",
    3: "sentiment",
    4: "summarization",
    5: "ner",
    6: "code_debug",
    7: "logic",
    8: "code_gen",
}

# category -> {difficulty -> role}
ROUTING = {
    "factual":       {"easy": "small",  "medium": "small",  "hard": "medium"},
    "math":          {"easy": "small",  "medium": "medium", "hard": "large"},
    "sentiment":     {"easy": "small",  "medium": "small",  "hard": "small"},
    "summarization": {"easy": "small",  "medium": "small",  "hard": "medium"},
    "ner":           {"easy": "small",  "medium": "small",  "hard": "medium"},
    "code_debug":    {"easy": "code",   "medium": "code",   "hard": "code"},
    "logic":         {"easy": "medium", "medium": "large",  "hard": "large"},
    "code_gen":      {"easy": "code",   "medium": "code",   "hard": "code"},
}

# Token caps per category keep the total spend low (ranking is by tokens).
MAX_TOKENS = {
    "factual": 400,
    "math": 1200,
    "sentiment": 150,
    "summarization": 350,
    "ner": 400,
    "code_debug": 1400,
    "logic": 1400,
    "code_gen": 1400,
}

SYSTEM_PROMPTS = {
    "factual": "Answer accurately and concisely. No filler, no preamble.",
    "math": (
        "Solve step by step, briefly. End with the final answer on its own "
        "line as: Answer: <value>"
    ),
    "sentiment": (
        "Classify the sentiment (positive/negative/neutral) and justify in "
        "one short sentence."
    ),
    "summarization": (
        "Summarise exactly as requested. Obey every length and format "
        "constraint strictly. Output only the summary."
    ),
    "ner": (
        "Extract the requested entities and label each (person, organization, "
        "location, date, ...). Output only the labelled entities."
    ),
    "code_debug": (
        "Identify the bug(s) briefly, then provide the corrected code in a "
        "single code block."
    ),
    "logic": (
        "Reason carefully but concisely; verify every constraint is "
        "satisfied. End with the final answer on its own line as: "
        "Answer: <value>"
    ),
    "code_gen": (
        "Write correct, clean code that meets the spec. Output the code in "
        "one code block with minimal explanation."
    ),
}

CLASSIFY_SYSTEM = (
    "Classify the user task. Reply with ONLY compact JSON, no other text: "
    '{"c":<1-8>,"d":"<e|m|h>"} where c is: 1 factual knowledge, '
    "2 math reasoning, 3 sentiment classification, 4 summarization, "
    "5 named entity recognition, 6 code debugging, 7 logic puzzle, "
    "8 code generation; d is difficulty: e easy, m medium, h hard."
)

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def clean(text):
    if not text:
        return ""
    return THINK_RE.sub("", text).strip()


def usage_dict(resp):
    u = getattr(resp, "usage", None)
    return {
        "prompt": getattr(u, "prompt_tokens", 0) or 0,
        "completion": getattr(u, "completion_tokens", 0) or 0,
        "total": getattr(u, "total_tokens", 0) or 0,
    }


def add_usage(acc, usage):
    for k in ("prompt", "completion", "total"):
        acc[k] += usage[k]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    def __init__(self, client, roles):
        self.client = client
        self.roles = roles

    async def _chat(self, model, system, user, max_tokens, retries=2):
        last_err = None
        for attempt in range(retries + 1):
            try:
                resp = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        temperature=0.0,
                        max_tokens=max_tokens,
                    ),
                    timeout=PER_REQUEST_TIMEOUT_S,
                )
                return clean(resp.choices[0].message.content), usage_dict(resp)
            except Exception as e:  # noqa: BLE001 — retry on any transport error
                last_err = e
                if attempt < retries:
                    await asyncio.sleep(1.5 * (attempt + 1))
        raise last_err

    async def classify(self, prompt, stats):
        """Route with the smallest model; fall back to safe defaults."""
        snippet = prompt[:CLASSIFY_MAX_PROMPT_CHARS]
        try:
            raw, usage = await self._chat(
                self.roles["classifier"], CLASSIFY_SYSTEM, snippet,
                max_tokens=200, retries=1,
            )
            add_usage(stats["tokens"], usage)
            match = re.search(r'\{[^{}]*"c"\s*:\s*(\d)[^{}]*\}', raw)
            if match:
                obj = json.loads(match.group(0))
                category = CATEGORIES.get(int(obj.get("c", 1)), "factual")
                difficulty = {"e": "easy", "m": "medium", "h": "hard"}.get(
                    str(obj.get("d", "m"))[:1].lower(), "medium"
                )
                return category, difficulty
        except Exception as e:  # noqa: BLE001
            print(f"classification failed ({e}); using defaults", file=sys.stderr)
        return "factual", "medium"

    async def solve(self, task):
        prompt = task.get("prompt", "")
        t0 = time.monotonic()
        stats = {
            "task_id": task.get("task_id"),
            "tokens": {"prompt": 0, "completion": 0, "total": 0},
        }
        category, difficulty = await self.classify(prompt, stats)
        role = ROUTING[category][difficulty]
        model = self.roles[role]
        print(f"[{task.get('task_id')}] {category}/{difficulty} -> {model}")
        try:
            answer, usage = await self._chat(
                model, SYSTEM_PROMPTS[category], prompt,
                max_tokens=MAX_TOKENS[category],
            )
        except Exception as e:  # noqa: BLE001
            print(f"[{task.get('task_id')}] primary model failed ({e}); "
                  f"falling back to {self.roles['medium']}", file=sys.stderr)
            model = self.roles["medium"]
            answer, usage = await self._chat(
                model, SYSTEM_PROMPTS[category], prompt,
                max_tokens=MAX_TOKENS[category], retries=1,
            )
        add_usage(stats["tokens"], usage)
        stats.update(
            category=category,
            difficulty=difficulty,
            model=model,
            elapsed_s=round(time.monotonic() - t0, 2),
        )
        return answer, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run():
    start = time.monotonic()
    api_key, base_url, models = get_config()
    roles = build_roles(models)
    print(f"allowed models: {models}")
    print(f"role assignment: {roles}")

    with open(INPUT_PATH) as f:
        tasks = json.load(f)

    client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0)
    agent = Agent(client, roles)
    semaphore = asyncio.Semaphore(CONCURRENCY)
    results = {t["task_id"]: "" for t in tasks}
    task_stats = {}

    async def worker(task):
        async with semaphore:
            remaining = MAX_RUNTIME_S - (time.monotonic() - start)
            if remaining <= 5:
                print(f"[{task['task_id']}] skipped, out of time budget",
                      file=sys.stderr)
                return
            try:
                answer, stats = await asyncio.wait_for(
                    agent.solve(task), timeout=min(remaining, 90)
                )
                results[task["task_id"]] = answer
                task_stats[task["task_id"]] = stats
            except Exception as e:  # noqa: BLE001
                print(f"[{task['task_id']}] failed: {e}", file=sys.stderr)

    await asyncio.gather(*(worker(t) for t in tasks))

    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(
            [{"task_id": t["task_id"], "answer": results[t["task_id"]]}
             for t in tasks],
            f, ensure_ascii=False, indent=2,
        )

    elapsed = time.monotonic() - start
    stats_rows = [task_stats[t["task_id"]] for t in tasks
                  if t["task_id"] in task_stats]
    metrics = {
        "total_elapsed_s": round(elapsed, 2),
        "total_tokens": sum(s["tokens"]["total"] for s in stats_rows),
        "total_prompt_tokens": sum(s["tokens"]["prompt"] for s in stats_rows),
        "total_completion_tokens": sum(
            s["tokens"]["completion"] for s in stats_rows),
        "tasks": stats_rows,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    answered = sum(1 for v in results.values() if v)
    print(f"done: {answered}/{len(tasks)} answered, "
          f"{metrics['total_tokens']} tokens, {elapsed:.1f}s "
          f"-> {OUTPUT_PATH} (metrics: {METRICS_PATH})")


if __name__ == "__main__":
    asyncio.run(run())
