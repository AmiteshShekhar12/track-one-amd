"""
AMD Hackathon Track 1 — General-Purpose AI Agent (hybrid local + Fireworks).

Pipeline per task:
  1. Classify the prompt into one of 8 categories + a difficulty level.
     Runs on the bundled LOCAL model when available (zero recorded tokens),
     otherwise on the smallest allowed Fireworks model.
  2. Route the prompt: easy tasks stay on the local model; everything else
     goes to the Fireworks model best suited for that category/difficulty.
  3. Write all answers to /output/results.json (+ metrics.json).

All Fireworks model IDs come from ALLOWED_MODELS at runtime; nothing is
hardcoded. Local weights are bundled in the image (no Ollama / no runtime
pre-installed on the judging VM) and loaded with llama-cpp-python.

Flags (env):
  USE_GEMMA=false     skip gemma-* models (they are on-demand on Fireworks —
                      deploy at https://app.fireworks.ai/models first; a 404
                      means "not deployed", not "banned")
  USE_LOCAL=false     disable the local model entirely
  LOCAL_MODEL_PATH    path to the bundled GGUF weights
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

def env_flag(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


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


load_dotenv()  # must run before the path constants below are resolved

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
LOCAL_CONTEXT = 4096


def get_config():
    api_key = os.environ.get("FIREWORKS_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("FIREWORKS_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    models_raw = os.environ.get("ALLOWED_MODELS") or os.environ.get("MODELS", "")
    models = [m.strip() for m in models_raw.split(",") if m.strip()]
    if not env_flag("USE_GEMMA", True):
        non_gemma = [m for m in models if "gemma" not in m.lower()]
        if non_gemma:
            print(f"USE_GEMMA=false: skipping {len(models) - len(non_gemma)} "
                  "gemma model(s)")
            models = non_gemma
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
    (r"gemma.*a\d+b", 0),                  # MoE, few active params — cheapest
    (r"gemma.*(nvfp4|fp4|int4|awq)", 1),   # quantised dense gemma
    (r"gemma", 2),
    (r"qwen", 2),
    (r"minimax", 3),                       # large reasoning model
    (r"kimi|k2", 4),                       # largest (code-specialised)
]
DEFAULT_RANK = 2

CODE_PATTERN = re.compile(r"code|coder|kimi|k2", re.IGNORECASE)


def rank_model(model_id):
    for pattern, rank in SIZE_PATTERNS:
        if re.search(pattern, model_id, re.IGNORECASE):
            return rank
    return DEFAULT_RANK


def build_roles(models):
    """Map roles -> Fireworks model IDs from whatever ALLOWED_MODELS contains."""
    ordered = sorted(models, key=rank_model)
    code_models = [m for m in ordered if CODE_PATTERN.search(m)]
    roles = {
        "classifier": ordered[0],                 # smallest remote model
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
# Local model (bundled GGUF weights, llama-cpp-python)
# ---------------------------------------------------------------------------

class LocalModel:
    """Small local model. Its tokens are not recorded by the judging proxy,
    so every task it absorbs is free. Calls are serialised (llama.cpp
    context is not safe for concurrent generation)."""

    def __init__(self, path):
        from llama_cpp import Llama  # imported lazily: optional dependency
        self.llm = Llama(
            model_path=path,
            n_ctx=LOCAL_CONTEXT,
            n_threads=os.cpu_count(),
            verbose=False,
        )
        self.name = f"local:{os.path.basename(path)}"
        self.lock = asyncio.Lock()

    async def chat(self, system, user, max_tokens):
        async with self.lock:
            resp = await asyncio.to_thread(
                self.llm.create_chat_completion,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
            )
        usage = resp.get("usage", {})
        return clean(resp["choices"][0]["message"]["content"]), {
            "prompt": usage.get("prompt_tokens", 0),
            "completion": usage.get("completion_tokens", 0),
            "total": usage.get("total_tokens", 0),
        }


def load_local_model():
    if not env_flag("USE_LOCAL", True):
        print("USE_LOCAL=false: local model disabled")
        return None
    path = os.environ.get("LOCAL_MODEL_PATH", "/app/models/model.gguf")
    if not os.path.exists(path):
        print(f"local model not found at {path}; running remote-only",
              file=sys.stderr)
        return None
    try:
        model = LocalModel(path)
        print(f"local model loaded: {model.name}")
        return model
    except Exception as e:  # noqa: BLE001
        print(f"failed to load local model ({e}); running remote-only",
              file=sys.stderr)
        return None


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

# category -> {difficulty -> role}. "local" degrades to "small" when the
# local model is unavailable.
ROUTING = {
    "factual":       {"easy": "local",  "medium": "small",  "hard": "medium"},
    "math":          {"easy": "small",  "medium": "medium", "hard": "large"},
    "sentiment":     {"easy": "local",  "medium": "local",  "hard": "small"},
    "summarization": {"easy": "local",  "medium": "small",  "hard": "medium"},
    "ner":           {"easy": "local",  "medium": "small",  "hard": "medium"},
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
        "Classify the sentiment using the labels offered in the task and "
        "justify in one short sentence. If the text mixes clearly negative "
        "and clearly positive aspects, do NOT label it Negative — call it "
        "Mixed, Neutral, or Positive (weigh the overall outcome) and make "
        "the reason acknowledge both sides."
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
    "Classify the user task into a category c and difficulty d. Categories: "
    "1=factual knowledge (explain a concept, definition, how something works), "
    "2=mathematical reasoning (arithmetic, percentages, word problems), "
    "3=sentiment classification (label a text positive/negative/neutral), "
    "4=summarisation (condense a passage), "
    "5=named entity recognition (extract people/orgs/locations/dates), "
    "6=code debugging (find and fix bugs in given code), "
    "7=logical puzzle (constraints to satisfy, deduction), "
    "8=code generation (write new code from a description). "
    "Difficulty: e=easy, m=medium, h=hard.\n"
    "Examples:\n"
    'Task: "What is photosynthesis and how does it work?" -> {"c":1,"d":"e"}\n'
    'Task: "A phone costs $500, gets 20% off, then 8% tax. Final price?" -> {"c":2,"d":"m"}\n'
    'Task: "Label the sentiment of this tweet and explain: ..." -> {"c":3,"d":"e"}\n'
    'Task: "Condense this article into two sentences: ..." -> {"c":4,"d":"m"}\n'
    'Task: "List every person, company and date mentioned: ..." -> {"c":5,"d":"e"}\n'
    'Task: "This function returns the wrong total, fix it: def f(..." -> {"c":6,"d":"m"}\n'
    'Task: "Three friends each own a different pet; from these clues, who owns the cat?" -> {"c":7,"d":"m"}\n'
    'Task: "Implement a function that reverses a linked list." -> {"c":8,"d":"m"}\n'
    "Reply with ONLY the JSON object, no other text."
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
    def __init__(self, client, roles, local=None):
        self.client = client
        self.roles = roles
        self.local = local

    async def _remote_chat(self, model, system, user, max_tokens, retries=2):
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
        """Classify locally when possible (free tokens), else use the
        smallest remote model; fall back to safe defaults."""
        snippet = prompt[:CLASSIFY_MAX_PROMPT_CHARS]
        try:
            if self.local:
                raw, usage = await self.local.chat(
                    CLASSIFY_SYSTEM, snippet, max_tokens=60)
                add_usage(stats["local_tokens"], usage)
            else:
                raw, usage = await self._remote_chat(
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
            "local_tokens": {"prompt": 0, "completion": 0, "total": 0},
        }
        category, difficulty = await self.classify(prompt, stats)
        role = ROUTING[category][difficulty]
        max_tokens = MAX_TOKENS[category]
        system = SYSTEM_PROMPTS[category]

        if role == "local" and self.local:
            model_name = self.local.name
            print(f"[{task.get('task_id')}] {category}/{difficulty} -> {model_name}")
            try:
                # bound the local call (including lock-queue time) so a slow
                # CPU or a pile-up of local tasks degrades to a cheap remote
                # call instead of an empty answer
                answer, usage = await asyncio.wait_for(
                    self.local.chat(system, prompt, max_tokens), timeout=60)
                add_usage(stats["local_tokens"], usage)
            except Exception as e:  # noqa: BLE001
                print(f"[{task.get('task_id')}] local answer failed ({e!r}); "
                      f"falling back to {self.roles['small']}", file=sys.stderr)
                model_name = self.roles["small"]
                answer, usage = await self._remote_chat(
                    model_name, system, prompt, max_tokens, retries=1)
                add_usage(stats["tokens"], usage)
        else:
            model_name = self.roles["small" if role == "local" else role]
            print(f"[{task.get('task_id')}] {category}/{difficulty} -> {model_name}")
            try:
                answer, usage = await self._remote_chat(
                    model_name, system, prompt, max_tokens)
                add_usage(stats["tokens"], usage)
            except Exception as e:  # noqa: BLE001
                if self.local:
                    print(f"[{task.get('task_id')}] remote failed ({e}); "
                          "answering locally", file=sys.stderr)
                    model_name = self.local.name
                    answer, usage = await self.local.chat(
                        system, prompt, max_tokens)
                    add_usage(stats["local_tokens"], usage)
                else:
                    print(f"[{task.get('task_id')}] primary model failed "
                          f"({e}); falling back to {self.roles['medium']}",
                          file=sys.stderr)
                    model_name = self.roles["medium"]
                    answer, usage = await self._remote_chat(
                        model_name, system, prompt, max_tokens, retries=1)
                    add_usage(stats["tokens"], usage)

        stats.update(
            category=category,
            difficulty=difficulty,
            model=model_name,
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
    local = load_local_model()
    print(f"allowed models: {models}")
    print(f"role assignment: {roles}")

    with open(INPUT_PATH) as f:
        tasks = json.load(f)

    client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0)
    agent = Agent(client, roles, local=local)
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
                    agent.solve(task), timeout=min(remaining, 240)
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
        "total_local_tokens": sum(
            s["local_tokens"]["total"] for s in stats_rows),
        "tasks": stats_rows,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    answered = sum(1 for v in results.values() if v)
    print(f"done: {answered}/{len(tasks)} answered, "
          f"{metrics['total_tokens']} remote tokens "
          f"(+{metrics['total_local_tokens']} free local), {elapsed:.1f}s "
          f"-> {OUTPUT_PATH} (metrics: {METRICS_PATH})")


if __name__ == "__main__":
    asyncio.run(run())
