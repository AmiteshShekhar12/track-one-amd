"""
Track 1 agent — LOCAL-ONLY variant. Zero API calls, zero billable tokens.

Everything (classification + answering) runs on a GGUF model bundled in the
Docker image via llama-cpp-python. The judging proxy records 0 tokens, so
after passing the accuracy gate this ranks at the very top of the
token-efficiency leaderboard.

Design notes:
- Sequential processing (llama.cpp context is single-stream); a global
  deadline plus per-category token caps keep the run inside the 10-minute
  limit, and results.json is rewritten after every task so a crash or
  timeout still leaves valid, complete-so-far output.
- No `openai` dependency and no network use at runtime at all.
"""

import json
import os
import re
import sys
import time

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def env_flag(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")
METRICS_PATH = os.environ.get(
    "METRICS_PATH",
    os.path.join(os.path.dirname(OUTPUT_PATH) or ".", "metrics.json"),
)
LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "/app/models/model.gguf")

MAX_RUNTIME_S = float(os.environ.get("MAX_RUNTIME_S", "570"))
LOCAL_CONTEXT = 8192
CLASSIFY_MAX_PROMPT_CHARS = 1500

CATEGORIES = {
    1: "factual", 2: "math", 3: "sentiment", 4: "summarization",
    5: "ner", 6: "code_debug", 7: "logic", 8: "code_gen",
}

MAX_TOKENS = {
    "factual": 700, "math": 900, "sentiment": 150, "summarization": 400,
    "ner": 350, "code_debug": 1200, "logic": 1000, "code_gen": 1200,
}

SYSTEM_PROMPTS = {
    "factual": (
        "Answer every part of the question completely and accurately. If it "
        "asks for a difference or comparison, explicitly describe BOTH "
        "sides. If it asks 'why' or 'what is each used for', answer that "
        "for every item mentioned. Cover every sub-question; be direct, but "
        "never omit a requested detail."
    ),
    "math": (
        "Solve step by step, showing the key intermediate values. "
        "Double-check the arithmetic, answer every part of the question, "
        "and end with the final answer(s) on their own line as: "
        "Answer: <value>"
    ),
    "sentiment": (
        "Classify the sentiment using the labels offered in the task and "
        "justify in one short sentence. If the text mixes clearly negative "
        "and clearly positive aspects, do NOT label it Negative — call it "
        "Mixed, Neutral, or Positive (weigh the overall outcome) and make "
        "the reason explicitly acknowledge both the negative and the "
        "positive aspects."
    ),
    "summarization": (
        "Summarise exactly as requested. Obey every length and format "
        "constraint strictly: count your sentences, bullets and words "
        "before answering. Cover ALL the main points of the passage — both "
        "positives/opportunities and negatives/challenges, plus any "
        "response or outlook mentioned. When the passage lists several items (e.g. challenges or benefits), name each listed item explicitly even in short bullets — compress wording, never drop an item. Output only the summary."
    ),
    "ner": (
        "Extract every distinct named entity and label each one using "
        "EXACTLY the label names requested in the task. Output one entity "
        "per line in the format: <entity text> - <LABEL> "
        "(e.g. 'Marie Curie - PERSON'). Include the entity text verbatim, "
        "keep multi-word names and full dates as ONE entity (e.g. 'March 15 2023 - DATE'), extract only proper named entities (no generic phrases like 'a research lab'), and do not abbreviate the labels."
    ),
    "code_debug": (
        "Identify the bug(s) briefly, then provide the corrected code in a "
        "single code block."
    ),
    "logic": (
        "Reason carefully; verify every constraint is satisfied before "
        "answering. End with the final answer on its own line as: "
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


# ---------------------------------------------------------------------------
# Local model
# ---------------------------------------------------------------------------

class LocalModel:
    def __init__(self, path):
        from llama_cpp import Llama
        self.llm = Llama(
            model_path=path,
            n_ctx=LOCAL_CONTEXT,
            n_threads=os.cpu_count(),
            verbose=False,
        )
        self.name = f"local:{os.path.basename(path)}"

    def chat(self, system, user, max_tokens):
        resp = self.llm.create_chat_completion(
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


def classify(model, prompt):
    snippet = prompt[:CLASSIFY_MAX_PROMPT_CHARS]
    try:
        raw, usage = model.chat(CLASSIFY_SYSTEM, snippet, max_tokens=60)
        match = re.search(r'\{[^{}]*"c"\s*:\s*(\d)[^{}]*\}', raw)
        if match:
            obj = json.loads(match.group(0))
            category = CATEGORIES.get(int(obj.get("c", 1)), "factual")
            difficulty = {"e": "easy", "m": "medium", "h": "hard"}.get(
                str(obj.get("d", "m"))[:1].lower(), "medium")
            return category, difficulty, usage
    except Exception as e:  # noqa: BLE001
        print(f"classification failed ({e}); using defaults", file=sys.stderr)
        usage = {"prompt": 0, "completion": 0, "total": 0}
    return "factual", "medium", usage


# ---------------------------------------------------------------------------
# Deterministic format-constraint verification (free retries, big gate wins)
# ---------------------------------------------------------------------------

NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
             "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def _num(tok):
    return NUM_WORDS.get(tok.lower()) or (int(tok) if tok.isdigit() else None)


def extract_constraints(prompt):
    """Pull 'exactly N sentences/bullets, each no longer than M words'-style
    constraints out of the task text."""
    cons = {}
    m = re.search(r"(?:exactly\s+)?(\w+)\s+sentences?\b", prompt, re.IGNORECASE)
    if m and "exactly" in prompt.lower() and _num(m.group(1)):
        cons["sentences"] = _num(m.group(1))
    m = re.search(r"(?:exactly\s+)?(\w+)\s+bullet(?:\s+point)?s?\b",
                  prompt, re.IGNORECASE)
    if m and _num(m.group(1)):
        cons["bullets"] = _num(m.group(1))
    m = re.search(r"no (?:longer|more) than (\w+) words", prompt, re.IGNORECASE)
    if m and _num(m.group(1)):
        cons["max_words_per_item"] = _num(m.group(1))
    return cons


def split_sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def split_bullets(text):
    return [re.sub(r"^\s*([-*•]|\d+[.)])\s+", "", l).strip()
            for l in text.splitlines()
            if re.match(r"^\s*([-*•]|\d+[.)])\s+", l)]


def constraint_violations(answer, cons):
    problems = []
    if "bullets" in cons:
        items = split_bullets(answer)
        if len(items) != cons["bullets"]:
            problems.append(
                f"it must contain exactly {cons['bullets']} bullet points "
                f"(yours had {len(items)}); format each as '- <text>' and "
                "merge related items into one bullet per theme rather "
                "than dropping any")
    elif "sentences" in cons:
        n = len(split_sentences(answer))
        if n != cons["sentences"]:
            problems.append(
                f"it must contain exactly {cons['sentences']} sentences "
                f"(yours had {n})")
    if "max_words_per_item" in cons:
        items = split_bullets(answer) or split_sentences(answer)
        for item in items:
            words = len(re.findall(r"[\w'-]+", item))
            if words > cons["max_words_per_item"]:
                problems.append(
                    f"every point must be at most "
                    f"{cons['max_words_per_item']} words (one of yours has "
                    f"{words})")
                break
    return problems


def write_results(tasks, results):
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(
            [{"task_id": t["task_id"], "answer": results.get(t["task_id"], "")}
             for t in tasks],
            f, ensure_ascii=False, indent=2,
        )


def main():
    start = time.monotonic()
    with open(INPUT_PATH) as f:
        tasks = json.load(f)

    model = LocalModel(LOCAL_MODEL_PATH)
    print(f"local model loaded: {model.name} | {len(tasks)} task(s)")

    results, stats_rows = {}, []
    write_results(tasks, results)  # valid output exists from second zero

    for task in tasks:
        remaining = MAX_RUNTIME_S - (time.monotonic() - start)
        if remaining <= 15:
            print(f"[{task['task_id']}] skipped, out of time budget",
                  file=sys.stderr)
            continue
        t0 = time.monotonic()
        local_tokens = {"prompt": 0, "completion": 0, "total": 0}
        try:
            category, difficulty, usage = classify(model, task.get("prompt", ""))
            for k in local_tokens:
                local_tokens[k] += usage[k]
            # shrink generation if the clock is running down
            cap = min(MAX_TOKENS[category], max(96, int(remaining * 6)))
            print(f"[{task['task_id']}] {category}/{difficulty} "
                  f"(cap {cap})")
            prompt_text = task.get("prompt", "")
            answer, usage = model.chat(
                SYSTEM_PROMPTS[category], prompt_text, cap)
            for k in local_tokens:
                local_tokens[k] += usage[k]

            # summarisation coverage scaffold: small models drop listed
            # items under tight word limits, so extract the item list first
            # and then write the summary conditioned on it
            if category == "summarization":
                items, usage = model.chat(
                    "List every distinct key item the passage mentions, as "
                    "short comma-separated phrases grouped by theme (e.g. "
                    "benefits: ...; challenges: ...; responses: ...). "
                    "Output only the list.",
                    prompt_text, 200)
                for k in local_tokens:
                    local_tokens[k] += usage[k]
                revised, usage = model.chat(
                    SYSTEM_PROMPTS[category],
                    f"{prompt_text}\n\nKey items your summary must "
                    f"explicitly name (do not drop any):\n{items}\n\n"
                    "Allocate one sentence/bullet per theme group and "
                    "name every item inside its theme's sentence/"
                    "bullet — merge, never drop.",
                    cap)
                for k in local_tokens:
                    local_tokens[k] += usage[k]
                if revised:
                    answer = revised

            # verify explicit format constraints; one corrective retry —
            # local tokens are free and format misses are automatic fails
            cons = extract_constraints(prompt_text)
            problems = constraint_violations(answer, cons) if cons else []
            if problems:
                print(f"[{task['task_id']}] format retry: {problems}")
                feedback = (
                    f"{prompt_text}\n\nYour previous answer was:\n{answer}\n\n"
                    f"It violates the required format: {'; '.join(problems)}. "
                    "Rewrite the answer so it satisfies every constraint "
                    "exactly, keeping the content coverage.")
                retry, usage = model.chat(
                    SYSTEM_PROMPTS[category], feedback, cap)
                for k in local_tokens:
                    local_tokens[k] += usage[k]
                if not constraint_violations(retry, cons):
                    answer = retry
            results[task["task_id"]] = answer
        except Exception as e:  # noqa: BLE001
            print(f"[{task['task_id']}] failed: {e}", file=sys.stderr)
            category, difficulty = "?", "?"
        write_results(tasks, results)  # crash/timeout-safe incremental output
        stats_rows.append({
            "task_id": task["task_id"],
            "category": category,
            "difficulty": difficulty,
            "model": model.name,
            "tokens": {"prompt": 0, "completion": 0, "total": 0},
            "local_tokens": local_tokens,
            "elapsed_s": round(time.monotonic() - t0, 2),
        })

    elapsed = time.monotonic() - start
    metrics = {
        "total_elapsed_s": round(elapsed, 2),
        "total_tokens": 0,   # zero billable/remote tokens by construction
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_local_tokens": sum(
            s["local_tokens"]["total"] for s in stats_rows),
        "tasks": stats_rows,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    answered = sum(1 for t in tasks if results.get(t["task_id"], ""))
    print(f"done: {answered}/{len(tasks)} answered, 0 billable tokens "
          f"(+{metrics['total_local_tokens']} local), {elapsed:.1f}s "
          f"-> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
