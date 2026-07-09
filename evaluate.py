"""
Evaluation pipeline for the Track 1 agent.

For each task it:
  1. Generates an *ideal* reference answer by calling a bigger model
     (JUDGE_MODEL, defaults to the largest model in ALLOWED_MODELS).
  2. Asks the LLM judge to score the agent's answer against that reference
     (0.0-1.0 semantic correctness).

It then merges in the agent's own run metrics (tokens + per-response latency
from metrics.json) and reports:
  - accuracy (mean judge score) and pass rate
  - tokens used by the agent (total and per task)
  - time elapsed per response and total time elapsed

Usage:
  python evaluate.py            # uses ./input, ./output by default via env
Reads:  INPUT_PATH, OUTPUT_PATH, METRICS_PATH, EVAL_OUTPUT_PATH, JUDGE_MODEL
"""

import asyncio
import json
import os
import re
import sys
import time

from openai import AsyncOpenAI

from main import get_config, rank_model, clean

INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")
METRICS_PATH = os.environ.get(
    "METRICS_PATH",
    os.path.join(os.path.dirname(OUTPUT_PATH) or ".", "metrics.json"),
)
EVAL_OUTPUT_PATH = os.environ.get(
    "EVAL_OUTPUT_PATH",
    os.path.join(os.path.dirname(OUTPUT_PATH) or ".", "evaluation.json"),
)

JUDGE_TIMEOUT_S = 60
CONCURRENCY = 4
PASS_THRESHOLD = 0.7

IDEAL_SYSTEM = (
    "You are a domain expert. Produce the ideal, correct, concise answer to "
    "the task. This answer will be used as the gold reference for grading."
)

JUDGE_SYSTEM = (
    "You are a strict but fair grader. Compare a candidate answer against a "
    "gold reference answer for the given task. Judge semantic correctness, "
    "not wording: different phrasing, ordering or style is fine if the "
    "substance matches. For code, judge whether the candidate code is "
    "functionally correct. For summaries, judge faithfulness and whether "
    "format/length constraints were obeyed. Do NOT explain or think out "
    "loud. Your entire reply must be exactly one line of compact JSON: "
    '{"score": <float 0.0-1.0>, "reason": "<one short sentence>"}'
)

JUDGE_TEMPLATE = """TASK:
{prompt}

GOLD REFERENCE ANSWER:
{ideal}

CANDIDATE ANSWER:
{answer}"""


async def chat(client, model, system, user, max_tokens, temperature=0.0):
    resp = await asyncio.wait_for(
        client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        ),
        timeout=JUDGE_TIMEOUT_S,
    )
    usage = getattr(resp, "usage", None)
    total = (getattr(usage, "total_tokens", 0) or 0) if usage else 0
    return clean(resp.choices[0].message.content), total


def parse_score(raw):
    match = re.search(r'\{[^{}]*"score"[^{}]*\}', raw, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            score = max(0.0, min(1.0, float(obj.get("score", 0.0))))
            return score, str(obj.get("reason", ""))
        except (ValueError, TypeError):
            pass
    # last resort: any bare float in the reply
    match = re.search(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", raw)
    if match:
        return float(match.group(1)), "unparsed judge reply"
    return 0.0, f"could not parse judge reply: {raw[:120]}"


async def evaluate_task(client, judge_model, task, answer):
    task_id = task["task_id"]
    prompt = task["prompt"]
    row = {"task_id": task_id, "score": 0.0, "reason": "", "judge_tokens": 0}
    if not answer:
        row["reason"] = "empty answer"
        return row
    try:
        ideal, t1 = await chat(
            client, judge_model, IDEAL_SYSTEM, prompt, max_tokens=1500
        )
        verdict, t2 = await chat(
            client, judge_model, JUDGE_SYSTEM,
            JUDGE_TEMPLATE.format(prompt=prompt, ideal=ideal, answer=answer),
            max_tokens=600,
        )
        row["score"], row["reason"] = parse_score(verdict)
        row["ideal"] = ideal
        row["judge_tokens"] = t1 + t2
    except Exception as e:  # noqa: BLE001
        row["reason"] = f"evaluation error: {e}"
    return row


async def run():
    start = time.monotonic()
    api_key, base_url, models = get_config()
    judge_model = os.environ.get("JUDGE_MODEL") or sorted(models, key=rank_model)[-1]
    print(f"judge/reference model: {judge_model}")

    with open(INPUT_PATH) as f:
        tasks = json.load(f)
    with open(OUTPUT_PATH) as f:
        answers = {r["task_id"]: r.get("answer", "") for r in json.load(f)}

    agent_metrics = {"tasks": []}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            agent_metrics = json.load(f)
    else:
        print(f"warning: {METRICS_PATH} not found; token/latency columns "
              "will be empty (run main.py first)", file=sys.stderr)
    per_task_metrics = {s["task_id"]: s for s in agent_metrics.get("tasks", [])}

    client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=1)
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def worker(task):
        async with semaphore:
            return await evaluate_task(
                client, judge_model, task, answers.get(task["task_id"], "")
            )

    rows = await asyncio.gather(*(worker(t) for t in tasks))

    # merge agent-side metrics into each row
    for row in rows:
        m = per_task_metrics.get(row["task_id"], {})
        row["category"] = m.get("category")
        row["model"] = m.get("model")
        row["tokens"] = m.get("tokens", {}).get("total")
        row["elapsed_s"] = m.get("elapsed_s")

    scores = [r["score"] for r in rows]
    accuracy = sum(scores) / len(scores) if scores else 0.0
    passed = sum(1 for s in scores if s >= PASS_THRESHOLD)
    report = {
        "judge_model": judge_model,
        "accuracy": round(accuracy, 4),
        "pass_rate": round(passed / len(rows), 4) if rows else 0.0,
        "pass_threshold": PASS_THRESHOLD,
        "agent_total_tokens": agent_metrics.get("total_tokens"),
        "agent_total_elapsed_s": agent_metrics.get("total_elapsed_s"),
        "judge_total_tokens": sum(r["judge_tokens"] for r in rows),
        "eval_elapsed_s": round(time.monotonic() - start, 2),
        "tasks": rows,
    }
    os.makedirs(os.path.dirname(EVAL_OUTPUT_PATH) or ".", exist_ok=True)
    with open(EVAL_OUTPUT_PATH, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # console summary
    print(f"\n{'task':<8}{'score':>6}{'tokens':>8}{'time(s)':>9}  "
          f"{'category':<15}reason")
    for r in rows:
        print(f"{r['task_id']:<8}{r['score']:>6.2f}"
              f"{(r['tokens'] if r['tokens'] is not None else '-'):>8}"
              f"{(r['elapsed_s'] if r['elapsed_s'] is not None else '-'):>9}  "
              f"{(r['category'] or '-'):<15}{r['reason'][:60]}")
    print(f"\naccuracy: {accuracy:.2%} | pass rate (>= {PASS_THRESHOLD}): "
          f"{passed}/{len(rows)} | agent tokens: "
          f"{agent_metrics.get('total_tokens', '-')} | agent time: "
          f"{agent_metrics.get('total_elapsed_s', '-')}s")
    print(f"report -> {EVAL_OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(run())
