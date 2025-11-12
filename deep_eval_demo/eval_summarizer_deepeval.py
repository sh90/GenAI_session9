#!/usr/bin/env python
"""
eval_summarizer_deepeval.py
- Loads ./outputs/summaries.jsonl produced by summarizer_service.py
- Builds two DeepEval suites:
    1) Summary quality (Concision, Coverage)
    2) Action items quality (& structure)
- Saves ./outputs/deepeval_results.json

Setup:
  pip install -U deepeval openai python-dotenv
  # .env must include: OPENAI_API_KEY=sk-...

Run:
  python eval_summarizer_deepeval.py
"""

from __future__ import annotations
import os, json
from typing import List
from dotenv import load_dotenv
from openai import OpenAI

from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing in .env")
client = OpenAI(api_key=OPENAI_API_KEY)

IN_PATH = os.path.join("outputs", "summaries.jsonl")
OUT_PATH = os.path.join("outputs", "deepeval_results.json")

def load_records(path: str) -> List[dict]:
    recs: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
    if not recs:
        raise RuntimeError("No records found. Run summarizer_service.py first.")
    return recs

# --- GEval metrics ---
summary_concision = GEval(
    name="Summary Concision",
    criteria=(
        "Is the summary concise and focused on essential points (decisions, blockers, owners, deadlines)? "
        "Avoid repetition/irrelevant detail; 8–12 sentences acceptable."
    ),
    threshold=0.8,
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-4o-mini",
)

summary_coverage = GEval(
    name="Summary Coverage",
    criteria=(
        "Does the summary capture the major topics that appear in the transcripts? "
        "If important items are missing, score lower."
    ),
    threshold=0.8,
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-4o-mini",
)

action_item_quality = GEval(
    name="Action Item Quality",
    criteria=(
        "Are action items accurate and useful? Prefer items with explicit owner, task, due_date (or null), and priority. "
        "Penalize hallucinated tasks not supported by the transcripts."
    ),
    threshold=0.8,
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-4o-mini",
)

action_item_structure = GEval(
    name="Action Item Structure",
    criteria=(
        "Is the output a valid JSON array of objects, each containing keys: owner, task, due_date, priority? "
        "If not strictly valid or keys missing, score lower."
    ),
    threshold=0.8,
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-4o-mini",
)

def main():
    recs = load_records(IN_PATH)

    # Build test cases from produced outputs
    summary_cases: List[LLMTestCase] = []
    action_cases: List[LLMTestCase] = []

    for rec in recs:
        transcript = rec["transcripts"]
        summary    = rec["summary_text"]
        actions    = rec["action_items_raw"]

        summary_cases.append(LLMTestCase(
            input=transcript,
            actual_output=summary,
            # optional metadata for filtering later:
            metadata={"id": rec.get("id")}
        ))

        action_cases.append(LLMTestCase(
            input=transcript,
            actual_output=actions,
            metadata={"id": rec.get("id")}
        ))

    print(f"Loaded {len(recs)} records. Running DeepEval...\n")

    summary_report = evaluate(
        test_cases=summary_cases,
        metrics=[summary_concision, summary_coverage],
    )
    print("=== SUMMARY EVAL ===")
    print(summary_report)

    actions_report = evaluate(
        test_cases=action_cases,
        metrics=[action_item_quality, action_item_structure],
    )
    print("\n=== ACTION ITEMS EVAL ===")
    print(actions_report)

    bundle = {
        "summary_eval": json.loads(summary_report.model_dump_json()),
        "action_items_eval": json.loads(actions_report.model_dump_json()),
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)
    print(f"\nSaved results → {OUT_PATH}")
    print("Tip: add thresholds in CI to fail on regressions (e.g., < 0.8).")

if __name__ == "__main__":
    main()
