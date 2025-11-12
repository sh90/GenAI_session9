# DeepEval metrics are LLM-as-judge.
from __future__ import annotations
import os
from dotenv import load_dotenv
from openai import OpenAI
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

load_dotenv()
client = OpenAI()  # requires OPENAI_API_KEY in .env

def answer(q: str) -> str:
    resp = client.responses.create(model="gpt-4o", input=q, temperature=0)
    return (resp.output_text or "").strip()

# Build test cases with retrieval_context (list[str])
inputs_and_ctx = [
    (
        "What is gradient descent?",
        [
            "Gradient descent is an optimization algorithm that iteratively updates parameters by moving "
            "in the opposite direction of the gradient of the objective function to minimize it."
        ],
    ),
    (
        "Summarize: RAG evaluation best practices.",
        [
            "Best practices for evaluating RAG: check answer faithfulness against retrieved context, "
            "measure retrieval precision and recall, ensure source traceability, and run regression/eval harnesses."
        ],
    ),
]

test_cases = []
for q, passages in inputs_and_ctx:
    actual = answer(q)
    tc = LLMTestCase(
        input=q,
        actual_output=actual,
        # IMPORTANT: FaithfulnessMetric requires this:
        retrieval_context=passages,   # <-- list[str], not a single string, and not 'context'
        # expected_output="(optional reference answer)"
    )
    test_cases.append(tc)

metrics = [
    AnswerRelevancyMetric(model="gpt-4o-mini", threshold=0.5),
    FaithfulnessMetric(model="gpt-4o-mini", threshold=0.5),
]

if __name__ == "__main__":
    report = evaluate(test_cases=test_cases, metrics=metrics)
    print("\n=== DeepEval Summary ===")
    print(report)

    # Optional per-test details (works on recent DeepEval versions)
    try:
        for i, r in enumerate(report.test_results):
            print(f"\n--- Test #{i} ---")
            print("Input: ", r.input)
            print("Actual:", r.actual_output)
            for mname, mval in r.metrics_score.items():
                print(f"{mname}: {mval:.3f}")
    except Exception:
        pass
