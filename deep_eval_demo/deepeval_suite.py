# DeepEval is a powerful open-source LLM evaluation framework.
# Example taken from: https://deepeval.com/tutorials/summarization-agent/development
from __future__ import annotations
import os
from dotenv import load_dotenv
from openai import OpenAI
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

load_dotenv()
client = OpenAI()

def answer(q: str) -> str:
    resp = client.responses.create(model="gpt-4o", input=q, temperature=0)
    return resp.output_text

tests = [
    LLMTestCase(
        input="What is gradient descent?",
        expected_output="An iterative optimization method that updates parameters by moving opposite to the gradient.",
        context="ML basics"
    ),
    LLMTestCase(
        input="Summarize: RAG evaluation best practices.",
        expected_output="Mention faithfulness, retrieval precision/recall, tracing, regression harnesses.",
        context="RAG eval"
    ),
]

metrics = [
    AnswerRelevancyMetric(model="gpt-4o-mini"),
    FaithfulnessMetric(model="gpt-4o-mini"),
]

if __name__ == "__main__":
    report = evaluate(test_cases=tests, metrics=metrics)
    print(report)
