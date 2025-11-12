# rag_eval_end_to_end.py
# ------------------------------------------------------------
# pip install -U ragas datasets openai python-dotenv
# (Optional) If you don't have Git installed and ragas tries to import git:
#   this script silences the GitPython warning automatically.
#
# Requires: .env with OPENAI_API_KEY=sk-...
#
# Outputs:
#   - ragas_report.json    (aggregate metrics as JSON)
#   - ragas_scores.csv     (per-sample metrics as CSV)
#   - ragas_scores.json    (per-sample metrics as JSON)
# ------------------------------------------------------------
from __future__ import annotations
import os, json
from typing import List, Dict, Any
import logging
logging.getLogger("langchain_core").setLevel(logging.ERROR)
from langchain_openai import ChatOpenAI

# Silence GitPython dependency noise if git isn't installed
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from datasets import Dataset

# ✅ New, recommended way to wire OpenAI into Ragas
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing. Put it in your .env")

# Build OpenAI client and a Ragas-compatible LLM via the factory
client = OpenAI(api_key=OPENAI_API_KEY)
lc_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
ragas_llm = LangchainLLMWrapper(lc_llm)
# --------------------------------------------------------------------
# 1) Tiny but realistic dataset (extend with your real RAG logs later)
# Fields Ragas expects:
#   - question: str
#   - answer: str (your system’s answer)
#   - contexts: List[str] (what you retrieved)
#   - ground_truth: str (short reference answer/summary)
# --------------------------------------------------------------------
samples: List[Dict[str, Any]] = [
    {
        "question": "What are key best practices to evaluate RAG in production?",
        "answer":   "Use faithfulness checks, measure retrieval precision/recall, trace sources, and run regressions.",
        "contexts": [
            "Best practices include: measure faithfulness between answer and retrieved text; context recall/precision; latency; traceability.",
            "Tooling examples: Ragas, Arize Phoenix, DeepEval, Promptfoo."
        ],
        "ground_truth": "Evaluate faithfulness and retrieval metrics; track latency and traceability."
    },
    {
        "question": "How do rerankers improve retrieval quality in RAG?",
        "answer":   "Rerankers rescore retrieved chunks to push the most relevant passages up, improving precision.",
        "contexts": [
            "Rerankers use cross-encoders to re-evaluate top-k candidates, often improving precision at k.",
            "They can be applied after BM25 or vector retrieval to refine the result list."
        ],
        "ground_truth": "By rescoring candidates to surface the most relevant contexts, rerankers improve precision."
    },
    {
        "question": "Why does chunk size matter for RAG?",
        "answer":   "Too small chunks lose context; too large chunks add noise. A balanced size improves retrieval and grounding.",
        "contexts": [
            "Chunk size affects semantic cohesion; overly small chunks drop context.",
            "Overly large chunks contain irrelevant text that may hurt precision."
        ],
        "ground_truth": "Balanced chunk size preserves context while controlling noise."
    },
    {
        "question": "Name two open-source tools for evaluating RAG systems.",
        "answer":   "Ragas and DeepEval are two good open-source options.",
        "contexts": [
            "Ragas provides faithfulness, answer relevancy, and context metrics.",
            "DeepEval supports building custom test cases and LLM-based evaluators."
        ],
        "ground_truth": "Ragas and DeepEval are open-source RAG evaluation tools."
    },
    {
        "question": "What does faithfulness measure in a RAG answer?",
        "answer":   "It measures how well the answer is supported by the retrieved context, penalizing hallucinations.",
        "contexts": [
            "Faithfulness evaluates the alignment of claims with retrieved evidence.",
            "Lower faithfulness implies potential hallucination."
        ],
        "ground_truth": "The extent to which the answer’s claims are grounded in the provided context."
    },
    {
        "question": "How can we catch regressions after changing retrieval configs?",
        "answer":   "Create a small benchmark set and run regression checks comparing metrics before and after changes.",
        "contexts": [
            "Benchmark datasets enable repeatable evaluation across system versions.",
            "Track metrics like context precision/recall and faithfulness over time."
        ],
        "ground_truth": "Use a benchmark dataset and compare metrics across versions."
    },
    {
        "question": "When should we use hybrid retrieval?",
        "answer":   "Use hybrid when queries mix lexical and semantic signals; it can improve recall and robustness.",
        "contexts": [
            "Hybrid combines BM25 with embeddings to capture exact terms and semantic similarity.",
            "It helps when users use rare keywords or domain terms."
        ],
        "ground_truth": "When queries benefit from both keyword matching and semantic similarity."
    },
    {
        "question": "What is context recall in RAG evaluation?",
        "answer":   "It’s the fraction of ground-truth information covered by the retrieved contexts.",
        "contexts": [
            "Context recall quantifies how completely retrieval captured the needed info.",
            "Precision focuses on how relevant retrieved contexts are; recall focuses on completeness."
        ],
        "ground_truth": "The completeness of the retrieved contexts relative to the reference."
    },
]

ds = Dataset.from_list(samples)

# --------------------------------------------------------------------
# 2) Evaluate with Ragas
#    - We pass the LLM explicitly (new API).
#    - If you want fewer console messages, set show_progress=False.
# --------------------------------------------------------------------
report = evaluate(
    dataset=ds,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=ragas_llm,
    show_progress=True,
)

# --------------------------------------------------------------------
# 3) Display + persist (handle non-JSON-serializable EvaluationResult)
# --------------------------------------------------------------------
print("\n=== Aggregate Metrics ===")
# robust access to aggregate scores
agg = None
for attr in ("to_dict", "model_dump", "dict"):
    if hasattr(report, attr):
        try:
            agg = getattr(report, attr)()
            break
        except Exception:
            pass
if agg is None:
    # fall back to best-effort string
    agg = {"__repr__": str(report)}
print(json.dumps(agg, indent=2, ensure_ascii=False))

# Save aggregate JSON
with open("ragas_report.json", "w", encoding="utf-8") as f:
    json.dump(agg, f, ensure_ascii=False, indent=2)

# Per-sample table (CSV + JSON). Different versions expose different helpers; try a few.
per_sample_saved = False
for method in ("to_pandas", "to_dataframe", "to_df"):
    if hasattr(report, method):
        try:
            df = getattr(report, method)()
            df.to_csv("ragas_scores.csv", index=False)
            # also save JSON records
            try:
                df.to_json("ragas_scores.json", orient="records", force_ascii=False, indent=2)
            except TypeError:
                # some pandas versions don't support indent in to_json
                df.to_json("ragas_scores.json", orient="records", force_ascii=False)
            per_sample_saved = True
            break
        except Exception:
            pass

if not per_sample_saved:
    # Fallback: try to pull per-sample scores if available
    rows = getattr(report, "sample_scores", None)
    if rows is None:
        rows = getattr(report, "scores", None)
    if rows is not None:
        with open("ragas_scores.json", "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

print("\nSaved:")
print(" - ragas_report.json (aggregate)")
print(" - ragas_scores.csv / ragas_scores.json (per-sample)")
