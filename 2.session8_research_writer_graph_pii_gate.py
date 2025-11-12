# session8_research_writer_graph_pii_gate.py
from __future__ import annotations
import os, json
from typing import TypedDict, List, Literal, Optional, Dict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

# --- PII (Presidio) ---
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
UA = os.getenv("USER_AGENT", "GenAI-Session8/1.1 (+contact: you@example.com)")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in environment/.env")
if not TAVILY_API_KEY:
    raise RuntimeError("TAVILY_API_KEY is not set in environment/.env")

# ---------- LLMs ----------
LLM_PLAN   = ChatOpenAI(model="gpt-4o-mini", temperature=0)
LLM_WRITE  = ChatOpenAI(model="gpt-4o",      temperature=0)
LLM_REVIEW = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------- Tools ----------
tavily = TavilySearchAPIWrapper()  # uses TAVILY_API_KEY
splitter = RecursiveCharacterTextSplitter(chunk_size=1400, chunk_overlap=120)

# ---------- Presidio (PII) ----------
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

DEFAULT_MASK = OperatorConfig(
    operator_name="mask",
    params={
        "masking_char": "*",
        "chars_to_mask": 100,
        "from_end": False,  # required by your Presidio version
    },
)

def pii_redact(text: str) -> Dict[str, any]:
    """
    Detect PII and return {text, findings, count}.
    findings: [{"entity_type": "...", "start": int, "end": int, "score": float}, ...]
    """
    if not text:
        return {"text": "", "findings": [], "count": 0}
    results = analyzer.analyze(text=text, entities=None, language="en")
    out = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators={"DEFAULT": DEFAULT_MASK}
    )
    findings = [
        {
            "entity_type": r.entity_type,
            "start": r.start,
            "end": r.end,
            "score": getattr(r, "score", None)
        }
        for r in results
    ]
    return {"text": out.text, "findings": findings, "count": len(findings)}

# ---------- State ----------
class RWState(TypedDict, total=False):
    # inputs
    query: str
    # sanitized inputs
    query_sanitized: str
    query_pii_count: int
    query_pii_findings: List[Dict[str, any]]

    # research
    urls: List[str]
    chunks: List[str]

    # writing/review
    draft: str
    feedback: str
    decision: Literal["ok", "revise"]

    # output sanitization
    draft_redacted: str
    draft_pii_count: int
    draft_pii_findings: List[Dict[str, any]]

    # misc
    error: Optional[str]

# ---------- helpers ----------
def _safe_json_list(text: str, fallback: List[str]) -> List[str]:
    try:
        v = json.loads(text)
        if isinstance(v, list) and all(isinstance(x, str) for x in v):
            return v
    except Exception:
        pass
    return fallback

# ---------- Nodes ----------
def pii_sanitize_query(state: RWState) -> RWState:
    """Redact PII from the incoming user query before ANY LLM/tool call."""
    original = state.get("query", "")
    red = pii_redact(original)
    # If redaction changed the text, we still preserve intent but remove PII
    return {
        "query_sanitized": red["text"] or original,
        "query_pii_count": red["count"],
        "query_pii_findings": red["findings"],
    }

def plan(state: RWState) -> RWState:
    """Turn the (sanitized) question into 3–5 focused search queries and fetch top URLs."""
    q = state.get("query_sanitized") or state.get("query") or ""
    prompt = f"""Break the question into 3–5 focused web queries.
Question: {q}
Return as a JSON list of strings only."""
    res = LLM_PLAN.invoke(prompt)
    subqueries = _safe_json_list(res.content, [q])

    urls: List[str] = []
    for sq in subqueries[:5]:
        try:
            hits = tavily.results(sq, max_results=3) or []
            for h in hits:
                url = h.get("url") or h.get("link")
                if url and url not in urls:
                    urls.append(url)
        except Exception:
            continue
    return {"urls": urls[:8]}

def fetch(state: RWState) -> RWState:
    """Load and chunk pages."""
    urls = state.get("urls", [])
    texts: List[str] = []
    for u in urls[:8]:
        try:
            docs = WebBaseLoader(u, requests_kwargs={"headers": {"User-Agent": UA}}).load()
            text = docs[0].page_content if docs else ""
            if text:
                for d in splitter.create_documents([text]):
                    if d.page_content:
                        texts.append(d.page_content)
        except Exception:
            continue
    return {"chunks": texts[:12]}

def write(state: RWState) -> RWState:
    """Write a short answer using the gathered chunks with inline citations."""
    q = state.get("query_sanitized") or state.get("query") or ""
    ch = state.get("chunks", [])
    ctx = "\n\n".join([f"[{i+1}] {c[:900]}" for i, c in enumerate(ch)])
    prompt = f"""Answer the question using ONLY the evidence below.
Question: {q}

Evidence:
{ctx}

Write 8–12 sentences with 3–5 bullet takeaways.
Cite inline as [n]. At end, add "Sources: [1] ... [2] ...".
Keep it tight and factual."""
    res = LLM_WRITE.invoke(prompt)
    return {"draft": res.content}

def review(state: RWState) -> RWState:
    """Quality review with pass/fail decision."""
    draft = state.get("draft", "")
    prompt = (
        'You are a strict editor. Check: accuracy (use only provided citations), clarity, and length.\n'
        'If acceptable, respond with JSON: {"decision":"ok","feedback":"..."}.\n'
        'If not, respond with JSON: {"decision":"revise","feedback":"what to fix ..."}.\n'
        f"Draft:\n{draft}"
    )
    res = LLM_REVIEW.invoke(prompt)
    try:
        payload = json.loads(res.content)
        decision = payload.get("decision", "ok")
        feedback = payload.get("feedback", "Looks good.")
    except Exception:
        decision, feedback = "ok", "Looks good."
    return {"decision": decision, "feedback": feedback}

def rewrite(state: RWState) -> RWState:
    """Revise draft according to feedback."""
    draft = state.get("draft", "")
    fb    = state.get("feedback", "")
    prompt = f"""Revise the draft per feedback (preserve citations).
Feedback: {fb}
Draft:
{draft}
"""
    res = LLM_WRITE.invoke(prompt)
    return {"draft": res.content}

def pii_redact_output(state: RWState) -> RWState:
    """Redact the final draft before publishing/logging to prevent PII leakage."""
    draft = state.get("draft", "")
    red = pii_redact(draft)
    return {
        "draft_redacted": red["text"] or draft,
        "draft_pii_count": red["count"],
        "draft_pii_findings": red["findings"],
    }

def finalize(_: RWState) -> RWState:
    """No-op node; could persist/publish. Output is in state."""
    return {}

# ---------- Router ----------
def route_after_review(state: RWState) -> Literal["pii_redact_output", "rewrite"]:
    # If edit needed, loop back; otherwise go to PII redaction before finalize
    return "pii_redact_output" if state.get("decision") == "ok" else "rewrite"

# ---------- Graph ----------
graph = StateGraph(RWState)
graph.add_node("pii_sanitize_query", pii_sanitize_query)
graph.add_node("plan",    plan)
graph.add_node("fetch",   fetch)
graph.add_node("write",   write)
graph.add_node("review",  review)
graph.add_node("rewrite", rewrite)
graph.add_node("pii_redact_output", pii_redact_output)
graph.add_node("finalize", finalize)

graph.set_entry_point("pii_sanitize_query")
graph.add_edge("pii_sanitize_query", "plan")
graph.add_edge("plan",   "fetch")
graph.add_edge("fetch",  "write")
graph.add_edge("write",  "review")
graph.add_conditional_edges("review", route_after_review, {
    "pii_redact_output": "pii_redact_output",
    "rewrite":           "rewrite",
})
graph.add_edge("rewrite", "review")
graph.add_edge("pii_redact_output", "finalize")
graph.add_edge("finalize", END)

# ---------- Run with persistent checkpoint ----------
if __name__ == "__main__":
    user_query = (
        "Write a short brief on best practices to evaluate RAG systems in production. "
        "If you need to contact me, my email is john.smith@example.com and phone is +1-415-555-1234."
    )
    start_state: RWState = {"query": user_query}
    config = {"configurable": {"thread_id": "sess8-pii-demo"}}

    # Keep compile + stream INSIDE the context so the DB stays open
    with SqliteSaver.from_conn_string("session8_runs.sqlite") as checkpointer:
        app = graph.compile(checkpointer=checkpointer)

        print("=== STREAM ===")
        for ev in app.stream(start_state, config=config):
            print(ev)

        print("\n--- FINAL STATE ---")
        final_state = app.get_state(config).values
        print(json.dumps({
            "query_original": final_state.get("query", ""),
            "query_sanitized": final_state.get("query_sanitized", ""),
            "query_pii_count": final_state.get("query_pii_count", 0),
            "draft_redacted": final_state.get("draft_redacted", ""),
            "draft_pii_count": final_state.get("draft_pii_count", 0),
        }, indent=2, ensure_ascii=False))
        print("\n--- OUTPUT (SAFE) ---\n")
        print(final_state.get("draft_redacted", "<no draft>"))
