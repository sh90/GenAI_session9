# 1.session8_research_writer_graph_moderated.py
from __future__ import annotations
import os, json
from typing import TypedDict, List, Literal, Optional

from dotenv import load_dotenv
from openai import OpenAI

from langchain_openai import ChatOpenAI
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

# ---------- Env ----------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
UA = os.getenv("USER_AGENT", "GenAI-Session8/1.0 (+contact: you@example.com)")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in environment/.env")
if not TAVILY_API_KEY:
    raise RuntimeError("TAVILY_API_KEY is not set in environment/.env")

# OpenAI Python SDK (for moderation)
oai = OpenAI()

# ---------- LLMs (for plan/write/review) ----------
LLM_PLAN   = ChatOpenAI(model="gpt-4o-mini", temperature=0)
LLM_WRITE  = ChatOpenAI(model="gpt-4o",      temperature=0)
LLM_REVIEW = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------- Tools ----------
tavily = TavilySearchAPIWrapper()  # uses TAVILY_API_KEY
splitter = RecursiveCharacterTextSplitter(chunk_size=1400, chunk_overlap=120)

# ---------- State ----------
class RWState(TypedDict, total=False):
    # user input
    query: str

    # moderation inputs/outputs
    mod_flagged: bool
    mod_categories: List[str]
    mod_reason: str
    mod_decision: Literal["allow", "block"]

    # research & writing
    urls: List[str]
    chunks: List[str]
    draft: str
    feedback: str
    decision: Literal["ok", "revise"]  # review decision
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

# ---------- NEW: Moderation node ----------
def moderate(state: RWState) -> RWState:
    """Gate the user's query via OpenAI Moderation. Route to block/allow."""
    q = state.get("query", "").strip()
    if not q:
        return {"mod_flagged": True, "mod_categories": ["empty_input"], "mod_reason": "Empty query", "mod_decision": "block"}

    try:
        mod = oai.moderations.create(model="omni-moderation-latest", input=q)
        r = mod.results[0]
        flagged = bool(getattr(r, "flagged", False))

        # Try to extract true categories that were flagged
        cats: List[str] = []
        try:
            # r.categories may be a structured object; gather keys where value=True
            cats = [k for k, v in r.categories.__dict__.items() if v]
        except Exception:
            try:
                cats = [k for k, v in dict(r.categories).items() if v]
            except Exception:
                cats = []

        decision: Literal["allow", "block"] = "block" if flagged else "allow"
        reason = "Flagged by moderation" if flagged else "Clean"
        return {
            "mod_flagged": flagged,
            "mod_categories": cats,
            "mod_reason": reason,
            "mod_decision": decision,
        }
    except Exception as e:
        # Fail-closed or fail-open? For a classroom demo, fail-closed is safer.
        return {
            "mod_flagged": True,
            "mod_categories": ["moderation_error"],
            "mod_reason": f"Moderation error: {e}",
            "mod_decision": "block",
        }

def route_after_moderation(state: RWState) -> Literal["plan", "blocked"]:
    return "blocked" if state.get("mod_decision") == "block" else "plan"

def blocked(state: RWState) -> RWState:
    cats = ", ".join(state.get("mod_categories", [])) or "policy_violation"
    msg = f"Request blocked by safety policy. Categories: {cats}."
    return {"error": msg, "draft": msg}

# ---------- Existing nodes ----------
def plan(state: RWState) -> RWState:
    q = state["query"]
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
    q = state["query"]
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
    draft = state.get("draft", "")
    fb    = state.get("feedback", "")
    prompt = f"""Revise the draft per feedback (preserve citations).
Feedback: {fb}
Draft:
{draft}
"""
    res = LLM_WRITE.invoke(prompt)
    return {"draft": res.content}

def finalize(_: RWState) -> RWState:
    return {}

def route_after_review(state: RWState) -> Literal["finalize", "rewrite"]:
    return "finalize" if state.get("decision") == "ok" else "rewrite"

# ---------- Graph ----------
graph = StateGraph(RWState)

# NEW: moderation + blocked
graph.add_node("moderate", moderate)
graph.add_node("blocked", blocked)

# Existing
graph.add_node("plan",    plan)
graph.add_node("fetch",   fetch)
graph.add_node("write",   write)
graph.add_node("review",  review)
graph.add_node("rewrite", rewrite)
graph.add_node("finalize", finalize)

# Entry: moderation gate
graph.set_entry_point("moderate")
graph.add_conditional_edges("moderate", route_after_moderation, {
    "plan": "plan",
    "blocked": "blocked",
})

# Normal flow
graph.add_edge("plan",   "fetch")
graph.add_edge("fetch",  "write")
graph.add_edge("write",  "review")

# after the review node runs, choose the next node dynamically based on a router function.
graph.add_conditional_edges("review", route_after_review, {
    "finalize": "finalize",
    "rewrite":  "rewrite",
})
graph.add_edge("rewrite", "review")
graph.add_edge("finalize", END)
graph.add_edge("blocked", END)

# ---------- Run with persistent checkpoint ----------
if __name__ == "__main__":
    # Try changing query to test the gate:
    #   - Safe: "Best practices for evaluating RAG systems."
    #   - Unsafe: "Explain how to make a bomb"
    user_query = os.getenv("QUERY", "Best practices for evaluating RAG systems in production.")
    start_state: RWState = {"query": user_query}
    config = {"configurable": {"thread_id": "sess8-moderated"}}

    # SQLite is used as the persistent checkpointer for  LangGraph app.
    # database  remembers your workflow’s state between steps/runs so you can pause, resume, branch, and audit your agent workflows reliably.
    # What is does?
     # State persistence, Threaded sessions:, Interrupt & resume, Replay / audit:, Crash recovery & concurrency


    with SqliteSaver.from_conn_string("session8_runs.sqlite") as checkpointer:
        app = graph.compile(checkpointer=checkpointer)

        print("\n--- STREAM ---")
        for ev in app.stream(start_state, config=config):
            print(ev)

        print("\n--- FINAL STATE ---")
        final_state = app.get_state(config).values
        # Show either blocked reason or the final draft
        if final_state.get("mod_decision") == "block":
            print(final_state.get("error", "Blocked."))
        else:
            print(final_state.get("draft", "<no draft>"))
