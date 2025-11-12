#!/usr/bin/env python
"""
summarizer_service.py
- Reads UTF-8 transcripts from ./transcripts/*.txt
- Produces (summary_text, action_items_raw_json_string)
- Writes one record per transcripts to ./outputs/summaries.jsonl

Setup:
  pip install -U openai python-dotenv
  # .env must include: OPENAI_API_KEY=sk-...

Run:
  python summarizer_service.py
"""

from __future__ import annotations
import os, re, json, glob
from typing import Tuple, List
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing in .env")
client = OpenAI(api_key=OPENAI_API_KEY)

OUT_DIR = "outputs"
OUT_PATH = os.path.join(OUT_DIR, "summaries.jsonl")

SUMMARY_SYS = (
    "You are a meeting summarizer. Write a concise, factual summary "
    "(8–12 sentences). Focus on decisions, blockers, owners, and deadlines. "
    "No bullet points. No hallucinations."
)

ACTION_SYS = (
    "Extract clear action items from the meeting as STRICT JSON array. "
    "Format:\n[\n  {\"owner\":\"<name>\", \"task\":\"<what>\", \"due_date\":\"YYYY-MM-DD or null\", \"priority\":\"High|Medium|Low\"},\n  ...\n]\n"
    "Only return raw JSON (no markdown fences, no extra text)."
)

def _strip_fences(s: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", (s or "").strip(), flags=re.I|re.M)

def summarize_meeting(transcript: str, model_summary="gpt-4o-mini", model_actions="gpt-4o-mini") -> Tuple[str, str]:
    # summary text
    r1 = client.chat.completions.create(
        model=model_summary,
        messages=[{"role":"system","content":SUMMARY_SYS},
                  {"role":"user","content":transcript}],
        temperature=0
    )
    summary_text = (r1.choices[0].message.content or "").strip()

    # action items (raw JSON string)
    r2 = client.chat.completions.create(
        model=model_actions,
        messages=[{"role":"system","content":ACTION_SYS},
                  {"role":"user","content":transcript}],
        temperature=0
    )
    raw_actions = _strip_fences(r2.choices[0].message.content)

    return summary_text, raw_actions

def load_transcripts(folder="transcripts") -> List[tuple[str, str]]:
    """
    Returns list of (file_id, text)
    """
    paths = sorted(glob.glob(os.path.join(folder, "*.txt")))
    out = []
    for p in paths:
        fid = os.path.splitext(os.path.basename(p))[0]
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            out.append((fid, f.read().strip()))
    if not out:
        # fallback single example so pipeline runs
        out = [("sample",
                "Team discussed hallucinations, compared GPT-4o vs Claude, "
                "decided to try a hybrid approach. Maya to build a similarity "
                "metric for confidence scoring; also run a hybrid trial. "
                "Ethan to sync with design for fallback UX. Prompt tuning planned.")
               ]
    return out

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pairs = load_transcripts("transcripts")
    written = 0
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for fid, text in pairs:
            print(f"Processing {fid} ...")
            summary_text, action_items_raw = summarize_meeting(text)
            rec = {
                "id": fid,
                "transcripts": text,
                "summary_text": summary_text,
                "action_items_raw": action_items_raw,  # keep raw; eval will judge structure & quality
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
    print(f"\nWrote {written} records → {OUT_PATH}")

if __name__ == "__main__":
    main()
