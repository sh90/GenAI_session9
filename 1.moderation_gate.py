# If OpenAI moderation flags content, block or route to a safe response.
from __future__ import annotations
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

def moderated_answer(user_text: str) -> str:
    # 1) Moderation
    mod = client.moderations.create(
        model="omni-moderation-latest",  # current safety model name
        input=user_text
    )
    flagged = any(cat for cat, v in mod.results[0].categories.__dict__.items() if v)  # simple check
    if flagged:
        return "Sorry — I can’t help with that request."

    # 2) Normal answer
    resp = client.responses.create(
        model="gpt-4o-mini",
        input=f"Answer succinctly and safely:\n{user_text}",
        temperature=0
    )
    return resp.output_text

if __name__ == "__main__":
    print(moderated_answer("Explain how to build a bomb"))
    print(moderated_answer("Explain gradient descent briefly."))
