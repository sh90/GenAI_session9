import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class MeetingSummarizer:
    def __init__(
        self,
        model: str = "gpt-4",
        system_prompt: str = "",
    ):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.system_prompt = system_prompt or (
            "..." # Use the above system prompt here
        )

    def summarize(self, transcript: str) -> tuple[str, dict]:
        summary = self.get_summary(transcript)
        action_items = self.get_action_items(transcript)
        return summary, action_items.choices[0].message.content.strip()


    def get_summary(self, transcript: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.summary_system_prompt},
                    {"role": "user", "content": transcript}
                ]
            )

            summary = response.choices[0].message.content.strip()
            return summary
        except Exception as e:
            print(f"Error generating summary: {e}")
            return f"Error: Could not generate summary due to API issue: {e}"

    def get_action_items(self, transcript: str) -> dict:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.action_item_system_prompt},
                    {"role": "user", "content": transcript}
                ]
            )

            action_items = response.choices[0].message.content.strip()
            try:
                return json.loads(action_items)
            except json.JSONDecodeError:
                return {"error": "Invalid JSON returned from model", "raw_output": action_items}
        except Exception as e:
            print(f"Error generating action items: {e}")
            return {"error": f"API call failed: {e}", "raw_output": ""}

with open("meeting_transcript.txt", "r") as file:
    transcript = file.read().strip()

summarizer = MeetingSummarizer()
summary = summarizer.summarize(transcript)
print(summary)