from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.dataset import EvaluationDataset
from deepeval_meeting_summarizer import MeetingSummarizer # import your summarizer here

dataset = EvaluationDataset()
dataset.pull(alias="MeetingSummarizer Dataset")

summarizer = MeetingSummarizer() # Initialize with your best config
summary_test_cases = []
action_item_test_cases = []
for golden in dataset.goldens:
    summary, action_items = summarizer.summarize(golden.input)
    summary_test_case = LLMTestCase(
        input=golden.input,
        actual_output=summary
    )
    action_item_test_case = LLMTestCase(
        input=golden.input,
        actual_output=str(action_items)
    )
    summary_test_cases.append(summary_test_case)
    action_item_test_cases.append(action_item_test_case)