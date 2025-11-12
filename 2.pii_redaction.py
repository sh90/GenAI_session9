# pii_redaction.py
from __future__ import annotations
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# Optional: use spaCy for higher quality (uncomment and install if you want)
# from presidio_analyzer.nlp_engine import SpacyNlpEngine
# nlp_engine = SpacyNlpEngine(models=[{"lang_code": "en", "model_name": "en_core_web_sm"}])
# analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def redact(text: str) -> str:
    results = analyzer.analyze(text=text, entities=None, language="en")
    # NOTE: include from_end explicitly to satisfy your Presidio version
    default_mask = OperatorConfig(
        operator_name="mask",
        params={
            "masking_char": "*",
            "chars_to_mask": 100,
            "from_end": False  # <— this fixes: Expected parameter from_end
        }
    )
    out = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators={"DEFAULT": default_mask}
    )
    return out.text

if __name__ == "__main__":
    s = "My name is John Smith, email john.smith@example.com, phone +1-415-555-1234."
    print("REDACTED:", redact(s))
