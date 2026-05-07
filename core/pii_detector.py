from __future__ import annotations
import re
from dataclasses import dataclass

_PATTERNS: dict[str, re.Pattern] = {
    "IBAN":        re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7,25}\b"),
    "SSN":         re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d{4}[- ]){3}\d{4}\b"),
    "EMAIL":       re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    "PHONE":       re.compile(
        r"\b"
        r"(?:"
        r"(?:\+?1[-.\s])(?:\(?\d{3}\)?[-.\s])\d{3}[-.\s]\d{4}"
        r"|"
        r"(?:\(?\d{3}\)?[-.\s])\d{3}[-.\s]\d{4}"
        r")\b"
    ),
}


@dataclass
class PIIScanResult:
    has_pii: bool
    categories: list[str]
    redacted_text: str
    span_count: int


class PIIDetector:
    def __init__(
        self,
        use_ner: bool = True,
        allowed_categories: list[str] | None = None,
    ):
        self.use_ner = use_ner
        self.allowed_categories: set[str] = set(allowed_categories or [])
        self._nlp = None
        if use_ner:
            self._load_ner()

    def _load_ner(self):
        try:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
        except (ImportError, OSError):
            self._nlp = None

    def scan(self, text: str) -> PIIScanResult:
        redacted = text
        found_categories: list[str] = []
        span_count = 0

        # Gate 1 — regex
        for category, pattern in _PATTERNS.items():
            if category in self.allowed_categories:
                continue
            matches = pattern.findall(redacted)
            if matches:
                found_categories.append(category)
                span_count += len(matches)
                redacted = pattern.sub(f"[{category}]", redacted)

        # Gate 2 — NER
        if self.use_ner and self._nlp:
            doc = self._nlp(redacted)
            ner_map = {"PERSON": "PERSON", "GPE": "LOCATION", "LOC": "LOCATION"}
            # Collect (start, end, replacement) tuples for all relevant entities
            replacements = []
            for ent in doc.ents:
                ner_category = ner_map.get(ent.label_)
                if not ner_category or ner_category in self.allowed_categories:
                    continue
                replacements.append((ent.start_char, ent.end_char, ner_category, ent.text))
            # Apply in reverse order to preserve char offsets
            for start, end, ner_category, ent_text in sorted(replacements, key=lambda x: x[0], reverse=True):
                if ner_category not in found_categories:
                    found_categories.append(ner_category)
                span_count += 1
                redacted = redacted[:start] + f"[{ner_category}]" + redacted[end:]

        return PIIScanResult(
            has_pii=bool(found_categories),
            categories=found_categories,
            redacted_text=redacted,
            span_count=span_count,
        )
