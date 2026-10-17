import re
from collections.abc import Iterable

_EMAIL = re.compile(r"\b[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_ADDRESS = re.compile(
    r"\b(?:via|viale|piazza|corso|largo|vicolo|strada|salita|street|st\.?|road|avenue|ave\.?)\s+"
    r"[\wÀ-ÿ'’-]+(?:\s+[\wÀ-ÿ'’-]+){0,4}(?:\s+(?:n\.?\s*)?\d+[A-Za-z]?)?",
    re.IGNORECASE,
)
_PHONE_CANDIDATE = re.compile(r"(?<!\w)(?:\+|00)?[\d][\d\s()./-]{6,}\d(?!\w)")
_NAME_INTRO = re.compile(
    r"(?i:\b(?:mi chiamo|my name is|i am|i'm|this is|sono)\s+)"
    r"([A-ZÀ-Þ][\wÀ-ÿ'’-]*(?:\s+[A-ZÀ-Þ][\wÀ-ÿ'’-]*){0,2})"
)
_NAME_SEQUENCE = re.compile(r"\b([A-ZÀ-Þ][a-zà-ÿ'’-]+(?:\s+[A-ZÀ-Þ][a-zà-ÿ'’-]+)+)\b")
_DATE = re.compile(r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$")
_NON_PERSON_PHRASES = {
    "amd ai", "communityconnect", "care support", "community support", "digital support",
    "education", "food support", "healthcare", "housing support", "language support",
    "transport support", "genoa", "genova", "liguria", "italy", "italia",
    "saturday sunday", "new year", "good morning",
}


def deidentify_text(text: str, known_names: Iterable[str] = ()) -> tuple[str, dict[str, str]]:
    """Mask common personal identifiers before text is sent to a remote model."""
    redactions: dict[str, str] = {}

    def replace(pattern: re.Pattern, label: str, value: str, *, should_redact=None) -> str:
        def substitute(match: re.Match) -> str:
            original = match.group(0)
            if should_redact and not should_redact(original):
                return original
            token = f"[{label}_{len(redactions) + 1}]"
            redactions[token] = original
            return token
        return pattern.sub(substitute, value_text)

    value_text = text
    value_text = replace(_EMAIL, "EMAIL", value_text)
    value_text = replace(_ADDRESS, "ADDRESS", value_text)
    value_text = replace(
        _PHONE_CANDIDATE,
        "PHONE",
        value_text,
        should_redact=lambda value: len(re.sub(r"\D", "", value)) >= 9 and not _DATE.fullmatch(value.strip()),
    )
    value_text = replace(_NAME_INTRO, "PERSON", value_text)

    def replace_name_sequence(match: re.Match) -> str:
        original = match.group(1)
        if original.casefold() in _NON_PERSON_PHRASES:
            return original
        token = f"[PERSON_{len(redactions) + 1}]"
        redactions[token] = original
        return token

    value_text = _NAME_SEQUENCE.sub(replace_name_sequence, value_text)

    for name in sorted({item.strip() for item in known_names if item and item.strip()}, key=len, reverse=True):
        pattern = re.compile(re.escape(name), re.IGNORECASE)
        value_text = replace(pattern, "PERSON", value_text)

    return value_text, redactions


def restore_redactions(text: str, redactions: dict[str, str]) -> str:
    for token, original in redactions.items():
        text = text.replace(token, original)
    return text
