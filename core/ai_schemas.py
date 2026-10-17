import json
import math
from typing import Any

ALLOWED_CATEGORIES = {
    "Education", "Digital Support", "Food Support", "Healthcare", "Care Support",
    "Housing Support", "Language Support", "Transport Support", "Community Support",
}
ALLOWED_LANGUAGES = {"it", "en"}
ALLOWED_URGENCY = {"normal", "soon", "urgent"}
TRANSLATION_LANGUAGES = {"it", "en", "es", "fr"}
ALLOWED_DAYS = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
ALLOWED_PARTS_OF_DAY = {"morning", "afternoon", "evening"}


def _object(text: str, keys: set[str]) -> dict[str, Any]:
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("AI response did not match the required JSON object")
    return value


def validate_intake(text: str) -> dict[str, Any]:
    result = _object(text, {
        "category", "urgency", "skills", "availability_slots", "number_constraints",
        "language", "place", "stated_lang", "summary",
    })
    if result["category"] is not None and (
        not isinstance(result["category"], str) or result["category"] not in ALLOWED_CATEGORIES
    ):
        raise ValueError("Intake category is not allowed")
    if not isinstance(result["urgency"], str) or result["urgency"] not in ALLOWED_URGENCY:
        raise ValueError("Intake urgency is not allowed")
    if not isinstance(result["skills"], list) or len(result["skills"]) > 8:
        raise ValueError("Intake skills must be a short list")
    if any(not isinstance(skill, str) or not skill.strip() or len(skill) > 100 for skill in result["skills"]):
        raise ValueError("Intake skills contain an invalid value")
    slots = result["availability_slots"]
    if not isinstance(slots, list) or len(slots) > 21:
        raise ValueError("Intake availability slots are invalid")
    seen_slots = set()
    for slot in slots:
        if not isinstance(slot, dict) or set(slot) != {"day", "part_of_day"}:
            raise ValueError("Intake availability slot has an invalid shape")
        if slot["day"] not in ALLOWED_DAYS or slot["part_of_day"] not in ALLOWED_PARTS_OF_DAY:
            raise ValueError("Intake availability slot has an unsupported value")
        key = (slot["day"], slot["part_of_day"])
        if key in seen_slots:
            raise ValueError("Intake availability slots must be unique")
        seen_slots.add(key)
    number_constraints = result["number_constraints"]
    if not isinstance(number_constraints, list) or len(number_constraints) > 8:
        raise ValueError("Intake number constraints are invalid")
    for number in number_constraints:
        if not isinstance(number, dict) or set(number) != {"field", "value", "range", "unit"}:
            raise ValueError("Intake number constraint has an invalid shape")
        if not isinstance(number["field"], str) or not number["field"].strip() or len(number["field"]) > 40:
            raise ValueError("Intake number constraint field is invalid")
        if not isinstance(number["unit"], str) or not number["unit"].strip() or len(number["unit"]) > 24:
            raise ValueError("Intake number constraint unit is invalid")
        value = number["value"]
        bounds = number["range"]
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
            raise ValueError("Intake number constraint value is invalid")
        if bounds is not None and (
            not isinstance(bounds, list) or len(bounds) != 2
            or any(isinstance(bound, bool) or not isinstance(bound, (int, float)) or not math.isfinite(bound) or bound < 0 for bound in bounds)
            or bounds[0] > bounds[1]
        ):
            raise ValueError("Intake number constraint range is invalid")
        if (value is None) == (bounds is None):
            raise ValueError("Provide an exact number or a range, but not both")
    if result["language"] is not None and (
        not isinstance(result["language"], str) or result["language"] not in ALLOWED_LANGUAGES
    ):
        raise ValueError("Intake language is not supported")
    if result["stated_lang"] is not None and (
        not isinstance(result["stated_lang"], str) or result["stated_lang"] not in ALLOWED_LANGUAGES
    ):
        raise ValueError("Requested spoken language is not supported")
    if result["place"] is not None and (
        not isinstance(result["place"], str) or not result["place"].strip() or len(result["place"]) > 120
    ):
        raise ValueError("Intake place is invalid")
    if not isinstance(result["summary"], str) or not result["summary"].strip() or len(result["summary"]) > 500:
        raise ValueError("Intake summary is invalid")
    return result


def validate_triage(text: str) -> dict[str, Any]:
    result = _object(text, {"emergency", "medical_advice", "legal_advice", "reason"})
    if any(not isinstance(result[key], bool) for key in ("emergency", "medical_advice", "legal_advice")):
        raise ValueError("Triage flags must be booleans")
    if not isinstance(result["reason"], str) or len(result["reason"]) > 240:
        raise ValueError("Triage reason is invalid")
    return result


def validate_translation(text: str) -> dict[str, str]:
    result = _object(text, {"translation"})
    if not isinstance(result["translation"], str) or not result["translation"].strip() or len(result["translation"]) > 4_000:
        raise ValueError("Translation is invalid")
    return result


def validate_document_summary(text: str) -> dict[str, Any]:
    result = _object(text, {"summary", "actions", "help_request", "deadline", "emergency", "needs_professional_advice"})
    if not isinstance(result["summary"], str) or not result["summary"].strip() or len(result["summary"]) > 1_000:
        raise ValueError("Document summary is invalid")
    if not isinstance(result["actions"], list) or len(result["actions"]) > 8:
        raise ValueError("Document actions are invalid")
    for action in result["actions"]:
        if not isinstance(action, dict) or set(action) != {"action", "deadline"}:
            raise ValueError("Document action has an invalid shape")
        if not isinstance(action["action"], str) or not action["action"].strip() or len(action["action"]) > 300:
            raise ValueError("Document action is invalid")
        if not isinstance(action["deadline"], str) or len(action["deadline"]) > 100:
            raise ValueError("Document deadline is invalid")
    if not isinstance(result["help_request"], str) or len(result["help_request"]) > 500:
        raise ValueError("Prefilled help request is invalid")
    if not isinstance(result["deadline"], str) or len(result["deadline"]) > 100:
        raise ValueError("Document overall deadline is invalid")
    if not isinstance(result["emergency"], bool) or not isinstance(result["needs_professional_advice"], bool):
        raise ValueError("Document safety flags must be booleans")
    return result
