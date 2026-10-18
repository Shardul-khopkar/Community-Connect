import math
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


_INITIAL_WEIGHTS = {
    "semantic": 0.35,
    "location": 0.20,
    "availability": 0.20,
    "language": 0.10,
    "reliability": 0.15,
}
_INITIAL_SEMANTIC_LO = 0.25
_INITIAL_SEMANTIC_HI = 0.75
EARTH_RADIUS_KM = 6371.0


def _load_calibration() -> tuple[dict[str, float], float, float]:
    path = Path(__file__).resolve().parents[1] / "data" / "matching_calibration.json"
    if not path.exists():
        return dict(_INITIAL_WEIGHTS), _INITIAL_SEMANTIC_LO, _INITIAL_SEMANTIC_HI
    calibration = json.loads(path.read_text(encoding="utf-8"))
    weights = calibration.get("weights")
    bounds = calibration.get("semantic_bounds")
    if not isinstance(weights, dict) or set(weights) != set(_INITIAL_WEIGHTS):
        raise ValueError("Matching calibration must contain weights for every scoring component")
    if not isinstance(bounds, dict):
        raise ValueError("Matching calibration is missing semantic bounds")
    weights = {name: float(value) for name, value in weights.items()}
    lo = float(bounds["lo"])
    hi = float(bounds["hi"])
    if any(not math.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("Matching calibration weights must be finite and nonnegative")
    if not math.isclose(sum(weights.values()), 1.0, rel_tol=0, abs_tol=1e-6):
        raise ValueError("Matching calibration weights must sum to one")
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        raise ValueError("Matching calibration semantic bounds are invalid")
    return weights, lo, hi


WEIGHTS, SEMANTIC_LO, SEMANTIC_HI = _load_calibration()

_SYNONYMS = {
    "rental": "rent",
    "rentals": "rent",
    "affitto": "rent",
    "affitti": "rent",
    "lease": "rent",
    "leases": "rent",
    "room": "room",
    "rooms": "room",
    "stanza": "room",
    "stanze": "room",
    "tutoring": "tutoring",
    "ripetizioni": "tutoring",
    "lift": "lift",
    "passaggio": "lift",
    "ride": "lift",
    "elderly": "elderly",
    "care": "care",
    "assistenza": "care",
    "anziani": "elderly",
}
_MANUAL_STOPWORDS = {"and", "find", "finding", "for", "help", "need", "service", "services", "support"}


def normalize_text(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value.casefold())
    unaccented = "".join(char for char in folded if not unicodedata.combining(char))
    unaccented = re.sub(r"\bassistenza(?:\s+agli)?\s+anziani\b", "elderly care", unaccented)
    words = re.findall(r"[a-z0-9]+", unaccented)
    normalized = [_SYNONYMS.get(word, word) for word in words]
    return " ".join(normalized)


def manual_relevance_component(request: Mapping[str, Any], helper: Mapping[str, Any]) -> float | None:
    def tokens(record: Mapping[str, Any], fields: Sequence[str]) -> set[str]:
        values = []
        for field in fields:
            value = record.get(field)
            if isinstance(value, str):
                values.append(value)
            elif isinstance(value, Sequence):
                values.extend(item for item in value if isinstance(item, str))
        return {
            word
            for value in values
            for word in normalize_text(value).split()
            if word not in _MANUAL_STOPWORDS
        }

    requested = tokens(request, ("category", "skill", "skills", "skill_tags"))
    offered = tokens(helper, ("categories", "skills", "skill_tags", "bio"))
    if not requested or not offered:
        return None
    return len(requested & offered) / len(requested)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float | None:
    if not left or len(left) != len(right):
        return None
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return None
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


def semantic_component(cosine: float | None, lo: float = SEMANTIC_LO, hi: float = SEMANTIC_HI) -> float | None:
    if cosine is None:
        return None
    if hi <= lo:
        raise ValueError("Semantic calibration hi must be greater than lo")
    return max(0.0, min(1.0, (cosine - lo) / (hi - lo)))


def location_component(
    request_place: Mapping[str, Any] | None,
    helper_place: Mapping[str, Any] | None,
    max_travel_km: float | None,
) -> tuple[float | None, float | None]:
    if request_place is None or helper_place is None or max_travel_km is None:
        return None, None
    if max_travel_km <= 0:
        raise ValueError("max_travel_km must be positive")
    latitude_one = math.radians(float(request_place["lat"]))
    latitude_two = math.radians(float(helper_place["lat"]))
    latitude_delta = latitude_two - latitude_one
    longitude_delta = math.radians(float(helper_place["lon"]) - float(request_place["lon"]))
    arc = (
        math.sin(latitude_delta / 2) ** 2
        + math.cos(latitude_one) * math.cos(latitude_two) * math.sin(longitude_delta / 2) ** 2
    )
    distance_km = 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, arc)))
    return max(0.0, min(1.0, 1 - distance_km / max_travel_km)), distance_km


def availability_component(request_slots: Sequence[Mapping[str, str]], helper_slots: Sequence[Mapping[str, str]]) -> float | None:
    if not request_slots:
        return None
    requested = {(slot["day"], slot["part_of_day"]) for slot in request_slots}
    offered = {(slot["day"], slot["part_of_day"]) for slot in helper_slots}
    return len(requested & offered) / len(requested)


def language_component(user_lang: str | None, stated_lang: str | None, spoken_languages: Sequence[str], translation_ok: bool) -> float | None:
    targets = {language.casefold() for language in (user_lang, stated_lang) if language}
    if not targets:
        return None
    spoken = {language.casefold() for language in spoken_languages}
    return 1.0 if targets & spoken else 0.6


def reliability_component(ratings: Sequence[float], response_rate: float | None) -> float | None:
    if not ratings or response_rate is None:
        return None
    if any(rating < 1 or rating > 5 for rating in ratings):
        raise ValueError("Ratings must be between 1 and 5")
    if not 0 <= response_rate <= 1:
        raise ValueError("response_rate must be between 0 and 1")
    count = len(ratings)
    bayesian_rating = (5 * 3.5 + sum(ratings)) / (5 + count)
    return 0.7 * (bayesian_rating - 1) / 4 + 0.3 * response_rate


def eligibility_reasons(request: Mapping[str, Any], helper: Mapping[str, Any]) -> list[str]:
    reasons = []
    if helper.get("active") is not True:
        reasons.append("helper_inactive_or_unconfirmed")
    if request.get("emergency") or request.get("medical"):
        reasons.append("request_requires_emergency_or_medical_support")
    category = request.get("category")
    categories = helper.get("categories") or []
    if category and _explicit_constraints(request).get("category") and category not in categories:
        reasons.append("category_not_offered")
    stated_language = request.get("stated_lang")
    spoken_languages = {str(value).casefold() for value in helper.get("spoken_languages") or []}
    if stated_language and stated_language.casefold() not in spoken_languages and not request.get("translation_ok"):
        reasons.append("stated_language_not_offered")
    return reasons


def _explicit_constraints(request: Mapping[str, Any]) -> dict[str, Any]:
    explicit = request.get("explicit_constraints") or {}
    return explicit if isinstance(explicit, Mapping) else {}


def exact_constraint_results(request: Mapping[str, Any], helper: Mapping[str, Any]) -> list[dict[str, Any]]:
    explicit = _explicit_constraints(request)
    results = []
    spoken = {str(value).casefold() for value in helper.get("spoken_languages") or []}
    helper_skills = {normalize_text(str(value)) for value in helper.get("skills") or []}
    helper_categories = {normalize_text(str(value)) for value in helper.get("categories") or []}
    numeric_offers = {
        (normalize_text(str(offer.get("field", ""))), normalize_text(str(offer.get("unit", "")))): offer.get("value")
        for offer in helper.get("numeric_offers") or []
        if isinstance(offer, Mapping)
    }

    if explicit.get("stated_lang"):
        value = request.get("stated_lang")
        results.append({"constraint": "stated_lang", "value": value, "matched": bool(value and value.casefold() in spoken)})
    if explicit.get("place_id"):
        value = request.get("place_id")
        results.append({"constraint": "place_id", "value": value, "matched": bool(value and value == helper.get("place_id"))})
    if explicit.get("availability_slots"):
        requested = {(slot["day"], slot["part_of_day"]) for slot in request.get("availability_slots") or []}
        offered = {(slot["day"], slot["part_of_day"]) for slot in helper.get("availability_slots") or []}
        results.append({"constraint": "availability_slots", "value": sorted(requested), "matched": bool(requested) and requested <= offered})
    if explicit.get("skills"):
        values = request.get("skills") or request.get("skill_tags") or []
        targets = helper_skills | helper_categories
        results.append({
            "constraint": "skills",
            "value": values,
            "matched": bool(values) and all(normalize_text(str(value)) in targets for value in values),
        })
    if explicit.get("category"):
        value = request.get("category")
        results.append({
            "constraint": "category",
            "value": value,
            "matched": bool(value and normalize_text(str(value)) in helper_categories),
        })
    for number in request.get("number_constraints") or []:
        if not explicit.get("numbers"):
            continue
        number_field = normalize_text(str(number["field"]))
        number_unit = normalize_text(str(number.get("unit") or ""))
        candidate_value = numeric_offers.get((number_field, number_unit), helper.get(number["field"]))
        expected = number.get("value")
        bounds = number.get("range")
        matched = candidate_value is not None and (
            candidate_value == expected if expected is not None else
            bounds is not None and bounds[0] <= candidate_value <= bounds[1]
        )
        results.append({"constraint": number["field"], "value": expected if expected is not None else bounds, "matched": bool(matched)})
    return results


def calculate_match(request: Mapping[str, Any], helper: Mapping[str, Any], *, cosine: float | None, request_place: Mapping[str, Any] | None, helper_place: Mapping[str, Any] | None, backend: str | None, model: str | None) -> dict[str, Any]:
    rejected = eligibility_reasons(request, helper)
    if rejected:
        return {"eligible": False, "rejected_reasons": rejected}

    semantic = semantic_component(cosine)
    location, distance_km = location_component(request_place, helper_place, helper.get("max_travel_km"))
    availability = availability_component(request.get("availability_slots") or [], helper.get("availability_slots") or [])
    languages = helper.get("spoken_languages") or []
    user_lang = request.get("user_lang") if request.get("user_lang_confirmed") else None
    language = language_component(user_lang, request.get("stated_lang"), languages, bool(request.get("translation_ok")))
    reliability = reliability_component(helper.get("ratings") or [], helper.get("response_rate"))
    relevance = manual_relevance_component(request, helper) if semantic is None else None
    components = {
        "semantic": semantic, "relevance": relevance, "location": location,
        "availability": availability, "language": language, "reliability": reliability,
    }
    included = {name: value for name, value in components.items() if value is not None}
    score_weights = (
        {name: 1.0 for name in included if name != "semantic"}
        if semantic is None else WEIGHTS
    )
    weight_total = sum(score_weights.get(name, 0.0) for name in included)
    raw_score = sum(score_weights.get(name, 0.0) * value for name, value in included.items()) / weight_total if weight_total else None
    constraint_results = exact_constraint_results(request, helper)
    exact_hits = sum(result["matched"] for result in constraint_results)
    exact_all = bool(constraint_results) and exact_hits == len(constraint_results)
    confidence = 100 if exact_all else min(99, round(raw_score * 100)) if raw_score is not None else None
    label = "Exact match" if exact_all else "Strong" if confidence is not None and confidence >= 80 else "Good" if confidence is not None and confidence >= 60 else "Possible" if confidence is not None and confidence >= 40 else None
    return {
        "eligible": True,
        "confidence": confidence,
        "label": label,
        "raw_score": raw_score,
        "semantic_cosine": cosine,
        "semantic_calibration": {"lo": SEMANTIC_LO, "hi": SEMANTIC_HI},
        "components": components,
        "weights": {name: score_weights.get(name) if value is not None else None for name, value in components.items()},
        "distance_km": distance_km,
        "lang_exact": int(bool(user_lang and user_lang.casefold() in {str(value).casefold() for value in languages})),
        "exact_all": exact_all,
        "exact_hits": exact_hits,
        "exact_constraints": constraint_results,
        "backend": backend,
        "model": model,
    }


def rank_matches(matches: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
        components = item["components"]
        helper = item["helper"]
        confidence = item.get("confidence")
        availability = components.get("availability")
        reliability = components.get("reliability")
        return (
            -(confidence if confidence is not None else -1),
            -item.get("lang_exact", 0),
            -item.get("exact_hits", 0),
            item.get("distance_km") if item.get("distance_km") is not None else math.inf,
            -(availability if availability is not None else -1),
            -(reliability if reliability is not None else -1),
            helper.get("average_response_seconds") if helper.get("average_response_seconds") is not None else math.inf,
            str(helper.get("id", "")),
        )

    ranked = [dict(item) for item in matches]
    ranked.sort(key=sort_key)
    criteria = ("confidence", "lang_exact", "exact_hits", "distance_km", "availability", "reliability", "response_time", "helper_id")
    for index, item in enumerate(ranked):
        item["rank"] = index + 1
        if index == 0:
            item["ranked_above_previous_by"] = None
        else:
            previous_key = sort_key(ranked[index - 1])
            current_key = sort_key(item)
            item["ranked_above_previous_by"] = next(
                (criteria[position] for position, (current, prior) in enumerate(zip(current_key, previous_key)) if current != prior),
                "tie",
            )
        if index == len(ranked) - 1:
            item["ranked_above_next_by"] = None
        else:
            next_key = sort_key(ranked[index + 1])
            current_key = sort_key(item)
            item["ranked_above_next_by"] = next(
                (criteria[position] for position, (current, following) in enumerate(zip(current_key, next_key)) if current != following),
                "tie",
            )
    return ranked