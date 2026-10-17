import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.scoring import (
    availability_component,
    cosine_similarity,
    language_component,
    location_component,
    reliability_component,
    semantic_component,
)

DATA_PATH = ROOT / "tests" / "golden_matches.csv"
REPORT_PATH = ROOT / "data" / "matching_evaluation.json"
CALIBRATION_PATH = ROOT / "data" / "matching_calibration.json"
MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COMPONENTS = ("semantic", "location", "availability", "language", "reliability")
INITIAL_WEIGHTS = {
    "semantic": 0.35,
    "location": 0.20,
    "availability": 0.20,
    "language": 0.10,
    "reliability": 0.15,
}
INITIAL_LO = 0.25
INITIAL_HI = 0.75

REQUESTS = [
    ("it", "Education", "Cerco ripetizioni di matematica per preparare un esame.", "ripetizioni di matematica", "math tutoring"),
    ("en", "Education", "I need math tutoring to prepare for an exam.", "math tutoring", "ripetizioni di matematica"),
    ("it", "Housing Support", "Cerco una stanza in affitto vicino all'universita.", "stanza in affitto", "room to rent"),
    ("en", "Housing Support", "I need a room to rent near the university.", "room to rent", "stanza in affitto"),
    ("it", "Transport Support", "Mi serve un passaggio per arrivare a una visita.", "passaggio per una visita", "lift to an appointment"),
    ("en", "Transport Support", "I need a lift to get to an appointment.", "lift to an appointment", "passaggio per una visita"),
    ("it", "Digital Support", "Ho bisogno di aiuto con SPID e moduli online.", "aiuto con SPID e moduli online", "help with SPID and online forms"),
    ("en", "Digital Support", "I need help with SPID and online forms.", "help with SPID and online forms", "aiuto con SPID e moduli online"),
    ("it", "Food Support", "Cerco aiuto per fare la spesa questa settimana.", "aiuto per fare la spesa", "help with grocery shopping"),
    ("en", "Food Support", "I need help with grocery shopping this week.", "help with grocery shopping", "aiuto per fare la spesa"),
]

SLOT = {"day": "saturday", "part_of_day": "morning"}
REQUEST_SLOTS = [SLOT, {"day": "saturday", "part_of_day": "afternoon"}]
VARIANTS = ("exact", "exact", "exact", "related", "related", "related", "borderline", "borderline", "borderline", "unrelated", "unrelated", "unrelated")
DISTANCES = (0.0, 0.0, 0.5, 1.5, 2.5, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 13.0)
OVERLAPS = (2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0)


def _helper_text(variant, request_skill, translated_skill, language):
    if variant == "exact":
        return request_skill
    if variant == "related":
        return f"Practical local support for {translated_skill}." if language == "it" else f"Supporto pratico per {translated_skill}."
    if variant == "borderline":
        return "Supporto alla comunita e orientamento locale." if language == "it" else "Community support and local guidance."
    return "Giardinaggio e cura degli animali." if language == "it" else "Gardening and pet care."


def _feature_context(request_index, variant_index, variant):
    user_lang = REQUESTS[request_index][0]
    tie_copy = variant_index in (0, 1)
    context_index = 0 if tie_copy else variant_index
    distance = DISTANCES[context_index]
    origin = {"lat": 44.4, "lon": 8.9}
    helper_place = {"lat": 44.4 + distance / 111.195, "lon": 8.9}
    location, actual_distance = location_component(origin, helper_place, 12.0)
    overlap_count = OVERLAPS[context_index]
    helper_slots = REQUEST_SLOTS[:overlap_count]
    availability = availability_component(REQUEST_SLOTS, helper_slots)
    languages = [user_lang] if context_index in (0, 1, 2, 4, 6, 8) else (["en"] if user_lang == "it" else ["it"])
    language = language_component(user_lang, None, languages, False)
    ratings_by_index = (
        [5, 5, 4], [5, 4, 4], [4, 4, 4], [4, 3, 4], [4, 3, 3], [3, 3, 4],
        [3, 3, 3], [3, 2, 3], [2, 3, 3], [2, 2, 3], [1, 2, 2], [1, 1, 2],
    )
    ratings = ratings_by_index[context_index]
    response_rate = (0.98, 0.90, 0.85, 0.80, 0.72, 0.65, 0.60, 0.55, 0.50, 0.40, 0.30, 0.25)[context_index]
    reliability = reliability_component(ratings, response_rate)
    explicit_matches = {
        "skill": variant == "exact",
        "place": actual_distance == 0,
        "availability": overlap_count == len(REQUEST_SLOTS),
    }
    exact_hits = sum(explicit_matches.values())
    exact_all = bool(explicit_matches) and all(explicit_matches.values())
    return {
        "components": {
            "location": location,
            "availability": availability,
            "language": language,
            "reliability": reliability,
        },
        "exact_all": exact_all,
        "exact_hits": exact_hits,
        "lang_exact": int(user_lang in languages),
        "user_lang": user_lang,
        "distance_km": actual_distance,
        "response_time": 30 + context_index * 15,
        "helper_id": f"helper-{request_index:02d}-{variant_index:02d}",
        "explicit_matches": explicit_matches,
        "spoken_languages": languages,
        "ratings": ratings,
        "response_rate": response_rate,
        "max_travel_km": 12.0,
        "category": REQUESTS[request_index][1],
        "request_id": f"request-{request_index:02d}",
        "helper_categories": [REQUESTS[request_index][1]],
        "helper_active": True,
        "request_place": origin,
        "helper_place": helper_place,
        "request_slots": REQUEST_SLOTS,
        "helper_slots": helper_slots,
    }


def write_golden_draft(path=DATA_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("request", "helper", "label", "features"))
        writer.writeheader()
        for request_index, (language, _category, request_text, request_skill, translated_skill) in enumerate(REQUESTS):
            for variant_index, variant in enumerate(VARIANTS):
                helper_text = _helper_text(variant, request_skill, translated_skill, language)
                features = _feature_context(request_index, variant_index, variant)
                # Labels use an explicit skill-relevance rubric; no model output is used.
                label = int(variant in {"exact", "related"})
                writer.writerow({
                    "request": request_text,
                    "helper": helper_text,
                    "label": label,
                    "features": json.dumps(features, ensure_ascii=False, sort_keys=True),
                })


def load_rows(path=DATA_PATH):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"request", "helper", "label", "features"}
        if not required <= set(reader.fieldnames or ()):
            raise ValueError("Golden CSV requires request, helper, label, and features columns")
        rows = []
        for row in reader:
            if row["label"] not in {"0", "1"}:
                raise ValueError("Golden labels must be 0 or 1")
            context = json.loads(row["features"])
            rows.append({
                "request": row["request"],
                "helper": row["helper"],
                "label": int(row["label"]),
                "features": context,
            })
    if len(rows) < 120:
        raise ValueError(f"Golden set has {len(rows)} rows; at least 120 are required")
    return rows


def split_rows(rows, seed=42):
    requests = list(dict.fromkeys(row["features"]["request_id"] for row in rows))
    random.Random(seed).shuffle(requests)
    train_size = int(len(requests) * 0.70)
    train_requests = set(requests[:train_size])
    return (
        [row for row in rows if row["features"]["request_id"] in train_requests],
        [row for row in rows if row["features"]["request_id"] not in train_requests],
    )


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate a percentile from an empty set")
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def fit_weights(rows):
    import numpy as np
    from scipy.optimize import minimize

    matrix = np.array([[row["components"][name] for name in COMPONENTS] for row in rows], dtype=float)
    labels = np.array([row["label"] for row in rows], dtype=float)

    def objective(parameters):
        logits = parameters[0] + matrix @ parameters[1:]
        losses = np.logaddexp(0.0, logits) - labels * logits
        regularization = 1e-6 * float(parameters[1:] @ parameters[1:])
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
        gradient = np.concatenate((
            [float(np.sum(probabilities - labels))],
            matrix.T @ (probabilities - labels),
        ))
        gradient[0] /= len(labels)
        gradient[1:] = gradient[1:] / len(labels) + 2e-6 * parameters[1:]
        return float(np.mean(losses) + regularization), gradient

    initial = np.array([0.0] + [1.0] * len(COMPONENTS), dtype=float)
    bounds = [(None, None)] + [(0.0, None)] * len(COMPONENTS)
    result = minimize(objective, initial, jac=True, method="L-BFGS-B", bounds=bounds)
    if not result.success:
        raise RuntimeError(f"Logistic regression did not converge: {result.message}")
    coefficients = result.x[1:]
    total = float(coefficients.sum())
    if not math.isfinite(total) or total <= 0:
        raise RuntimeError("Fitted logistic coefficients cannot be rescaled to sum to one")
    return {name: float(value / total) for name, value in zip(COMPONENTS, coefficients)}, float(result.x[0])


def _confidence(components, weights, exact_all):
    included = [(weights[name], components.get(name)) for name in COMPONENTS if components.get(name) is not None]
    denominator = sum(weight for weight, _value in included)
    if exact_all:
        return 100
    if not denominator:
        return None
    score = sum(weight * value for weight, value in included) / denominator
    return min(99, round(score * 100))


def _sort_key(item):
    fields = item["features"]
    components = item["components"]
    return (
        -(item["confidence"] if item["confidence"] is not None else -1),
        -fields["lang_exact"],
        -fields["exact_hits"],
        fields["distance_km"] if fields["distance_km"] is not None else math.inf,
        -(components["availability"] if components["availability"] is not None else -1),
        -(components["reliability"] if components["reliability"] is not None else -1),
        fields["response_time"] if fields["response_time"] is not None else math.inf,
        fields["helper_id"],
    )


def _dcg(labels, cutoff):
    return sum((2 ** label - 1) / math.log2(rank + 1) for rank, label in enumerate(labels[:cutoff], 1))


def ranking_metrics(rows, weights, lo, hi):
    groups = defaultdict(list)
    for row in rows:
        semantic = semantic_component(row["cosine"], lo, hi)
        components = {**row["features"]["components"], "semantic": semantic}
        confidence = _confidence(components, weights, row["features"]["exact_all"])
        groups[row["features"]["request_id"]].append({
            "label": row["label"],
            "confidence": confidence,
            "features": row["features"],
            "components": components,
        })

    precision_at_one = []
    precision_at_three = []
    ndcg_at_five = []
    false_shown = 0
    total_bad = 0
    for candidates in groups.values():
        candidates.sort(key=_sort_key)
        labels = [item["label"] for item in candidates]
        precision_at_one.append(sum(labels[:1]))
        precision_at_three.append(sum(labels[:3]) / 3)
        ideal_dcg = _dcg(sorted(labels, reverse=True), 5)
        ndcg_at_five.append(_dcg(labels, 5) / ideal_dcg if ideal_dcg else 0.0)
        false_shown += sum(item["label"] == 0 and item["confidence"] is not None and item["confidence"] >= 60 for item in candidates)
        total_bad += labels.count(0)
    return {
        "precision_at_1": sum(precision_at_one) / len(precision_at_one),
        "precision_at_3": sum(precision_at_three) / len(precision_at_three),
        "ndcg_at_5": sum(ndcg_at_five) / len(ndcg_at_five),
        "false_match_rate": false_shown / total_bad if total_bad else 0.0,
        "false_matches_shown": false_shown,
        "bad_pairs": total_bad,
    }


def embed_rows(rows):
    from core.embedding_cache import load_or_create_embeddings_with_backend

    texts = []
    for row in rows:
        texts.extend((row["request"], row["helper"]))
    result = load_or_create_embeddings_with_backend(texts, MODEL_ID)
    if result is None:
        raise RuntimeError("Cannot rank: AMD and the local multilingual embedding model are unavailable")
    vectors = result["vectors"]
    for index, row in enumerate(rows):
        row["cosine"] = cosine_similarity(vectors[index * 2], vectors[index * 2 + 1])
        if row["cosine"] is None:
            raise RuntimeError("Cannot rank: an embedding pair is missing or invalid")
    return result


def evaluate(path=DATA_PATH, seed=42):
    rows = load_rows(path)
    backend = embed_rows(rows)
    train, test = split_rows(rows, seed)
    bad_cosines = [row["cosine"] for row in train if row["label"] == 0]
    good_cosines = [row["cosine"] for row in train if row["label"] == 1]
    if not bad_cosines or not good_cosines:
        raise ValueError("Training split must contain both good and bad pairs")
    lo = percentile(bad_cosines, 0.05)
    hi = percentile(good_cosines, 0.95)
    if hi <= lo:
        raise ValueError(f"Fitted semantic bounds are invalid: lo={lo}, hi={hi}")
    for row in rows:
        row["components"] = {
            **row["features"]["components"],
            "semantic": semantic_component(row["cosine"], lo, hi),
        }
    new_weights, intercept = fit_weights(train)
    old_metrics = ranking_metrics(test, INITIAL_WEIGHTS, INITIAL_LO, INITIAL_HI)
    new_metrics = ranking_metrics(test, new_weights, lo, hi)
    report = {
        "dataset_rows": len(rows),
        "train_rows": len(train),
        "test_rows": len(test),
        "seed": seed,
        "embedding_backend": backend["backend"],
        "embedding_model": backend["model"],
        "semantic_bounds": {"lo": lo, "hi": hi},
        "weights": new_weights,
        "logistic_intercept": intercept,
        "metrics": {"old": old_metrics, "new": new_metrics},
    }
    CALIBRATION_PATH.write_text(json.dumps({
        "weights": new_weights,
        "semantic_bounds": {"lo": lo, "hi": hi},
    }, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def print_report(report):
    old = report["metrics"]["old"]
    new = report["metrics"]["new"]
    print("CommunityConnect matching evaluation (held-out test set)")
    print(f"Pairs: {report['dataset_rows']}  Train: {report['train_rows']}  Test: {report['test_rows']}  Seed: {report['seed']}")
    print(f"Embeddings: {report['embedding_backend']} / {report['embedding_model']}")
    print(f"Semantic bounds: lo={report['semantic_bounds']['lo']:.6f}, hi={report['semantic_bounds']['hi']:.6f}")
    print("Fitted weights: " + ", ".join(f"{name}={value:.6f}" for name, value in report["weights"].items()))
    print(f"{'Metric':<22}{'Old scorer':>15}{'New scorer':>15}")
    print("-" * 52)
    for metric, label in (("precision_at_1", "Precision@1"), ("precision_at_3", "Precision@3"), ("ndcg_at_5", "NDCG@5"), ("false_match_rate", "False-match rate")):
        print(f"{label:<22}{old[metric]:>14.1%}{new[metric]:>15.1%}")
    print(f"Bad pairs shown at >=60: old {old['false_matches_shown']}/{old['bad_pairs']}; new {new['false_matches_shown']}/{new['bad_pairs']}")


def main():
    parser = argparse.ArgumentParser(description="Draft and evaluate CommunityConnect match scoring.")
    parser.add_argument("--write-golden-draft", action="store_true", help="Generate the rule-labeled 120-pair review draft.")
    parser.add_argument("--labels-reviewed", action="store_true", help="Confirm the labels were reviewed and edited by a human.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.write_golden_draft:
        write_golden_draft()
        print(f"Wrote 120 rule-labeled pairs to {DATA_PATH}")
        return
    if not args.labels_reviewed:
        parser.error("review/edit tests/golden_matches.csv first, then pass --labels-reviewed")
    try:
        report = evaluate(seed=args.seed)
    except (ImportError, RuntimeError) as error:
        parser.exit(2, f"Evaluation stopped: {error}\n")
    print_report(report)


if __name__ == "__main__":
    main()