import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from communityconnect import _local_triage, intake_request, local_intake, triage_request

DATA_PATH = ROOT / "data" / "evaluation_requests.json"


def load_cases():
    cases = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    if not 30 <= len(cases) <= 50:
        raise ValueError("The labeled evaluation set must contain 30 to 50 cases")
    for case in cases:
        if set(case) != {"text", "language", "category", "emergency"}:
            raise ValueError("Every case needs text, language, category, and emergency labels")
        if case["language"] not in {"it", "en"} or not isinstance(case["emergency"], bool):
            raise ValueError("Evaluation labels must use Italian/English and boolean emergency values")
    return cases


def evaluate(backend):
    cases = load_cases()
    intake_correct = 0
    emergency_true_positive = 0
    emergency_false_negative = 0
    emergency_false_positive = 0
    category_errors = Counter()
    backends = Counter()
    for case in cases:
        if backend == "local":
            intake = local_intake(case["text"])
            safety = _local_triage(case["text"])
        else:
            intake = intake_request(case["text"])
            safety = triage_request(case["text"])
        predicted_category = intake["category"]
        backends[intake.get("backend", "local")] += 1
        expected_emergency = case["emergency"]
        predicted_emergency = safety["emergency"]
        intake_correct += predicted_category == case["category"]
        if predicted_category != case["category"]:
            category_errors[(case["category"], predicted_category)] += 1
        emergency_true_positive += expected_emergency and predicted_emergency
        emergency_false_negative += expected_emergency and not predicted_emergency
        emergency_false_positive += not expected_emergency and predicted_emergency

    emergency_total = emergency_true_positive + emergency_false_negative
    return {
        "cases": len(cases),
        "backend": backend,
        "intake_accuracy": intake_correct / len(cases),
        "intake_correct": intake_correct,
        "emergency_recall": emergency_true_positive / emergency_total if emergency_total else 0.0,
        "emergency_true_positive": emergency_true_positive,
        "emergency_total": emergency_total,
        "emergency_false_negative": emergency_false_negative,
        "emergency_false_positive": emergency_false_positive,
        "category_errors": category_errors,
        "backends": backends,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate request intake accuracy and emergency recall.")
    parser.add_argument("--backend", choices=("local", "configured"), default="local")
    args = parser.parse_args()
    result = evaluate(args.backend)
    print("CommunityConnect labeled-request evaluation")
    print(f"Backend: {result['backend']}")
    print("Call backend counts: " + ", ".join(f"{name}={count}" for name, count in sorted(result["backends"].items())))
    print(f"Cases: {result['cases']}")
    print(f"Intake accuracy: {result['intake_correct']}/{result['cases']} = {result['intake_accuracy']:.1%}")
    print(
        "Emergency recall: "
        f"{result['emergency_true_positive']}/{result['emergency_total']} = {result['emergency_recall']:.1%}"
    )
    print(f"Emergency false negatives: {result['emergency_false_negative']}")
    print(f"Emergency false positives: {result['emergency_false_positive']}")
    if result["category_errors"]:
        print("Category errors:")
        for (expected, predicted), count in sorted(result["category_errors"].items()):
            print(f"  expected {expected}; predicted {predicted}: {count}")


if __name__ == "__main__":
    main()
