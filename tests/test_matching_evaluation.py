import collections
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_matching import (
    _confidence,
    load_rows,
    percentile,
    split_rows,
    write_golden_draft,
)
from core.scoring import availability_component, language_component, location_component, reliability_component


class MatchingEvaluationTests(unittest.TestCase):
    def test_draft_has_120_bilingual_pairs_and_rule_labeled_balance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "golden_matches.csv"
            write_golden_draft(path)
            rows = load_rows(path)

        self.assertEqual(len(rows), 120)
        self.assertEqual(collections.Counter(row["label"] for row in rows), {0: 60, 1: 60})
        self.assertEqual(len({row["request"] for row in rows}), 10)
        self.assertEqual(len({row["features"]["request_id"] for row in rows}), 10)
        self.assertEqual(sum(row["features"]["exact_all"] for row in rows), 20)
        for row in rows:
            features = row["features"]
            location, _distance = location_component(
                features["request_place"], features["helper_place"], features["max_travel_km"],
            )
            self.assertAlmostEqual(features["components"]["location"], location)
            self.assertEqual(
                features["components"]["availability"],
                availability_component(features["request_slots"], features["helper_slots"]),
            )
            self.assertEqual(
                features["components"]["language"],
                language_component(features["user_lang"], None, features["spoken_languages"], False),
            )
            self.assertEqual(
                features["components"]["reliability"],
                reliability_component(features["ratings"], features["response_rate"]),
            )

    def test_each_request_has_a_deterministic_confidence_tie_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "golden_matches.csv"
            write_golden_draft(path)
            rows = load_rows(path)
        tied_groups = 0
        grouped = collections.defaultdict(list)
        for row in rows:
            grouped[row["features"]["request_id"]].append(row)
        for candidates in grouped.values():
            first, second = candidates[:2]
            first_features = dict(first["features"])
            second_features = dict(second["features"])
            first_features.pop("helper_id")
            second_features.pop("helper_id")
            self.assertEqual(first["helper"], second["helper"])
            self.assertEqual(first_features, second_features)
            tied_groups += 1
        self.assertEqual(tied_groups, 10)

    def test_split_holds_out_whole_request_groups_at_70_30(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "golden_matches.csv"
            write_golden_draft(path)
            rows = load_rows(path)
        train, test = split_rows(rows)

        self.assertEqual((len(train), len(test)), (84, 36))
        self.assertFalse(
            {row["features"]["request_id"] for row in train}
            & {row["features"]["request_id"] for row in test}
        )
        self.assertEqual(split_rows(rows), (train, test))

    def test_percentile_interpolates_and_exact_score_is_not_free(self):
        self.assertEqual(percentile([0.0, 0.5, 1.0], 0.05), 0.05)
        weights = {"semantic": 0.2, "location": 0.2, "availability": 0.2, "language": 0.2, "reliability": 0.2}
        components = {name: 1.0 for name in weights}
        self.assertEqual(_confidence(components, weights, False), 99)
        self.assertEqual(_confidence(components, weights, True), 100)


if __name__ == "__main__":
    unittest.main()