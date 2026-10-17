import unittest

from core.scoring import (
    availability_component,
    calculate_match,
    cosine_similarity,
    eligibility_reasons,
    language_component,
    location_component,
    normalize_text,
    rank_matches,
    reliability_component,
    semantic_component,
    SEMANTIC_HI,
    SEMANTIC_LO,
)


def _request(**overrides):
    return {
        "category": "Housing Support",
        "skills": ["rental room"],
        "availability_slots": [{"day": "monday", "part_of_day": "afternoon"}],
        "place_id": "genova",
        "user_lang": "it",
        "user_lang_confirmed": True,
        "stated_lang": None,
        "translation_ok": False,
        "explicit_constraints": {},
        **overrides,
    }


def _helper(**overrides):
    return {
        "id": "helper-a",
        "active": True,
        "categories": ["Housing Support"],
        "skills": ["rent room"],
        "spoken_languages": ["en"],
        "availability_slots": [{"day": "monday", "part_of_day": "afternoon"}],
        "place_id": "genova",
        "max_travel_km": 10,
        "ratings": [],
        "response_rate": None,
        "average_response_seconds": None,
        "numeric_offers": [],
        **overrides,
    }


class ScoringComponentTests(unittest.TestCase):
    def test_normalization_handles_cross_language_synonyms_and_accents(self):
        pairs = [
            ("rental", "affitto"),
            ("room", "stanza"),
            ("tutoring", "ripetizioni"),
            ("lift", "passaggio"),
            ("elderly care", "assistenza anziani"),
        ]
        for english, italian in pairs:
            with self.subTest(english=english):
                self.assertEqual(normalize_text(english), normalize_text(italian))
        self.assertEqual(normalize_text("Città, Genova!"), "citta genova")

    def test_semantic_calibration_clips_and_maps_bounds(self):
        midpoint = (SEMANTIC_LO + SEMANTIC_HI) / 2
        self.assertEqual(semantic_component(SEMANTIC_LO), 0)
        self.assertEqual(semantic_component(midpoint), 0.5)
        self.assertEqual(semantic_component(SEMANTIC_HI), 1)
        self.assertEqual(semantic_component(SEMANTIC_LO - 1), 0)
        self.assertEqual(semantic_component(SEMANTIC_HI + 1), 1)
        self.assertIsNone(semantic_component(None))

    def test_cosine_rejects_missing_mismatched_and_zero_vectors(self):
        self.assertIsNone(cosine_similarity([], [1]))
        self.assertIsNone(cosine_similarity([1], [1, 2]))
        self.assertIsNone(cosine_similarity([0], [1]))
        self.assertEqual(cosine_similarity([1, 0], [1, 0]), 1)

    def test_location_score_decreases_with_distance_and_missing_is_none(self):
        origin = {"lat": 44.4, "lon": 8.9}
        near = {"lat": 44.41, "lon": 8.9}
        far = {"lat": 44.5, "lon": 8.9}
        near_score, near_distance = location_component(origin, near, 20)
        far_score, far_distance = location_component(origin, far, 20)
        self.assertGreater(near_score, far_score)
        self.assertLess(near_distance, far_distance)
        self.assertEqual(location_component(origin, far, None), (None, None))

    def test_availability_overlap_is_monotonic_and_empty_request_is_missing(self):
        monday = {"day": "monday", "part_of_day": "morning"}
        tuesday = {"day": "tuesday", "part_of_day": "evening"}
        request_slots = [monday, tuesday]
        self.assertEqual(availability_component(request_slots, []), 0)
        self.assertEqual(availability_component(request_slots, [monday]), 0.5)
        self.assertEqual(availability_component(request_slots, [monday, tuesday]), 1)
        self.assertIsNone(availability_component([], [monday]))

    def test_language_and_reliability_missing_data_are_not_penalties(self):
        self.assertEqual(language_component("it", None, ["en"], False), 0.6)
        self.assertEqual(language_component("it", None, ["it"], False), 1)
        self.assertEqual(language_component("it", "en", ["it"], False), 1)
        self.assertIsNone(language_component(None, None, [], False))
        self.assertIsNone(reliability_component([], 1.0))
        self.assertIsNone(reliability_component([5], None))
        self.assertGreater(reliability_component([5], 1.0), reliability_component([1], 0.0))

    def test_hard_filters_return_reasons_and_translation_does_not_filter_user_language(self):
        self.assertEqual(eligibility_reasons(_request(), _helper()), [])
        self.assertIn("stated_language_not_offered", eligibility_reasons(
            _request(stated_lang="it"), _helper(),
        ))
        self.assertEqual(eligibility_reasons(
            _request(stated_lang="it", translation_ok=True), _helper(),
        ), [])
        self.assertIn("category_not_offered", eligibility_reasons(
            _request(explicit_constraints={"category": True}), _helper(categories=["Education"]),
        ))
        self.assertEqual(eligibility_reasons(
            _request(explicit_constraints={"category": False}), _helper(categories=["Education"]),
        ), [])
        self.assertEqual(eligibility_reasons(
            _request(category=None, explicit_constraints={}), _helper(categories=["Education"]),
        ), [])
        self.assertIn("helper_inactive_or_unconfirmed", eligibility_reasons(
            _request(), _helper(active=None),
        ))

    def test_score_renormalizes_only_over_present_components_and_exact_is_capped(self):
        result = calculate_match(
            _request(explicit_constraints={"skills": True}, skills=["rental room"]),
            _helper(skills=["rental room"]),
            cosine=None,
            request_place=None,
            helper_place=None,
            backend=None,
            model=None,
        )
        self.assertEqual(result["confidence"], 100)
        self.assertTrue(result["exact_all"])
        self.assertIsNone(result["components"]["semantic"])
        self.assertIsNone(result["components"]["location"])
        self.assertIsNone(result["weights"]["semantic"])

        fuzzy = calculate_match(
            _request(), _helper(), cosine=0.75,
            request_place={"lat": 44.4, "lon": 8.9},
            helper_place={"lat": 44.4, "lon": 8.9},
            backend="local-cpu", model="test-model",
        )
        self.assertLessEqual(fuzzy["confidence"], 99)
        self.assertEqual(fuzzy["backend"], "local-cpu")

    def test_removing_place_chip_removes_location_component_from_score(self):
        origin = {"lat": 44.4, "lon": 8.9}
        arguments = {
            "cosine": 0.5,
            "helper_place": origin,
            "backend": "local-cpu",
            "model": "test-model",
        }
        with_place = calculate_match(
            _request(place_id="genova", availability_slots=[], user_lang=None, user_lang_confirmed=False),
            _helper(max_travel_km=10), request_place=origin, **arguments,
        )
        without_place = calculate_match(
            _request(place_id=None, availability_slots=[], user_lang=None, user_lang_confirmed=False),
            _helper(max_travel_km=10), request_place=None, **arguments,
        )

        self.assertIsNotNone(with_place["components"]["location"])
        self.assertIsNone(without_place["components"]["location"])
        self.assertNotEqual(with_place["raw_score"], without_place["raw_score"])

    def test_nice_to_have_place_keeps_score_component_but_leaves_exact_set(self):
        origin = {"lat": 44.4, "lon": 8.9}
        arguments = {
            "cosine": 0.5, "request_place": origin, "helper_place": origin,
            "backend": "local-cpu", "model": "test-model",
        }
        request = _request(place_id="genova", availability_slots=[], user_lang=None, user_lang_confirmed=False)
        required = calculate_match(
            _request(**{**request, "explicit_constraints": {"place_id": True}}),
            _helper(max_travel_km=10), **arguments,
        )
        nice_to_have = calculate_match(
            _request(**{**request, "explicit_constraints": {"place_id": False}}),
            _helper(max_travel_km=10), **arguments,
        )

        self.assertEqual(required["components"]["location"], nice_to_have["components"]["location"])
        self.assertTrue(required["exact_all"])
        self.assertFalse(nice_to_have["exact_all"])
        self.assertEqual(required["confidence"], 100)
        self.assertLessEqual(nice_to_have["confidence"], 99)

    def test_stated_language_is_hard_filter_but_written_language_is_not(self):
        request = _request(user_lang="it")
        english_only = _helper(spoken_languages=["en"])
        self.assertEqual(eligibility_reasons(request, english_only), [])
        rejected = eligibility_reasons(_request(stated_lang="it"), english_only)
        self.assertEqual(rejected, ["stated_language_not_offered"])

    def test_unconfirmed_language_suggestion_is_not_a_scoring_signal(self):
        helper = _helper(spoken_languages=["it"])
        arguments = {
            "cosine": 0.5,
            "request_place": None,
            "helper_place": None,
            "backend": "amd",
            "model": "test-model",
        }
        suggested = calculate_match(_request(user_lang_confirmed=False), helper, **arguments)
        confirmed = calculate_match(_request(user_lang_confirmed=True), helper, **arguments)
        self.assertIsNone(suggested["components"]["language"])
        self.assertEqual(suggested["lang_exact"], 0)
        self.assertEqual(confirmed["components"]["language"], 1)
        self.assertEqual(confirmed["lang_exact"], 1)

    def test_explicit_numeric_requirement_does_not_match_a_different_offer(self):
        request = _request(
            number_constraints=[{"field": "hours", "value": 2, "range": None, "unit": "hours"}],
            explicit_constraints={"numbers": True},
        )
        helper = _helper(numeric_offers=[{"field": "hours", "value": 1, "unit": "hours"}])
        exact_helper = _helper(numeric_offers=[{"field": "hours", "value": 2, "unit": "hours"}])
        rejected_number = calculate_match(
            request, helper, cosine=0.9, request_place=None, helper_place=None, backend="amd", model="test",
        )
        exact_number = calculate_match(
            request, exact_helper, cosine=0.5, request_place=None, helper_place=None, backend="amd", model="test",
        )
        self.assertFalse(rejected_number["exact_all"])
        self.assertEqual(rejected_number["exact_hits"], 0)
        self.assertLess(rejected_number["confidence"], 100)
        self.assertTrue(exact_number["exact_all"])
        self.assertEqual(exact_number["confidence"], 100)

    def test_required_skill_chip_uses_saved_skill_tags_for_exact_matching(self):
        result = calculate_match(
            _request(skills=[], skill_tags=["rental room"], explicit_constraints={"skills": True}),
            _helper(skills=["rental room"]),
            cosine=0.5, request_place=None, helper_place=None, backend="local-cpu", model="test-model",
        )
        self.assertTrue(result["exact_all"])
        self.assertEqual(result["exact_hits"], 1)


class MatchRankingTests(unittest.TestCase):
    @staticmethod
    def _item(helper_id, confidence, lang_exact=0, exact_hits=0, distance=None):
        return {
            "confidence": confidence,
            "lang_exact": lang_exact,
            "exact_hits": exact_hits,
            "distance_km": distance,
            "components": {"availability": None, "reliability": None},
            "helper": {"id": helper_id, "average_response_seconds": None},
        }

    def test_confidence_precedes_language_and_tie_break_chain_is_deterministic(self):
        higher = self._item("b", 81, lang_exact=0)
        lower = self._item("a", 80, lang_exact=1)
        language_tie = self._item("c", 80, lang_exact=0, distance=4)
        ranked = rank_matches([lower, language_tie, higher])
        self.assertEqual([item["confidence"] for item in ranked], [81, 80, 80])
        self.assertEqual(ranked[1]["ranked_above_previous_by"], "confidence")
        self.assertEqual(ranked[2]["ranked_above_previous_by"], "lang_exact")
        self.assertEqual([item["helper"]["id"] for item in ranked], ["b", "a", "c"])
        self.assertEqual(rank_matches([lower, language_tie, higher]), ranked)


if __name__ == "__main__":
    unittest.main()