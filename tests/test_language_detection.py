import unittest
from unittest.mock import patch

from core.language_detection import detect_user_language


class UserLanguageDetectionTests(unittest.TestCase):
    def test_ambiguous_short_italian_phrase_requires_confirmation(self):
        result = detect_user_language("affitto stanza")
        self.assertEqual(result["suggested_language"], "it")
        self.assertTrue(result["needs_confirmation"])

    def test_required_bilingual_examples_are_detected(self):
        examples = {
            "rentals": "en",
            "cerco una stanza in affitto": "it",
            "ripetizioni matematica": "it",
            "need a lift to hospital": "en",
        }
        for text, expected in examples.items():
            with self.subTest(text=text):
                self.assertEqual(detect_user_language(text)["suggested_language"], expected)

    def test_disagreement_with_ai_intake_requires_confirmation(self):
        result = detect_user_language("I need help finding a rental", "it")
        self.assertEqual(result["suggested_language"], "en")
        self.assertTrue(result["ai_disagrees"])
        self.assertTrue(result["needs_confirmation"])


if __name__ == "__main__":
    unittest.main()