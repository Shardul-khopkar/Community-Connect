import re
from functools import lru_cache

from lingua import Language, LanguageDetectorBuilder


SUPPORTED_USER_LANGUAGES = {"it", "en"}


@lru_cache(maxsize=1)
def _detector():
    return LanguageDetectorBuilder.from_languages(Language.ITALIAN, Language.ENGLISH).build()


def detect_user_language(text: str, ai_language: str | None = None) -> dict:
    words = re.findall(r"[a-zà-ÿ]+", text.casefold())
    detector = _detector()
    values = detector.compute_language_confidence_values(text)
    ranked = sorted(values, key=lambda item: item.value, reverse=True)
    suggestion = ranked[0].language.iso_code_639_1.name.lower() if ranked else None
    top_confidence = ranked[0].value if ranked else 0.0
    runner_up = ranked[1].value if len(ranked) > 1 else 0.0
    short_text = len(words) <= 2
    disagreement = ai_language in SUPPORTED_USER_LANGUAGES and suggestion != ai_language
    needs_confirmation = (
        not suggestion
        or top_confidence < 0.80
        or top_confidence - runner_up < 0.20
        or short_text
        or disagreement
    )
    return {
        "suggested_language": suggestion,
        "needs_confirmation": needs_confirmation,
        "detector_confidence": top_confidence,
        "ai_language": ai_language,
        "ai_disagrees": disagreement,
    }