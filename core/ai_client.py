import json
import math
import os
import re
import time
from threading import Lock
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from core.privacy import deidentify_text, restore_redactions

_STATUS_CACHE_SECONDS = 60
_status_cache = {"checked_at": 0.0, "base_url": "", "status": None}
_status_lock = Lock()
_last_call = None
_metrics_lock = Lock()


def _record_call(backend: str, model: str, latency_s: float, tokens: int | None, operation: str) -> dict:
    global _last_call
    tokens_per_second = round(tokens / latency_s, 2) if tokens is not None and latency_s > 0 else None
    record = {
        "backend": backend,
        "model": model,
        "latency_s": round(latency_s, 4),
        "tokens": tokens,
        "tokens_per_second": tokens_per_second,
        "operation": operation,
    }
    with _metrics_lock:
        _last_call = record
    return record


def _base_url() -> str:
    base_url = os.environ.get("AI_BASE_URL", "").strip().rstrip("/")
    if not base_url:
        return ""
    parsed = urlsplit(base_url)
    if parsed.scheme == "https":
        return base_url
    if parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}:
        return base_url
    return ""


def _local_intake(text: str, has_image: bool) -> str:
    from communityconnect import understand_request

    category, skill, availability = understand_request(text)
    italian_markers = {"cerco", "aiuto", "per", "con", "ripetizioni", "spesa", "accompagnamento"}
    words = set(re.findall(r"[a-zà-ÿ]+", text.casefold()))
    language = "it" if words & italian_markers else "en"
    summary = text.strip()[:240]
    if has_image:
        summary = f"Image was not analyzed in local mode. {summary}".strip()
    return json.dumps({
        "category": category,
        "urgency": "normal",
        "skills": [skill],
        "availability": availability,
        "language": language,
        "summary": summary,
    }, ensure_ascii=False)


def _local_result(text: str, has_image: bool, started: float, fallback_text: str | None = None) -> dict:
    result = {
        "text": fallback_text if fallback_text is not None else _local_intake(text, has_image),
        "latency_s": round(time.perf_counter() - started, 4),
        "tokens": 0,
        "model": "keyword-rule-fallback",
        "backend": "local",
    }
    result["metrics"] = _record_call("local", result["model"], result["latency_s"], None, "generate")
    return result


def ask_ai(
    text: str,
    *,
    image_url: str | None = None,
    image_base64: str | None = None,
    max_new_tokens: int = 256,
    timeout: float | None = None,
    fallback_text: str | None = None,
    known_names: tuple[str, ...] | list[str] = (),
) -> dict:
    """Call the AMD inference service, retry once, then return local rule-based intake."""
    started = time.perf_counter()
    safe_text, redactions = deidentify_text(text, known_names)
    base_url = _base_url()
    api_key = os.environ.get("AI_API_KEY", "")
    if not base_url or not api_key:
        return _local_result(text, bool(image_url or image_base64), started, fallback_text)

    payload = json.dumps({
        "text": safe_text,
        "image_url": image_url,
        "image_base64": image_base64,
        "max_new_tokens": max(1, min(int(max_new_tokens), 1_024)),
    }).encode("utf-8")
    api_request = Request(
        f"{base_url}/generate",
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json", "x-api-key": api_key},
        method="POST",
    )
    timeout = timeout if timeout is not None else float(os.environ.get("AI_TIMEOUT_SECONDS", "20"))
    for _attempt in range(2):
        try:
            with urlopen(api_request, timeout=timeout) as response:
                result = json.loads(response.read())
            if not isinstance(result, dict) or not isinstance(result.get("text"), str):
                raise ValueError("AI server response is missing text")
            result["text"] = restore_redactions(result["text"], redactions)
            result["backend"] = "amd"
            result.setdefault("model", "unknown")
            with _status_lock:
                _status_cache.update({
                    "checked_at": time.monotonic(),
                    "base_url": base_url,
                    "status": {"backend": "amd", "model": result["model"]},
                })
            result["metrics"] = _record_call(
                "amd", result["model"], float(result.get("latency_s", time.perf_counter() - started)),
                int(result["tokens"]) if isinstance(result.get("tokens"), int) else None, "generate",
            )
            return result
        except (HTTPError, URLError, OSError, TimeoutError, ValueError, json.JSONDecodeError):
            continue
    with _status_lock:
        _status_cache.update({
            "checked_at": time.monotonic(),
            "base_url": base_url,
            "status": {"backend": "local", "model": "keyword-rule-fallback"},
        })
    return _local_result(text, bool(image_url or image_base64), started, fallback_text)


def embed_texts(texts: list[str], *, timeout: float | None = None) -> list[list[float]] | None:
    """Return server embeddings, or None when the AMD service is unavailable."""
    base_url = _base_url()
    api_key = os.environ.get("AI_API_KEY", "")
    if not base_url or not api_key:
        return None
    timeout = timeout if timeout is not None else float(os.environ.get("AI_TIMEOUT_SECONDS", "20"))
    redacted = [deidentify_text(text)[0] for text in texts]
    vectors: list[list[float]] = []
    for offset in range(0, len(redacted), 32):
        batch = redacted[offset:offset + 32]
        body = json.dumps({"texts": batch}).encode("utf-8")
        api_request = Request(
            f"{base_url}/embed",
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json", "x-api-key": api_key},
            method="POST",
        )
        result = None
        for _attempt in range(2):
            try:
                with urlopen(api_request, timeout=timeout) as response:
                    result = json.loads(response.read())
                if not isinstance(result, dict) or not isinstance(result.get("vectors"), list):
                    raise ValueError("Embedding response is invalid")
                expected_model = os.environ.get(
                    "AI_EMBEDDING_MODEL_ID",
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                )
                if result.get("model") != expected_model:
                    raise ValueError("Embedding model does not match the configured cache model")
                if len(result["vectors"]) != len(batch):
                    raise ValueError("Embedding response length does not match input")
                if any(not isinstance(vector, list) or not vector or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in vector) for vector in result["vectors"]):
                    raise ValueError("Embedding response contains an invalid vector")
                break
            except (HTTPError, URLError, OSError, TimeoutError, ValueError, json.JSONDecodeError):
                result = None
        if result is None:
            return None
        vectors.extend([[float(value) for value in vector] for vector in result["vectors"]])
        _record_call(
            "amd", result["model"], float(result.get("latency_s", 0.0)), None, "embed",
        )
    return vectors


def _status_with_metrics(status: dict) -> dict:
    with _metrics_lock:
        last_call = dict(_last_call) if _last_call else None
    return {**status, "last_call": last_call}


def get_backend_status() -> dict:
    base_url = _base_url()
    if not base_url or not os.environ.get("AI_API_KEY"):
        return _status_with_metrics({"backend": "local", "model": "keyword-rule-fallback"})

    now = time.monotonic()
    with _status_lock:
        cached = _status_cache["status"]
        if cached and _status_cache["base_url"] == base_url and now - _status_cache["checked_at"] < _STATUS_CACHE_SECONDS:
            status = cached
        else:
            status = {"backend": "local", "model": "keyword-rule-fallback"}
            try:
                health_request = Request(f"{base_url}/health", headers={"x-api-key": os.environ["AI_API_KEY"]})
                with urlopen(health_request, timeout=1.0) as response:
                    health = json.loads(response.read())
                if isinstance(health, dict) and health.get("ready") is True and health.get("authorized") is True:
                    status = {"backend": "amd", "model": health.get("model", "unknown")}
            except (HTTPError, URLError, OSError, TimeoutError, ValueError, json.JSONDecodeError):
                pass
            _status_cache.update({"checked_at": now, "base_url": base_url, "status": status})
    return _status_with_metrics(status)
