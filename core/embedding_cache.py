import hashlib
import json
import math
import os
import sqlite3
from contextlib import closing
from pathlib import Path
from time import time

from core.privacy import deidentify_text

_DEFAULT_DB = Path(__file__).resolve().parent.parent / "communityconnect.db"
_DEFAULT_MODEL_VERSION = "v2"


def _database_path() -> Path:
    default_path = "/tmp/communityconnect_embeddings.db" if os.environ.get("VERCEL") == "1" else str(_DEFAULT_DB)
    return Path(os.environ.get("AI_EMBEDDING_CACHE_DB", default_path))


def _cache_key(model_id: str, text: str, model_version: str = _DEFAULT_MODEL_VERSION) -> str:
    safe_text, _ = deidentify_text(text)
    return hashlib.sha256(f"{model_id}\0{model_version}\0{safe_text}".encode("utf-8")).hexdigest()


def _connect() -> sqlite3.Connection:
    path = _database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.execute(
        """create table if not exists ai_embedding_cache (
            cache_key text primary key,
            model_id text not null,
            model_version text not null default 'v2',
            backend text not null default 'unknown',
            vector_json text not null,
            updated_at real not null
        )"""
    )
    columns = {row[1] for row in connection.execute("pragma table_info(ai_embedding_cache)")}
    if "model_version" not in columns:
        connection.execute("alter table ai_embedding_cache add column model_version text not null default 'legacy'")
    if "backend" not in columns:
        connection.execute("alter table ai_embedding_cache add column backend text not null default 'unknown'")
    return connection


def _get_cached_embedding_records(
    texts: list[str], model_id: str, model_version: str,
) -> list[tuple[list[float], str] | None]:
    keys = [_cache_key(model_id, text, model_version) for text in texts]
    if not keys:
        return []
    placeholders = ",".join("?" for _ in keys)
    with closing(_connect()) as connection:
        with connection:
            rows = connection.execute(
                f"select cache_key, vector_json, backend from ai_embedding_cache where model_id = ? and model_version = ? and cache_key in ({placeholders})",
                [model_id, model_version, *keys],
            ).fetchall()
    by_key = {key: (json.loads(vector_json), backend) for key, vector_json, backend in rows}
    return [by_key.get(key) for key in keys]


def get_cached_embeddings(texts: list[str], model_id: str, model_version: str = _DEFAULT_MODEL_VERSION) -> list[list[float] | None]:
    records = _get_cached_embedding_records(texts, model_id, model_version)
    return [record[0] if record else None for record in records]


def store_embeddings(
    texts: list[str],
    vectors: list[list[float]],
    model_id: str,
    model_version: str = _DEFAULT_MODEL_VERSION,
    backend: str = "amd",
) -> None:
    if len(texts) != len(vectors):
        raise ValueError("Embedding text and vector counts differ")
    records = []
    for text, vector in zip(texts, vectors):
        values = [float(value) for value in vector]
        if not values or any(not math.isfinite(value) for value in values):
            raise ValueError("Embedding vectors must contain finite values")
        records.append((
            _cache_key(model_id, text, model_version), model_id, model_version, backend,
            json.dumps(values, allow_nan=False), time(),
        ))
    if records:
        with closing(_connect()) as connection:
            with connection:
                connection.executemany(
                          """insert into ai_embedding_cache (cache_key, model_id, model_version, backend, vector_json, updated_at)
                              values (?, ?, ?, ?, ?, ?)
                       on conflict(cache_key) do update set
                       model_id = excluded.model_id,
                              model_version = excluded.model_version,
                              backend = excluded.backend,
                       vector_json = excluded.vector_json,
                       updated_at = excluded.updated_at""",
                    records,
                )


def load_or_create_embeddings_with_backend(
    texts: list[str], model_id: str, model_version: str = _DEFAULT_MODEL_VERSION, *, allow_remote: bool = True,
) -> dict | None:
    records = _get_cached_embedding_records(texts, model_id, model_version)
    amd_records = [record if record and record[1] == "amd" else None for record in records]
    cached = [record[0] if record else None for record in amd_records]
    backends = [record[1] if record else None for record in amd_records]
    missing_indexes = [index for index, vector in enumerate(cached) if vector is None]
    if not missing_indexes:
        sources = sorted(set(backends))
        return {
            "vectors": cached, "backends": backends,
            "backend": sources[0] if len(sources) == 1 else "mixed",
            "model": model_id, "model_version": model_version,
        }

    missing_texts = [texts[index] for index in missing_indexes]
    vectors = None
    backend = "local-cpu"
    if allow_remote:
        from core.ai_client import embed_texts

        vectors = embed_texts(missing_texts)
        backend = "amd"
    if vectors is None:
        return None
    store_embeddings(missing_texts, vectors, model_id, model_version, backend)
    for index, vector in zip(missing_indexes, vectors):
        cached[index] = vector
        backends[index] = backend
    if any(vector is None for vector in cached):
        return None
    sources = sorted(set(backends))
    return {
        "vectors": cached, "backends": backends,
        "backend": sources[0] if len(sources) == 1 else "mixed",
        "model": model_id, "model_version": model_version,
    }


def load_or_create_embeddings(texts: list[str], model_id: str) -> list[list[float] | None] | None:
    result = load_or_create_embeddings_with_backend(texts, model_id)
    return result["vectors"] if result else None
