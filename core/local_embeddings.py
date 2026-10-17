from threading import Lock


_model = None
_model_id = None
_model_lock = Lock()


class LocalEmbeddingUnavailable(RuntimeError):
    pass


def embed_texts_local(texts: list[str], model_id: str) -> list[list[float]]:
    global _model, _model_id
    try:
        with _model_lock:
            if _model is None or _model_id != model_id:
                from sentence_transformers import SentenceTransformer

                _model = SentenceTransformer(model_id, device="cpu")
                _model_id = model_id
            active_model = _model
        encoded = active_model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in vector] for vector in encoded.tolist()]
    except Exception as error:
        raise LocalEmbeddingUnavailable("The local CPU embedding model is unavailable") from error