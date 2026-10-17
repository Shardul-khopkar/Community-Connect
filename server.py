import base64
import hmac
import ipaddress
import os
import socket
import threading
import time
from contextlib import asynccontextmanager
from io import BytesIO
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, model_validator

MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-4-31B-it")
EMBEDDING_MODEL_ID = os.environ.get(
    "EMBEDDING_MODEL_ID",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
API_KEY = os.environ.get("AI_API_KEY", "")
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000

model = None
processor = None
embedding_model = None
generation_lock = threading.Lock()
embedding_lock = threading.Lock()


class GenerateRequest(BaseModel):
    text: str = Field(default="", max_length=12_000)
    image_url: str | None = Field(default=None, max_length=2_048)
    image_base64: str | None = Field(default=None, max_length=14_000_000)
    max_new_tokens: int = Field(default=256, ge=1, le=1_024)

    @model_validator(mode="after")
    def only_one_image_source(self):
        if self.image_url and self.image_base64:
            raise ValueError("Provide image_url or image_base64, not both")
        return self


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=32)


class NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


def _validate_public_https_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise HTTPException(status_code=400, detail="image_url must be a public HTTPS URL")
    try:
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM)
        if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
            raise HTTPException(status_code=400, detail="image_url must resolve to a public address")
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail="image_url could not be validated") from error


def _read_url_image(url: str) -> bytes:
    _validate_public_https_url(url)
    opener = build_opener(NoRedirectHandler)
    image_request = Request(url, headers={"Accept": "image/*"}, method="GET")
    try:
        with opener.open(image_request, timeout=8) as response:
            content_type = response.headers.get_content_type()
            if not content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="image_url did not return an image")
            if int(response.headers.get("Content-Length", "0") or 0) > MAX_IMAGE_BYTES:
                raise HTTPException(status_code=413, detail="image exceeds the size limit")
            content = response.read(MAX_IMAGE_BYTES + 1)
    except HTTPError as error:
        raise HTTPException(status_code=400, detail="image_url could not be fetched") from error
    except (OSError, URLError, ValueError) as error:
        raise HTTPException(status_code=400, detail="image_url could not be fetched") from error
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="image exceeds the size limit")
    return content


def _decode_image(payload: GenerateRequest):
    from PIL import Image, UnidentifiedImageError

    if payload.image_base64:
        try:
            image_bytes = base64.b64decode(payload.image_base64, validate=True)
        except (ValueError, base64.binascii.Error) as error:
            raise HTTPException(status_code=400, detail="image_base64 is not valid base64") from error
    elif payload.image_url:
        image_bytes = _read_url_image(payload.image_url)
    else:
        return None

    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="image exceeds the size limit")
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
        with Image.open(BytesIO(image_bytes)) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="image dimensions exceed the limit")
            image.load()
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail="image data is not a supported image") from error


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global model, processor
    if len(API_KEY) < 32:
        raise RuntimeError("Set AI_API_KEY to a random value of at least 32 characters.")
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access the AMD GPU through its ROCm build.")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID,
        dtype="auto",
        device_map="cuda",
    )
    model.eval()
    yield
    model = None
    processor = None


app = FastAPI(title="CommunityConnect AMD AI", lifespan=lifespan)


@app.get("/health")
def health(x_api_key: str = Header(default="")):
    ready = model is not None and processor is not None
    authorized = bool(API_KEY) and hmac.compare_digest(x_api_key, API_KEY)
    return {"status": "ready" if ready else "loading", "ready": ready, "authorized": authorized, "model": MODEL_ID}


@app.post("/embed")
def embed(payload: EmbedRequest, x_api_key: str = Header(default="")):
    global embedding_model
    if not API_KEY or not hmac.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key")
    if any(not text.strip() or len(text) > 2_000 for text in payload.texts):
        raise HTTPException(status_code=422, detail="Each embedding text must contain 1 to 2,000 characters")

    with embedding_lock:
        if embedding_model is None:
            from sentence_transformers import SentenceTransformer

            embedding_model = SentenceTransformer(EMBEDDING_MODEL_ID, device="cpu")
        active_embedding_model = embedding_model

    started = time.perf_counter()
    vectors = active_embedding_model.encode(
        payload.texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return {
        "vectors": vectors.tolist(),
        "model": EMBEDDING_MODEL_ID,
        "latency_s": round(time.perf_counter() - started, 4),
    }


@app.post("/generate")
def generate(payload: GenerateRequest, x_api_key: str = Header(default="")):
    if not API_KEY or not hmac.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key")
    if not payload.text.strip() and not payload.image_url and not payload.image_base64:
        raise HTTPException(status_code=422, detail="Provide text or an image")
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model is not ready")

    started = time.perf_counter()
    image = _decode_image(payload)
    content = []
    if image is not None:
        content.append({"type": "image", "url": image})
    if payload.text.strip():
        content.append({"type": "text", "text": payload.text.strip()})
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    input_tokens = inputs["input_ids"].shape[-1]

    import torch

    with generation_lock, torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=payload.max_new_tokens)
        torch.cuda.synchronize()
    latency_s = time.perf_counter() - started
    generated_tokens = output[0][input_tokens:]
    generated_text = processor.decode(generated_tokens, skip_special_tokens=True).strip()
    return {
        "text": generated_text,
        "latency_s": round(latency_s, 4),
        "tokens": int(generated_tokens.shape[-1]),
        "model": MODEL_ID,
    }
