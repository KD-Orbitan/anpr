"""Run: python -m uvicorn api:app --host 127.0.0.1 --port 8000"""
import base64
import binascii
from functools import lru_cache
from threading import Lock

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from plateocr.inference import Pipeline

app = FastAPI(title="Vietnamese ANPR", version="0.1.0")
lock = Lock()


class ImageRequest(BaseModel):
    image_base64: str


@lru_cache(maxsize=1)
def pipeline():
    return Pipeline()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bool(pipeline.cache_info().currsize)}


@app.post("/anpr")
def anpr(request: ImageRequest):
    if len(request.image_base64) > 20 * 1024 * 1024:
        raise HTTPException(413, "Image payload is too large")
    try:
        raw = base64.b64decode(request.image_base64, validate=True)
        if not raw:
            raise ValueError("Empty image")
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Cannot decode image")
    except (ValueError, binascii.Error, cv2.error) as error:
        raise HTTPException(400, "Invalid base64 image") from error
    with lock:
        return {"plates": pipeline().predict(image)}
