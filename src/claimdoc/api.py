"""REST API: document image in, validated claim JSON out.

    CLAIMDOC_SYSTEM=lora CLAIMDOC_ADAPTER=outputs/qwen2vl-claims-lora uvicorn claimdoc.api:app --port 8000
    curl -F "file=@receipt.jpg" -F "doc_type=receipt" localhost:8000/extract
"""
from __future__ import annotations

import io
import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .claim import receipt_to_claim

app = FastAPI(title="Claim Document AI", version="0.1.0")
_state: dict = {}


def extractor():
    if "ex" not in _state:
        system = os.getenv("CLAIMDOC_SYSTEM", "lora")
        if system in ("lora", "zero-shot"):
            from .vlm import QwenVLExtractor

            _state["ex"] = QwenVLExtractor(os.getenv("CLAIMDOC_MODEL", "Qwen/Qwen2-VL-2B-Instruct"),
                                           adapter=os.getenv("CLAIMDOC_ADAPTER") if system == "lora" else None)
        else:
            from .baseline import OcrLlmExtractor, claude_llm, openai_llm

            _state["ex"] = OcrLlmExtractor({"ocr-claude": claude_llm, "ocr-openai": openai_llm}[system](), name=system)
    return _state["ex"]


@app.get("/health")
def health():
    return {"status": "ok", "loaded": "ex" in _state}


@app.post("/extract")
async def extract(file: UploadFile = File(...), doc_type: str = Form("receipt")):
    if doc_type not in ("receipt", "form"):
        raise HTTPException(400, "doc_type must be 'receipt' or 'form'")
    from PIL import Image, UnidentifiedImageError

    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except UnidentifiedImageError as e:
        raise HTTPException(422, "Not a readable image") from e
    data, status, secs = extractor().extract(img, doc_type)
    out = {"doc_type": doc_type, "json_status": status, "seconds": round(secs, 2), "data": data}
    if doc_type == "receipt" and isinstance(data, dict):
        out["claim"] = receipt_to_claim(data).to_dict()
    return out
