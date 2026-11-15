"""Claims desk demo:  streamlit run app/streamlit_app.py  (GPU machine or Colab with a tunnel)"""
from __future__ import annotations

import os

import streamlit as st
from PIL import Image

from claimdoc.claim import receipt_to_claim

st.set_page_config(page_title="Claim Document AI", page_icon="🧾", layout="wide")
st.sidebar.title("🧾 Claim Document AI")
system = st.sidebar.selectbox("Extractor", ["lora", "zero-shot", "ocr-claude"])
adapter = st.sidebar.text_input("LoRA adapter", os.getenv("CLAIMDOC_ADAPTER", "outputs/qwen2vl-claims-lora"))
doc_type = st.sidebar.radio("Document", ["receipt", "form"])


@st.cache_resource(show_spinner="Loading model…")
def get(system: str, adapter: str):
    if system == "ocr-claude":
        from claimdoc.baseline import OcrLlmExtractor, claude_llm
        return OcrLlmExtractor(claude_llm(), name="Tesseract + Claude")
    from claimdoc.vlm import QwenVLExtractor
    return QwenVLExtractor("Qwen/Qwen2-VL-2B-Instruct", adapter=adapter if system == "lora" else None)


up = st.file_uploader("Receipt or claim form", type=["png", "jpg", "jpeg"])
if up:
    img = Image.open(up).convert("RGB")
    c1, c2 = st.columns([1, 1])
    c1.image(img, use_container_width=True)
    with st.spinner("Extracting…"):
        data, status, secs = get(system, adapter).extract(img, doc_type)
    c2.caption(f"{secs:.1f}s · JSON {status}")
    c2.json(data or {})
    if doc_type == "receipt" and isinstance(data, dict):
        claim = receipt_to_claim(data)
        (c2.success if claim.status == "auto_ready" else c2.warning)(f"Claim status: {claim.status}")
        for chk in claim.checks:
            c2.write(("✅ " if chk["ok"] else "❌ ") + f"{chk['check']}: {chk['detail']}")
