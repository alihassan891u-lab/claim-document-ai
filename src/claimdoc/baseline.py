"""The "old way" baseline: OCR (Tesseract) -> plain text -> LLM -> JSON.

This is how most document pipelines are built today. Its weakness is that the
LLM never sees the layout: columns, alignment and which number belongs to
which label are lost or scrambled by OCR. The fine-tuned VLM reads the image.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable

from .parsing import parse_json
from .prompts import OCR_LLM, PROMPTS


def tesseract_lines(image, psm: int = 6) -> str:
    """OCR with line grouping in reading order (top-to-bottom, left-to-right)."""
    import pytesseract
    from PIL import Image

    img = Image.open(image) if isinstance(image, str) else image
    d = pytesseract.image_to_data(img.convert("L"), config=f"--psm {psm}", output_type=pytesseract.Output.DICT)
    lines: dict[tuple, list[tuple[int, str]]] = {}
    for i, word in enumerate(d["text"]):
        if word.strip() and float(d["conf"][i]) > 0:
            key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
            lines.setdefault(key, []).append((d["left"][i], word))
    ordered = sorted(lines.items(), key=lambda kv: kv[0])
    return "\n".join(" ".join(w for _, w in sorted(ws)) for _, ws in ordered)


def claude_llm(model: str | None = None) -> Callable[[str], str]:
    import anthropic

    client, m = anthropic.Anthropic(), model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
    return lambda prompt: "".join(b.text for b in client.messages.create(
        model=m, max_tokens=2048, temperature=0, messages=[{"role": "user", "content": prompt}]).content if b.type == "text")


def openai_llm(model: str | None = None) -> Callable[[str], str]:
    from openai import OpenAI

    client, m = OpenAI(), model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return lambda prompt: client.chat.completions.create(model=m, temperature=0, messages=[{"role": "user", "content": prompt}]
                                                         ).choices[0].message.content or ""


def hf_llm(model_id: str = "Qwen/Qwen2.5-1.5B-Instruct") -> Callable[[str], str]:
    """Free local LLM for the baseline (no API key)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

    def run(prompt: str) -> str:
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=1024, do_sample=False)
        return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)

    return run


class OcrLlmExtractor:
    def __init__(self, llm: Callable[[str], str], ocr: Callable[[Any], str] = tesseract_lines, name: str = "tesseract+llm"):
        self.llm, self.ocr, self.name = llm, ocr, name

    def extract(self, image, doc_type: str) -> tuple[Any, str, float]:
        t0 = time.time()
        text = self.ocr(image)
        prompt = OCR_LLM.format(doc={"receipt": "receipt", "form": "form"}[doc_type], ocr=text, instruction=PROMPTS[doc_type])
        data, status = parse_json(self.llm(prompt))
        return data, status, time.time() - t0


def describe(extractor) -> str:
    return json.dumps({"name": getattr(extractor, "name", type(extractor).__name__)})
