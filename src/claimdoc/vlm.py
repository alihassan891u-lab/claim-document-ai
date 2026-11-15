"""Qwen2-VL / Qwen2.5-VL document extractor (image in, JSON out). Needs a GPU + transformers.

    ex = QwenVLExtractor("Qwen/Qwen2-VL-2B-Instruct", adapter="outputs/qwen2vl-claims-lora")
    data, status = ex.extract(image, "receipt")
"""
from __future__ import annotations

import time
from typing import Any

from .parsing import parse_json
from .prompts import PROMPTS

# Receipts and forms are text-dense; cap pixels so a T4 (16 GB) fits, but keep enough for small print.
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1024 * 28 * 28


def load_model(model_id: str, adapter: str | None = None, four_bit: bool = False):
    import torch
    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:  # older transformers
        from transformers import Qwen2VLForConditionalGeneration as AutoVLM

    kw: dict[str, Any] = {"device_map": "auto"}
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    kw["torch_dtype"] = torch.bfloat16 if bf16 else torch.float16
    if four_bit:
        from transformers import BitsAndBytesConfig

        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=kw["torch_dtype"], bnb_4bit_use_double_quant=True)
    model = AutoVLM.from_pretrained(model_id, **kw)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    processor = AutoProcessor.from_pretrained(model_id, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    model.eval()
    return model, processor


def chat(prompt: str, answer: str | None = None) -> list[dict]:
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return msgs


class QwenVLExtractor:
    name = "qwen-vl"

    def __init__(self, model_id: str = "Qwen/Qwen2-VL-2B-Instruct", adapter: str | None = None, four_bit: bool = False,
                 max_new_tokens: int = 1024):
        self.model, self.processor = load_model(model_id, adapter, four_bit)
        self.max_new_tokens = max_new_tokens
        self.name = f"{model_id.split('/')[-1]}{'+lora' if adapter else ' zero-shot'}"

    def extract(self, image, doc_type: str) -> tuple[Any, str, float]:
        import torch
        from PIL import Image

        img = Image.open(image).convert("RGB") if isinstance(image, str) else image.convert("RGB")
        text = self.processor.apply_chat_template(chat(PROMPTS[doc_type]), tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[img], return_tensors="pt").to(self.model.device)
        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        reply = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        data, status = parse_json(reply)
        return data, status, time.time() - t0
