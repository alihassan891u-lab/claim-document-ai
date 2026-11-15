"""QLoRA fine-tuning of Qwen2-VL on (document image -> JSON) pairs. Fits a free Colab T4.

Loss is computed on the assistant's JSON only: prompt tokens, image tokens and padding are masked to -100,
so the model learns to *answer*, not to repeat the instructions.

    python -m claimdoc.finetune --out outputs/qwen2vl-claims-lora --epochs 2
"""
from __future__ import annotations

import argparse
import json
import random

from .prompts import PROMPTS
from .vlm import chat

IGNORE = -100


def mask_labels(input_ids: list[int], marker: list[int], pad_id: int | None, image_ids: set[int] = frozenset()) -> list[int]:
    """Copy input_ids to labels, masking everything up to and including the LAST assistant marker,
    plus padding and image placeholder tokens."""
    labels = list(input_ids)
    start = -1
    n = len(marker)
    for i in range(len(input_ids) - n, -1, -1):
        if input_ids[i:i + n] == marker:
            start = i + n
            break
    if start == -1:  # no answer in this sequence (should not happen): learn nothing from it
        return [IGNORE] * len(labels)
    for i in range(len(labels)):
        if i < start or labels[i] == pad_id or labels[i] in image_ids:
            labels[i] = IGNORE
    return labels


def target_text(example) -> str:
    return json.dumps(example.target, ensure_ascii=False, separators=(",", ":"))


class Collator:
    def __init__(self, processor):
        self.p = processor
        tok = processor.tokenizer
        self.marker = tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
        self.pad_id = tok.pad_token_id
        self.image_ids = {tok.convert_tokens_to_ids(t) for t in ("<|image_pad|>", "<|vision_start|>", "<|vision_end|>")}

    def __call__(self, batch):
        import torch
        from PIL import Image

        texts, images = [], []
        for ex in batch:
            img = ex.image if not isinstance(ex.image, str) else Image.open(ex.image)
            images.append(img.convert("RGB"))
            texts.append(self.p.apply_chat_template(chat(PROMPTS[ex.doc_type], target_text(ex)), tokenize=False))
        enc = self.p(text=texts, images=images, return_tensors="pt", padding=True)
        enc["labels"] = torch.tensor([mask_labels(ids, self.marker, self.pad_id, self.image_ids)
                                      for ids in enc["input_ids"].tolist()])
        return enc


def train(examples: list, out_dir: str, model_id: str = "Qwen/Qwen2-VL-2B-Instruct", epochs: float = 2, lr: float = 2e-4,
          r: int = 16, batch: int = 1, grad_accum: int = 8, seed: int = 0):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Trainer, TrainingArguments

    from .vlm import load_model

    random.Random(seed).shuffle(examples)
    model, processor = load_model(model_id, four_bit=True)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    bf16 = torch.cuda.is_bf16_supported()
    args = TrainingArguments(output_dir=out_dir, num_train_epochs=epochs, per_device_train_batch_size=batch,
                             gradient_accumulation_steps=grad_accum, learning_rate=lr, lr_scheduler_type="cosine",
                             warmup_ratio=0.05, logging_steps=10, save_strategy="epoch", bf16=bf16, fp16=not bf16,
                             gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
                             remove_unused_columns=False, report_to="none", seed=seed, optim="paged_adamw_8bit")
    trainer = Trainer(model=model, args=args, train_dataset=examples, data_collator=Collator(processor))
    trainer.train()
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    return out_dir


def main():
    from .datasets import cord_from_hf, funsd_from_dir

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/qwen2vl-claims-lora")
    ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--funsd", default="data/funsd", help="FUNSD folder (optional)")
    ap.add_argument("--epochs", type=float, default=2)
    a = ap.parse_args()
    ex = list(cord_from_hf("train"))
    try:
        ex += list(funsd_from_dir(a.funsd, "training_data"))
    except FileNotFoundError:
        print("FUNSD not found, training on CORD only")
    train(ex, a.out, a.model, a.epochs)


if __name__ == "__main__":
    main()
