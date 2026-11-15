"""Load CORD-v2 (receipts) and FUNSD (forms) as (image, target JSON) pairs.

CORD-v2: 1,000 receipt photos (800/100/100), ground truth `gt_parse` with
  menu items (nm, cnt, price, ...), sub_total and total groups. CC BY 4.0.
FUNSD:   199 scanned forms (149 train / 50 test), entities labelled
  question / answer / header / other, with links. We turn the links into
  key-value pairs: {"fields": [{"key": "Date", "value": "March 3, 1992"}, ...]}.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DROP_CORD_KEYS = {"void_menu"}  # crossed-out items are not part of the claim


@dataclass
class Example:
    id: str
    doc_type: str          # "receipt" | "form"
    image: Any             # PIL.Image or path
    target: dict


def clean_cord(gt: dict) -> dict:
    """Drop void items, make `menu` always a list (CORD stores one item as a dict)."""
    out = {k: v for k, v in gt.items() if k not in DROP_CORD_KEYS}
    if isinstance(out.get("menu"), dict):
        out["menu"] = [out["menu"]]
    for item in out.get("menu", []):
        if isinstance(item.get("sub"), dict):
            item["sub"] = [item["sub"]]
    return out


def cord_from_hf(split: str = "test"):
    """Requires `datasets` (Colab). Yields Examples from naver-clova-ix/cord-v2."""
    from datasets import load_dataset

    ds = load_dataset("naver-clova-ix/cord-v2", split=split)
    for i, row in enumerate(ds):
        gt = json.loads(row["ground_truth"])
        yield Example(f"cord_{split}_{i:04d}", "receipt", row["image"], clean_cord(gt["gt_parse"]))


def cord_from_jsonl(path: str | Path, image_dir: str | Path | None = None):
    """Local mirror format: one JSON per line with ground_truth (str or dict) and an image file name."""
    for i, line in enumerate(Path(path).read_text().splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        gt = row["ground_truth"]
        gt = json.loads(gt) if isinstance(gt, str) else gt
        img = row.get("image_file") or row.get("file_name")
        yield Example(str(row.get("id", f"cord_{i:04d}")), "receipt",
                      str(Path(image_dir) / img) if image_dir and img else img, clean_cord(gt["gt_parse"]))


def funsd_pairs(form: list[dict]) -> dict:
    """FUNSD entity list -> {"fields": [{"key", "value"}], "headers": [...]}, following question->answer links."""
    by_id = {e["id"]: e for e in form}
    pairs, seen = [], set()
    for e in form:
        for a, b in e.get("linking", []):
            if a not in by_id or b not in by_id:
                continue
            la, lb = by_id[a]["label"], by_id[b]["label"]
            if la == "answer" and lb == "question":
                a, b = b, a
            elif not (la == "question" and lb == "answer"):
                continue
            if (a, b) in seen:
                continue
            seen.add((a, b))
            k, v = by_id[a]["text"].strip(), by_id[b]["text"].strip()
            if k and v:
                pairs.append((a, b, k, v))
    pairs.sort(key=lambda p: (by_id[p[0]]["box"][1], by_id[p[0]]["box"][0], p[1]))  # reading order
    headers = [e["text"].strip() for e in sorted(form, key=lambda e: (e["box"][1], e["box"][0]))
               if e["label"] == "header" and e["text"].strip()]
    return {"headers": headers, "fields": [{"key": k, "value": v} for _, _, k, v in pairs]}


def funsd_from_dir(root: str | Path, split: str = "testing_data"):
    """Original FUNSD zip layout: dataset/{training_data,testing_data}/{annotations,images}."""
    base = Path(root)
    if (base / "dataset").exists():
        base = base / "dataset"
    for ann in sorted((base / split / "annotations").glob("*.json")):
        form = json.loads(ann.read_text())["form"]
        img = base / split / "images" / f"{ann.stem}.png"
        yield Example(f"funsd_{ann.stem}", "form", str(img), funsd_pairs(form))
