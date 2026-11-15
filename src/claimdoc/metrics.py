"""Field-level precision / recall / F1 for document extraction.

Both prediction and ground truth are flattened into (field path, value) pairs,
e.g. ("menu.nm", "es jeruk"), ("total.total_price", "48.000"),
("fields", "date => 3/12/92"). List positions are ignored, so a model that
reads items in a different order isn't punished, but each value counts once
(multiset matching). This is the standard field-level F1 used for CORD.

`normalize="strict"` compares exact strings; "loose" lowercases, collapses
spaces and drops trailing punctuation, which is what an operations team cares about.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any


def norm(v: Any, mode: str = "loose") -> str:
    s = str(v)
    if mode == "strict":
        return s.strip()
    s = re.sub(r"\s+", " ", s.casefold()).strip()
    return s.strip(" .:;,")


def flatten(obj: Any, prefix: str = "", mode: str = "loose") -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        if set(obj) == {"key", "value"}:  # FUNSD key-value pair: the pair is one field
            return [(prefix or "fields", f"{norm(obj['key'], mode)} => {norm(obj['value'], mode)}")]
        for k, v in obj.items():
            out += flatten(v, f"{prefix}.{k}" if prefix else str(k), mode)
    elif isinstance(obj, list):
        for v in obj:
            out += flatten(v, prefix, mode)
    elif obj is not None and str(obj).strip() != "":
        out.append((prefix, norm(obj, mode)))
    return out


def field_prf(pred: Any, gold: Any, mode: str = "loose", ignore: tuple[str, ...] = ()) -> tuple[int, int, int]:
    """Returns (true positives, predicted fields, gold fields)."""
    p = Counter(f for f in flatten(pred, mode=mode) if not f[0].startswith(ignore) or not ignore)
    g = Counter(f for f in flatten(gold, mode=mode) if not f[0].startswith(ignore) or not ignore)
    return sum((p & g).values()), sum(p.values()), sum(g.values())


def corpus_scores(pairs: list[tuple[Any, Any]], mode: str = "loose") -> dict:
    """Micro-averaged over all documents, plus share of documents extracted perfectly and per-field F1."""
    tp = n_pred = n_gold = perfect = 0
    per_field: dict[str, list[int]] = {}
    for pred, gold in pairs:
        t, p, g = field_prf(pred, gold, mode)
        tp, n_pred, n_gold = tp + t, n_pred + p, n_gold + g
        perfect += int(t == p == g)
        pc, gc = Counter(flatten(pred, mode=mode)), Counter(flatten(gold, mode=mode))
        for path in {k for k, _ in gc} | {k for k, _ in pc}:
            sub_p = Counter({k: v for k, v in pc.items() if k[0] == path})
            sub_g = Counter({k: v for k, v in gc.items() if k[0] == path})
            agg = per_field.setdefault(path, [0, 0, 0])
            agg[0] += sum((sub_p & sub_g).values())
            agg[1] += sum(sub_p.values())
            agg[2] += sum(sub_g.values())
    prec = tp / n_pred if n_pred else 0.0
    rec = tp / n_gold if n_gold else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    def _f1(a):
        p_, r_ = (a[0] / a[1] if a[1] else 0.0), (a[0] / a[2] if a[2] else 0.0)
        return round(2 * p_ * r_ / (p_ + r_), 4) if p_ + r_ else 0.0

    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "documents": len(pairs),
            "perfect_documents": perfect, "fields_gold": n_gold,
            "per_field_f1": {k: {"f1": _f1(v), "gold": v[2]} for k, v in sorted(per_field.items(), key=lambda kv: -kv[1][2])}}
