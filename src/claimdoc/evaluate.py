"""Run extractors over a test split and compare them on the same documents.

    python -m claimdoc.evaluate --systems zero-shot,lora,ocr-llm --limit 100
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .claim import receipt_to_claim
from .metrics import corpus_scores


def run_system(extractor, examples: list, out_jsonl: str | Path | None = None) -> dict:
    rows, pairs = [], []
    statuses = {"ok": 0, "repaired": 0, "failed": 0}
    t0 = time.time()
    auto_ready = 0
    for ex in examples:
        try:
            pred, status, secs = extractor.extract(ex.image, ex.doc_type)
        except Exception as e:  # one bad document must not kill a 100-document run
            pred, status, secs = None, "failed", 0.0
            print(f"  ! {ex.id}: {type(e).__name__}: {e}")
        statuses[status] += 1
        pred = pred if isinstance(pred, (dict, list)) else {}
        pairs.append((pred, ex.target))
        row = {"id": ex.id, "doc_type": ex.doc_type, "status": status, "seconds": round(secs, 2), "pred": pred, "gold": ex.target}
        if ex.doc_type == "receipt" and isinstance(pred, dict):
            claim = receipt_to_claim(pred)
            row["claim_status"] = claim.status
            auto_ready += claim.status == "auto_ready"
        rows.append(row)
    if out_jsonl:
        Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)
        Path(out_jsonl).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    receipts = sum(ex.doc_type == "receipt" for ex in examples)
    return {"system": getattr(extractor, "name", "system"), "loose": corpus_scores(pairs, "loose"),
            "strict": {k: v for k, v in corpus_scores(pairs, "strict").items() if k != "per_field_f1"},
            "json": statuses, "avg_seconds": round((time.time() - t0) / max(len(examples), 1), 2),
            "receipts_auto_ready": f"{auto_ready}/{receipts}" if receipts else None}


def table(results: list[dict], title: str) -> str:
    lines = [f"### {title}", "", "| System | Field F1 | Precision | Recall | Strict F1 | Perfect docs | JSON failed | s/doc |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lo = r["loose"]
        lines.append(f"| {r['system']} | {lo['f1']:.3f} | {lo['precision']:.3f} | {lo['recall']:.3f} | {r['strict']['f1']:.3f} | "
                     f"{lo['perfect_documents']}/{lo['documents']} | {r['json']['failed']} | {r['avg_seconds']} |")
    return "\n".join(lines) + "\n"


def main():
    from .datasets import cord_from_hf, funsd_from_dir

    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="zero-shot,lora,ocr-llm")
    ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--adapter", default="outputs/qwen2vl-claims-lora")
    ap.add_argument("--baseline-llm", default="hf", help="hf | claude | openai")
    ap.add_argument("--funsd", default="data/funsd")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    sets = {"CORD receipts (test)": list(cord_from_hf("test"))[: a.limit]}
    if Path(a.funsd).exists():
        sets["FUNSD forms (test)"] = list(funsd_from_dir(a.funsd, "testing_data"))[: a.limit]

    def build(name):
        if name == "zero-shot":
            from .vlm import QwenVLExtractor
            return QwenVLExtractor(a.model)
        if name == "lora":
            from .vlm import QwenVLExtractor
            return QwenVLExtractor(a.model, adapter=a.adapter)
        if name == "ocr-llm":
            from .baseline import OcrLlmExtractor, claude_llm, hf_llm, openai_llm
            llm = {"hf": hf_llm, "claude": claude_llm, "openai": openai_llm}[a.baseline_llm]()
            return OcrLlmExtractor(llm, name=f"Tesseract + {a.baseline_llm} LLM")
        raise ValueError(name)

    report, md = {}, []
    for sys_name in a.systems.split(","):
        ext = build(sys_name)
        for set_name, exs in sets.items():
            res = run_system(ext, exs, Path(a.out) / f"pred_{sys_name}_{set_name.split()[0].lower()}.jsonl")
            report.setdefault(set_name, []).append(res)
            print(set_name, res["system"], res["loose"]["f1"])
        del ext
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass
    for set_name, results in report.items():
        md.append(table(results, set_name))
    Path(a.out).mkdir(exist_ok=True)
    Path(a.out, "comparison.md").write_text("\n".join(md))
    Path(a.out, "comparison.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("\n".join(md))


if __name__ == "__main__":
    main()
