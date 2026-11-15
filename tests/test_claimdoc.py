import glob
import json
import re

import pytest

from claimdoc.baseline import OcrLlmExtractor, tesseract_lines
from claimdoc.claim import parse_amount, receipt_to_claim
from claimdoc.datasets import Example, clean_cord, cord_from_jsonl, funsd_from_dir, funsd_pairs
from claimdoc.evaluate import run_system, table
from claimdoc.finetune import IGNORE, mask_labels, target_text
from claimdoc.metrics import corpus_scores, field_prf, flatten
from claimdoc.parsing import parse_json

GOLD = {"menu": [{"nm": "BASO TAHU", "cnt": "1", "price": "43,181"}, {"nm": "ES JERUK", "cnt": "1", "price": "13,000"}],
        "sub_total": {"subtotal_price": "56,181", "tax_price": "5,618"}, "total": {"total_price": "61,799"}}


# ------------------------------------------------------------------ datasets
def test_clean_cord():
    g = clean_cord({"menu": {"nm": "A", "sub": {"nm": "+Hot"}}, "void_menu": {"nm": "X"}, "total": {"total_price": "1"}})
    assert g == {"menu": [{"nm": "A", "sub": [{"nm": "+Hot"}]}], "total": {"total_price": "1"}}


def test_cord_jsonl(tmp_path):
    p = tmp_path / "cord.jsonl"
    p.write_text(json.dumps({"image_file": "r0.jpg", "ground_truth": json.dumps({"gt_parse": {"menu": {"nm": "TEA"}}})}) + "\n")
    ex = list(cord_from_jsonl(p, tmp_path))[0]
    assert ex.doc_type == "receipt" and ex.target == {"menu": [{"nm": "TEA"}]} and ex.image.endswith("r0.jpg")


FORM = [
    {"id": 0, "text": "Registration No.", "box": [94, 169, 191, 186], "label": "question", "linking": [[0, 1]]},
    {"id": 1, "text": "533", "box": [209, 169, 236, 182], "label": "answer", "linking": [[0, 1]]},   # same link listed twice
    {"id": 2, "text": "28", "box": [184, 119, 208, 134], "label": "answer", "linking": [[3, 2]]},
    {"id": 3, "text": "DATE", "box": [100, 119, 150, 134], "label": "question", "linking": [[3, 2]]},
    {"id": 4, "text": "SCOPE:", "box": [99, 50, 149, 70], "label": "header", "linking": [[4, 3]]},     # header->question: not a field
    {"id": 5, "text": "EXCELLENT", "box": [300, 300, 380, 320], "label": "question", "linking": []},  # unanswered
]


def test_funsd_pairs():
    t = funsd_pairs(FORM)
    assert t["fields"] == [{"key": "DATE", "value": "28"}, {"key": "Registration No.", "value": "533"}]  # reading order
    assert t["headers"] == ["SCOPE:"]


def test_funsd_dir(tmp_path):
    d = tmp_path / "dataset" / "testing_data"
    (d / "annotations").mkdir(parents=True)
    (d / "annotations" / "f1.json").write_text(json.dumps({"form": FORM}))
    ex = list(funsd_from_dir(tmp_path, "testing_data"))[0]
    assert ex.id == "funsd_f1" and ex.image.endswith("f1.png") and len(ex.target["fields"]) == 2


# ------------------------------------------------------------------- metrics
def test_field_f1():
    assert field_prf(GOLD, GOLD) == (9, 9, 9)   # 2 items x (nm, cnt, price) + subtotal + tax + total
    shuffled = {**GOLD, "menu": list(reversed(GOLD["menu"]))}
    assert field_prf(shuffled, GOLD)[0] == 9                                   # item order does not matter
    wrong = json.loads(json.dumps(GOLD))
    wrong["total"]["total_price"] = "61,709"
    wrong["menu"].append({"nm": "EXTRA"})
    tp, p, g = field_prf(wrong, GOLD)
    assert (tp, p, g) == (8, 10, 9)
    assert field_prf({"menu": [{"nm": "es jeruk."}]}, {"menu": [{"nm": "ES JERUK"}]})[0] == 1   # loose
    assert field_prf({"menu": [{"nm": "es jeruk."}]}, {"menu": [{"nm": "ES JERUK"}]}, mode="strict")[0] == 0
    assert field_prf({}, GOLD) == (0, 0, 9)


def test_funsd_pairs_are_single_fields():
    f = flatten({"fields": [{"key": "Date:", "value": "3/12/92"}]})
    assert f == [("fields", "date => 3/12/92")]
    s = corpus_scores([({"fields": [{"key": "DATE", "value": "28"}]}, {"fields": [{"key": "Date", "value": "28"}, {"key": "To", "value": "X"}]})])
    assert s["precision"] == 1.0 and s["recall"] == 0.5 and s["perfect_documents"] == 0


def test_corpus_scores_per_field():
    s = corpus_scores([(GOLD, GOLD), ({"total": {"total_price": "1"}}, GOLD)])
    assert s["perfect_documents"] == 1 and s["per_field_f1"]["menu.nm"]["gold"] == 4
    assert 0 < s["f1"] < 1


# ------------------------------------------------------------------- parsing
@pytest.mark.parametrize("raw,status", [
    ('{"a": 1}', "ok"),
    ('```json\n{"a": 1}\n```', "ok"),
    ('Sure! Here is the JSON: {"a": 1} Hope it helps.', "repaired"),
    ("{'a': 1}", "repaired"),
    ('{"a": [1, 2,], }', "repaired"),
    ('{"menu": [{"nm": "TEA", "price": "10,0', "repaired"),   # cut off by max_new_tokens
])
def test_parse_json(raw, status):
    obj, st = parse_json(raw)
    assert st == status and obj is not None


def test_parse_json_truncated_keeps_complete_fields():
    obj, _ = parse_json('{"menu": [{"nm": "TEA", "price": "10"}, {"nm": "COF')
    assert obj["menu"][0] == {"nm": "TEA", "price": "10"}
    assert parse_json("no json here") == (None, "failed")


# --------------------------------------------------------------------- claim
@pytest.mark.parametrize("s,v", [("48.000", 48000), ("1,250.00", 1250), ("12,49", 12.49), ("Rp. 39.000", 39000),
                                 ("@21,000", 21000), ("-5,000", -5000), ("1.250.000", 1250000), ("25.000,50", 25000.5),
                                 ("3.5", 3.5), ("0", 0), ("abc", None), (None, None)])
def test_parse_amount(s, v):
    assert parse_amount(s) == v


def test_claim_checks():
    ok = receipt_to_claim(GOLD)
    assert ok.status == "auto_ready" and ok.total == 61799 and len(ok.checks) == 2
    bad = json.loads(json.dumps(GOLD))
    bad["total"]["total_price"] = "68,799"          # misread digit
    r = receipt_to_claim(bad)
    assert r.status == "needs_review" and "68,799.00" in r.checks[1]["detail"]
    assert receipt_to_claim({"menu": [{"nm": "A", "price": "10"}]}).status == "needs_review"   # no total
    svc = {"menu": [{"nm": "A", "price": "100"}], "sub_total": {"subtotal_price": "100", "service_price": "10",
                                                                "discount_price": "-20"}, "total": {"total_price": "90"}}
    assert receipt_to_claim(svc).status == "auto_ready"
    assert receipt_to_claim({"menu": {"nm": "A", "price": "25.000"}, "total": {"total_price": "25.000"}}).status == "auto_ready"


# ------------------------------------------------------------ finetune masks
def test_mask_labels():
    marker = [7, 8]
    ids = [1, 7, 8, 2, 50, 50, 3, 7, 8, 11, 12, 13, 0, 0]  # earlier marker in prompt; answer after the LAST one
    lab = mask_labels(ids, marker, pad_id=0, image_ids={50})
    assert lab == [IGNORE] * 9 + [11, 12, 13, IGNORE, IGNORE]
    assert mask_labels([1, 2, 3], marker, 0) == [IGNORE] * 3
    assert target_text(Example("x", "receipt", None, {"total": {"total_price": "1"}})) == '{"total":{"total_price":"1"}}'


# ----------------------------------------------------- OCR + LLM baseline (real Tesseract)
@pytest.fixture(scope="module")
def receipt_png(tmp_path_factory):
    from PIL import Image, ImageDraw, ImageFont

    fonts = glob.glob("/usr/share/fonts/**/DejaVuSansMono.ttf", recursive=True)
    if not fonts:
        pytest.skip("no TTF font")
    f = ImageFont.truetype(fonts[0], 22)
    img = Image.new("RGB", (520, 380), "white")
    d = ImageDraw.Draw(img)
    for i, line in enumerate(["1 BASO TAHU        43,181", "1 ES JERUK         13,000", "SUBTOTAL           56,181",
                              "TAX                 5,618", "TOTAL              61,799"]):
        d.text((20, 20 + i * 50), line, fill="black", font=f)
    p = tmp_path_factory.mktemp("img") / "r.png"
    img.save(p)
    return str(p)


def rule_llm(prompt: str) -> str:
    """Stand-in for the LLM: parses the OCR text block with regexes (tests the pipeline, not an LLM)."""
    ocr = prompt.split("may contain OCR errors).\n\n", 1)[1].split("\n\n", 1)[0]
    out = {"menu": [], "sub_total": {}, "total": {}}
    for line in ocr.splitlines():
        if m := re.match(r"(\d+) (.+?) ([\d.,]+)$", line):
            out["menu"].append({"nm": m.group(2), "cnt": m.group(1), "price": m.group(3)})
        elif m := re.match(r"SUBTOTAL ([\d.,]+)", line):
            out["sub_total"]["subtotal_price"] = m.group(1)
        elif m := re.match(r"TAX ([\d.,]+)", line):
            out["sub_total"]["tax_price"] = m.group(1)
        elif m := re.match(r"TOTAL ([\d.,]+)", line):
            out["total"]["total_price"] = m.group(1)
    return "```json\n" + json.dumps(out) + "\n```"


def test_tesseract_reads_receipt(receipt_png):
    text = tesseract_lines(receipt_png)
    assert "ES JERUK 13,000" in text and "TOTAL 61,799" in text


def test_ocr_llm_pipeline_and_eval(receipt_png):
    ex = OcrLlmExtractor(rule_llm, name="tesseract+rules")
    data, status, _ = ex.extract(receipt_png, "receipt")
    assert status == "ok" and field_prf(data, GOLD) == (9, 9, 9)

    class Broken:
        name = "broken"

        def extract(self, image, doc_type):
            return parse_json("I can't read this")[0], "failed", 0.1

    docs = [Example("r1", "receipt", receipt_png, GOLD)]
    good, bad = run_system(ex, docs), run_system(Broken(), docs)
    assert good["loose"]["f1"] == 1.0 and good["receipts_auto_ready"] == "1/1"
    assert bad["loose"]["f1"] == 0.0 and bad["json"]["failed"] == 1
    md = table([good, bad], "CORD")
    assert "| tesseract+rules | 1.000 |" in md
