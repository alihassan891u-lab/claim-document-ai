"""From extracted receipt JSON to a claim-ready expense record with automatic checks.

An insurance or expense claim needs numbers it can trust. After extraction we
parse every amount (handling "43.636", "1,250.00", "12,49", "Rp. 39.000"),
then check the receipt's own arithmetic. A receipt that adds up can be
auto-processed; one that doesn't goes to a human with the reason.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


def parse_amount(s: str | None) -> float | None:
    """Locale-tolerant money parser.

    "48.000" -> 48000 (dot as thousands separator), "1,250.00" -> 1250.0, "12,49" -> 12.49,
    "Rp. 39.000" -> 39000, "@21,000" -> 21000, "-5,000" -> -5000.
    """
    if s is None:
        return None
    t = str(s).strip()
    neg = t.startswith("-") or (t.startswith("(") and t.endswith(")"))
    t = re.sub(r"[^\d.,]", "", t)
    if not t or not re.search(r"\d", t):
        return None
    if "." in t and "," in t:  # the later separator is the decimal point
        dec = "." if t.rfind(".") > t.rfind(",") else ","
        t = t.replace("," if dec == "." else ".", "").replace(dec, ".")
    elif "." in t or "," in t:
        sep = "." if "." in t else ","
        parts = t.split(sep)
        if len(parts) > 2 or (len(parts) == 2 and len(parts[1]) == 3):
            t = t.replace(sep, "")             # thousands grouping: 48.000 / 1,250 / 1.250.000
        elif len(parts) == 2:
            t = parts[0] + "." + parts[1]      # decimal: 12,49 / 3.5
    try:
        v = float(t)
    except ValueError:
        return None
    return -v if neg else v


@dataclass
class LineItem:
    name: str
    qty: float | None
    amount: float | None


@dataclass
class ClaimRecord:
    items: list[LineItem]
    subtotal: float | None
    tax: float | None
    service: float | None
    discount: float | None
    total: float | None
    checks: list[dict] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.total is None:
            return "needs_review"
        return "auto_ready" if all(c["ok"] for c in self.checks) else "needs_review"

    def to_dict(self) -> dict:
        return {"status": self.status, "total": self.total, "subtotal": self.subtotal, "tax": self.tax,
                "service": self.service, "discount": self.discount,
                "items": [{"name": i.name, "qty": i.qty, "amount": i.amount} for i in self.items], "checks": self.checks}


def _qty(s) -> float | None:
    m = re.search(r"\d+(?:[.,]\d+)?", str(s or ""))
    return float(m.group(0).replace(",", ".")) if m else None


def _close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= max(tol * max(abs(a), abs(b)), 1.0)


def receipt_to_claim(parsed: dict, tol: float = 0.01) -> ClaimRecord:
    menu = parsed.get("menu") or []
    menu = [menu] if isinstance(menu, dict) else menu
    items = [LineItem(str(m.get("nm", "")).strip(), _qty(m.get("cnt")), parse_amount(m.get("price"))) for m in menu
             if isinstance(m, dict)]
    st = parsed.get("sub_total") or {}
    tt = parsed.get("total") or {}
    rec = ClaimRecord(items, parse_amount(st.get("subtotal_price")), parse_amount(st.get("tax_price")),
                      parse_amount(st.get("service_price")), parse_amount(st.get("discount_price")),
                      parse_amount(tt.get("total_price")))

    priced = [i.amount for i in items if i.amount is not None]
    if rec.total is None:
        rec.checks.append({"check": "total_present", "ok": False, "detail": "no total found on the document"})
    if priced and rec.subtotal is not None:
        s = sum(priced)
        rec.checks.append({"check": "items_sum_to_subtotal", "ok": _close(s, rec.subtotal, tol),
                           "detail": f"items {s:,.2f} vs subtotal {rec.subtotal:,.2f}"})
    if rec.subtotal is not None and rec.total is not None:
        expected = rec.subtotal + (rec.tax or 0) + (rec.service or 0) - abs(rec.discount or 0)
        rec.checks.append({"check": "subtotal_plus_tax_equals_total", "ok": _close(expected, rec.total, tol),
                           "detail": f"subtotal+tax+service-discount {expected:,.2f} vs total {rec.total:,.2f}"})
    elif priced and rec.total is not None:
        s = sum(priced)
        rec.checks.append({"check": "items_sum_to_total", "ok": _close(s, rec.total, tol),
                           "detail": f"items {s:,.2f} vs total {rec.total:,.2f}"})
    return rec
