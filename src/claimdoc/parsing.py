"""Get a JSON object out of a model reply, even a messy one.

Small vision-language models often wrap JSON in code fences, add a sentence,
use single quotes, leave trailing commas or get cut off at max_new_tokens.
Every failure here would count as a fully wrong document, so we repair what
can be repaired safely and report whether repair was needed.
"""
from __future__ import annotations

import json
import re


def _close_truncated(s: str) -> str:
    """Close brackets/strings of a reply cut off mid-JSON."""
    stack, in_str, esc = [], False, False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    if in_str:
        s += '"'
    s = re.sub(r",\s*$", "", s.rstrip())
    s = re.sub(r':\s*$', ': null', s)
    return s + "".join(reversed(stack))


def parse_json(text: str) -> tuple[dict | list | None, str]:
    """Returns (object or None, status) where status is ok | repaired | failed."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```\s*$", "", t)
    try:
        return json.loads(t), "ok"
    except json.JSONDecodeError:
        pass
    start = min([i for i in (t.find("{"), t.find("[")) if i != -1], default=-1)
    if start == -1:
        return None, "failed"
    cand = t[start:]
    end = max(cand.rfind("}"), cand.rfind("]"))
    attempts = [cand[: end + 1]] if end != -1 else []
    attempts.append(cand)
    for a in attempts:
        for fix in (lambda x: x,
                    lambda x: re.sub(r",\s*([}\]])", r"\1", x),
                    lambda x: re.sub(r",\s*([}\]])", r"\1", x.replace("'", '"')),
                    lambda x: _close_truncated(re.sub(r",\s*([}\]])", r"\1", x))):
            try:
                return json.loads(fix(a)), "repaired"
            except json.JSONDecodeError:
                continue
    return None, "failed"
