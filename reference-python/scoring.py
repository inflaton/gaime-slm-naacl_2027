"""Reference scorer for the ten-field extraction task. Standard library only.

Retains the study parser, including Python dictionary literals and legacy
schema-invalid coercion. This module contains no dataset or filesystem paths.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

FIELDS = ["INTERMEDIARY", "MERCHANT_NAME", "DYNAMIC_DESCRIPTOR", "STREET", "ZIP",
          "CITY", "STATE", "COUNTRY", "PHONE", "WEBSITE"]
FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def canon(v: str) -> str:
    v = str(v).upper()
    for ch in ("*", " ", ",", "-"):
        v = v.replace(ch, "")
    return v


def parse(text):
    """-> (fields dict or None, status) with status in parsed|schema_invalid|parse_failed."""
    if text is None:
        return None, "parse_failed"
    s = text.strip()
    if "</think>" in s:
        s = s.rsplit("</think>", 1)[1].strip()
    obj = None
    for cand in (s, (FENCE.search(s) or [None, None])[1],
                 s[s.find("{"): s.rfind("}") + 1] if "{" in s and "}" in s else None):
        if not cand:
            continue
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if isinstance(obj, dict):
            break
        obj = None
    if not isinstance(obj, dict):
        # the prior-study artifacts store parsed outputs as Python dict reprs (single quotes); the prior-study scorer
        # falls back to ast.literal_eval the same way
        try:
            import ast
            lit = ast.literal_eval(s)
            obj = lit if isinstance(lit, dict) else None
        except Exception:
            obj = None
    if not isinstance(obj, dict):
        return None, "parse_failed"
    valid = set(obj) == set(FIELDS) and all(isinstance(v, str) for v in obj.values())
    fields = {k: ("" if obj.get(k) is None else str(obj.get(k, ""))) for k in FIELDS}
    return fields, "parsed" if valid else "schema_invalid"


def score(gold_rows, preds, truncated=None):
    """gold_rows: [{id, target}]; preds: {id: raw text}; truncated: {id: bool}."""
    counts = Counter()
    tp, n_gold, n_pred = Counter(), Counter(), Counter()
    raw_exact = canon_exact = valid_and_exact = 0
    out_len = []
    dup = Counter(preds.get("__dup__", []))
    for g in gold_rows:
        gold = json.loads(g["target"])
        gold = {k: ("" if gold.get(k) is None else str(gold.get(k, ""))) for k in FIELDS}
        raw = preds.get(g["id"])
        if raw is None:
            counts["missing"] += 1
            fields, status = None, "parse_failed"
        else:
            fields, status = parse(raw)
            out_len.append(len(raw))
        counts[status] += 1
        if truncated and truncated.get(g["id"]):
            counts["truncated"] += 1
        p = fields or {k: "" for k in FIELDS}
        for k in FIELDS:
            gv, pv = gold[k], p[k]
            if gv:
                n_gold[k] += 1
            if pv:
                n_pred[k] += 1
            if gv and pv and canon(gv) == canon(pv):
                tp[k] += 1
        if fields is not None:
            if all(gold[k] == p[k] for k in FIELDS):
                raw_exact += 1
                if status == "parsed":
                    valid_and_exact += 1
            if all(canon(gold[k]) == canon(p[k]) for k in FIELDS):
                canon_exact += 1

    def prf(t, g, pr):
        P = t / pr if pr else 0.0
        R = t / g if g else 0.0
        F = 2 * P * R / (P + R) if P + R else 0.0
        return {"tp": t, "gold": g, "pred": pr, "precision": round(P, 4), "recall": round(R, 4), "f1": round(F, 4)}

    n = len(gold_rows)
    per_field = {k: prf(tp[k], n_gold[k], n_pred[k]) for k in FIELDS}
    overall = prf(sum(tp.values()), sum(n_gold.values()), sum(n_pred.values()))
    return {"n": n, "overall": overall, "per_field": per_field,
            "record": {"raw_exact": raw_exact, "raw_exact_rate": round(raw_exact / n, 4),
                       "canonical_exact": canon_exact, "canonical_exact_rate": round(canon_exact / n, 4),
                       "schema_valid_and_raw_exact": valid_and_exact,
                       "schema_valid_and_raw_exact_rate": round(valid_and_exact / n, 4)},
            "outputs": {"schema_valid": counts["parsed"], "schema_invalid": counts["schema_invalid"],
                        "parse_failed": counts["parse_failed"] - counts["missing"], "missing": counts["missing"],
                        "duplicate_ids": sum(dup.values()), "truncated": counts["truncated"],
                        "schema_valid_rate": round(counts["parsed"] / n, 4),
                        "mean_output_chars": round(sum(out_len) / len(out_len), 1) if out_len else None}}

