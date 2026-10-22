"""Reference paired record bootstrap using NumPy default_rng and linear percentiles.

This module requires caller-supplied gold and prediction records. The public
artifact includes only synthetic records; published intervals cannot be
recomputed from aggregate scores alone.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

from scoring import FIELDS, canon, parse

ANALYSIS_PROTOCOL_VERSION = "paired-record-f1-bonferroni-v1"
METRIC_NAMES = ("f1", "precision", "recall", "raw_exact_rate")


def validate_gold(gold_rows):
    """Return the unique reserved IDs; ambiguity or an empty denominator is an error."""
    ids = set()
    for row in gold_rows:
        rid = row["id"]
        if rid in ids:
            raise ValueError(f"duplicate gold id: {rid!r}")
        ids.add(rid)
    if not ids:
        raise ValueError("gold must contain at least one reserved row")
    return ids


def load_gold(path):
    with open(path, encoding="utf-8") as source:
        rows = [json.loads(line) for line in source]
    validate_gold(rows)
    return rows


def load(path):
    """Load raw predictions without silently accepting duplicate records."""
    out = {}
    with open(path, encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            row = json.loads(line)
            rid = row["id"]
            if rid in out:
                raise ValueError(f"{path}:{line_number}: duplicate prediction id: {rid!r}")
            output = row.get("output")
            if output is not None and not isinstance(output, str):
                raise ValueError(f"{path}:{line_number}: output for {rid!r} must be text or null")
            out[rid] = output
    return out


def coverage(gold_ids, preds, label):
    unknown = set(preds) - gold_ids
    if unknown:
        examples = sorted(map(repr, unknown))[:5]
        raise ValueError(f"{label} has {len(unknown)} prediction IDs outside gold: {examples}")
    missing = len(gold_ids - set(preds))
    null = sum(value is None for value in preds.values())
    return {"prediction_rows": len(preds), "missing_ids": missing, "null_outputs": null,
            "missing_or_null_outputs": missing + null}


def per_record(gold_rows, preds):
    """Return tp, gold-count, pred-count and raw exact; absent/null outputs fail."""
    tp = np.zeros(len(gold_rows))
    ng = np.zeros(len(gold_rows))
    npd = np.zeros(len(gold_rows))
    ex = np.zeros(len(gold_rows))
    for i, row in enumerate(gold_rows):
        gold = json.loads(row["target"])
        gold = {k: ("" if gold.get(k) is None else str(gold.get(k, ""))) for k in FIELDS}
        fields, _ = parse(preds.get(row["id"]))
        prediction = fields or {k: "" for k in FIELDS}
        for key in FIELDS:
            gv, pv = gold[key], prediction[key]
            ng[i] += bool(gv)
            npd[i] += bool(pv)
            tp[i] += bool(gv and pv and canon(gv) == canon(pv))
        ex[i] = fields is not None and all(gold[k] == prediction[k] for k in FIELDS)
    return tp, ng, npd, ex


def metrics(tp, ng, npd, ex, idx=None):
    if idx is not None:
        tp, ng, npd, ex = tp[idx], ng[idx], npd[idx], ex[idx]
    true_positive, gold_count, pred_count = tp.sum(), ng.sum(), npd.sum()
    precision = true_positive / pred_count if pred_count else 0.0
    recall = true_positive / gold_count if gold_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return 100 * f1, 100 * precision, 100 * recall, 100 * ex.mean()


def reading(lo, hi):
    """Describe direction only; an ordinary CI does not establish equivalence or NI."""
    direction = "favours A" if lo > 0 else "favours B" if hi < 0 else "direction unresolved"
    return f"{direction}; exploratory unadjusted 95% interval"


def f1_noninferiority(f1_diffs, margin, alpha=0.05, family_size=None):
    """Make the F1-only decision using the full-precision Bonferroni lower bound."""
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError("margin must be a finite positive number of percentage points")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")
    if family_size is None:
        return {"enabled": False, "metric": "f1", "passed": None,
                "reason": "descriptive comparison; no predeclared NI family supplied"}
    if isinstance(family_size, bool) or not isinstance(family_size, int) or family_size < 1:
        raise ValueError("NI family size must be a positive integer")
    per_test_alpha = alpha / family_size
    percentile = 100 * per_test_alpha
    lower = float(np.percentile(f1_diffs, percentile, method="linear"))
    return {"enabled": True, "metric": "f1", "difference": "A minus B",
            "null_hypothesis": "delta_f1_pp <= -margin_pp", "margin_pp": margin,
            "method": "one-sided percentile paired bootstrap; Bonferroni family adjustment",
            "family_size": family_size, "family_wise_alpha": alpha,
            "per_test_alpha": per_test_alpha, "lower_percentile": percentile,
            "lower_bound_pp": lower, "threshold_pp": -margin,
            "decision_rule": "unrounded lower_bound_pp > -margin_pp",
            "passed": lower > -margin}


def exact_match_test(a_exact, b_exact):
    corrected = int(((a_exact == 1) & (b_exact == 0)).sum())
    regressed = int(((a_exact == 0) & (b_exact == 1)).sum())
    k, n = min(corrected, regressed), corrected + regressed
    sign_p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"corrected_by_a": corrected, "regressed_by_a": regressed,
            "sign_test_p": round(sign_p, 4),
            "interpretation": "exploratory two-sided exact-match sign test against zero; not an F1 NI test"}


def compare(gold_rows, pa, pb, *, resamples=50000, seed=42, margin=1.0,
            alpha=0.05, ni_family_size=None):
    gold_ids = validate_gold(gold_rows)
    cov_a, cov_b = coverage(gold_ids, pa, "A"), coverage(gold_ids, pb, "B")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    # Validate NI arguments before spending time on resampling.
    f1_noninferiority([0.0], margin, alpha, ni_family_size)
    a_records, b_records = per_record(gold_rows, pa), per_record(gold_rows, pb)
    ma, mb = metrics(*a_records), metrics(*b_records)
    rng = np.random.default_rng(seed)
    n = len(gold_rows)
    diffs = np.empty((resamples, len(METRIC_NAMES)))
    for r in range(resamples):
        idx = rng.integers(0, n, n)
        diffs[r] = np.subtract(metrics(*a_records, idx), metrics(*b_records, idx))
    lo, hi = np.percentile(diffs, [2.5, 97.5], axis=0, method="linear")
    return {
        "analysis_protocol_version": ANALYSIS_PROTOCOL_VERSION,
        "n_evaluated": n, "gold_rows": n,
        "denominator_policy": "all reserved gold IDs; missing/null outputs scored as failures",
        "n_common": len(set(pa) & set(pb)),
        "only_in_a": len(set(pa) - set(pb)), "only_in_b": len(set(pb) - set(pa)),
        "coverage": {"a": cov_a, "b": cov_b},
        "metrics_a": {name: round(float(value), 2) for name, value in zip(METRIC_NAMES, ma)},
        "metrics_b": {name: round(float(value), 2) for name, value in zip(METRIC_NAMES, mb)},
        "a_minus_b": {
            name: {"delta": round(float(x - y), 2),
                   "ci95": [round(float(low), 2), round(float(high), 2)],
                   "reading": reading(low, high)}
            for name, x, y, low, high in zip(METRIC_NAMES, ma, mb, lo, hi)
        },
        "record_exact": exact_match_test(a_records[3], b_records[3]),
        "noninferiority": f1_noninferiority(diffs[:, 0], margin, alpha, ni_family_size),
        "bootstrap": {"resamples": resamples, "seed": seed, "unit": "record",
                      "margin_pp": margin, "percentile_method": "linear",
                      "ci95_interpretation": "exploratory unadjusted two-sided intervals"},
        "normalisation": "prior-study light_eval: upper-case, remove '*', ' ', ',', '-'",
    }

