"""Provider-neutral selective decision studies. No inference or execution authority.

A frozen classifier/threshold is measured against caller-supplied held-out labels.
The Hoeffding bound is conditional on independent representative labeled examples;
this software cannot certify those assumptions or the truth of the supplied labels.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

from .metrics import expected_calibration_error, log_loss, brier_score
from .receipts import canonical_json

MAX_SAMPLES = 10000
MAX_CLASSES = 255
MAX_STUDY_BYTES = 8 * 1024 * 1024


def parse_study(raw: bytes) -> dict:
    """Reject ambiguous duplicate fields before evidence becomes a Python dict."""
    if len(raw) > MAX_STUDY_BYTES:
        raise ValueError("study exceeds 8 MiB")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result

    def nonfinite(_):
        raise ValueError("non-finite JSON number")

    try:
        return json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)
    except RecursionError as exc:
        raise ValueError("study nesting exceeds parser limit") from exc


def _object(value, fields, name):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError(f"{name} must have exactly the documented fields")


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{name} must be a nonempty string of at most 256 characters")


def _number(value, name, lo=0.0, hi=1.0):
    if type(value) not in (int, float) or not lo <= value <= hi or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number in [{lo}, {hi}]")
    return float(value)


def _names(value, name, low=1, high=64):
    if not isinstance(value, list) or not low <= len(value) <= high:
        raise ValueError(f"{name} must have {low} to {high} entries")
    for item in value:
        _text(item, name)
    if len(set(value)) != len(value):
        raise ValueError(f"{name} entries must be unique")


def assess_decisions(request: dict) -> dict:
    """Assess one prespecified threshold, including every prespecified cohort.

    Highest outcome is ELIGIBLE_FOR_SHADOW, never deployment authorization.
    Digests bind supplied inputs; they are not dataset or identity attestations.
    """
    _object(request, ("model_id", "task_id", "question_revision", "dataset_revision", "label_kind",
                      "held_out", "policy_frozen", "classes", "policy",
                      "evidence", "samples"), "request")
    for key in ("model_id", "task_id", "dataset_revision"):
        _text(request[key], key)
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", request["dataset_revision"]):
        raise ValueError("dataset_revision must be a lowercase immutable 40/64-hex revision")
    if not isinstance(request["question_revision"], str) or not re.fullmatch(r"[0-9a-f]{64}", request["question_revision"]):
        raise ValueError("question_revision must be the 64-hex digest of the frozen task questions")
    model_id = request["model_id"]
    if model_id != model_id.strip() or re.split(r"[-:/@]", model_id.lower())[-1] in ("latest", "preview", "main", "master"):
        raise ValueError("model_id must be a caller-asserted version; common moving aliases and surrounding whitespace are rejected")
    if request["label_kind"] not in ("OBSERVED", "SYNTHETIC"):
        raise ValueError("label_kind must be OBSERVED or SYNTHETIC")
    if type(request["held_out"]) is not bool or type(request["policy_frozen"]) is not bool:
        raise ValueError("held_out and policy_frozen must be booleans")
    classes = request["classes"]
    _names(classes, "classes", 2, MAX_CLASSES)
    policy = request["policy"]
    _object(policy, ("threshold", "max_error", "alpha", "min_accepted_per_group",
                     "required_groups", "required_evidence"), "policy")
    threshold = _number(policy["threshold"], "threshold", 0.5)
    max_error = _number(policy["max_error"], "max_error", 0.0, 0.5)
    alpha = _number(policy["alpha"], "alpha", 0.000001, 0.25)
    minimum = policy["min_accepted_per_group"]
    if type(minimum) is not int or not 1 <= minimum <= MAX_SAMPLES:
        raise ValueError("min_accepted_per_group must be an integer from 1 to 10000")
    groups = policy["required_groups"]
    _names(groups, "required_groups")
    _names(policy["required_evidence"], "required_evidence")
    evidence = request["evidence"]
    _object(evidence, policy["required_evidence"], "evidence")
    if any(v not in ("PASS", "FAIL", "UNAVAILABLE") for v in evidence.values()):
        raise ValueError("evidence values must be PASS, FAIL, or UNAVAILABLE")
    samples = request["samples"]
    if not isinstance(samples, list) or not 1 <= len(samples) <= MAX_SAMPLES:
        raise ValueError("samples must contain 1 to 10000 records")
    if len(samples) * len(classes) > 100000:
        raise ValueError("study exceeds 100000 probability cells")
    seen = set()
    rows = []
    for sample in samples:
        _object(sample, ("id", "group", "probabilities", "label"), "sample")
        _text(sample["id"], "sample id")
        if sample["id"] in seen:
            raise ValueError("duplicate sample id")
        seen.add(sample["id"])
        if sample["group"] not in groups:
            raise ValueError("sample group must be prespecified in required_groups")
        if sample["label"] not in classes:
            raise ValueError("sample label must occur in classes")
        probs = sample["probabilities"]
        _object(probs, classes, "probabilities")
        for value in probs.values():
            _number(value, "probability")
        if not math.isclose(math.fsum(probs.values()), 1.0, abs_tol=1e-6, rel_tol=0):
            raise ValueError("probabilities must sum to one; no silent normalization")
        top = max(probs.values())
        winners = [name for name in classes if probs[name] == top]
        # Stable class-order tie-break defines the event for calibration metrics.
        # Execution selection still abstains on every tie.
        predicted = winners[0]
        rows.append({"group": sample["group"], "confidence": top,
                     "correct": int(predicted == sample["label"]),
                     "accepted": len(winners) == 1 and top >= threshold,
                     "brier": math.fsum((probs[c] - int(c == sample["label"])) ** 2 for c in classes),
                     "log_loss": -math.log(max(probs[sample["label"]], 1e-15))})

    # Union-bound allocation covers every declared group, including empty ones.
    group_alpha = alpha / len(groups)
    group_reports = []
    for group in groups:
        subset = [row for row in rows if row["group"] == group]
        accepted = [row for row in subset if row["accepted"]]
        n = len(accepted)
        errors = sum(1 - row["correct"] for row in accepted)
        risk = errors / n if n else None
        upper = min(1.0, risk + math.sqrt(math.log(1 / group_alpha) / (2 * n))) if n else None
        group_reports.append({"group": group, "n": len(subset), "accepted": n,
                              "errors": errors, "coverage": n / len(subset) if subset else 0.0,
                              "selective_error": risk, "error_upper_bound": upper,
                              "alpha": group_alpha,
                              "passes": n >= minimum and upper is not None and upper <= max_error})
    probabilities = [row["confidence"] for row in rows]
    correctness = [row["correct"] for row in rows]
    reasons = []
    if request["label_kind"] != "OBSERVED":
        reasons.append("SYNTHETIC_LABELS")
    if not request["held_out"]:
        reasons.append("HELD_OUT_NOT_ASSERTED")
    if not request["policy_frozen"]:
        reasons.append("POLICY_NOT_FROZEN")
    if any(value != "PASS" for value in evidence.values()):
        reasons.append("REQUIRED_EVIDENCE_NOT_PASS")
    if not all(report["passes"] for report in group_reports):
        reasons.append("GROUP_RISK_OR_SAMPLE_SIZE")
    verdict = "BLOCK" if "FAIL" in evidence.values() else "REVIEW" if reasons else "ELIGIBLE_FOR_SHADOW"
    accepted_count = sum(row["accepted"] for row in rows)
    return {
        "schema": "szl.decision.study.v1", "verdict": verdict, "reasons": reasons,
        "execution_authorized": False, "model_id": request["model_id"],
        "task_id": request["task_id"], "dataset_revision": request["dataset_revision"],
        "input_sha256": hashlib.sha256(canonical_json(request).encode()).hexdigest(),
        "policy_sha256": hashlib.sha256(canonical_json({
            key: request[key] for key in ("model_id", "task_id", "question_revision", "classes", "policy")
        }).encode()).hexdigest(),
        "label_kind": request["label_kind"], "n": len(rows), "accepted": accepted_count,
        "abstained": len(rows) - accepted_count, "coverage": accepted_count / len(rows),
        "metrics": {"multiclass_brier": math.fsum(row["brier"] for row in rows) / len(rows),
                    "multiclass_log_loss": math.fsum(row["log_loss"] for row in rows) / len(rows),
                    "top_label_ece": expected_calibration_error(probabilities, correctness),
                    "top_label_brier": brier_score(probabilities, correctness),
                    "top_label_log_loss": log_loss(probabilities, correctness)},
        "groups": group_reports, "bound": "ONE_SIDED_HOEFFDING_BONFERRONI",
        "evidence_status": evidence.copy(),
        "assumptions": "Caller asserts labels, independent representative held-out samples, and a frozen model, prompt, threshold and group definition. Not independently verified. Repeated adaptive evaluations require new data or alpha accounting.",
    }
