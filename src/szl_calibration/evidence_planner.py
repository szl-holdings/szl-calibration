"""MODELED, one-step evidence acquisition proposals; never execution authority.

Joint scenarios preserve explicitly supplied correlations. Acquisitions are
modeled as a truthful PASS/FAIL observation, or an independent unavailable result.
The planner neither obtains evidence nor promotes modeled outcomes to evidence.
"""
from __future__ import annotations

import hashlib
import math

from .decisions import MAX_STUDY_BYTES, assess_decisions, parse_study
from .receipts import canonical_json

MAX_PLAN_BYTES = MAX_STUDY_BYTES
MAX_SCENARIOS = 256
MAX_ACTIONS = 64
VERDICTS = ("BLOCK", "REVIEW", "ELIGIBLE_FOR_SHADOW")


def parse_plan(raw: bytes) -> dict:
    """Parse bounded JSON, rejecting duplicate fields and non-finite numbers."""
    return parse_study(raw)


def _object(value, fields, name):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError(f"{name} must have exactly the documented fields")


def _name(value, name):
    if not isinstance(value, str) or not value or value != value.strip() or len(value) > 256:
        raise ValueError(f"{name} must be a nonempty trimmed string of at most 256 characters")


def _probability(value, name):
    if type(value) not in (int, float) or not 0 <= value <= 1 or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite probability")


def _units(value, name, minimum):
    if type(value) is not int or not minimum <= value <= 1_000_000:
        raise ValueError(f"{name} must be an integer from {minimum} to 1000000")


def _digest(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _distribution(scenarios, all_pass_verdict):
    total = math.fsum(s["probability"] for s in scenarios)
    return {
        verdict: math.fsum(s["probability"] for s in scenarios
                           if ("BLOCK" if "FAIL" in s["evidence"].values()
                               else all_pass_verdict) == verdict) / total
        for verdict in VERDICTS
    }


def _entropy(distribution):
    return -math.fsum(p * math.log2(p) for p in distribution.values() if p > 0)


def plan_evidence(request: dict) -> dict:
    """Propose one affordable acquisition from caller-declared scenario weights.

    Scores are expected final-verdict entropy reduction per integer cost unit.
    Ties use expected direct evidence resolution per cost, cost, then action ID.
    No budget is spent; no evidence is changed; no action is authorized.
    """
    _object(request, ("schema", "probability_basis", "study", "budget_units",
                      "scenarios", "actions"), "request")
    if request["schema"] != "szl.evidence.plan.request.v1":
        raise ValueError("unsupported evidence planning schema")
    if request["probability_basis"] != "MODELED":
        raise ValueError("scenario probabilities must be explicitly MODELED")
    _units(request["budget_units"], "budget_units", 0)
    study = request["study"]
    assessment = assess_decisions(study)
    required = study["policy"]["required_evidence"]
    evidence = study["evidence"]
    scenarios = request["scenarios"]
    if not isinstance(scenarios, list) or not 1 <= len(scenarios) <= MAX_SCENARIOS:
        raise ValueError("scenarios must contain 1 to 256 records")
    seen_ids, seen_states = set(), set()
    for scenario in scenarios:
        _object(scenario, ("id", "probability", "evidence"), "scenario")
        _name(scenario["id"], "scenario id")
        if scenario["id"] in seen_ids:
            raise ValueError("duplicate scenario id")
        seen_ids.add(scenario["id"])
        _probability(scenario["probability"], "scenario probability")
        if scenario["probability"] == 0:
            raise ValueError("scenario probabilities must be positive")
        _object(scenario["evidence"], required, "scenario evidence")
        if any(status not in ("PASS", "FAIL") for status in scenario["evidence"].values()):
            raise ValueError("scenario evidence must be PASS or FAIL")
        state = tuple(scenario["evidence"][key] for key in required)
        if state in seen_states:
            raise ValueError("duplicate scenario evidence assignment")
        seen_states.add(state)
    if not math.isclose(math.fsum(s["probability"] for s in scenarios), 1.0,
                        abs_tol=1e-9, rel_tol=0):
        raise ValueError("scenario probabilities must sum to one")

    actions = request["actions"]
    if not isinstance(actions, list) or not 0 <= len(actions) <= MAX_ACTIONS:
        raise ValueError("actions must contain 0 to 64 records")
    seen_ids, seen_targets = set(), set()
    for action in actions:
        _object(action, ("id", "evidence", "cost_units", "requires_pass",
                         "unavailable_probability"), "action")
        _name(action["id"], "action id")
        _name(action["evidence"], "action evidence")
        if action["id"] in seen_ids:
            raise ValueError("duplicate action id")
        if action["evidence"] in seen_targets:
            raise ValueError("duplicate action evidence target")
        if action["evidence"] not in required:
            raise ValueError("action evidence must be a required evidence item")
        seen_ids.add(action["id"])
        seen_targets.add(action["evidence"])
        _units(action["cost_units"], "cost_units", 1)
        _probability(action["unavailable_probability"], "unavailable_probability")
        prerequisites = action["requires_pass"]
        if not isinstance(prerequisites, list) or len(prerequisites) > len(required):
            raise ValueError("requires_pass must be a bounded list of evidence names")
        for name in prerequisites:
            _name(name, "prerequisite")
            if name not in required or name == action["evidence"]:
                raise ValueError("prerequisites must name other required evidence")
        if len(prerequisites) != len(set(prerequisites)):
            raise ValueError("duplicate prerequisite")

    consistent = [s for s in scenarios if all(
        status == "UNAVAILABLE" or s["evidence"][name] == status
        for name, status in evidence.items())]
    if not consistent:
        raise ValueError("observed evidence lies outside declared scenario support")
    # Conditioning is explicit: impossible scenarios are removed, not treated as
    # negative observations. The retained mass is returned for model diagnostics.
    retained_mass = math.fsum(s["probability"] for s in consistent)
    all_pass_study = dict(study, evidence={name: "PASS" for name in required})
    all_pass_verdict = assess_decisions(all_pass_study)["verdict"]
    prior = _distribution(consistent, all_pass_verdict)
    entropy = _entropy(prior)
    candidates = []
    for action in actions:
        target = action["evidence"]
        unavailable = float(action["unavailable_probability"])
        outcomes = []
        for status in ("PASS", "FAIL", "UNAVAILABLE"):
            subset = consistent if status == "UNAVAILABLE" else [
                s for s in consistent if s["evidence"][target] == status]
            mass = math.fsum(s["probability"] for s in subset) / retained_mass
            probability = unavailable if status == "UNAVAILABLE" else (1 - unavailable) * mass
            posterior = _distribution(subset, all_pass_verdict) if subset and probability > 0 else None
            outcomes.append({"outcome": status, "probability": probability,
                             "verdict_probabilities": posterior,
                             "entropy_bits": _entropy(posterior) if posterior is not None else None})
        expected_entropy = math.fsum(o["probability"] * o["entropy_bits"]
                                     for o in outcomes if o["entropy_bits"] is not None)
        gain = max(0.0, entropy - expected_entropy)  # roundoff only; never a trust score
        reasons = []
        if assessment["verdict"] == "BLOCK":
            reasons.append("CURRENT_DECISION_BLOCKED")
        if evidence[target] != "UNAVAILABLE":
            reasons.append("EVIDENCE_ALREADY_PRESENT")
        if any(evidence[name] != "PASS" for name in action["requires_pass"]):
            reasons.append("PREREQUISITE_NOT_PASS")
        if action["cost_units"] > request["budget_units"]:
            reasons.append("OVER_BUDGET")
        if unavailable == 1:
            reasons.append("NO_MODELED_RESOLUTION")
        candidates.append({
            "id": action["id"], "evidence": target, "cost_units": action["cost_units"],
            "requires_pass": list(action["requires_pass"]),
            "eligible_for_proposal": not reasons, "exclusion_reasons": reasons,
            "expected_entropy_bits": expected_entropy, "information_gain_bits": gain,
            "information_gain_per_cost": gain / action["cost_units"],
            "expected_direct_resolution_per_cost": (1 - unavailable) / action["cost_units"],
            "outcomes": outcomes,
        })
    candidates.sort(key=lambda c: (
        not c["eligible_for_proposal"], -round(c["information_gain_per_cost"], 12),
        -round(c["expected_direct_resolution_per_cost"], 12), c["cost_units"], c["id"]))
    selected = next((c for c in candidates if c["eligible_for_proposal"]), None)
    if selected:
        reason = "EXPECTED_INFORMATION_GAIN" if selected["information_gain_bits"] > 1e-12 else "REQUIRED_EVIDENCE_PENDING"
    else:
        reason = "CURRENT_DECISION_BLOCKED" if assessment["verdict"] == "BLOCK" else "NO_ELIGIBLE_ACQUISITION"
    return {
        "schema": "szl.evidence.plan.v1", "evidence_class": "MODELED",
        "execution_authorized": False, "trust_ceiling": 0.97,
        "input_sha256": _digest(request), "study_sha256": assessment["input_sha256"],
        "scenario_sha256": _digest(scenarios), "policy_sha256": assessment["policy_sha256"],
        "current_assessment": assessment,
        "modeled_verdict_probabilities": prior, "modeled_entropy_bits": entropy,
        "retained_scenario_mass": retained_mass, "consistent_scenarios": len(consistent),
        "budget_units": request["budget_units"], "selected_action_id": selected["id"] if selected else None,
        "selection_reason": reason, "candidates": candidates,
        "assumptions": "Caller-declared joint scenario weights; truthful PASS/FAIL acquisition; unavailable outcomes independent of scenario. Probabilities are MODELED, not observed correctness or authorization. One-step ranking is not a globally optimal budget policy. Replan after a separately verified observation and fresh source binding. No budget is reserved or spent.",
    }
