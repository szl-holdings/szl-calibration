import copy
import itertools
import json
import math
from pathlib import Path

import pytest

from szl_calibration.decisions import assess_decisions
from szl_calibration.evidence_planner import MAX_PLAN_BYTES, parse_plan, plan_evidence


def request():
    return {
        "schema": "szl.evidence.plan.request.v1", "probability_basis": "MODELED",
        "budget_units": 1,
        "study": {
            "model_id": "sample-classifier-v1", "task_id": "sample-release-triage-v1",
            "question_revision": "c" * 64, "dataset_revision": "d" * 64,
            "label_kind": "SYNTHETIC", "held_out": False, "policy_frozen": True,
            "classes": ["review", "route"],
            "policy": {"threshold": .95, "max_error": .5, "alpha": .05,
                       "min_accepted_per_group": 1, "required_groups": ["sample"],
                       "required_evidence": ["source", "runtime"]},
            "evidence": {"source": "UNAVAILABLE", "runtime": "UNAVAILABLE"},
            "samples": [{"id": "sample-1", "group": "sample",
                         "probabilities": {"review": .04, "route": .96}, "label": "route"}],
        },
        "scenarios": [
            {"id": "both-pass", "probability": .45, "evidence": {"source": "PASS", "runtime": "PASS"}},
            {"id": "runtime-fails", "probability": .45, "evidence": {"source": "PASS", "runtime": "FAIL"}},
            {"id": "source-fails", "probability": .05, "evidence": {"source": "FAIL", "runtime": "PASS"}},
            {"id": "both-fail", "probability": .05, "evidence": {"source": "FAIL", "runtime": "FAIL"}},
        ],
        "actions": [
            {"id": "check-source", "evidence": "source", "cost_units": 1,
             "requires_pass": [], "unavailable_probability": 0},
            {"id": "check-runtime", "evidence": "runtime", "cost_units": 1,
             "requires_pass": [], "unavailable_probability": 0},
        ],
    }


def candidate(result, target):
    return next(c for c in result["candidates"] if c["evidence"] == target)


def test_known_information_gain_and_synthetic_boundary():
    result = plan_evidence(request())
    h = lambda p: -p * math.log2(p) - (1 - p) * math.log2(1 - p)
    assert result["modeled_entropy_bits"] == pytest.approx(h(.45))
    assert candidate(result, "source")["information_gain_bits"] == pytest.approx(h(.45) - .9)
    assert candidate(result, "runtime")["information_gain_bits"] == pytest.approx(h(.45) - .5 * h(.1))
    assert result["selected_action_id"] == "check-runtime"
    assert result["current_assessment"]["verdict"] == "REVIEW"
    assert result["modeled_verdict_probabilities"]["ELIGIBLE_FOR_SHADOW"] == 0
    assert result["execution_authorized"] is False
    assert result["evidence_class"] == "MODELED"
    assert result["trust_ceiling"] == .97


def test_unknown_acquisition_retains_prior_and_reduces_gain():
    req = request()
    original = candidate(plan_evidence(req), "runtime")
    req["actions"][1]["unavailable_probability"] = .4
    result = plan_evidence(req)
    runtime = candidate(result, "runtime")
    unknown = next(o for o in runtime["outcomes"] if o["outcome"] == "UNAVAILABLE")
    assert unknown["probability"] == .4
    assert unknown["verdict_probabilities"] == result["modeled_verdict_probabilities"]
    assert unknown["entropy_bits"] == result["modeled_entropy_bits"]
    assert runtime["information_gain_bits"] == pytest.approx(.6 * original["information_gain_bits"])
    assert math.fsum(o["probability"] for o in runtime["outcomes"]) == pytest.approx(1)


def test_always_unavailable_and_over_budget_never_selected():
    req = request()
    req["actions"][0]["unavailable_probability"] = 1
    req["actions"][1]["cost_units"] = 2
    result = plan_evidence(req)
    assert result["selected_action_id"] is None
    assert "NO_MODELED_RESOLUTION" in candidate(result, "source")["exclusion_reasons"]
    assert "OVER_BUDGET" in candidate(result, "runtime")["exclusion_reasons"]
    req["budget_units"] = 2
    assert plan_evidence(req)["selected_action_id"] == "check-runtime"
    req["budget_units"] = 0
    assert plan_evidence(req)["selected_action_id"] is None


def test_only_declared_pass_satisfies_prerequisite():
    req = request()
    req["actions"][1]["requires_pass"] = ["source"]
    result = plan_evidence(req)
    assert result["selected_action_id"] == "check-source"
    assert "PREREQUISITE_NOT_PASS" in candidate(result, "runtime")["exclusion_reasons"]
    req["study"]["evidence"]["source"] = "PASS"
    result = plan_evidence(req)
    assert result["selected_action_id"] == "check-runtime"
    assert result["retained_scenario_mass"] == pytest.approx(.9)
    assert result["consistent_scenarios"] == 2
    assert result["modeled_verdict_probabilities"]["BLOCK"] == pytest.approx(.5)
    assert "EVIDENCE_ALREADY_PRESENT" in candidate(result, "source")["exclusion_reasons"]


def test_declared_failure_stops_proposals():
    req = request()
    req["study"]["evidence"]["source"] = "FAIL"
    result = plan_evidence(req)
    assert result["current_assessment"]["verdict"] == "BLOCK"
    assert result["selected_action_id"] is None
    assert all("CURRENT_DECISION_BLOCKED" in c["exclusion_reasons"] for c in result["candidates"])


def test_correlated_scenarios_do_not_become_independent():
    req = request()
    req["scenarios"] = [req["scenarios"][0], req["scenarios"][3]]
    for scenario in req["scenarios"]:
        scenario["probability"] = .5
    result = plan_evidence(req)
    assert result["modeled_entropy_bits"] == pytest.approx(1)
    assert all(c["information_gain_bits"] == pytest.approx(1) for c in result["candidates"])
    assert result["selected_action_id"] == "check-runtime"  # deterministic lexical tie
    req["actions"].reverse()
    assert plan_evidence(req)["selected_action_id"] == "check-runtime"


def test_modeled_certainty_never_replaces_missing_obligations():
    req = request()
    req["scenarios"] = [req["scenarios"][0]]
    req["scenarios"][0]["probability"] = 1
    result = plan_evidence(req)
    assert result["modeled_entropy_bits"] == 0
    assert result["selected_action_id"] is not None
    assert result["selection_reason"] == "REQUIRED_EVIDENCE_PENDING"
    assert result["current_assessment"]["evidence_status"] == {"source": "UNAVAILABLE", "runtime": "UNAVAILABLE"}


def test_fully_observed_evidence_has_no_pending_acquisition():
    req = request()
    req["study"]["evidence"] = {"source": "PASS", "runtime": "PASS"}
    result = plan_evidence(req)
    assert result["selected_action_id"] is None
    assert result["current_assessment"]["verdict"] == "REVIEW"


def test_small_declared_support_is_refused_after_contradicting_observation():
    req = request()
    req["scenarios"] = [req["scenarios"][0]]
    req["scenarios"][0]["probability"] = 1
    req["study"]["evidence"]["runtime"] = "FAIL"
    with pytest.raises(ValueError, match="outside declared scenario support"):
        plan_evidence(req)


def test_cost_changes_proposal_without_authorization():
    req = request()
    req["budget_units"] = 20
    req["actions"][1]["cost_units"] = 20
    assert plan_evidence(req)["selected_action_id"] == "check-source"


def test_no_mutation_and_full_input_digest_binding():
    req = request()
    original = copy.deepcopy(req)
    result = plan_evidence(req)
    assert req == original
    result["current_assessment"]["evidence_status"]["runtime"] = "FAIL"
    result["candidates"][0]["requires_pass"].append("source")
    assert req == original
    first = plan_evidence(req)
    req["budget_units"] += 1
    second = plan_evidence(req)
    assert first["input_sha256"] != second["input_sha256"]
    assert first["study_sha256"] == second["study_sha256"]
    req["scenarios"][0]["probability"] -= .01
    req["scenarios"][1]["probability"] += .01
    third = plan_evidence(req)
    assert second["scenario_sha256"] != third["scenario_sha256"]
    assert second["policy_sha256"] == third["policy_sha256"]


@pytest.mark.parametrize("value", [True, -1, 1.1, "0.5", None, float("inf"), float("nan")])
def test_malformed_outcome_probabilities_rejected(value):
    req = request()
    req["actions"][0]["unavailable_probability"] = value
    with pytest.raises(ValueError):
        plan_evidence(req)


@pytest.mark.parametrize("field,value", [("cost_units", True), ("cost_units", 0),
                                         ("cost_units", 1.5), ("cost_units", 1_000_001),
                                         ("requires_pass", ["source"]),
                                         ("requires_pass", ["runtime", "runtime"]),
                                         ("requires_pass", ["unknown"]),
                                         ("requires_pass", [["runtime"]]),
                                         ("evidence", "unknown")])
def test_malformed_actions_rejected(field, value):
    req = request()
    req["actions"][0][field] = value
    with pytest.raises(ValueError):
        plan_evidence(req)


@pytest.mark.parametrize("duplicate", ["action_id", "action_target", "scenario_id", "scenario_state"])
def test_duplicate_evidence_and_ids_rejected(duplicate):
    req = request()
    if duplicate == "action_id":
        req["actions"][1]["id"] = req["actions"][0]["id"]
    elif duplicate == "action_target":
        req["actions"][1]["evidence"] = req["actions"][0]["evidence"]
    elif duplicate == "scenario_id":
        req["scenarios"][1]["id"] = req["scenarios"][0]["id"]
    else:
        req["scenarios"][1]["evidence"] = dict(req["scenarios"][0]["evidence"])
    with pytest.raises(ValueError, match="duplicate"):
        plan_evidence(req)


@pytest.mark.parametrize("value", [0, -.1, True, .8, float("nan")])
def test_invalid_scenario_mass_rejected(value):
    req = request()
    req["scenarios"][0]["probability"] = value
    with pytest.raises(ValueError):
        plan_evidence(req)


def test_exact_schema_and_declared_probability_basis_required():
    for field, value in (("probability_basis", "MEASURED"), ("schema", "other"),
                         ("budget_units", True), ("budget_units", -1),
                         ("unexpected", 1), ("scenarios", [])):
        req = request()
        req[field] = value
        with pytest.raises(ValueError):
            plan_evidence(req)
    req = request()
    req["scenarios"][0]["evidence"]["runtime"] = "UNAVAILABLE"
    with pytest.raises(ValueError):
        plan_evidence(req)
    req = request()
    req["actions"] = []
    assert plan_evidence(req)["selected_action_id"] is None


def test_strict_parser_and_size_bound():
    encoded = json.dumps(request()).encode()
    assert plan_evidence(parse_plan(encoded))["selected_action_id"] == "check-runtime"
    with pytest.raises(ValueError, match="duplicate"):
        parse_plan(encoded.replace(b'"budget_units": 1', b'"budget_units": 1, "budget_units": 100'))
    with pytest.raises(ValueError, match="non-finite"):
        parse_plan(b'{"number": NaN}')
    with pytest.raises(ValueError, match="exceeds"):
        parse_plan(b" " * (MAX_PLAN_BYTES + 1))


def test_joint_scenario_entropy_properties_over_all_three_check_worlds():
    req = request()
    names = ["source", "runtime", "receipt"]
    req["study"]["policy"]["required_evidence"] = names
    req["study"]["evidence"] = dict.fromkeys(names, "UNAVAILABLE")
    req["scenarios"] = [{"id": str(i), "probability": 1 / 8, "evidence": dict(zip(names, statuses))}
                        for i, statuses in enumerate(itertools.product(("PASS", "FAIL"), repeat=3))]
    result = plan_evidence(req)
    for c in result["candidates"]:
        assert 0 <= c["information_gain_bits"] <= result["modeled_entropy_bits"]
        assert math.fsum(o["probability"] for o in c["outcomes"]) == pytest.approx(1)
        for outcome in c["outcomes"]:
            if outcome["verdict_probabilities"] is not None:
                assert math.fsum(outcome["verdict_probabilities"].values()) == pytest.approx(1)


def test_twenty_case_invented_experiment_against_fixed_order():
    # SIMULATED closed-world exercise, not a vendor evaluation or held-out study.
    req = request()
    proposal = plan_evidence(req)
    selected = next(a for a in req["actions"] if a["id"] == proposal["selected_action_id"])
    fixed_first = req["actions"][0]
    blocked = {"planner": 0, "fixed_order": 0}
    counts = [9, 9, 1, 1]
    for scenario, count in zip(req["scenarios"], counts):
        for method, action in (("planner", selected), ("fixed_order", fixed_first)):
            observed = copy.deepcopy(req["study"])
            observed["evidence"][action["evidence"]] = scenario["evidence"][action["evidence"]]
            assessment = assess_decisions(observed)
            if assessment["verdict"] == "BLOCK":
                blocked[method] += count
            assert assessment["execution_authorized"] is False
    assert blocked == {"planner": 10, "fixed_order": 2}


def test_committed_example_uses_invented_synthetic_study():
    path = Path(__file__).resolve().parents[1] / "examples" / "evidence-plan.json"
    req = parse_plan(path.read_bytes())
    result = plan_evidence(req)
    assert req == request()
    assert result["selected_action_id"] == "check-runtime"
    assert result["current_assessment"]["verdict"] == "REVIEW"
