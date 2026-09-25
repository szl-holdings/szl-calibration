import copy
import json
import math

import pytest
from fastapi.testclient import TestClient

from szl_calibration.decisions import assess_decisions
from szl_calibration import service
from szl_calibration.receipts import ReceiptChain
from szl_calibration.decisions import parse_study
from szl_calibration.decision_cli import main as decision_main


def study(n=1000):
    return {
        "model_id": "own-model@" + "b" * 40, "task_id": "route-triage-v1",
        "question_revision": "c" * 64,
        "dataset_revision": "a" * 64, "label_kind": "OBSERVED",
        "held_out": True, "policy_frozen": True, "classes": ["review", "route"],
        "policy": {"threshold": .95, "max_error": .05, "alpha": .05,
                   "min_accepted_per_group": 100,
                   "required_groups": ["normal"], "required_evidence": ["source_verified"]},
        "evidence": {"source_verified": "PASS"},
        "samples": [{"id": str(i), "group": "normal",
                     "probabilities": {"review": .01, "route": .99}, "label": "route"}
                    for i in range(n)],
    }


def test_known_bound_and_brier():
    result = assess_decisions(study())
    assert result["verdict"] == "ELIGIBLE_FOR_SHADOW"
    assert result["execution_authorized"] is False
    assert result["groups"][0]["error_upper_bound"] == pytest.approx(math.sqrt(math.log(20) / 2000))
    assert result["metrics"]["multiclass_brier"] == pytest.approx(.0002)
    assert result["metrics"]["top_label_brier"] == pytest.approx(.0001)


def test_high_confidence_small_sample_never_certifies():
    result = assess_decisions(study(2))
    assert result["verdict"] == "REVIEW"
    assert result["groups"][0]["error_upper_bound"] > .8


def test_empty_required_group_is_not_averaged_away():
    req = study()
    req["policy"]["required_groups"].append("adversarial")
    result = assess_decisions(req)
    assert result["verdict"] == "REVIEW"
    assert result["groups"][1]["error_upper_bound"] is None
    assert result["groups"][0]["alpha"] == .025


def test_wrong_high_confidence_group_cannot_hide_behind_good_majority():
    req = study(2000)
    req["policy"]["required_groups"].append("adversarial")
    for row in req["samples"][-100:]:
        row.update(group="adversarial", label="review")
    result = assess_decisions(req)
    assert result["groups"][0]["passes"] is True
    assert result["groups"][1]["selective_error"] == 1
    assert result["verdict"] == "REVIEW"


def test_ties_and_low_confidence_abstain_even_at_threshold():
    req = study(2)
    req["policy"]["threshold"] = .5
    req["samples"][0]["probabilities"] = {"review": .5, "route": .5}
    result = assess_decisions(req)
    assert result["abstained"] == 1
    req["policy"]["threshold"] = 1
    assert assess_decisions(req)["accepted"] == 0


@pytest.mark.parametrize("field,value", [("label_kind", "SYNTHETIC"),
                                         ("held_out", False), ("policy_frozen", False)])
def test_unverified_evaluation_design_remains_review(field, value):
    req = study()
    req[field] = value
    assert assess_decisions(req)["verdict"] == "REVIEW"


@pytest.mark.parametrize("status,verdict", [("FAIL", "BLOCK"), ("UNAVAILABLE", "REVIEW")])
def test_model_confidence_never_overrides_hard_evidence(status, verdict):
    req = study()
    req["evidence"]["source_verified"] = status
    assert assess_decisions(req)["verdict"] == verdict


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -.1, 1.1, ".99", None])
def test_malformed_probabilities_rejected(value):
    req = study(1)
    req["samples"][0]["probabilities"]["route"] = value
    with pytest.raises(ValueError):
        assess_decisions(req)


def test_distribution_shape_sum_and_duplicate_rejection():
    for mutate in (
        lambda r: r["samples"][0]["probabilities"].update(route=.8),
        lambda r: r["samples"][0]["probabilities"].update(unknown=0),
        lambda r: r["samples"].append(copy.deepcopy(r["samples"][0])),
        lambda r: r["samples"][0].update(label="unknown"),
        lambda r: r["samples"][0].update(group="unknown"),
        lambda r: r["policy"].update(min_accepted_per_group=True),
        lambda r: r.update(model_id="jev-latest"),
        lambda r: r.update(dataset_revision="main"),
    ):
        req = study(1)
        mutate(req)
        with pytest.raises(ValueError):
            assess_decisions(req)


def test_input_and_policy_binding_and_no_mutation():
    req = study(1)
    original = copy.deepcopy(req)
    first = assess_decisions(req)
    assert req == original
    req["samples"][0]["label"] = "review"
    second = assess_decisions(req)
    assert first["input_sha256"] != second["input_sha256"]
    assert first["policy_sha256"] == second["policy_sha256"]
    req["policy"]["threshold"] = .96
    assert assess_decisions(req)["policy_sha256"] != second["policy_sha256"]


def test_api_receipt_binds_study_and_corruption_fails_closed(monkeypatch):
    chain = ReceiptChain()
    monkeypatch.setattr(service, "CHAIN", chain)
    client = TestClient(service.app)
    response = client.post("/v1/decisions/assess", json=study(1))
    assert response.status_code == 200
    assert response.json()["receipt"]["signature"] == "UNSIGNED_HONEST"
    assert chain.verify()
    chain._items[0].payload["verdict"] = "FAKE_PASS"
    assert client.post("/v1/decisions/assess", json=study(1)).status_code == 503
    assert len(chain) == 1


def test_invalid_api_request_leaves_receipts_unchanged(monkeypatch):
    chain = ReceiptChain()
    monkeypatch.setattr(service, "CHAIN", chain)
    client = TestClient(service.app)
    req = study(1)
    req["samples"][0]["probabilities"]["route"] = True
    assert client.post("/v1/decisions/assess", json=req).status_code == 422
    assert len(chain) == 0


@pytest.mark.parametrize("model", ["jev-latest ", "latest", "JEV-LATEST", "main", "model:preview"])
def test_common_aliases_and_whitespace_are_rejected(model):
    req = study(1)
    req["model_id"] = model
    with pytest.raises(ValueError):
        assess_decisions(req)


def test_tie_break_defines_metrics_but_never_acceptance():
    req = study(3)
    req["classes"] = ["review", "route", "abstain"]
    for row in req["samples"]:
        row["probabilities"] = {"review": .4, "route": .4, "abstain": .2}
        row["label"] = "review"
    first = assess_decisions(req)
    req["classes"] = ["route", "review", "abstain"]
    second = assess_decisions(req)
    assert first["accepted"] == second["accepted"] == 0
    assert first["metrics"]["top_label_brier"] == pytest.approx(.36)
    assert second["metrics"]["top_label_brier"] == pytest.approx(.16)
    assert first["metrics"]["multiclass_brier"] == second["metrics"]["multiclass_brier"]


def test_duplicate_fail_pass_evidence_rejected_by_cli_and_http(monkeypatch, tmp_path, capsys):
    chain = ReceiptChain()
    monkeypatch.setattr(service, "CHAIN", chain)
    client = TestClient(service.app)
    raw = json.dumps(study()).replace('"source_verified": "PASS"',
                                      '"source_verified": "FAIL", "source_verified": "PASS"')
    with pytest.raises(ValueError, match="duplicate"):
        parse_study(raw.encode())
    response = client.post("/v1/decisions/assess", content=raw, headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert len(chain) == 0
    path = tmp_path / "duplicate.json"
    path.write_text(raw, encoding="utf-8")
    assert decision_main([str(path)]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "INVALID_STUDY"


def test_content_type_and_stream_body_limits(monkeypatch):
    monkeypatch.setattr(service, "CHAIN", ReceiptChain())
    monkeypatch.setattr(service, "MAX_STUDY_BYTES", 100)
    client = TestClient(service.app)
    assert client.post("/v1/decisions/assess", content="{}").status_code == 415
    assert client.post("/v1/decisions/assess", content=" " * 101,
                       headers={"Content-Type": "application/json"}).status_code == 413


def test_nonfinite_raw_json_rejected_before_evaluation():
    with pytest.raises(ValueError, match="non-finite"):
        parse_study(b'{"arbitrary":NaN}')
