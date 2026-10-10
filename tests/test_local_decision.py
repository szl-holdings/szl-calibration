"""Finite local questions are typed from a supplied table. No model is called."""
import math

import pytest
from fastapi.testclient import TestClient

from szl_calibration import service
from szl_calibration.local_decision import answer_local_question, question_revision
from szl_calibration.receipts import ReceiptChain


def question(kind="choice", evidence="The package arrived with a broken screen."):
    body = {
        "type": kind,
        "name": "route",
        "instructions": "Select only from the declared options.",
        "evidence": evidence,
    }
    if kind == "choice":
        body["choices"] = [
            {"value": "review", "description": "Send the report to a person."},
            {"value": "hold", "description": "Do not select an effect."},
        ]
    elif kind == "score":
        body["levels"] = [
            {"label": "none", "description": "No damage is described."},
            {"label": "minor", "description": "Cosmetic damage is described."},
            {"label": "severe", "description": "Function is described as lost."},
        ]
    return body


def request(kind="choice", distribution=None, evidence="The package arrived with a broken screen.", backend="recording-table"):
    if distribution is None:
        distribution = {"review": 0.75, "hold": 0.25}
    return {
        "schema": "szl.local-decision/v1",
        "backend": backend,
        "question": question(kind, evidence),
        "distribution": distribution,
    }


def test_choice_uses_the_unique_supplied_winner_and_grants_nothing():
    answer = answer_local_question(request())
    assert answer["type"] == "choice"
    assert answer["choice"] == "review"
    assert answer["confidence"] == 0.75
    assert answer["execution_authorized"] is False
    assert answer["model_called"] is False
    assert answer["provider"] == "NONE"
    assert answer["question_sha256"] == question_revision(question())


def test_tie_refuses_instead_of_breaking_the_tie():
    answer = answer_local_question(request(distribution={"review": 0.5, "hold": 0.5}))
    assert answer["type"] == "refusal"
    assert answer["reason"] == "TIE"
    assert "choice" not in answer
    assert answer["execution_authorized"] is False


def test_score_is_the_probability_weighted_level_index():
    answer = answer_local_question(request(
        "score",
        {"none": 0.25, "minor": 0.5, "severe": 0.25},
    ))
    assert answer["score"] == 1.0
    assert answer["confidence"] == 0.5
    assert [row["value"] for row in answer["probabilities"]] == [0, 1, 2]


def test_predicate_returns_only_the_supplied_probability():
    answer = answer_local_question(request("predicate", {"probability": 0.95}))
    assert answer["type"] == "predicate"
    assert answer["probability"] == 0.95
    assert "choice" not in answer
    assert answer["execution_authorized"] is False


def test_missing_evidence_or_unadmitted_backend_refuses():
    empty = answer_local_question(request(evidence="  "))
    assert empty["reason"] == "EVIDENCE_ABSENT"
    foreign = answer_local_question(request(backend="gpt-6-luna"))
    assert foreign["reason"] == "BACKEND_NOT_ADMITTED"
    assert foreign["model_called"] is False


def test_malformed_distribution_is_rejected_without_normalization():
    with pytest.raises(ValueError, match="sum to one"):
        answer_local_question(request(distribution={"review": 0.7, "hold": 0.2}))
    with pytest.raises(ValueError, match="match the declared"):
        answer_local_question(request(distribution={"review": 1.0, "other": 0.0}))


def test_question_revision_changes_when_the_question_changes():
    first = question_revision(question())
    changed = question()
    changed["instructions"] = "Select a different declared option."
    assert question_revision(changed) != first
    assert len(first) == 64


def test_local_route_records_an_unsigned_receipt_and_rejects_junk(monkeypatch):
    chain = ReceiptChain()
    monkeypatch.setattr(service, "CHAIN", chain)
    client = TestClient(service.app)
    response = client.post("/v1/decisions/local", json=request())
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]["choice"] == "review"
    assert body["receipt"]["signature"] == "UNSIGNED_HONEST"
    assert chain.verify()
    assert client.post(
        "/v1/decisions/local",
        content=b"{}",
        headers={"Content-Type": "application/json"},
    ).status_code == 422
    assert client.post("/v1/decisions/local", content=b"{}").status_code == 415
    assert len(chain) == 1
    chain._items[0].payload["choice"] = "forged"
    assert client.post("/v1/decisions/local", json=request()).status_code == 503


def test_weighted_score_matches_the_declared_three_level_example():
    answer = answer_local_question(request(
        "score",
        {"none": 0.1, "minor": 0.7, "severe": 0.2},
    ))
    assert answer["score"] == pytest.approx(1.1)
    assert math.isfinite(answer["score"])
