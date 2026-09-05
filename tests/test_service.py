"""Exercise the documented HTTP contract, including rejected batches."""
import math

import pytest
from fastapi.testclient import TestClient

from szl_calibration import service
from szl_calibration.receipts import ReceiptChain


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(service, "CHAIN", ReceiptChain())
    with TestClient(service.app) as client:
        yield client


@pytest.mark.parametrize("route", ["/v1/score", "/v1/calibration/score"])
def test_score_routes_match_hand_checked_reference(client, route):
    response = client.post(route, json={"probabilities": [0.1, 0.9], "labels": [0, 1]})
    assert response.status_code == 200
    result = response.json()
    assert result["metrics"] == pytest.approx({
        "ece": 0.1, "mce": 0.1, "brier": 0.01,
        "log_loss": -math.log(0.9), "auroc": 1.0, "n": 2,
    })
    assert result["receipt"]["signature"] == "UNSIGNED_HONEST"
    assert client.get("/v1/receipts/verify").json() == {
        "count": 1, "chain_valid": True, "storage": "PROCESS_LOCAL",
    }
    assert client.get("/healthz").status_code == 200
    assert "szl_calibration_requests_total" in client.get("/metrics").text


@pytest.mark.parametrize("body", [
    {"probabilities": [0.1], "labels": [0, 1]},
    {"probabilities": [1.1], "labels": [1]},
    {"probabilities": [0.1], "labels": [2]},
    {"probabilities": [], "labels": []},
    {"probabilities": [0.1], "labels": [0], "n_bins": 0},
])
def test_invalid_batches_return_422_without_receipt(client, body):
    assert client.post("/v1/score", json=body).status_code == 422
    assert len(service.CHAIN) == 0


def test_verification_detects_tampered_receipt(client):
    client.post("/v1/score", json={"probabilities": [0.1, 0.9], "labels": [0, 1]})
    service.CHAIN._items[0].payload["metrics"]["ece"] = 0
    assert client.get("/v1/receipts/verify").status_code == 503
