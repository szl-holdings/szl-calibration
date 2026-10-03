import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from szl_calibration import service
from szl_calibration.evidence_cli import main
from szl_calibration.receipts import ReceiptChain


def example():
    return json.loads((Path(__file__).parents[1] / "examples/evidence-plan.json").read_text(encoding="utf-8"))


@pytest.fixture
def api(monkeypatch):
    chain = ReceiptChain()
    monkeypatch.setattr(service, "CHAIN", chain)
    with TestClient(service.app) as client:
        yield client, chain


def test_http_plan_is_bound_receipted_and_never_authorizes(api):
    client, chain = api
    response = client.post("/v1/evidence/plan", json=example())
    assert response.status_code == 200
    body = response.json()
    assert body["plan"]["execution_authorized"] is False
    assert body["receipt"]["signature"] == "UNSIGNED_HONEST"
    retained = json.loads(chain.to_jsonl())
    assert retained["kind"] == "evidence.plan.v1"
    assert retained["payload"] == body["plan"]
    assert chain.verify()
    # Reading health or verification never creates a receipt.
    assert client.get("/healthz").status_code == 200
    assert client.get("/v1/receipts/verify").status_code == 200
    assert len(chain) == 1


@pytest.mark.parametrize("raw", ['{"budget_units":0,"budget_units":10}', '{"x":NaN}', '[', '{}'])
def test_invalid_plans_do_not_write_receipts(api, raw):
    client, chain = api
    response = client.post("/v1/evidence/plan", content=raw, headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert response.json()["detail"] == "invalid evidence plan"
    assert len(chain) == 0


def test_http_body_and_media_limits(api, monkeypatch):
    client, chain = api
    monkeypatch.setattr(service, "MAX_PLAN_BYTES", 100)
    assert client.post("/v1/evidence/plan", content="{}").status_code == 415
    assert client.post("/v1/evidence/plan", content=" " * 101,
                       headers={"Content-Type": "application/json"}).status_code == 413
    assert len(chain) == 0


def test_corrupt_chain_refuses_further_plans(api):
    client, chain = api
    assert client.post("/v1/evidence/plan", json=example()).status_code == 200
    chain._items[0].payload["execution_authorized"] = True
    assert client.post("/v1/evidence/plan", json=example()).status_code == 503
    assert len(chain) == 1


def test_cli_completes_calculation_without_authorizing(tmp_path, capsys):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(example()), encoding="utf-8")
    assert main([str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["plan"]["execution_authorized"] is False
    assert result["receipt"]["kind"] == "evidence.plan.v1"


def test_cli_invalid_input_has_stable_error_no_echo(tmp_path, capsys):
    path = tmp_path / "plan.json"
    path.write_text('{"private-payload":NaN}', encoding="utf-8")
    assert main([str(path)]) == 1
    output = capsys.readouterr().out
    assert "private-payload" not in output
    assert json.loads(output)["evidence_class"] == "BLOCKED"
