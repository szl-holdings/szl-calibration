import copy
import io
import json
import time

import pytest

from szl_calibration import jev
from szl_calibration.jev_cli import main


def request():
    return {"model": "jev-1.13.0", "state": {"synthetic": True}, "questions": {
        "yes": {"type": "noul", "instructions": "Is synthetic true?"},
        "route": {"type": "choice", "instructions": "Select a route.", "criteria": {"a": "First", "b": "Second"}},
        "grade": {"type": "score", "instructions": "Grade this state.", "criteria": ["Low", "High"]},
    }}


def response():
    return {"model": "jev-1.13.0", "usage": {"input_tokens": 12, "output_tokens": 0}, "answers": {
        "yes": {"type": "noul", "noul": .9},
        "route": {"type": "choice", "choice": "a", "probabilities": {"a": .8, "b": .2}, "confidence": .6},
        "grade": {"type": "score", "score": .75, "probabilities": {"0": .25, "1": .75},
                  "legend": {"0": "Low", "1": "High"}, "confidence": .5},
    }}


def transport(monkeypatch, *, status=200, headers=None, raw=None, delay=0):
    data = json.dumps(response()).encode() if raw is None else raw
    calls = []
    class Reply:
        def __init__(self):
            self.status = status
            self.buffer = io.BytesIO(data)
        def getheader(self, name, default=None):
            values = {"Content-Type": "application/json", "x-typesafe-request-id": "req-test"}
            values.update(headers or {})
            return values.get(name, default)
        def read(self, size):
            return self.buffer.read(size)
    class Connection:
        def __init__(self, host, **kwargs):
            assert host == "api.typesafe.ai"
            assert kwargs["context"].verify_mode != 0
        def connect(self):
            if delay:
                time.sleep(delay)
        def request(self, method, path, **kwargs):
            calls.append((method, path, kwargs))
        def getresponse(self):
            return Reply()
        def close(self):
            pass
    monkeypatch.setattr(jev.http.client, "HTTPSConnection", Connection)
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-credential-never-log")
    return calls


def test_all_primitives_valid_and_detached():
    original = response()
    result = jev.validate_response(request(), original)
    original["answers"]["yes"]["noul"] = .1
    assert result["answers"]["yes"]["noul"] == .9


@pytest.mark.parametrize("model", ["jev-latest", "jev-preview", "jev-1.13.0 ", "http://evil.test", None])
def test_moving_or_invalid_models_rejected(model):
    req = request()
    req["model"] = model
    with pytest.raises(jev.JevError, match="versioned_model_required"):
        jev.validate_request(req)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -1, 2, ".9", 10 ** 400])
def test_invalid_noul_values_rejected(value):
    rep = response()
    rep["answers"]["yes"]["noul"] = value
    with pytest.raises(jev.JevError):
        jev.validate_response(request(), rep)


@pytest.mark.parametrize("mutate", [
    lambda r: r.update(model="jev-1.14.0"),
    lambda r: r["answers"].pop("yes"),
    lambda r: r["answers"].update(extra={"type": "noul", "noul": .5}),
    lambda r: r["answers"]["route"].update(choice="b"),
    lambda r: r["answers"]["route"]["probabilities"].update(a=.5),
    lambda r: r["answers"]["grade"].update(score=.3),
    lambda r: r["answers"]["grade"].update(legend={"0": "Changed", "1": "High"}),
    lambda r: r["usage"].update(input_tokens=True),
    lambda r: r["answers"]["route"].update(confidence=float("nan")),
])
def test_malformed_answers_never_become_validated_evidence(mutate):
    rep = response()
    mutate(rep)
    with pytest.raises(jev.JevError):
        jev.validate_response(request(), rep)


@pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'not JSON'])
def test_wire_json_is_unambiguous(raw):
    with pytest.raises(jev.JevError):
        jev.parse_json(raw)


def test_transport_has_one_attempt_fixed_origin_and_no_state_in_receipt(monkeypatch):
    calls = transport(monkeypatch)
    result = jev.evaluate(request())
    assert len(calls) == 1
    assert calls[0][:2] == ("POST", "/v1/systemone")
    assert result["status"] == "LIVE_PROVIDER_RESPONSE_VALIDATED"
    assert result["action_authorized"] is False
    assert result["calibration_status"] == "NOT_ASSESSED"
    assert "state" not in result
    assert "test-credential-never-log" not in json.dumps(result)


@pytest.mark.parametrize("status,code", [(302, "redirect_rejected"), (401, "provider_http_401"), (429, "provider_http_429"), (500, "provider_http_500")])
def test_http_errors_not_retried_and_body_never_exposed(monkeypatch, status, code):
    calls = transport(monkeypatch, status=status, raw=b'private error body test-credential-never-log')
    with pytest.raises(jev.JevError) as error:
        jev.evaluate(request())
    assert str(error.value) == code
    assert len(calls) == 1


def test_missing_key_never_connects(monkeypatch):
    calls = transport(monkeypatch)
    monkeypatch.delenv("TYPESAFE_API_KEY")
    with pytest.raises(jev.JevError, match="missing_api_key"):
        jev.evaluate(request())
    assert calls == []


def test_size_content_type_and_duplicate_body_rejected(monkeypatch):
    for opts in ({"raw": b'x' * 1025}, {"headers": {"Content-Type": "text/html"}},
                 {"raw": b'{"model":"first","model":"last"}'},
                 {"headers": {"Content-Length": "99999999"}}):
        transport(monkeypatch, **opts)
        with pytest.raises(jev.JevError):
            jev.evaluate(request(), max_response_bytes=1024)


def test_timeout_records_unknown_outcome_and_does_not_send_after_delayed_connect(monkeypatch):
    calls = transport(monkeypatch, delay=.15)
    started = time.monotonic()
    with pytest.raises(jev.JevError, match="provider_timeout_outcome_unknown"):
        jev.evaluate(request(), timeout=.05)
    assert time.monotonic() - started < .5
    time.sleep(.2)
    assert calls == []


def test_cli_offline_and_atomic_receipt_never_overwrites(monkeypatch, tmp_path, capsys):
    calls = transport(monkeypatch)
    monkeypatch.delenv("TYPESAFE_API_KEY")
    source = tmp_path / "input.json"
    source.write_text(json.dumps(request()), encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    args = ["--request", str(source), "--receipt", str(receipt)]
    assert main(args) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "REQUEST_VALIDATED_OFFLINE"
    assert output["live_request_performed"] is False
    assert calls == []
    original = receipt.read_bytes()
    assert main(args) == 2
    assert receipt.read_bytes() == original
    assert json.loads(capsys.readouterr().err)["error"] == "receipt_write_failed"
