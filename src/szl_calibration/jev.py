"""Optional, bounded TypeSafe transport; no network or credentials at import time.

This original adapter implements a conservative subset of the public v1 contract.
Provider predictions are not empirical calibration or authorization to act.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import math
import os
import queue
import re
import ssl
import threading
import time
from datetime import datetime, timezone
from typing import Any

HOST = "api.typesafe.ai"
PATH = "/v1/systemone"
PINNED_MODEL = "jev-1.13.0"
ADAPTER_VERSION = "szl-jev-1"
MAX_REQUEST_BYTES = 262_144
MAX_RESPONSE_BYTES = 1_048_576
_MODEL = re.compile(r"jev-(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\Z")


class JevError(ValueError):
    """A stable error code; never includes input, credentials or response bodies."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _fail(code: str) -> None:
    raise JevError(code)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate_json_key")
        result[key] = value
    return result


def _constant(_: str) -> None:
    _fail("non_finite_json_number")


def parse_json(raw: bytes | str) -> Any:
    """Strict JSON, including duplicate-key and NaN/Infinity rejection."""
    try:
        return json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant)
    except JevError:
        raise
    except (ValueError, UnicodeError, RecursionError):
        raise JevError("invalid_json") from None


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                          separators=(",", ":")).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise JevError("invalid_json_value") from None


def _keys(value: Any, required: set[str], optional: set[str] | None = None) -> None:
    if type(value) is not dict or not required <= value.keys() or value.keys() - required - (optional or set()):
        _fail("invalid_fields")


def _text(value: Any, maximum: int = 8192) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        _fail("invalid_text")


def _json_state(value: Any, depth: int = 0) -> None:
    if depth > 20:
        _fail("state_too_deep")
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            _fail("invalid_state_number")
        return
    if type(value) is list:
        for item in value:
            _json_state(item, depth + 1)
        return
    if type(value) is dict and all(isinstance(k, str) for k in value):
        for item in value.values():
            _json_state(item, depth + 1)
        return
    _fail("invalid_state")


def validate_request(request: Any) -> dict[str, Any]:
    """Validate and snapshot a conservative request. Does not read environment."""
    _keys(request, {"model", "state", "questions"})
    if not isinstance(request["model"], str) or not _MODEL.fullmatch(request["model"]):
        _fail("versioned_model_required")
    if type(request["state"]) not in (str, dict, list):
        _fail("invalid_state")
    _json_state(request["state"])
    questions = request["questions"]
    if type(questions) is not dict or not 1 <= len(questions) <= 128:
        _fail("invalid_question_count")
    for name, question in questions.items():
        _text(name, 128)
        _keys(question, {"type", "instructions"}, {"criteria"})
        _text(question["instructions"], 16384)
        kind = question["type"]
        if kind == "noul":
            criteria = question.get("criteria", {})
            _keys(criteria, set(), {"true", "false"})
            for value in criteria.values():
                if value is not None:
                    _text(value)
        elif kind == "choice":
            criteria = question.get("criteria")
            if type(criteria) is not dict or not 2 <= len(criteria) <= 255:
                _fail("invalid_choice_criteria")
            for label, value in criteria.items():
                _text(label, 256)
                if value is not None:
                    _text(value)
        elif kind == "score":
            criteria = question.get("criteria")
            if type(criteria) is not list or not 2 <= len(criteria) <= 255:
                _fail("invalid_score_criteria")
            for value in criteria:
                _text(value)
        else:
            _fail("unsupported_question_type")
    payload = canonical_bytes(request)
    if len(payload) > MAX_REQUEST_BYTES:
        _fail("request_too_large")
    return parse_json(payload)


def _number(value: Any, lower: float, upper: float) -> float:
    if type(value) not in (int, float):
        _fail("invalid_number")
    try:
        numeric = float(value)
    except (OverflowError, ValueError):
        raise JevError("invalid_number") from None
    if not math.isfinite(numeric) or not lower <= numeric <= upper:
        _fail("invalid_number")
    return numeric


def _distribution(value: Any, keys: set[str]) -> dict[str, float]:
    _keys(value, keys)
    probabilities = {key: _number(probability, 0.0, 1.0) for key, probability in value.items()}
    if not math.isclose(math.fsum(probabilities.values()), 1.0, rel_tol=0.0, abs_tol=1e-5):
        _fail("invalid_probability_sum")
    return probabilities


def validate_response(request: Any, response: Any) -> dict[str, Any]:
    """Reject malformed/incomplete answers; do not repair or normalize them."""
    request = validate_request(request)
    _keys(response, {"model", "answers", "usage"})
    if response["model"] != request["model"]:
        _fail("model_mismatch")
    _keys(response["answers"], set(request["questions"]))
    _keys(response["usage"], {"input_tokens", "output_tokens"})
    for value in response["usage"].values():
        if type(value) is not int or value < 0:
            _fail("invalid_usage")
    for name, question in request["questions"].items():
        answer = response["answers"][name]
        kind = question["type"]
        required = {"type", "noul"} if kind == "noul" else {"type", "confidence", "probabilities"}
        if kind == "choice":
            required.add("choice")
        elif kind == "score":
            required.update({"score", "legend"})
        _keys(answer, required)
        if answer["type"] != kind:
            _fail("answer_type_mismatch")
        if kind == "noul":
            _number(answer["noul"], 0.0, 1.0)
            continue
        _number(answer["confidence"], 0.0, 1.0)
        if kind == "choice":
            probabilities = _distribution(answer["probabilities"], set(question["criteria"]))
            choice = answer["choice"]
            if not isinstance(choice, str) or choice not in probabilities:
                _fail("invalid_choice")
            if not math.isclose(probabilities[choice], max(probabilities.values()), rel_tol=0.0, abs_tol=1e-6):
                _fail("choice_not_highest_probability")
        else:
            legend = {str(i): value for i, value in enumerate(question["criteria"])}
            if answer["legend"] != legend:
                _fail("score_legend_mismatch")
            probabilities = _distribution(answer["probabilities"], set(legend))
            score = _number(answer["score"], 0.0, float(len(legend) - 1))
            expected = math.fsum(int(i) * p for i, p in probabilities.items())
            if not math.isclose(score, expected, rel_tol=0.0, abs_tol=1e-5):
                _fail("score_expectation_mismatch")
    return parse_json(canonical_bytes(response))


def request_summary(request: Any) -> dict[str, Any]:
    request = validate_request(request)
    return {"model": request["model"], "question_count": len(request["questions"]),
            "request_sha256": hashlib.sha256(canonical_bytes(request)).hexdigest(),
            "questions_sha256": hashlib.sha256(canonical_bytes(request["questions"])).hexdigest()}


def evaluate(request: Any, *, timeout: float = 10.0,
             max_response_bytes: int = MAX_RESPONSE_BYTES) -> dict[str, Any]:
    """One explicit live attempt using only TYPESAFE_API_KEY and a fixed TLS host.

    A daemon worker bounds the caller's wait even if platform DNS/socket operations
    stall. A timeout is an unknown provider outcome and is never automatically retried.
    """
    request = validate_request(request)
    _number(timeout, 0.05, 30.0)
    if type(max_response_bytes) is not int or not 1 <= max_response_bytes <= MAX_RESPONSE_BYTES:
        _fail("invalid_response_limit")
    key = os.environ.get("TYPESAFE_API_KEY", "")
    if not key:
        _fail("missing_api_key")
    if len(key) > 4096 or any(ord(c) < 33 or ord(c) > 126 for c in key):
        _fail("invalid_api_key")
    payload = canonical_bytes(request)
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    cancelled = threading.Event()
    results: queue.Queue[Any] = queue.Queue(maxsize=1)
    connection_holder: list[Any] = []

    def perform() -> None:
        connection = None
        try:
            connection = http.client.HTTPSConnection(HOST, timeout=float(timeout), context=ssl.create_default_context())
            connection_holder.append(connection)
            connection.connect()
            if cancelled.is_set():
                return
            connection.request("POST", PATH, body=payload,
                               headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                                        "Accept": "application/json", "User-Agent": ADAPTER_VERSION})
            response = connection.getresponse()
            status = response.status
            if 300 <= status < 400:
                _fail("redirect_rejected")
            if status != 200:
                # A status is safe; an arbitrary server-provided error body is not.
                _fail("provider_http_" + str(status))
            content_type = response.getheader("Content-Type", "").split(";", 1)[0].strip().lower()
            if content_type != "application/json":
                _fail("unexpected_content_type")
            length = response.getheader("Content-Length")
            if length is not None:
                if not length.isdecimal() or int(length) > max_response_bytes:
                    _fail("response_too_large")
            chunks = bytearray()
            while not cancelled.is_set():
                chunk = response.read(min(16384, max_response_bytes + 1 - len(chunks)))
                if not chunk:
                    break
                chunks.extend(chunk)
                if len(chunks) > max_response_bytes:
                    _fail("response_too_large")
            if cancelled.is_set():
                return
            result = validate_response(request, parse_json(bytes(chunks)))
            # Returned text is confined to the requested IDs, labels, rubric and pinned model.
            # Redact a coincidental secret match instead of ever returning a bearer credential.
            if key in canonical_bytes(result).decode("utf-8"):
                _fail("credential_in_response")
            request_id = response.getheader("x-typesafe-request-id", "")
            if key in request_id or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", request_id):
                request_id = None
            results.put((result, request_id))
        except JevError as error:
            results.put(error)
        except Exception:
            # Never propagate HTTP-library exceptions, URLs, headers or server text.
            results.put(JevError("provider_transport_error"))
        finally:
            if connection is not None:
                connection.close()

    worker = threading.Thread(target=perform, name="szl-jev-request", daemon=True)
    worker.start()
    try:
        outcome = results.get(timeout=float(timeout))
    except queue.Empty:
        cancelled.set()
        # Interrupt the underlying socket where possible. A delayed DNS result checks
        # cancelled before sending. An already sent request can still consume credit.
        for connection in connection_holder:
            try:
                connection.close()
            except Exception:
                pass
        raise JevError("provider_timeout_outcome_unknown") from None
    if isinstance(outcome, JevError):
        raise outcome
    result, request_id = outcome
    return {"status": "LIVE_PROVIDER_RESPONSE_VALIDATED", "live_request_performed": True,
            "adapter_version": ADAPTER_VERSION, "recorded_at_utc": started_at,
            "elapsed_seconds": round(time.monotonic() - started, 6),
            **request_summary(request), "provider_request_id": request_id, "result": result,
            "receipt_authenticity": "UNSIGNED_CLIENT_OBSERVATION",
            "calibration_status": "NOT_ASSESSED", "action_authorized": False}
