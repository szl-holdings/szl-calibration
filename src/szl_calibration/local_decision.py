"""Type one finite local question from a caller-supplied distribution.

This is not a provider client. It does not call OpenAI, NVIDIA, Ollama, or any
other model. A probability, a choice, and a score are measurements of the
supplied table. None of them grants execution, merge, or publication.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

from .receipts import canonical_json

SCHEMA = "szl.local-decision/v1"
ANSWER_SCHEMA = "szl.local-decision-answer/v1"
BACKEND = "recording-table"
MAX_LOCAL_BYTES = 1_000_000
_ID = r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}"


def parse_local_decision(raw: bytes) -> dict:
    """Reject duplicate fields and non-finite numbers before typing."""
    if len(raw) > MAX_LOCAL_BYTES:
        raise ValueError("local decision exceeds 1 MiB")

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
        value = json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)
    except RecursionError as exc:
        raise ValueError("local decision nesting exceeds parser limit") from exc
    if not isinstance(value, dict):
        raise ValueError("local decision must be an object")
    return value


def question_revision(question: dict) -> str:
    """64-hex digest of the frozen question, suitable as a study question_revision."""
    return hashlib.sha256(canonical_json(question).encode("ascii")).hexdigest()


def answer_local_question(request: dict) -> dict:
    """Return a typed answer or a refusal. execution_authorized is always false."""
    if not isinstance(request, dict) or set(request) != {"schema", "backend", "question", "distribution"}:
        raise ValueError("local decision must have exactly the documented fields")
    if request["schema"] != SCHEMA:
        raise ValueError("schema is not admitted")
    backend = request["backend"]
    if not isinstance(backend, str) or not backend.strip() or len(backend) > 128:
        raise ValueError("backend must be a nonempty string of at most 128 characters")
    question = _question(request["question"])
    base = _base(question, backend)
    if backend != BACKEND:
        return _refusal(base, "BACKEND_NOT_ADMITTED")
    if not str(question["evidence"]).strip():
        return _refusal(base, "EVIDENCE_ABSENT")
    distribution = request["distribution"]
    if not isinstance(distribution, dict):
        raise ValueError("distribution must be an object")
    kind = question["type"]
    if kind == "predicate":
        return _predicate(base, distribution)
    if kind == "choice":
        return _choice(base, question, distribution)
    return _score(base, question, distribution)


def _question(value: dict) -> dict:
    if not isinstance(value, dict):
        raise ValueError("question must be an object")
    kind = value.get("type")
    if kind not in ("predicate", "choice", "score"):
        raise ValueError("question type is not admitted")
    fields = {"type", "name", "instructions", "evidence"}
    if kind == "choice":
        fields = fields | {"choices"}
    elif kind == "score":
        fields = fields | {"levels"}
    if set(value) != fields:
        raise ValueError("question must have exactly the documented fields")
    _identifier(value["name"], "name")
    _text(value["instructions"], "instructions", 4000)
    if not isinstance(value["evidence"], str) or len(value["evidence"]) > 100_000:
        raise ValueError("evidence must be a string of at most 100000 characters")
    if kind == "choice":
        _options(value["choices"], "choices", "value")
    elif kind == "score":
        _options(value["levels"], "levels", "label")
    return value


def _identifier(value: object, name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(_ID, value) is None:
        raise ValueError(f"{name} must be a bounded ASCII identifier")


def _text(value: object, name: str, limit: int) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{name} must be a nonempty string of at most {limit} characters")


def _options(value: object, name: str, key: str) -> None:
    if not isinstance(value, list) or not 2 <= len(value) <= 32:
        raise ValueError(f"{name} must contain 2 to 32 entries")
    seen = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {key, "description"}:
            raise ValueError(f"{name} entries must have exactly {key} and description")
        _text(item[key], key, 128)
        _text(item["description"], "description", 500)
        seen.append(item[key])
    if len(set(seen)) != len(seen):
        raise ValueError(f"{name} entries must be unique")


def _probability(value: object, name: str) -> float:
    if type(value) is bool or type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return number


def _distribution(value: dict, keys: list[str]) -> dict[str, float]:
    if set(value) != set(keys):
        raise ValueError("distribution keys must match the declared options")
    parsed = {key: _probability(value[key], "probability") for key in keys}
    if not math.isclose(math.fsum(parsed.values()), 1.0, abs_tol=1e-6, rel_tol=0):
        raise ValueError("probabilities must sum to one; no silent normalization")
    return parsed


def _base(question: dict, backend: str) -> dict:
    return {
        "schema": ANSWER_SCHEMA,
        "backend": backend,
        "backend_admitted": backend == BACKEND,
        "provider": "NONE",
        "model_called": False,
        "execution_authorized": False,
        "question_type": question["type"],
        "name": question["name"],
        "question_sha256": question_revision(question),
        "evidence_sha256": hashlib.sha256(question["evidence"].encode("utf-8")).hexdigest(),
    }


def _refusal(base: dict, reason: str) -> dict:
    answer = dict(base)
    answer["type"] = "refusal"
    answer["reason"] = reason
    return answer


def _predicate(base: dict, distribution: dict) -> dict:
    if set(distribution) != {"probability"}:
        raise ValueError("predicate distribution must contain only probability")
    answer = dict(base)
    answer["type"] = "predicate"
    answer["probability"] = _probability(distribution["probability"], "probability")
    return answer


def _choice(base: dict, question: dict, distribution: dict) -> dict:
    keys = [item["value"] for item in question["choices"]]
    parsed = _distribution(distribution, keys)
    top = max(parsed.values())
    winners = [key for key in keys if parsed[key] == top]
    if len(winners) != 1:
        answer = _refusal(base, "TIE")
        answer["probabilities"] = parsed
        return answer
    answer = dict(base)
    answer["type"] = "choice"
    answer["choice"] = winners[0]
    answer["confidence"] = top
    answer["probabilities"] = parsed
    return answer


def _score(base: dict, question: dict, distribution: dict) -> dict:
    labels = [item["label"] for item in question["levels"]]
    parsed = _distribution(distribution, labels)
    score = math.fsum(index * parsed[label] for index, label in enumerate(labels))
    answer = dict(base)
    answer["type"] = "score"
    answer["score"] = score
    answer["confidence"] = max(parsed.values())
    answer["probabilities"] = [
        {"label": label, "value": index, "probability": parsed[label]}
        for index, label in enumerate(labels)
    ]
    return answer
