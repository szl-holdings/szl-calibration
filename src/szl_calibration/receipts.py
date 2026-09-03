"""Hash-chained UNSIGNED_HONEST receipts. Proves integrity and order; never accuracy."""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

GENESIS = "SZL-CALIBRATION-GENESIS-V1"
UNSIGNED = "UNSIGNED_HONEST"


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass(frozen=True)
class Receipt:
    index: int
    timestamp_utc: str
    kind: str
    payload: dict
    prev_hash: str
    signature: str
    hash: str


def _digest(index: int, ts: str, kind: str, payload: dict, prev_hash: str, signature: str) -> str:
    body = canonical_json({"index": index, "timestamp_utc": ts, "kind": kind,
                           "payload": payload, "prev_hash": prev_hash, "signature": signature})
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


class ReceiptChain:
    """Append-only, thread-safe. Tamper-evident via prev_hash linkage."""

    def __init__(self, genesis: str = GENESIS):
        self._genesis = hashlib.sha256(genesis.encode()).hexdigest()
        self._items: list[Receipt] = []
        self._lock = threading.Lock()

    def append(self, kind: str, payload: dict) -> Receipt:
        with self._lock:
            prev = self._items[-1].hash if self._items else self._genesis
            ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
            h = _digest(len(self._items), ts, kind, payload, prev, UNSIGNED)
            r = Receipt(len(self._items), ts, kind, payload, prev, UNSIGNED, h)
            self._items.append(r)
            return r

    def verify(self) -> bool:
        prev = self._genesis
        for r in self._items:
            if r.prev_hash != prev:
                return False
            if _digest(r.index, r.timestamp_utc, r.kind, r.payload, r.prev_hash, r.signature) != r.hash:
                return False
            prev = r.hash
        return True

    def to_jsonl(self) -> str:
        return "\n".join(canonical_json(asdict(r)) for r in self._items)

    def __len__(self) -> int:
        return len(self._items)
