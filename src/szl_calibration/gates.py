"""Fail-closed safetensors weight-validation gate.

ALLOW  = structure valid AND NaN/Inf scan clean.
REVIEW = structure valid but scan UNAVAILABLE (numpy missing or no float tensors).
BLOCK  = any structural violation. Structure failures are never waived.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field

DTYPE_SIZE = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4,
              "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
FLOAT_DTYPES = {"F64", "F32", "F16", "BF16"}

ALLOW, REVIEW, BLOCK = "ALLOW", "REVIEW", "BLOCK"


@dataclass
class GateReport:
    verdict: str
    reasons: list[str] = field(default_factory=list)
    tensors: int = 0
    parameters: int = 0
    scan: str = "UNAVAILABLE"

    def to_json(self) -> str:
        return json.dumps({"verdict": self.verdict, "reasons": self.reasons,
                           "tensors": self.tensors, "parameters": self.parameters,
                           "scan": self.scan}, sort_keys=True)


def validate_safetensors(path: str, max_bytes: int = 64 * 1024**3) -> GateReport:
    rep = GateReport(verdict=BLOCK)
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        rep.reasons.append(f"unreadable: {exc}")
        return rep
    if len(raw) > max_bytes:
        rep.reasons.append(f"file exceeds max_bytes={max_bytes}")
        return rep
    if len(raw) < 9:
        rep.reasons.append("file too small for safetensors framing")
        return rep
    (n,) = struct.unpack("<Q", raw[:8])
    if n == 0 or n > len(raw) - 8:
        rep.reasons.append(f"header length {n} inconsistent with file size {len(raw)}")
        return rep
    try:
        header = json.loads(raw[8:8 + n].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        rep.reasons.append(f"header not valid UTF-8 JSON: {exc}")
        return rep
    buf = raw[8 + n:]
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dt = meta.get("dtype")
        shape = meta.get("shape") or []
        offs = meta.get("data_offsets")
        if dt not in DTYPE_SIZE:
            rep.reasons.append(f"{name}: unsupported dtype {dt!r}")
            return rep
        if not (isinstance(offs, list) and len(offs) == 2 and offs[0] <= offs[1] <= len(buf)):
            rep.reasons.append(f"{name}: data_offsets {offs!r} outside buffer {len(buf)}")
            return rep
        count = 1
        for d in shape:
            if not isinstance(d, int) or d < 0:
                rep.reasons.append(f"{name}: invalid shape {shape!r}")
                return rep
            count *= d
        if count * DTYPE_SIZE[dt] != offs[1] - offs[0]:
            rep.reasons.append(f"{name}: shape*dtype size != byte span")
            return rep
        rep.tensors += 1
        rep.parameters += count
    if rep.tensors == 0:
        rep.reasons.append("no tensors in header")
        return rep
    try:
        import numpy as np
    except ImportError:
        rep.verdict = REVIEW
        rep.reasons.append("numpy unavailable: NaN/Inf scan skipped, fail-closed to REVIEW")
        return rep
    bad = 0
    scanned = False
    for name, meta in header.items():
        if name == "__metadata__" or meta["dtype"] not in FLOAT_DTYPES:
            continue
        np_dt = {"F64": "<f8", "F32": "<f4", "F16": "<f2", "BF16": "u2"}[meta["dtype"]]
        arr = np.frombuffer(buf, dtype=np_dt, count=(meta["data_offsets"][1] - meta["data_offsets"][0]) // DTYPE_SIZE[meta["dtype"]], offset=meta["data_offsets"][0])
        scanned = True
        if meta["dtype"] == "BF16":
            arr = (arr.astype("u4") << 16).view("<f4")
        bad += int(np.isnan(arr).sum() + np.isinf(arr).sum())
    if not scanned:
        rep.verdict = REVIEW
        rep.scan = "NO_FLOAT_TENSORS"
        rep.reasons.append("no floating tensors to scan")
        return rep
    rep.scan = "CLEAN" if bad == 0 else "CONTAMINATED"
    if bad:
        rep.reasons.append(f"{bad} NaN/Inf values in floating tensors")
        return rep
    rep.verdict = ALLOW
    return rep
