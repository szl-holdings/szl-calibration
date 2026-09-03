"""FastAPI surface: /healthz, /metrics, /v1/calibration/score, /v1/calibration/receipts.

Every scored call appends an UNSIGNED_HONEST receipt. Metrics are Prometheus-native.
"""
from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from . import __version__, metrics as M
from .receipts import ReceiptChain

logging.basicConfig(level=logging.INFO, format='{"ts":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}')
log = logging.getLogger("szl.calibration")

app = FastAPI(title="szl-calibration", version=__version__)
CHAIN = ReceiptChain()

REQS = Counter("szl_calibration_requests_total", "Scored requests", ["route", "status"])
LAT = Histogram("szl_calibration_request_seconds", "Request latency", ["route"],
                buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5))
ECE_G = Histogram("szl_calibration_ece", "Observed ECE per scored batch",
                  buckets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5))


@app.middleware("http")
async def observe(request: Request, call_next):
    rid = request.headers.get("x-request-id", str(uuid.uuid4()))
    start = time.perf_counter()
    resp = await call_next(request)
    dt = time.perf_counter() - start
    REQS.labels(request.url.path, resp.status_code).inc()
    LAT.labels(request.url.path).observe(dt)
    log.info('{"rid":"%s","path":"%s","status":%d,"seconds":%.6f}', rid, request.url.path, resp.status_code, dt)
    resp.headers["x-request-id"] = rid
    return resp


class ScoreRequest(BaseModel):
    probabilities: list[float] = Field(min_length=1)
    labels: list[int] = Field(min_length=1)
    n_bins: int = Field(default=10, ge=1, le=1000)
    model_id: str | None = None


@app.get("/healthz")
def healthz():
    return {"status": "ok", "version": __version__, "receipts": len(CHAIN), "chain_valid": CHAIN.verify()}


@app.get("/metrics", response_class=PlainTextResponse)
def prom():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/calibration/score")
def score(req: ScoreRequest):
    out = {
        "ece": M.expected_calibration_error(req.probabilities, req.labels, req.n_bins),
        "mce": M.maximum_calibration_error(req.probabilities, req.labels, req.n_bins),
        "brier": M.brier_score(req.probabilities, req.labels),
        "log_loss": M.log_loss(req.probabilities, req.labels),
        "n": len(req.probabilities),
    }
    try:
        out["auroc"] = M.auroc(req.probabilities, req.labels)
    except ValueError:
        out["auroc"] = None  # single-class batch: honest null, never fabricated
    ECE_G.observe(out["ece"])
    r = CHAIN.append("calibration.score.v1", {"model_id": req.model_id, "metrics": out})
    return {"metrics": out, "receipt": {"index": r.index, "hash": r.hash, "prev_hash": r.prev_hash,
                                        "signature": r.signature}}


@app.get("/v1/calibration/receipts")
def receipts():
    return {"count": len(CHAIN), "chain_valid": CHAIN.verify(), "jsonl": CHAIN.to_jsonl()}
