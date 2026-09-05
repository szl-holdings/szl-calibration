# szl-calibration

**Calibration Intelligence Plane** — the bridge between the SZL formula corpus, governed
runtime, and observability. Exact calibration math on caller-supplied probabilities,
every scored batch receipted, Prometheus-native surface, fail-closed weight gates in CI.

Doctrine v11. Lambda = Conjecture 1 (advisory). Apache-2.0.

## Honesty labels

- **SOFTWARE.** Metrics are exact math on probabilities the caller supplies. No model
  weights are loaded by this service. No hardware, energy, or accuracy claims are made.
- Receipts are **UNSIGNED_HONEST**: a SHA-256 hash chain that proves integrity and order
  of the log, never the correctness of a score. Verification is offline and dependency-free.
- The safetensors gate is **fail-closed**: structural failures are BLOCK and cannot be
  waived; an unavailable NaN scan degrades to REVIEW, never to ALLOW.

## API

| Route | Purpose |
|---|---|
| `GET /healthz` | liveness + receipt-chain validity |
| `GET /metrics` | Prometheus exposition (requests, latency, ECE histogram) |
| `POST /v1/score` | ECE/MCE/Brier/log-loss/AUROC + receipt for a batch |
| `POST /v1/calibration/score` | backwards-compatible alias for the same scorer |
| `GET /v1/calibration/receipts` | full hash-chained receipt log (JSONL) |
| `GET /v1/receipts/verify` | chain validity; HTTP 503 if integrity fails |

Invalid batches return HTTP 422 and do not append a receipt. Receipts are stored
in memory for the current service process; restarting the process starts a new
chain. This service does not claim persistent or shared multi-worker history.

## Run

```bash
pip install -e '.[serve]'
uvicorn szl_calibration.service:app --host 127.0.0.1 --port 8080
```

## CI weight gate

```bash
python tests/make_fixtures.py
python -m szl_calibration.gate_cli tests/fixtures/tiny.safetensors --expect ALLOW
```

Exit codes: ALLOW 0 / BLOCK 1 / REVIEW 2 / expectation mismatch 3. Structure (header
framing, dtype/shape/offset consistency) is validated stdlib-only; NaN/Inf scanning uses
numpy when present and honestly degrades to REVIEW otherwise.

## Alerts

`deploy/grafana-alerts.yaml` ships p95-latency, ECE-drift, chain-broken (critical,
fail-closed), and memory alerts as a PrometheusRule-compatible manifest.
