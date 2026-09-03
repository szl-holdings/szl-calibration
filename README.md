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
| `POST /v1/calibration/score` | ECE/MCE/Brier/log-loss/AUROC + receipt for a batch |
| `GET /v1/calibration/receipts` | full hash-chained receipt log (JSONL) |

## Run

```bash
pip install -e '.[serve]'
uvicorn szl_calibration.service:app --config deploy/uvicorn.conf.py
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
