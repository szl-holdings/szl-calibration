# OPERATIONS — szl-calibration

How to run, verify, and deploy the calibration intelligence plane.
Honesty note: this document describes procedure only. Operational claims are valid
only after the verification steps below pass on the target machine.

## Layout

- `src/szl_calibration/metrics.py` — ECE, MCE, Brier score, log loss, reliability bins
- `src/szl_calibration/receipts.py` — hash-chained UNSIGNED_HONEST receipts (SHA-256)
- `src/szl_calibration/gates.py` — fail-closed safetensors validation (ALLOW / REVIEW / BLOCK)
- `src/szl_calibration/service.py` — FastAPI surface (health, score, receipts, metrics)
- `src/szl_calibration/gate_cli.py` — CLI entry for the weight gate
- `tests/` — pytest suite + safetensors fixture generator
- `deploy/` — `prometheus.yml`, `grafana-alerts.yaml`, `uvicorn.conf.py`
- `.github/workflows/ci.yml` — CI gate on push

## Prerequisites

- Python 3.11 or 3.12
- `pip install -e '.[dev]'` from the repository root (installs test/service tools,
  the `szl_calibration` package
  and runtime dependencies declared in `pyproject.toml`)

## Verify (do this first on any new machine)

1. `python tests/make_fixtures.py` — build the clean / NaN / truncated safetensors fixtures.
2. `python -m pytest tests/ -q` — full suite must be green.
3. Expected fail-closed behavior, asserted by the suite:
   - clean fixture → ALLOW
   - NaN/Inf weights → BLOCK
   - truncated header → BLOCK
   - receipt chain verification detects any tampered link

If any of these expectations fail, the deployment is not operational. Do not ship.

## Run the service

```
uvicorn szl_calibration.service:app --host 0.0.0.0 --port 8080
```

Production settings live in `deploy/uvicorn.conf.py`. Endpoints:

- `GET /healthz` — liveness
- `POST /v1/score` — calibration metrics for probability/label arrays
- `POST /v1/calibration/score` — compatible alias for existing clients
- `GET /v1/receipts/verify` — hash-chain integrity check
- `GET /metrics` — Prometheus exposition

## Observability

- Prometheus: `prometheus --config.file=deploy/prometheus.yml` (pre-wired scrape target)
- Grafana: provision alert rules from `deploy/grafana-alerts.yaml`
  (latency, calibration-drift, receipt-integrity, memory)

## CI

`.github/workflows/ci.yml` runs the test suite on main pushes and pull requests.
Require successful checks for the proposed commit before merging. Branch
protection is a separate repository setting; the workflow alone does not enforce it.

Receipts are process-local and reset on restart. The gate proves this mathematical
service on the observed machine, not hosted deployment, model quality, clinical
validity, durable storage, or external Prometheus/Grafana provisioning.

## Doctrine

Fail closed. Never fabricate a metric, a signature, or a passing gate.
Λ = Conjecture 1 (advisory). Apache-2.0. SZL Holdings.
