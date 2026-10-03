# szl-calibration

[![PyPI](https://img.shields.io/pypi/v/szl-calibration)](https://pypi.org/project/szl-calibration/) [![Python](https://img.shields.io/pypi/pyversions/szl-calibration)](https://pypi.org/project/szl-calibration/) [![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/szl-holdings/szl-calibration/badge)](https://scorecard.dev/viewer/?uri=github.com/szl-holdings/szl-calibration)

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
| `POST /v1/decisions/assess` | source-bound offline decision study, per-group risk and abstention; no execution authorization |
| `POST /v1/evidence/plan` | rank missing evidence acquisitions under a budget using explicitly MODELED joint scenarios; proposal only |

Invalid batches return HTTP 422 and do not append a receipt. Receipts are stored
in memory for the current service process; restarting the process starts a new
chain. This service does not claim persistent or shared multi-worker history.

Receipt payloads are detached from both caller inputs and returned receipt objects.
Non-finite JSON values are rejected. A broken chain makes `/healthz` and both
scoring routes return HTTP 503; no new receipt is appended. Append verifies the
existing chain while holding its lock, so verification cost grows with history.
The hash chain detects changes relative to retained receipts; it is unsigned and
does not prevent a privileged writer from replacing or truncating the entire log.

## Run

```bash
pip install -e '.[serve]'
uvicorn szl_calibration.service:app --host 127.0.0.1 --port 8080
```

## Typed decisions and optional Jev integration

[Decision studies](docs/DECISION_STUDIES.md) reuse the existing calibration metrics
for labeled multiclass predictions, bind the exact cohort and frozen policy in an
unsigned receipt, and measure selective error for every declared cohort. Missing
evidence and small samples remain `REVIEW`; failed prerequisites remain `BLOCK`.
The highest result is `ELIGIBLE_FOR_SHADOW`, never deployment or action authority.

The optional [Jev adapter](docs/JEV_ADAPTER.md) uses the documented TypeSafe API with
an explicit model version, strict response validation and a bounded request. It
does not change the service into a provider proxy. The included synthetic example
can be assessed offline without a key:

```bash
python -m szl_calibration.decision_cli examples/decision-study.json
```

The [evidence acquisition planner](docs/EVIDENCE_PLANNER.md) chooses one affordable
next observation using expected final-verdict entropy reduction per cost. Joint
scenarios retain caller-declared correlations; unavailable observations preserve
uncertainty. Existing failed evidence blocks proposals, and prerequisites must be
present before an acquisition is eligible. All probabilities remain **MODELED**;
the calculation never collects evidence, spends the budget, or authorizes action.

```bash
python -m szl_calibration.evidence_cli examples/evidence-plan.json
```

This command completes without a provider key. Exit zero means the planning
calculation completed; inspect the proposal and actual current assessment before
deciding what separately governed observation to collect.

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
