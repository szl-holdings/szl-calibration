# Changelog — szl-calibration

All entries reference commits visible on `main`. Dates are UTC.

## 2026-09-03

### Added
- `OPERATIONS.md` — run/verify/deploy procedures with explicit fail-closed
  expectations and a do-not-ship rule (commit `e1f053ad`).

### Changed
- `LICENSE` — replaced the short stub with the canonical Apache-2.0 full text
  (11,342 bytes), matching the estate convention (commit `a38bdaf5`).

## Initial structure (pre-changelog)

- `src/szl_calibration/`: `metrics.py` (ECE, MCE, Brier, log loss, reliability
  bins), `receipts.py` (SHA-256 hash-chained UNSIGNED_HONEST receipts),
  `gates.py` (fail-closed safetensors validation: ALLOW / REVIEW / BLOCK),
  `service.py` (FastAPI surface), `gate_cli.py` (CLI), `__init__.py`.
- `tests/`: pytest suite plus safetensors fixture generator (clean / NaN /
  truncated).
- `.github/workflows/ci.yml`: CI gate on push.
- `deploy/`: `prometheus.yml`, `grafana-alerts.yaml`, `uvicorn.conf.py`.

---

Format: newest first. Every entry must name its commit SHA. No entry without a
landed commit. Doctrine v11 · Apache-2.0 · SZL Holdings.
