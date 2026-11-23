# Leader-Study Upgrade Plan - W27 loop-6 - 2026-11-23

Status: PROPOSE-ONLY DRAFT - DO NOT MERGE. Owner-gated: merging requires governing approval and the estate signature-wall owner unlock.

Repo: szl-calibration - Calibration Intelligence Plane — exact calibration metrics (ECE/MCE/Brier/NLL/AUROC), UNSIGNED_HONEST hash-chained receipts, fail-closed safetensors weight gates, FastAPI + Prometheus surface. Doctrine v11. SOFTWARE honesty. Apache-2.0.

Frame: OUROBOROS loop-6 leader study. Governing seat: Kimi kimi-k3.1-pro (json_ok, think off). Advisory seats: Astra/Sol/Claude community rails (provenance unverified, advisory only). Leader evidence web-verified 2026-11-23.

## Leader moats to absorb
- Expected Calibration Error (ECE) with honest binning-limitation disclosure: The repo's primary metric, computed exactly on caller-supplied probabilities with explicit disclosure of binning limitations. Moat: Exact math plus honesty labeling (SOFTWARE, UNSIGNED_HONEST) and a hash-chained receipt per scored batch; invalid batches return 422 and append no receipt.
- Reliability diagrams: Standard visualization companion to ECE for diagnosing miscalibration across confidence bins. Moat: Not evidenced as a route or artifact in the facts; the repo exposes ECE/MCE/Brier/log-loss/AUROC and a Prometheus ECE histogram instead.
- Temperature scaling (single-parameter post-hoc, Guo 2017 lineage): Post-hoc recalibration via a single temperature parameter fitted on validation data. Moat: No recalibration/fitting route is evidenced; the service scores caller-supplied probabilities and loads no model weights.
- Platt scaling vs isotonic regression tradeoffs: Parametric vs non-parametric post-hoc calibration methods with different data-size and overfitting tradeoffs. Moat: No fitting or comparison endpoint is evidenced; the repo's decision/evidence routes are proposal-only and make no model calls.
- Brier score, NLL, AUROC as complementary metrics: Proper scoring rules and ranking quality reported alongside ECE/MCE for a fuller calibration picture. Moat: Implemented together at POST /v1/score (and the backwards-compatible alias) with a receipt per batch and strict input rejection of coerced numeric strings and JSON booleans as proba.
- sklearn calibration_curve + PyTorch T recalibration recipes: Reference implementations/recipes for computing calibration curves and applying temperature scaling. Moat: No such recipes are evidenced in the facts; the repo's differentiators are receipts, fail-closed safetensors gates, and the FastAPI + Prometheus surface.

## Honest gaps (grounded in repo facts)
- pushed_at is null, so there is no evidence in the facts of any pushed code, release, or CI run behind the described surface.
- No recalibration capability is evidenced: the API lists scoring, receipts, decision, and evidence routes only, so temperature scaling, Platt scaling, and isotonic regression remain unimplemented as far as the facts show.
- Reliability diagrams are named as a leader but no route, schema, or artifact for per-bin diagram data appears in the API table.
- The ECE binning-limitation disclosure is asserted in the description, but the facts give no bin count, binning strategy, or empty-bin/edge policy, so reproducibility of ECE/MCE cannot be assessed.
- AUROC and log-loss edge-case behavior (single-class batches, proba exactly 0 or 1, empty bins) is undocumented beyond the general 422 rule for invalid batches.
- Receipt retention, persistence across restarts, and concurrency/ordering guarantees are not described, even though /healthz and /v1/receipts/verify depend on chain validity and return 503 on integrity failure.
- The safetensors weight gate is described as fail-closed in CI, but no gate route, policy artifact, or CI configuration is present in the facts, and the service itself loads no model weights.
- The OpenSSF Scorecard and PyPI badges are displayed, but the facts contain no scores, published versions, or verification results.
- Decision and evidence routes (/v1/decisions/assess, /v1/decisions/local, /v1/evidence/plan) are proposal-only with MODELED joint scenarios, but the facts do not specify input validation, receipting, or how MODELED outputs are marked to prevent being read as measured results.
- Doctrine v11 and Lambda = Conjecture 1 (advisory) are referenced without any facts defining their enforcement, scope, or relationship to the scored batches.

## Dosed upgrade plan
- Dose 1 (documentation-only, no behavior change): publish the exact ECE/MCE binning specification, metric edge-case semantics (single-class AUROC, log-loss clipping policy), and worked offline receipt-verification examples; explicitly restate that UNSIGNED_HONEST receipts prove integrity/order only.
- Dose 2 (additive, backwards-compatible): add optional per-bin reliability-diagram data to the score response or a new route, computed by the same exact math and covered by the same receipt; keep 422/no-receipt semantics for invalid batches and preserve the alias route.
- Dose 3 (proposal-only design): specify an offline recalibration study endpoint that accepts caller-supplied validation probabilities/labels and returns temperature (Guo 2017 lineage), Platt, and isotonic parameters plus before/after ECE/MCE/Brier/NLL/AUROC, marked SOFTWARE and advisory with no execution authorization and no model-weight loading.
- Dose 4 (governance evidence): publish the fail-closed safetensors gate policy and CI wiring as artifacts, including the BLOCK-vs-REVIEW matrix and the rule that an unavailable NaN scan degrades to REVIEW, never ALLOW; do not claim gate results until CI evidence exists.
- Dose 5 (operational clarity, propose before implementing): document receipt-log persistence, retention, and restart/concurrency behavior, plus Prometheus metric cardinality limits for the ECE histogram, and define /healthz and /v1/receipts/verify degradation semantics without weakening the 503 integrity-failure behavior.
- Dose 6 (validation): add property-based and golden-vector tests for all metrics and boundary rejections (coerced numeric strings, JSON booleans as proba), and only then consider publishing a PyPI release consistent with the existing badges.
- Risk note: Lowest risk is documentation and additive, receipted read-only outputs; moderate risk is recalibration and persistence work because new fitting paths, storage, and concurrency can silently change metric semantics or receipt ordering, and any overstatement beyond SOFTWARE honesty (or treating MODELED scenarios as measured evidence) would violate the repo's stated doctrine.
- Propose-only note: All steps are proposals grounded solely in the supplied facts; no repo files, CI runs, releases, or benchmark results were inspected, and nothing here authorizes execution, deployment, or any claim of measured accuracy improvement.

## Advisory notes (unverified rails, non-binding)
- astra: frontier-bench: hardware-specific kernel optimization suites retrieval-bench: industry-standard gold-set evaluation pipelines szl-calibration: probabilistic forecasting uncertainty baselines
- sol: frontier-bench: Hardware-specific kernel optimization profiles retrieval-bench: End-to-end latency-throughput Pareto frontiers szl-calibration: Domain-specific uncertainty-aware calibration baselines
- claude: frontier-bench: Hardware-specific kernel optimization profiles retrieval-bench: Domain-specific synthetic query generation pipelines szl-calibration: Out-of-distribution uncertainty quantification frameworks

## DO NOT MERGE
This plan is a proposal artifact of the 10-pass OUROBOROS loop (loop-6 of 10). No code, no CI, no behavior changes in this file. A reviewer with authority may convert to a work order later.