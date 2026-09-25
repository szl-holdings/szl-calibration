# Typed decision studies

`POST /v1/decisions/assess` evaluates caller-supplied class distributions against
labeled examples, measures abstention, and appends an unsigned receipt. The route
does not make a provider call. The optional Jev CLI is a separate, explicit network
operation; see [JEV_ADAPTER.md](JEV_ADAPTER.md).

The highest study verdict is `ELIGIBLE_FOR_SHADOW`. This is permission to consider
an experiment, not an authorization to execute a tool, merge, publish, deploy,
control equipment, or make a consequential decision. Every result includes
`execution_authorized: false`. Run the service on loopback; authentication,
tenant isolation and deployment request-size limits must be supplied before
exposing this existing service beyond a trusted operator environment.

## Run the included synthetic example

```bash
pip install -e '.[dev]'
python -m szl_calibration.decision_cli examples/decision-study.json
uvicorn szl_calibration.service:app --host 127.0.0.1 --port 8080
```

In another terminal:

```bash
curl -H 'Content-Type: application/json' --data-binary @examples/decision-study.json http://127.0.0.1:8080/v1/decisions/assess
```

The example intentionally returns `REVIEW`: invented labels and tiny cohorts must
not produce a qualifying result. CLI exit codes are 0 for `ELIGIBLE_FOR_SHADOW`,
2 for a valid `REVIEW` or `BLOCK`, and 1 for an invalid study. HTTP 200 means a valid
assessment was produced; inspect its verdict. Invalid inputs return 422, oversized
bodies 413, unsupported content types 415, and a broken receipt chain 503. These
failures do not append a receipt. HTTP and CLI JSON ingestion reject duplicate
object keys, nonfinite numbers and bodies larger than 8 MiB.

## Exact input contract

All documented fields are required; unknown fields are rejected.

| Field | Meaning |
| --- | --- |
| `model_id` | Caller-asserted provider/model version or artifact identity; common moving aliases and surrounding whitespace rejected. This does not authenticate immutability or provider identity. |
| `task_id` | Versioned task name. |
| `question_revision` | Lowercase SHA-256 of the frozen instructions/question definition. |
| `dataset_revision` | Lowercase 40- or 64-hex immutable cohort revision. Syntactically checked, not fetched. |
| `label_kind` | `OBSERVED` or `SYNTHETIC`. Observed labels still need independent provenance. |
| `held_out`, `policy_frozen` | Boolean caller assertions about independent evaluation data and policy selection. |
| `classes` | Two to 255 unique outcome names. |
| `policy.threshold` | Prespecified maximum-class-probability threshold in [0.5,1]. |
| `policy.max_error` | Maximum accepted conditional error in [0,0.5]. |
| `policy.alpha` | Study-wide bound failure level, 0.000001 through 0.25. |
| `policy.min_accepted_per_group` | Integer 1 through 10,000. |
| `policy.required_groups` | Prespecified cohort names, including cohorts with zero observations. |
| `policy.required_evidence` | Named deterministic prerequisites. |
| `evidence` | Exactly those prerequisites, each `PASS`, `FAIL` or `UNAVAILABLE`. These are caller-supplied observations, never inferred from provider confidence. |
| `samples` | 1 through 10,000 `{id, group, probabilities, label}` records; at most 100,000 probability cells. |

Each distribution must have exactly the class keys and finite numeric values in
[0,1], summing to one within 1e-6. Booleans are not numbers. Duplicate sample IDs,
unknown groups/classes, missing keys and extra keys are rejected. The evaluator
does not normalize malformed output. Ties always abstain. Otherwise a sample is
accepted only when its maximum probability reaches the frozen threshold.

## What the mathematics says

For each required group, with `n` accepted examples, empirical error `r`, `G`
prespecified groups and overall failure level `alpha`, the reported upper bound is:

```text
U = min(1, r + sqrt(log(G / alpha) / (2 * n)))
```

This is a one-sided Hoeffding bound with a union-bound allocation over the declared
groups. Each group must independently meet the minimum count and `U <= max_error`.
An empty group has no numeric bound and cannot pass. A good majority group cannot
hide an unsafe or unmeasured minority group. [Hoeffding's original paper](https://doi.org/10.1080/01621459.1963.10500830).

The statistical interpretation requires independent, representative samples from
the relevant conditional population and a fixed classifier, prompt, threshold,
group definition and evaluation plan selected before observing evaluation labels.
The code cannot verify those assumptions. Deduplicating IDs does not eliminate
correlated content. Repeated adaptive studies on the same cohort need fresh data
or explicit multiple-testing/alpha accounting; a digest is not protection against
evaluation leakage. These are conservative finite-sample bounds, not estimates of
per-example correctness and not distribution-shift guarantees.

Multiclass Brier is the mean **sum** of squared class errors (not divided by the
number of classes); multiclass log loss uses the observed label's probability with
a 1e-15 floor. Existing binary metrics are reused to measure top-label confidence
against top-label correctness. ECE uses ten equal-width bins and does not itself
authorize a pass. Equal-probability ties are measured in top-label metrics
using the first tied class in the declared class order for the metric event, while
all ties abstain from selective decisions.

## Provenance and release boundaries

The receipt binds the canonical full input digest, frozen model/task/question/
class/policy digest, claimed dataset revision, group outcomes, deterministic
prerequisites and mathematical assumptions. It omits raw examples. It remains
`UNSIGNED_HONEST` and process-local, with all the existing chain limitations:
neither authentic label provenance nor persistent/multiworker evidence is created.

Synthetic labels, missing held-out/frozen assertions, unavailable prerequisites,
or insufficient group evidence give `REVIEW`. Any prerequisite explicitly marked
`FAIL` gives `BLOCK`, regardless of the probabilities.

For A11oyCode, Ayllu or the SZL router, start with owner-authored labeled proposal
tasks. Bind each study to the precise provider response version and prompt. Retain
failures, separate threshold selection from final evaluation, and observe results
in shadow mode before connecting an independent execution policy. A model's
reported confidence, a conformal-style rank, a formula proof or an unsigned receipt
must not be substituted for this missing empirical evidence.

Vendor-provided output is for application use under the applicable agreement.
This change neither trains a competing model nor publishes provider benchmarks.
Synthetic test fixtures are explicitly invented by SZL for software verification.
