# Budgeted evidence acquisition

`plan_evidence` proposes one missing evidence check to collect next. It ranks
affordable checks by their expected reduction in uncertainty about the eventual
decision verdict, divided by declared cost. The calculation is **MODELED**. It
does not call a provider, obtain evidence, spend budget, change a study, or
authorize execution. The current decision remains the result of
`assess_decisions` on the supplied study; modeled outcomes never enter that study.

This module belongs to the calibration/evidence planning layer. A controller can
use the proposal to request an external observation through its existing governed
action path. The controller must validate the actual observation and refresh the
source, policy, budget and evidence snapshot before another planning call.

## Run the invented example

From the repository root after installing the package:

```powershell
python -m szl_calibration.evidence_cli examples/evidence-plan.json
```

CLI exit code 0 means the local calculation completed. The output contains an
`UNSIGNED_HONEST` process-local receipt, `evidence_class: MODELED` and
`execution_authorized: false`. A missing proposal is a valid calculation result.
Exit code 1 means invalid input. No API key is needed.

The HTTP integration accepts the same JSON:

```powershell
python -m uvicorn szl_calibration.service:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
$planBody = Get-Content -Raw examples/evidence-plan.json
Invoke-RestMethod -Uri http://127.0.0.1:8000/v1/evidence/plan -Method Post -ContentType application/json -Body $planBody
```

The service appends a receipt on this POST only. This endpoint does not execute
the selected acquisition. Service authentication and production deployment are
outside this change.

## Input contract

The top-level object has exactly these fields:

| Field | Meaning |
|---|---|
| `schema` | Exactly `szl.evidence.plan.request.v1` |
| `probability_basis` | Exactly `MODELED` |
| `study` | An unchanged [decision study](DECISION_STUDIES.md) request |
| `budget_units` | Integer from 0 to 1,000,000; caller defines the unit |
| `scenarios` | 1–256 declared joint possible evidence states |
| `actions` | 0–64 available acquisitions; each targets one required evidence item |

Each scenario has exactly `id`, `probability`, and `evidence`. Its evidence map
contains every `study.policy.required_evidence` name with a `PASS` or `FAIL`
outcome. Probabilities must be finite and strictly positive, and sum to one
within an absolute tolerance of 1e-9. Scenario IDs and complete evidence
assignments must be unique. Joint states can encode correlations; independence
between required evidence items is not assumed.

Each action has exactly:

| Field | Meaning |
|---|---|
| `id` | Unique nonempty trimmed action name |
| `evidence` | A required evidence name; targets must be unique across actions |
| `cost_units` | Positive integer at most 1,000,000, in the budget's unit |
| `requires_pass` | Unique names of other required evidence items |
| `unavailable_probability` | Declared probability, from 0 to 1, that acquisition returns `UNAVAILABLE` |

An acquisition is modeled as a truthful observation of its target's scenario
state, except that it can return `UNAVAILABLE`. Unavailability is assumed
independent of the scenario. Noisy or adversarial instruments and alternative
tests for the same evidence item are outside this version's model.

The parser rejects duplicate JSON fields, NaN, infinity, and documents larger
than 8 MiB. Booleans do not count as numbers. All object shapes are exact; unknown
fields are refused. Invalid studies are refused by the existing decision study
validator. Scenario support that contradicts all supplied observations is
refused; the planner does not invent a posterior for an impossible observation.

## Calculation and selection

Known `PASS`/`FAIL` evidence first conditions the declared joint scenarios. The
output reports the retained prior mass, so excluded scenario mass is visible.
The remaining weights are explicitly normalized by that mass.

Let `V` be the eventual verdict after all required evidence is supplied, keeping
the study's samples and policy fixed. A scenario containing any `FAIL` yields
`BLOCK`. For an all-PASS scenario, `assess_decisions` determines whether the
study remains `REVIEW` or reaches `ELIGIBLE_FOR_SHADOW`. In particular, synthetic
labels, missing held-out assertions, or failed statistical gates remain review
conditions even inside the modeled all-PASS branch.

For action `a`, the score is:

```text
H(V) = -sum_v p(v) log2 p(v)
gain(a) = H(V) - sum_o p(o | a) H(V | o, a)
score(a) = gain(a) / cost_units(a)
```

The output includes each action's three possible outcomes, their modeled
probabilities and conditional verdict distributions. A zero-probability outcome
has a null posterior. The `UNAVAILABLE` outcome retains the prior distribution;
it is not evidence of failure or success. Tiny negative numerical gain is
clamped to zero.

Only actions targeting currently `UNAVAILABLE` evidence, with all prerequisites
currently `PASS`, within budget, and with some modeled chance of returning an
observation can be proposed. A current `BLOCK` prevents all proposals.
Prerequisites use supplied evidence, never modeled likelihoods.

Ranking uses gain per cost, then direct evidence resolution probability per cost,
then lower cost, then lexicographic action ID. The first two quantities are
rounded to 12 decimal places for deterministic tie handling. If modeled entropy
is zero, a missing obligation can still be proposed with reason
`REQUIRED_EVIDENCE_PENDING`: modeled certainty never substitutes for an actual
required observation. Each call selects at most one action. It does not reserve
the declared cost or optimize a multi-step schedule.

`input_sha256` binds the complete request, `study_sha256` binds the exact study,
`scenario_sha256` binds the declared model, and `policy_sha256` uses the existing
decision-study policy binding. These are content bindings, not signatures,
provenance attestations or proofs of freshness. The current assessment and
modeled distributions remain separate output fields.

The declared `trust_ceiling` is 0.97; the planner emits no trust estimate.
Probability 1 in a mathematical scenario model is not trust, correctness,
certification or permission. `execution_authorized` is always false.

## Reproducible simulated comparison

`examples/evidence-plan.json` contains only invented data. Both actions cost one
unit, and the budget is one. A source check passes in 18 of 20 hypothetical
cases; a runtime check fails in 10 of 20. The joint scenario counts are 9
both-pass, 9 runtime-fails, 1 source-fails and 1 both-fail. The fixed-order
baseline takes the first listed action, the source check. The planner chooses
the runtime check because its declared outcomes carry more information about
the final verdict.

The deterministic experiment enumerates those counts without a random draw:

```powershell
python -m pytest tests/test_evidence_planner.py -q
python -m pytest tests/test_evidence_planner.py -k twenty_case -q
```

Under this **SIMULATED** example, the planner exposes an explicit blocking
failure in 10/20 cases within the one-unit budget; the fixed-order baseline
exposes one in 2/20. The other cases remain `REVIEW`; none authorizes execution.
These figures are assertions in the test, derived from the invented scenario
counts. They are not a held-out evaluation, a TypeSafe benchmark, or a forecast
of operational improvement. Reversing the baseline order removes this example's
advantage. Misspecified probabilities can make the ranking worse.

## Prior art and limits

Shannon entropy and information gain are established methods; see
[Shannon, *A Mathematical Theory of Communication* (1948)](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf).
Adaptive acquisition has substantial prior art, including
[Golovin and Krause, *Adaptive Submodularity*](https://arxiv.org/abs/1003.3967).
This implementation does not establish adaptive submodularity or claim its
approximation guarantees. It uses a bounded, one-step heuristic over declared
joint scenarios; long-term dependencies and actions that unlock future checks
can make another schedule better.

The feature contributes an explicit proposal contract connecting missing
evidence, declared uncertainty, cost, prerequisites and the existing decision
assessment. It does not establish a new information-theoretic result. Scenario
weights require independent validation before any operational benefit is
claimed. Evidence authenticity, source freshness, concurrent budget reservation,
provider execution, durable receipts and controller admission remain separate
responsibilities.
