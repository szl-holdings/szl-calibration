# Optional TypeSafe/Jev adapter

This is an original bounded client for the [documented TypeSafe HTTP API](https://docs.typesafe.ai/api).
It adds an optional provider interface to SZL software; it does not implement Jev's
proprietary model or RLCD training. No vendor SDK or remote agent skill is executed
or installed by this module. Source references were checked on 19 September 2026.

## Run

The included example contains only invented build metadata. Validate it offline:

```bash
python -m szl_calibration.jev_cli --request examples/jev-request.json
```

For one explicit live request, configure `TYPESAFE_API_KEY` through your local
environment or secret manager, then run:

```bash
python -m szl_calibration.jev_cli --request examples/jev-request.json --live --timeout 10 --receipt jev-observation.json
```

Do not put the key in the input file, a command argument or a committed artifact.
`--live` sends the provided state and questions to TypeSafe and may consume account
credit. Review the data first. An existing receipt file is never overwritten.
The CLI prints a stable error code on failure and exits 2. It exits 0 after offline
validation or a schema-valid provider result; neither implies empirical accuracy.

## Contract and limits

The initial version supports nonempty string instructions, Noul criteria with
`true`/`false` descriptions, Choice maps with 2–255 named options, and ordered Score
arrays with 2–255 levels. This is deliberately a subset of the vendor's broader
SDK schema. Requests have 1–128 questions and at most 256 KiB of JSON; this byte
limit is not a token-count guarantee. The provider's own token limits still apply.

Use an explicit `jev-x.y.z` model, such as the documented `jev-1.13.0`. Moving
aliases are rejected, and the returned model must match. Responses must contain
every requested answer and no extras. The adapter validates types, finite bounds,
exact probability keys, distribution sums within 1e-5, Choice winner consistency,
and Score legend/expectation consistency. It does not repair or normalize invalid
answers. Duplicate JSON keys and nonfinite JSON values are rejected.

Transport uses only `https://api.typesafe.ai/v1/systemone`, validates TLS, follows
no redirects, performs no retries and accepts at most 1 MiB of JSON response.
A bounded worker limits caller wait to the selected timeout (0.05–30 seconds).
An in-flight timeout is `provider_timeout_outcome_unknown`: a sent request may
still consume credit or finish remotely. Do not automatically repeat it. A stalled
platform DNS call may leave a daemon worker until it returns; this client is for
bounded operator requests, not a high-concurrency untrusted proxy.

Provider error bodies and credentials are not returned. A successful observation
records input/question hashes, the actual returned model, provider request ID when
well formed, and validated answers. It omits raw state. It is an
`UNSIGNED_CLIENT_OBSERVATION`, with `calibration_status: NOT_ASSESSED` and
`action_authorized: false`. Request IDs and hashes do not authenticate the provider
or the user's evidence. Keep observations within the applicable account agreement.

## Connecting to a decision study

Collect labeled examples separately from policy selection. For each Choice task,
use the answer's `probabilities` as one study row with the ground-truth class. A
Noul probability `p` can represent `{ "false": 1-p, "true": p }` with explicitly
matched labels. Score is an expected ordinal level; do not treat its scalar as a
probability of correctness. Preserve the entire distribution for a suitable task.

The adapter's `questions_sha256` binds its canonical UTF-8 JSON question definition;
use that value as the study's `question_revision`. Preserve exact model identity and
dataset revision. The separate `/v1/decisions/assess` route performs no provider
requests. See [decision study assumptions and limits](DECISION_STUDIES.md).

Software tests use invented responses and a fake TLS connection. They cover malformed
outputs, redirects, HTTP errors, timeout, absent credentials and offline defaults.
Passing them is not a live provider witness. No vendor-output distillation, model
training or public comparative benchmark is included. Review the applicable
[customer agreement](https://typesafe.ai/legal/mca) for those separate uses.
