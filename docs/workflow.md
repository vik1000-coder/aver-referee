# A decision from logs to a result

Aver compares agent options against mistake prices and limits you choose. Its
CLI reads files and computes a test; your runner executes the agent. Start with
the synthetic demo to learn the output before using real observations.

## Install and try it

Download the wheel from [GitHub Releases](https://github.com/vik1000-coder/aver-referee/releases),
then follow the virtual-environment installation in the [README](https://github.com/vik1000-coder/aver-referee/blob/main/README.md).
Python 3.11 or later is required. Activate that environment before changing
directories: both `aver` and `python3 -m aver_cli` must resolve to this install.
The core decision workflow has no runtime dependencies or network calls.

```bash
aver decision capabilities --json
aver decision demo --rounds 3 --out-dir decision-demo
aver decision explain --result decision-demo/decision.json
aver decision memo --result decision-demo/decision.json --monthly-fee 0 --out demo-memo.json
```

The demo supplies synthetic outcomes and example prices; it does not execute an
agent, validate your data or authorize a release. A pass and a shown saving are
separate results. `report.savings.status` can remain `estimated` after a pass:
the saving is shown only when its lower bound is strictly above zero at the
stopping round. The chance of meeting the limits and the chance of showing a
saving are separate pilot-based forecasts, not promised outcomes.

## Prepare a real decision

Use one directory per decision. Export logs from the options you are comparing
on the same cases. Keep recorded costs, option identities and configuration
digests; missing costs or unreadable outcomes must remain missing or unknown,
never be silently made into successes. `read-logs --help` lists supported file
formats and price-table flags. These are file readers, not hosted integrations.

```bash
mkdir decision-workspace
cd decision-workspace
# Copy your export to logs.jsonl (or logs.json for a trace export).
aver decision next --json
```

`next` reads the files and their provenance, then prints one step in `data.step`.
It never writes files, executes a command or calls a model. Inspect
`data.step.command`, `data.step.argv`, `data.step.template`, `data.step.why` and
`data.step.person_decides` before acting. A template needs your actual runner
or human choices filled in; it is not an executable default.

| `next` exit | What your agent does |
|---|---|
| 0 | Execute the returned command using the intended environment, then ask `next` again. |
| 1 | Show the reason and requested human input. Wait for it; never invent it. |
| 2 | Report the completed pass, fail or inconclusive result and its explanation. A person decides whether to switch. |
| 3 | Stop and show the inconsistency. Restore valid inputs or start a new workspace; never patch a recorded result. |

Do not apply those exit codes to every command. For `look` (alias `round`), 0 is
Passes, 1 is Keep testing, Inconclusive or audit required, 2 is Fails, and 3 is
invalid input. Other decision commands use 0 for success and 3 for refusal or
invalid input. After a non-error round, consult `next` to decide the next step.

## Six steps, with people in control

| Step | Files and responsibility |
|---|---|
| Logs | `read-logs` produces `proposal.json`. A person reviews its purpose, kinds of work and kinds of mistakes. |
| Worth | `accept` produces `spec.json` and `pilot.jsonl` using the person's prices, monthly limits, volume and checking cost. Its `--tolerances` flag means at most N mistakes of a kind per month. |
| Options | A random audit sample is drawn. A person independently labels it in `audit-labels.jsonl`; `audit` produces `audited-spec.json`. `explore` compares options on that working specification. |
| The test | `plan` prices the proposed rounds. Review the cases, budget and forecasts before `freeze`. An infeasible plan cannot be frozen. |
| Rounds | `cases` writes `round-N-cases.jsonl`; your runner writes `round-N.jsonl`. `look` reads it, with the previous result after round 1. `explain` and `bundle` package the finished answer. |
| Record | The local bundle and economic memo are unsigned. A separate hosted deployment can save an eligible pass on a human-approved plan as a signed record. |

The CLI never chooses the person's prices or limits, labels its own audit, or
approves the switch. Plan locking freezes the inputs; do not edit them, rerun a
recorded round, stop on a favorable unfinished result or reuse pilot cases as
confirmation cases. Start a new directory when the decision's terms change.

The audit calibrates completion labels. It does not bound every confusion
between incident types or kinds of mistakes. Limits on those categories still
depend on accurate classification and transferable audit assumptions. The 5%
whole-plan error limit also depends on the declared sampling and grading
assumptions. Use independently reviewed or fixed-rule outcomes where needed.

## Connect your runner

Put an executable command in `harness-command.txt`, with `{cases}`, `{out}` and
`{round}` placeholders. Your agent executes the resulting command. For example:

```text
python3 /absolute/path/to/your_runner.py --cases {cases} --out {out} --round {round}
```

The manifest assigns a `case_id`, `candidate_id`, `configuration_sha256` and
kind of work (`stratum`) to each execution. Draw fresh work from the intended
population; use the same case content for the baseline and every candidate in
a paired plan. Emit one `aver-outcomes/v2` row for every manifest entry. The
outcome contract is shown by `aver decision look --help`:

```json
{"schema":"aver-outcomes/v2","case_id":"confirmation-credit_note-0","case_sha256":"SHA256_OF_CASE_CONTENT","candidate_id":"guarded","configuration_sha256":"COPY_FROM_MANIFEST","stratum":"credit_note","phase":"confirmation","origin":"ai-labeled","completed":true,"incident":false,"cost_usd":0.003,"mistake":null}
```

This illustrates the fields, not valid evidence: replace the digest placeholders
with actual SHA-256 values and every outcome with an observation. Grade new
cases using the agreed procedure and origin. Include failed and unknown runs;
do not filter them out. Aver rejects incomplete, changed or out-of-order rounds.

## Use a chosen model to propose labels

Your own labeling agent can help turn logs into a proposal. It cannot choose
the verdict or substitute for the independent human audit.

```bash
aver decision label-request --logs logs.jsonl --model your-chosen-model --include-content --out label-request.json
```

Review and redact the request before sending only its `messages` and
`response_schema` to that agent. The redactor removes credential and economic
fields, but is not a general personal-data anonymizer. Without
`--include-content`, the model receives IDs only and should mark labels unsure.
Your provider credentials, authorization and spending stay with your agent.

Save the response as an envelope with `schema: aver-label-response/v1`, the
request's `request_digest` and `model`, and its structured output in `response`.
The response must satisfy the supplied schema and cover every blinded run ID.
`unsure: true` becomes unknown, never a successful run. The model name is
self-reported; the digests do not prove which model ran.

```bash
aver decision read-logs --logs logs.jsonl --reader model-response --label-request label-request.json --label-response label-response.json --out proposal.json
aver decision next --json
```

For token-only exports, pass the same `--prices` table to both commands. To
automate an already-reviewed handoff, put your labeling command in
`labeler-command.txt` using `{request}` and `{out}`. It must save that envelope.
`next` prints the command; Aver never sends the request itself.

## Hosted records and the economic memo

This repository contains the offline CLI and decision libraries. It contains no
hosted service, account system, payment processing or signing private keys.
Paid hosted plans currently remain demonstrations and collect no payment.

On a separate hosted deployment, a person first approves a plan for an App.
An App-bound token can then perform its authorized machine operations:
`submit_evidence`, `run_check`, `read_verdict`, `save_record`, `export_config`.
Human signoff, plan approval, revocation and superseding remain human actions.
Inspect that deployment's `/openapi.json` and MCP discovery for availability.

```bash
aver decision hosted-request --operation submit_evidence --app APP_ID --plan-id PLAN_ID --round 1 --evidence round-1.jsonl --out hosted-round-1.json
aver decision hosted-request --operation read_verdict --app APP_ID --plan-id PLAN_ID --out hosted-progress.json
```

Your agent sends the packet's `request` field to `/api/mcp` with its authorized
bearer token; it does not send the outer wrapper. Keep tokens in a secret
manager. The CLI does not upload or store them. Retry the same request after a
transport failure: its key binds the App, operation, plan and complete payload.
The server recomputes the rounds and refuses an ineligible Check or record.

`memo` recomputes a local result and records the test spend and monthly fee you
enter. It includes logged run costs and recorded per-case checking costs;
setup time and unlogged costs are extra. Missing cost remains unknown. It is
unsigned, and a fail or unshown saving supplies no positive economic claim.
A signed hosted record carries neither your prices nor savings. Local
`report.release_eligibility` remains `not_evaluated`.

For offline record verification, install the wheel's optional `verify` extra,
obtain an authenticated copy of the issuer's public JWKS, and run
`aver verify record.jws --jwks issuer-jwks.json`. A URL instead fetches the keys
over the network. A signature establishes integrity and key possession, not
the truth of observations. Check expiry and revocation separately; an offline
check cannot establish current revocation without a supplied status list.
