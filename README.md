# aver

Choose a cheaper agent configuration with your own mistake prices and limits.

## Cheaper or more reliable: decide in your own terms

`aver decision` reads your AI pipeline's logs, works out what the agent is for and
the kinds of cases and mistakes, and lets you put a **price** on each mistake that
has one and a **limit** ("at most N a month", called a tolerance in CLI flags)
on the rest. It compares each option's monthly running cost plus priced mistakes, shows the price of
the test before it runs, and tests in rounds. The answer is **Passes**, **Fails**,
**Keep testing** or **Inconclusive**. A pass bounds the chance of wrongly passing
an option outside the limits at 5% over the whole plan, under the declared
sampling, grading and audit assumptions. It is not a promise about every run.

```bash
aver decision demo --scenario document-processing --out-dir decision-demo
aver decision explain --result decision-demo/decision.json
aver decision capabilities --json
```

The demo is synthetic. Method `aver-decision-binary-v3.1`; the CLI makes no
network calls in this workflow and never calls a model.

### Run it from your own logs (or let a coding agent run it)

One decision per directory. `aver decision next --json` prints the next command
and stops where a person must decide: the prices and tolerances, and the audit.

```bash
aver decision read-logs --logs logs.jsonl --out proposal.json
aver decision accept --proposal proposal.json --prices wrong_field=2 --tolerances unwanted_write=450 \
  --failure-limit 3.2% --monthly-uses 30000 --price-per-case 2 --out spec.json --pilot-out pilot.jsonl
aver decision audit --proposal proposal.json --spec spec.json --out audit-sample.json   # a person labels it
aver decision audit --spec spec.json --sample audit-sample.json --labels audit-labels.jsonl --out audited-spec.json
aver decision explore --spec audited-spec.json --pilot pilot.jsonl --out exploration.json
aver decision plan --spec audited-spec.json --exploration exploration.json --finalists 1 --rounds 3 --out plan.json
aver decision freeze --plan plan.json --out frozen.json
aver decision cases --plan frozen.json --round 1 --out round-1-cases.jsonl   # your harness runs these
aver decision look --plan frozen.json --evidence round-1.jsonl --out result-1.json
aver decision explain --result result-1.json --out explanation.json
```

Labels from the log reader need an independent human audit of a random sample
(cases the reader called fine, and cases it flagged). This calibrates completion
labels; it does not bound all confusion between kinds of mistakes or incidents.
Those limits still require accurate classification and transferable audit
assumptions. An
infeasible plan cannot be frozen; pilot cases cannot count twice. `look` exits 0
for Passes, 1 for Keep testing, Inconclusive or audit required, 2 for Fails and 3
for invalid input.

Passing your limits and showing a positive saving are separate results.
`report.savings.status` describes whether the saving is estimated or shown;
"shown" requires a strictly positive saving lower bound at the stopping round.
A pass can leave the saving estimated. The test's planning forecasts are
estimates under the pilot and sampling assumptions, not promised outcomes.
`report.release_eligibility` remains `not_evaluated`: a local pass is not release
authorization. In the signed-in
product a pass on a plan a person approved can be saved as a signed record that
anyone can verify; the server recomputes every round before signing.

### Work with your labeling agent and the hosted product

`label-request` prepares a blinded request for your chosen model. Review its
content before giving `messages` and `response_schema` to that agent. Save the
answer with the matching request digest and reported model name, then validate
it with `read-logs --reader model-response`. A person still audits its labels.
The model name is reported by your agent, not independently attested.

After a person approves a hosted plan, `hosted-request` prepares a retry-safe
MCP request file. Your agent authenticates and sends its `request` field to the
deployed endpoint. The CLI does not upload the file or store a bearer token.
Hosted discovery and availability depend on that deployment.

`memo` recomputes a result and exports an unsigned economic memo with the
recorded testing spend, the saving bound and a monthly fee you enter. Setup
effort and unlogged costs are extra. Signed records carry neither prices nor
savings. Paid hosted plans currently remain demonstrations; no payment is
collected.

```bash
aver decision memo --result decision-demo/decision.json --monthly-fee 0 --out memo.json
```

For repeated-run behavioral comparisons, use the existing workflow below.

## Measure your agent's repeatability

Agent regression tests are flaky tests: the same agent on the same tasks
disagrees with itself, and a one-run diff reads that as a regression. Here is
`aver floor` on the bundled fixture's refund review tasks, each run five times:

<!-- AVER_FLOOR_BEGIN -->
<!-- Fixture transcript generated and checked upstream. -->

```text
$ aver demo grok-refund --evidence-out evidence.jsonl
wrote evidence.jsonl (1480 run(s) of 148 refund review task(s), both sides)
$ aver floor evidence.jsonl
repeat-churn floor: incumbent
------------------------------------------------------------------------
action          issue_refund
tasks / runs    148 / 740   unreadable 0
churn rate      4.6%   [3.3%, 6.4%] Wilson
undecided at 0.80 6 task(s)   (4.1%)
byte-stable     120 task(s) gave the same answer every run
propensity      0.0-0.2   44  0.2-0.4   10  0.4-0.6    5  0.6-0.8    1  0.8-1.0   88
regime          noisy

this agent is stable on 96% of tasks; repeat churn 4.6% [3.3, 6.4].

stability map   per task, runs that withheld issue_refund (n/k); decided at 0.80
  stable-withholds     88 task(s)   reliably withholds issue_refund
  stable-fires         54 task(s)   reliably fires issue_refund
  coin-flip             6 task(s)   undecided: the runs do not agree

coin-flip tasks: the prompt and tools leave what this agent does here
under-determined. Fix these whatever model you run.
  tkt_00001    2/5
  tkt_00033    2/5
  tkt_00099    2/5
  tkt_00180    2/5
  tkt_00192    2/5
  tkt_00222    3/5

one-run diff    run 0 against run 1 of this same agent, on the same tasks
withheld then fired  8/89 = 9.0%   [4.6%, 16.7%]
fired then withheld  11
tolerance       5.0%
rule            point estimate vs tolerance (interval shown, not used)

a one-run diff of this agent against itself would report: BLOCK (8/89 = 9.0%)
Nothing changed between those two runs: that is this agent's own churn,
not a regression.
------------------------------------------------------------------------
next            aver check evidence.jsonl --policy aver.yaml referees a change
```

<!-- AVER_FLOOR_END -->

These numbers are not claims. They are the output of checked-in code on a
checked-in fixture of decision tuples from a repeat study of two models; the two
models in the fixture are named in the data and nowhere else. No prompts,
completions or tokens are in the package.

## Run it on your agent

Two commands. The first one runs **your** command, not ours: `aver repeat`
collects your harness's evidence, calls no model provider and scores no output.

```bash
# 1. Run each task five times. Your harness prints one evidence line per run;
#    aver collects. Resumable -- re-run it and it fills the gaps.
aver repeat --cmd "./run-one-task.sh" --tasks tasks.jsonl -k 5 \
            --side incumbent --out evidence.jsonl

# 2. How often does your agent disagree with itself, and on which tasks?
aver floor evidence.jsonl
```

A guarded action is any yes/no fact about a run that your harness can print: a
tool call fired, a file outside scope was touched, a destructive command ran, an
escalation happened. The evidence line names it, and `aver` reads whatever name
you give it:

```json
{"schema": "aver-evidence/v1", "task_id": "task_017", "side": "incumbent", "run": 0,
 "facts": {"ran_destructive_git_command": {"fired": false, "count": 0}}}
```

Already have traces? Import them instead of re-running anything:

```bash
aver import traces.json --format auto --actions issue_refund,escalate \
            --incumbent-model my-old-model --candidate-model my-new-model
```

`--format` takes `otel`, `langsmith`, `langfuse`, `promptfoo`, `jsonl`, or `auto`.
Those are **export formats**, not integrations: `aver` reads a file you already
have and does not talk to any service.

## What the stability map means

**Churn.** Your agent, run twice on the same input, does not always decide the
same way. `aver floor` measures how often, with a Wilson interval. This is a fact
about your agent, not about this tool, and it is the number that decides whether
any comparison you make is readable. An agent whose churn interval reaches past
your tolerance cannot be refereed on one run per task.

**The stability map.** Every task gets one of three labels, from the runs of that
task alone. *stable-withholds*: the runs agree the action is withheld.
*stable-fires*: they agree it fires. *coin-flip*: they do not agree enough to say
-- by default a task is decided at 4 of 5 runs, 3 of 3, or 8 of 10, so a 3-2 split
is a coin flip. `n/k` is how many of the task's readable runs withheld the action.
`aver floor` lists every coin-flip task by id; `--all` lists every task.

**Why the coin flips matter first.** A coin-flip task is one where the prompt and
tools leave what the agent does under-determined. Swapping the model does not fix
that, and comparing two models on it compares two coins. Fix those tasks whatever
model you run. They are counted as undecided, listed, and reported -- never
dropped quietly to make a denominator look better.

**The one-run diff line.** `aver floor` also shows what a single golden-trace
diff would say if you pointed it at the agent's first recorded run and its second:
the same agent, the same tasks, nothing changed. Whatever it reports is not a
regression; it is how much of any one-run diff on your agent is the agent
disagreeing with itself. Look at what it reports on the fixture above.

## Next: referee a change with `aver check`

Once the floor is known, add a candidate's runs to the same file and write your
acceptance rules down. `aver check` then compares the two sides using every
repeat, with an interval and a tolerance a person wrote down. It rules, or it
refuses and says what evidence is missing.

```bash
aver repeat --cmd "./run-one-task.sh" --tasks tasks.jsonl -k 5 \
            --side candidate --out evidence.jsonl
aver check evidence.jsonl --policy aver.yaml
```

On the bundled fixture, against the second of the two models in the fixture:

<!-- AVER_CHECK_BEGIN -->
<!-- Fixture transcript generated and checked upstream. -->

```text
Only now is the candidate compared: every repeat on both sides, an
interval, and a tolerance a person wrote down.

regime          this agent is stable on 96% of cases; repeat churn 4.6% [3.3, 6.4] — noisy
------------------------------------------------------------------------
action          issue_refund
  clause A  modal switch             7/85        [4.0%, 16.0%]   refuse_straddle
  clause B  expected harm      7.0% of 88        [1.9%, 12.9%]   refuse_straddle

------------------------------------------------------------------------
REFUSE

what to do next
  - 7 of 85 tasks where the current agent reliably withholds 'issue_refund' now reliably fire it. This needs a person: review these tasks and either change the candidate or record that the change is acceptable and re-approve the acceptance rules.
    cases: tkt_00012, tkt_00018, tkt_00108, tkt_00123, tkt_00132, tkt_00199, tkt_00239
  - The rise in how often 'issue_refund' fires straddles your 5% tolerance and the tasks average 5.0 runs per side. At small repeat counts this measure is biased upward (by up to 10 percentage points at 3 runs), so run each task 10 times per side and re-check.
    count: 10
```

<!-- AVER_CHECK_END -->

**Clause A, the modal switch.** Among the tasks where the incumbent *reliably*
withholds the guarded action, on what fraction does the candidate now *reliably*
take it? That ratio gets an interval. If the whole interval is inside your
tolerance, it passes; if the whole interval is above it, it blocks; if the
interval straddles your tolerance, the honest answer is that this evidence
cannot tell, and you get told how much more would settle it.

**Clause B, expected harm.** A candidate can keep every modal decision and still
become much more likely to take the action. Clause B measures how much of the
incumbent's withholding the candidate gave up, across the guarded tasks, with a
bootstrap interval. Zero under no change, negative if the candidate is stricter.
Set `harm_tolerance: null` on an action to switch it off, and your policy will
say that you did.

**Cures.** A refusal is not a failed build and it is never a shrug. Every refusal
carries what to do about it: which cases a person should look at, how many more
guarded cases would settle the interval, or that the repeats need raising because
the agent is near a coin flip. A refusal with no cure is a bug in this tool.

### Acceptance rules

`aver.yaml` is the rules, written down, by a person:

```yaml
schema: aver-policy/v3
decision_point: "refund review"
actions:
  - name: issue_refund
    direction: fire_is_harmful      # taking the action is the harmful event
    tolerance: 0.05                 # Clause A
    harm_tolerance: 0.05            # Clause B; null switches it off
    min_evidence: 5
  - name: escalate
    direction: withhold_is_harmful  # NOT taking the action is the harmful event
    tolerance: 0.05
decidability_threshold: 0.8
max_undecided_fraction: 0.25
confidence: 0.95
```

A key `aver` does not implement is a refusal, not a warning. A rule this tool
silently stored would be reported back to you as applied.

### Exit codes

| Code | Meaning | In CI |
|---|---|---|
| 0 | pass | green |
| 1 | refuse, with cures | **neutral, not red** -- the evidence cannot answer yet |
| 2 | block | red |
| 3 | usage or parse error | red |

Mapping 1 to pass defeats the point of the tool. A CI wrapper should report
refusal separately and require more evidence before granting a release.

## What this does not do

It does not call model providers or replay your traffic.
It does not enforce anything at run time. `repeat` and `check` read the outcomes
your harness records.
`decision read-logs` can propose labels from exported runs or validate your
labeling agent's saved response; those proposals need an independent human
audit. They are not an independent judge of output quality.

It also does not tell you that your agent is safe. It answers one narrow
question about one decision point, one guarded action, the tasks you gave it and
the two sides you compared, under rules you wrote. Everything outside that is
outside the answer.

## Signed records

`aver check --show-hosted` adds one line saying where a signed assurance record
could be issued from the same evidence. It is off by default.

### Verify a signed record, offline

If someone hands you a signed Aver record and an authenticated copy of the
issuer's public key set, you can check the signature without an account or a
network call:

```bash
aver verify record.jws --jwks issuer-jwks.json
```

It prints four facts on four lines, and it never lets one stand in for another:

- **signature** — the bytes are intact and were signed by a key that key set
  publishes. That is arithmetic, and it is all it is.
- **currency** — a record is current only while it is not expired, superseded,
  revoked or erased. Expiry is inside the record; revocation is not, so an
  offline check reports it as **not checked** unless you pass `--status-list`.
- **human review** — whether a named person ratified the acceptance rules.
- **release eligibility** — whether this record is a basis for a release at all.
  A demonstration over synthetic evidence verifies perfectly and authorises
  nothing.

To fetch keys instead, pass the issuer's `/api/verify/jwks.json` URL; that makes
a network request. Establish that the key set belongs to the intended issuer.
A valid signature does not establish that its underlying observations are true.

Ed25519 verification needs the `verify` extra. Without it, `aver verify` still
reads the record and reports the other three facts, and says plainly that the
signature was not checked.

## Install

Download the wheel from [GitHub Releases](https://github.com/vik1000-coder/aver-referee/releases).

Python 3.11 or later. The command is `aver`; the distribution is `aver-referee`.
The core CLI has no runtime dependencies. This project is not published to a
package index; install its wheel rather than an unrelated package named `aver`.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install /path/to/aver_referee-0.1.0-py3-none-any.whl
aver decision capabilities --json
aver decision demo --rounds 3 --out-dir decision-demo
```

These shell examples target macOS/Linux. On Windows, activate
`.venv\Scripts\Activate.ps1` in PowerShell instead.
Keep the environment activated when entering your decision directory: commands
printed by `next` use `python3 -m aver_cli`. If Windows has no `python3` alias,
execute the returned arguments with your activated environment's Python path.
A source checkout also supports
`python -m aver_cli` from its root.

To build your own wheel from this standalone source root:

```bash
python -m pip install build
python -m build --wheel
python -m pip install dist/aver_referee-0.1.0-py3-none-any.whl
```

For optional Ed25519 verification, install the same wheel with its extra:
`python -m pip install "/path/to/aver_referee-0.1.0-py3-none-any.whl[verify]"`.
See [the workflow guide](docs/workflow.md) and [contributing](CONTRIBUTING.md).


## Licence

Apache-2.0.
