# Contributing agents

This is the standalone Apache-2.0 Aver CLI, not the hosted product repository.
Read the [README](https://github.com/vik1000-coder/aver-referee/blob/main/README.md)
and [docs/workflow.md](docs/workflow.md), then run
`python3 -m aver_cli decision demo --rounds 3 --out-dir decision-demo`.

`aver_policy/` owns decision math, log readers and explanation;
`aver_cass/` owns CASS primitives; `aver_cli/` owns CLI commands and file workflows.
Tests are under `aver_cli/tests/`. Preserve schema identifiers, provenance,
round ordering and the documented human decisions. Never substitute mock
outcomes or automatic audit labels into a real decision.

Run the checks in [CONTRIBUTING.md](CONTRIBUTING.md) before proposing a change.
Core imports must remain standard-library-only plus these three packages;
signature verification alone uses the optional `verify` extra. No telemetry,
hidden network requests, provider SDKs or credentials belong here.

Decision methods are maintained with a TypeScript twin in the upstream private
product. A mathematical change requires matching upstream parity and method
validation before a public release; the CLI tests alone do not qualify it.
Propose it explicitly rather than silently changing a bound or error budget.
