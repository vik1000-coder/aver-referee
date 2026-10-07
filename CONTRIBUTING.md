# Contributing

Use Python 3.11 or later. The core CLI has no runtime dependencies. The optional
`verify` extra supplies Ed25519 verification for signed records.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install build
python -m pip install '.[verify]'
python -m unittest discover -s aver_cli/tests -t . -p 'test_*.py' -v
python -m build --wheel
aver decision demo --rounds 3 --out-dir decision-demo
```

Run source tests from the checkout root. They use the selected synthetic log
fixtures under `demo_data/agent_logs` as well as the bundled CLI fixtures.
Those logs and their hidden truth are demonstration data, not customer
observations. Three historical study tests skip when their private study output
is unavailable; that output is intentionally not part of this repository. Two
tests for an absent legacy engine also skip because that engine is present.

Also install the built wheel in a fresh environment, change to a directory
outside the source checkout, and run `aver decision capabilities --json` and
the decision demo. This checks that source imports are not hiding packaging
errors. The wheel's importer and demo fixtures are bundled; the separate source
log fixtures are needed for the complete source test suite.

Tests for a change should check meaningful behavior, including refusals,
missing data and invalid provenance where relevant. Keep prices, limits, audit
labels and switching decisions under the person's control. Preserve local
results as engineering evidence, never as release authorization.

The README's worked repeatability transcript is generated from the bundled
fixture upstream. Reproduce the values using `aver floor` and `aver demo`
before proposing edits to those numbers. The method is shared with an upstream
TypeScript implementation; method changes require parity and separate
validation there before release.

Release maintainers create the public tree from a reviewed allowlist with fresh
Git history. `EXPORT_MANIFEST.json` records source revision and SHA-256 hashes
for that export; it is provenance, not a signature or a promise about later
contributor edits. Release assets are the source ZIP, wheel and SHA256SUMS.
No package-index publishing workflow is enabled. Do not add private product
files, real customer logs, environment files, credentials or signing keys.

Changes are contributed under the repository's
[Apache-2.0 license](https://github.com/vik1000-coder/aver-referee/blob/main/LICENSE).
