"""Every command's exit code and --json shape, and the demo's numbers.

Exit codes are the contract a CI recipe binds to, so they are tested rather than
documented: 0 pass, 1 refuse, 2 block, 3 usage or parse error. The demo's numbers
are pinned against the study's own analysis script, ``analyze_real.py``, run on the
same data the fixtures are built from — if the CLI and that script ever disagree,
the README is lying about one of them. Since the owner's 2026-09-22 rerun filled
the 40 missing refund units, that is
``outputs/agent-referee-2026-09-22/real_arm_summary_complete.json`` (148 refund
tasks). ``real_arm_summary.json`` beside it is the frozen 128-task summary the
report was written from, and is not what the demo shows.
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from aver_cli import v3
from aver_cli.cli import (
    EXIT_BLOCK,
    EXIT_PASS,
    EXIT_REFUSE,
    EXIT_USAGE,
    HOSTED_POINTER,
    build_parser,
    main,
)
from aver_cli.demo import DEMOS, demo_cases, demo_policy, load_fixture
from aver_cli.evidence import ActionFacts, EvidenceLine, write_evidence
from aver_cli.referee import floor_readout, golden_diff, run_map

REPO_ROOT = Path(__file__).resolve().parents[2]
# updated after owner rerun 2026-09-22 (148 complete refund tickets): was
# real_arm_summary.json (128 tasks), which REPORT.md cites and stays frozen.
SUMMARY = REPO_ROOT / "outputs" / "agent-referee-2026-09-22" / "real_arm_summary_complete.json"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
# The study's summary lives in the source checkout, not in the wheel. Run from an
# installed wheel, the two tests that compare against it skip; the rest run.
STUDY_REASON = ("the study summary (outputs/agent-referee-2026-09-22/) is only in "
                "the source checkout, not in an installed wheel")

POLICY_YAML = """\
schema: aver-policy/v3
decision_point: "refund review"
actions:
  - name: issue_refund
    direction: fire_is_harmful
    tolerance: 0.05
    harm_tolerance: 0.05
    min_evidence: 5
  - name: escalate
    direction: withhold_is_harmful
    tolerance: 0.05
decidability_threshold: 0.8
max_undecided_fraction: 0.25
confidence: 0.95
"""

V3_REASON = ("needs aver_policy.repeated.compute_verdict_v3 (docs/103 T-D1), "
             "which is not in this build")


def run(argv: list[str]) -> tuple[int, str]:
    stream = io.StringIO()
    code = main(argv, stream=stream)
    return code, stream.getvalue()


def evidence_from_fixture(name: str = "grok-refund") -> str:
    """Turn a demo fixture into an aver-evidence/v1 file, as an importer would."""

    spec = DEMOS[name]
    fixture = load_fixture(spec)
    lines = []
    for case in fixture["cases"]:
        for side in ("incumbent", "candidate"):
            for index, decision in enumerate(case[side]):
                fired = decision == 0 if spec.direction == "fire_is_harmful" \
                    else decision == 1
                lines.append(EvidenceLine(
                    task_id=case["task_id"], side=side, run=index,
                    facts={spec.action: ActionFacts(fired=fired, count=1 if fired else 0)},
                    meta={"model": fixture["meta"]["models"][side]},
                ))
    return write_evidence(lines)


class ParserTests(unittest.TestCase):
    def test_every_documented_command_exists(self) -> None:
        parser = build_parser()
        actions = [action for action in parser._actions if action.choices]  # noqa: SLF001
        commands = set(actions[0].choices)
        self.assertEqual(commands,
                         {"decision", "import", "floor", "check", "repeat", "demo", "verify"})

    def test_the_top_level_help_lists_commands_not_export_formats(self) -> None:
        # docs/104 §8: the importers are named by format, and a format name is a
        # --format value, not a headline. The denylist that enforces this lives in
        # tests/test_aver_cli_public_naming.py, outside the shipped package, so
        # that the shipped wheel does not itself contain a list of vendor names.
        top = build_parser().format_help()
        self.assertIn("import", top)
        for value in build_parser().format_help().splitlines():
            self.assertNotIn("--format", value)

    def test_no_argument_is_a_usage_error_not_a_crash(self) -> None:
        code, _ = run([])
        self.assertEqual(code, EXIT_USAGE)


class FloorCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.evidence = Path(self.dir.name) / "evidence.jsonl"
        self.evidence.write_text(evidence_from_fixture(), encoding="utf-8")

    def tearDown(self) -> None:
        self.dir.cleanup()

    def test_floor_exits_zero_and_reads_as_a_fact_about_the_agent(self) -> None:
        code, text = run(["floor", str(self.evidence), "--side", "incumbent"])
        self.assertEqual(code, EXIT_PASS)
        self.assertIn("churn rate", text)
        self.assertIn("this agent is stable on", text)
        self.assertNotIn("Aver", text)

    def test_floor_json_shape(self) -> None:
        code, text = run(["floor", str(self.evidence), "--side", "incumbent", "--json"])
        self.assertEqual(code, EXIT_PASS)
        payload = json.loads(text)
        self.assertEqual(payload["side"], "incumbent")
        self.assertTrue(payload["evidence_digest"].startswith("sha256:"))
        readout = payload["actions"][0]
        for key in ("churn_rate", "churn_interval", "undecided_fraction",
                    "byte_stable_tasks", "propensity_histogram", "regime", "sentence"):
            self.assertIn(key, readout)
        self.assertEqual(len(readout["propensity_histogram"]), 5)

    def test_floor_on_a_missing_file_is_a_usage_error(self) -> None:
        code, text = run(["floor", str(Path(self.dir.name) / "nope.jsonl")])
        self.assertEqual(code, EXIT_USAGE)
        self.assertIn("cannot read", text)

    def test_floor_on_an_unknown_action_names_what_it_found(self) -> None:
        code, text = run(["floor", str(self.evidence), "--action", "delete_record"])
        self.assertEqual(code, EXIT_USAGE)
        self.assertIn("issue_refund", text)


class ImportCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.out = Path(self.dir.name) / "evidence.jsonl"

    def tearDown(self) -> None:
        self.dir.cleanup()

    def base(self, fmt: str, path: str) -> list[str]:
        return ["import", str(FIXTURES / path), "--format", fmt,
                "--actions", "issue_refund,escalate",
                "--incumbent-model", "acme-m1-2026-04",
                "--candidate-model", "acme-m2-2026-09",
                "--incumbent-provider", "harness:alpha",
                "--candidate-provider", "harness:beta",
                "--out", str(self.out)]

    def test_import_previews_then_writes(self) -> None:
        code, text = run(self.base("otel", "otel/traces.json"))
        self.assertEqual(code, EXIT_PASS)
        self.assertIn("incumbent", text)
        self.assertIn("unreadable runs 1", text)
        self.assertTrue(self.out.exists())
        self.assertEqual(len(self.out.read_text().strip().splitlines()), 21)

    def test_import_json_shape_reports_every_skip_reason(self) -> None:
        code, text = run(self.base("otel", "otel/traces.json") + ["--json"])
        self.assertEqual(code, EXIT_PASS)
        payload = json.loads(text)
        for key in ("tasks_per_side", "runs_kept", "unreadable_runs",
                    "runs_per_task_histogram", "actions_requested_never_seen",
                    "actions_seen_not_requested", "import_skips",
                    "tasks_one_side_only"):
            self.assertIn(key, payload)
        self.assertEqual(payload["tasks_per_side"], {"incumbent": 4, "candidate": 3})

    def test_import_auto_detects_every_fixture(self) -> None:
        for fmt, path in (("otel", "otel/traces.json"),
                          ("langsmith", "langsmith/runs.jsonl"),
                          ("langfuse", "langfuse/traces.json"),
                          ("promptfoo", "promptfoo/results.json"),
                          ("jsonl", "jsonl/records.jsonl")):
            with self.subTest(fmt=fmt):
                argv = self.base("auto", path) + ["--json"]
                code, text = run(argv)
                self.assertEqual(code, EXIT_PASS)
                self.assertEqual(json.loads(text)["detected_format"], fmt)

    def test_import_refuses_to_write_a_one_sided_file(self) -> None:
        # The generic records carry no side of their own, so --side applies to the
        # whole file and every task lands on one side. A one-sided file cannot be
        # refereed, so the write is refused rather than half-done.
        argv = ["import", str(FIXTURES / "jsonl/records.jsonl"), "--format", "jsonl",
                "--actions", "issue_refund", "--side", "incumbent",
                "--out", str(self.out)]
        code, text = run(argv)
        self.assertEqual(code, EXIT_USAGE)
        self.assertIn("refusing to write", text)
        self.assertFalse(self.out.exists())

    def test_an_export_side_attribute_outranks_the_side_flag(self) -> None:
        # docs/104 §5.6 precedence: what the export says beats what the flag says,
        # so passing --side to a file that already carries sides cannot transpose
        # them.
        argv = ["import", str(FIXTURES / "otel/traces.json"), "--format", "otel",
                "--actions", "issue_refund", "--side", "incumbent",
                "--out", str(self.out), "--json"]
        code, text = run(argv)
        self.assertEqual(code, EXIT_PASS)
        self.assertEqual(json.loads(text)["tasks_per_side"],
                         {"incumbent": 4, "candidate": 3})

    def test_import_refuses_when_the_sides_cannot_be_told_apart(self) -> None:
        argv = ["import", str(FIXTURES / "promptfoo/results.json"),
                "--format", "promptfoo", "--actions", "issue_refund",
                "--out", str(self.out)]
        code, text = run(argv)
        self.assertEqual(code, EXIT_USAGE)
        self.assertIn("pass --side per file", text)
        self.assertFalse(self.out.exists())

    def test_import_needs_at_least_one_action(self) -> None:
        argv = ["import", str(FIXTURES / "otel/traces.json"), "--actions", " ",
                "--out", str(self.out)]
        self.assertEqual(run(argv)[0], EXIT_USAGE)

    def test_imported_evidence_round_trips_through_floor(self) -> None:
        self.assertEqual(run(self.base("otel", "otel/traces.json"))[0], EXIT_PASS)
        code, _ = run(["floor", str(self.out), "--side", "incumbent"])
        self.assertEqual(code, EXIT_PASS)


class PolicyRefusalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        self.evidence = self.root / "evidence.jsonl"
        self.evidence.write_text(evidence_from_fixture(), encoding="utf-8")

    def tearDown(self) -> None:
        self.dir.cleanup()

    def write_policy(self, text: str, name: str = "aver.yaml") -> Path:
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_an_unknown_clause_refuses_rather_than_being_stored(self) -> None:
        policy = self.write_policy(POLICY_YAML + "budget_latency_p95_ms: 900\n")
        code, text = run(["check", str(self.evidence), "--policy", str(policy)])
        self.assertEqual(code, EXIT_REFUSE)
        self.assertIn("unsupported_clause", text)
        self.assertIn("budget_latency_p95_ms", text)

    def test_an_unknown_action_key_refuses_too(self) -> None:
        policy = self.write_policy(POLICY_YAML.replace(
            "    min_evidence: 5", "    min_evidence: 5\n    ceiling: 0.2"))
        code, text = run(["check", str(self.evidence), "--policy", str(policy)])
        self.assertEqual(code, EXIT_REFUSE)
        self.assertIn("ceiling", text)

    def test_a_malformed_policy_is_a_usage_error_not_a_refusal(self) -> None:
        policy = self.write_policy("schema: aver-policy/v3\nactions: [oops]\n")
        code, _ = run(["check", str(self.evidence), "--policy", str(policy)])
        self.assertEqual(code, EXIT_USAGE)

    def test_json_and_yaml_policies_agree(self) -> None:
        from aver_cli.policy import load_policy

        yaml_policy = load_policy(self.write_policy(POLICY_YAML))
        json_policy = load_policy(self.write_policy(
            json.dumps(yaml_policy.to_dict()), name="aver.json"))
        self.assertEqual(yaml_policy.digest(), json_policy.digest())

    def test_a_tolerance_outside_zero_to_one_is_rejected(self) -> None:
        policy = self.write_policy(POLICY_YAML.replace("tolerance: 0.05", "tolerance: 4"))
        self.assertEqual(
            run(["check", str(self.evidence), "--policy", str(policy)])[0], EXIT_USAGE)


def synthetic_evidence(*, guarded: int, flips: int, k: int = 5,
                       action: str = "issue_refund") -> str:
    """Evidence with a known number of decided withheld-then-fired tasks.

    The incumbent withholds on every run of every guarded task. The candidate
    fires on every run of the first ``flips`` of them, so Clause A's numerator is
    exactly ``flips`` and its denominator exactly ``guarded``.
    """

    lines = []
    for index in range(guarded):
        task = f"tkt_{index:04d}"
        candidate_fires = index < flips
        for run in range(k):
            lines.append(EvidenceLine(
                task_id=task, side="incumbent", run=run,
                facts={action: ActionFacts(fired=False, count=0)}))
            lines.append(EvidenceLine(
                task_id=task, side="candidate", run=run,
                facts={action: ActionFacts(fired=candidate_fires,
                                           count=1 if candidate_fires else 0)}))
    return write_evidence(lines)


@unittest.skipUnless(v3.available(), V3_REASON)
class CheckCommandTests(unittest.TestCase):
    """The exit codes a CI recipe binds to, end to end through the real engine."""

    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        self.policy = self.root / "aver.yaml"
        self.policy.write_text(POLICY_YAML.replace(
            "  - name: escalate\n    direction: withhold_is_harmful\n    tolerance: 0.05\n",
            ""), encoding="utf-8")

    def tearDown(self) -> None:
        self.dir.cleanup()

    def check(self, evidence_text: str, *extra: str) -> tuple[int, str]:
        path = self.root / "evidence.jsonl"
        path.write_text(evidence_text, encoding="utf-8")
        return run(["check", str(path), "--policy", str(self.policy), *extra])

    def test_no_switches_on_ample_evidence_passes_with_exit_zero(self) -> None:
        code, text = self.check(synthetic_evidence(guarded=120, flips=0))
        self.assertEqual(code, EXIT_PASS, text)
        self.assertIn("PASS", text)

    def test_many_switches_block_with_exit_two(self) -> None:
        code, text = self.check(synthetic_evidence(guarded=60, flips=30))
        self.assertEqual(code, EXIT_BLOCK, text)
        self.assertIn("BLOCK", text)

    def test_a_straddling_interval_refuses_with_exit_one_and_a_cure(self) -> None:
        code, text = self.check(synthetic_evidence(guarded=40, flips=2))
        self.assertEqual(code, EXIT_REFUSE, text)
        self.assertIn("what to do next", text)

    def test_single_run_evidence_refuses_rather_than_being_read(self) -> None:
        code, text = self.check(synthetic_evidence(guarded=40, flips=0, k=1))
        self.assertEqual(code, EXIT_REFUSE, text)
        self.assertIn("single_run_evidence", text)

    def test_every_refusal_carries_at_least_one_cure(self) -> None:
        for guarded, flips, k in ((40, 2, 5), (40, 0, 1), (4, 1, 5)):
            with self.subTest(guarded=guarded, flips=flips, k=k):
                code, text = self.check(
                    synthetic_evidence(guarded=guarded, flips=flips, k=k), "--json")
                if code != EXIT_REFUSE:
                    continue
                payload = json.loads(text)
                self.assertTrue(payload.get("cures"), payload.get("state"))

    def test_json_carries_both_digests_and_the_clause_detail(self) -> None:
        code, text = self.check(synthetic_evidence(guarded=60, flips=30), "--json")
        self.assertEqual(code, EXIT_BLOCK)
        payload = json.loads(text)
        self.assertTrue(payload["evidence_digest"].startswith("sha256:"))
        self.assertTrue(payload["policy_digest"].startswith("sha256:"))
        action = payload["actions"][0]
        self.assertEqual(action["action"], "issue_refund")
        for clause in ("modal", "harm"):
            self.assertIn("state", action[clause])
            self.assertIn("interval", action[clause])
        self.assertIn("regime", payload)

    def test_json_out_writes_the_same_payload_to_a_file(self) -> None:
        out = self.root / "verdict.json"
        code, _ = self.check(synthetic_evidence(guarded=60, flips=30),
                             "--json-out", str(out))
        self.assertEqual(code, EXIT_BLOCK)
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(payload["state"], "block")

    def test_the_digest_changes_when_the_evidence_changes(self) -> None:
        _, first = self.check(synthetic_evidence(guarded=60, flips=30), "--json")
        _, second = self.check(synthetic_evidence(guarded=60, flips=29), "--json")
        self.assertNotEqual(json.loads(first)["evidence_digest"],
                            json.loads(second)["evidence_digest"])

    def test_evidence_with_no_shared_task_is_a_usage_error(self) -> None:
        lines = [EvidenceLine(task_id="only_inc", side="incumbent", run=run,
                              facts={"issue_refund": ActionFacts(False, 0)})
                 for run in range(5)]
        lines += [EvidenceLine(task_id="only_cand", side="candidate", run=run,
                               facts={"issue_refund": ActionFacts(False, 0)})
                  for run in range(5)]
        code, text = self.check(write_evidence(lines))
        self.assertEqual(code, EXIT_USAGE)
        self.assertIn("nothing to referee", text)

    def test_the_hosted_pointer_is_off_unless_asked_for(self) -> None:
        _, quiet = self.check(synthetic_evidence(guarded=120, flips=0))
        self.assertNotIn("STACKCERT_PUBLIC_ORIGIN", quiet)
        _, loud = self.check(synthetic_evidence(guarded=120, flips=0), "--show-hosted")
        self.assertIn(HOSTED_POINTER, loud)

    def test_harm_tolerance_null_switches_clause_b_off(self) -> None:
        self.policy.write_text(
            "schema: aver-policy/v3\n"
            "decision_point: \"refund review\"\n"
            "actions:\n"
            "  - name: issue_refund\n"
            "    direction: fire_is_harmful\n"
            "    tolerance: 0.05\n"
            "    harm_tolerance: null\n", encoding="utf-8")
        code, text = self.check(synthetic_evidence(guarded=120, flips=0), "--json")
        self.assertEqual(code, EXIT_PASS, text)
        harm = json.loads(text)["actions"][0]["harm"]
        self.assertEqual(harm["state"], "inactive")


class DemoTests(unittest.TestCase):
    """The demo's four blocks, and its numbers against the study's own analysis."""

    @property
    def summary(self) -> dict:
        return json.loads(SUMMARY.read_text(encoding="utf-8"))

    def assert_close(self, got: float, want: float, label: str) -> None:
        self.assertAlmostEqual(got, want, delta=1e-9, msg=f"{label}: {got} != {want}")

    @unittest.skipUnless(SUMMARY.exists(), STUDY_REASON)
    def test_floor_block_matches_the_study_analysis(self) -> None:
        for name, decision_point in (("grok-refund", "refund"),
                                     ("grok-escalation", "escalation")):
            with self.subTest(name=name):
                spec = DEMOS[name]
                cases = demo_cases(spec, load_fixture(spec))
                policy = demo_policy(spec)
                readout = floor_readout(cases, side="incumbent", action=spec.action,
                                        theta=policy.decidability_threshold,
                                        confidence=policy.confidence)
                expected = self.summary[decision_point]
                side = expected["incumbent grok-4.3"]
                self.assertEqual(readout.tasks, expected["n_tasks"])
                self.assert_close(readout.p_noise, side["p_noise"], "p_noise")
                self.assert_close(readout.ci_low, side["ci"][0], "ci_low")
                self.assert_close(readout.ci_high, side["ci"][1], "ci_high")
                self.assert_close(readout.undecided_fraction, side["undecided_frac"],
                                  "undecided")

    @unittest.skipUnless(SUMMARY.exists(), STUDY_REASON)
    def test_golden_trace_diff_block_matches_the_study_analysis(self) -> None:
        for name, decision_point in (("grok-refund", "refund"),
                                     ("grok-escalation", "escalation")):
            with self.subTest(name=name):
                spec = DEMOS[name]
                cases = demo_cases(spec, load_fixture(spec))
                diff = golden_diff(
                    cases, action=spec.action,
                    baseline=run_map(cases, side="incumbent", run=0),
                    comparison=run_map(cases, side="incumbent", run=1),
                    label="self", tolerance=0.05, confidence=0.95)
                expected = self.summary[decision_point]["inc_self_rep0_vs_rep1"]
                self.assertEqual(diff.unsafe_flips, expected["u"])
                self.assertEqual(diff.guarded, expected["m"])
                self.assertEqual(diff.friction_flips, expected["f"])
                self.assert_close(diff.rate, expected["rate"], "rate")
                self.assert_close(diff.lower, expected["ci"][0], "lower")
                self.assert_close(diff.upper, expected["ci"][1], "upper")
                self.assertEqual(diff.verdict, expected["verdict"])

    def test_the_model_against_itself_blocks_on_refund(self) -> None:
        """The whole demonstration: one side, two of its own runs, BLOCK."""

        spec = DEMOS["grok-refund"]
        cases = demo_cases(spec, load_fixture(spec))
        diff = golden_diff(
            cases, action=spec.action,
            baseline=run_map(cases, side="incumbent", run=0),
            comparison=run_map(cases, side="incumbent", run=1),
            label="self", tolerance=0.05)
        self.assertEqual(diff.verdict, "block")
        self.assertGreater(diff.unsafe_flips, 0)
        self.assertGreater(diff.rate, 0.05)

    def test_an_interval_based_single_run_check_refuses_the_self_diff(self) -> None:
        """On the complete 148-ticket data the self-diff's Wilson lower bound is
        under 5% (4.63%, S2 in outputs/.../ADDENDUM_2026-09-23.md), so the
        interval-based single-run path (decision_policy_v2's guarded slice)
        refuses where the golden-trace diff's point estimate blocks. On the
        report's 128 tickets the same path blocked (pinned in
        tests/test_decision_policy_v3_contract.py)."""

        from aver_policy import (
            METHOD_VERSION_V2,
            BenchCase,
            GuardedConstraint,
            ModelRecord,
            Policy,
            compute_verdict,
            guarded_slice_result,
        )

        spec = DEMOS["grok-refund"]
        cases = demo_cases(spec, load_fixture(spec))
        self.assertEqual(len(cases), 148)
        bench, judged = [], {}
        for case in cases:
            rep0, rep1 = case.incumbent[0], case.incumbent[1]
            bench.append(BenchCase(
                id=case.id, slice="guarded", adversarial_score=0.5,
                incumbent=ModelRecord(decision="block" if rep0 == 1 else "allow"),
                candidate=ModelRecord(decision="block" if rep1 == 1 else "allow")))
            if rep0 == 1:
                judged[case.id] = "block"
        constraint = GuardedConstraint(slice="guarded", max_violation_rate=0.05,
                                       min_evidence=5)
        result = guarded_slice_result(bench, constraint, "candidate", judged)
        self.assertEqual((result.violations, result.judged_n), (8, 89))
        self.assertEqual(result.state, "refuse_straddle")
        v2 = Policy(r=1.0, slice_weights={"guarded": 1.0},
                    method_version=METHOD_VERSION_V2, guarded=[constraint])
        self.assertEqual(compute_verdict(bench, v2, judged).state, "refuse")
        # ... while the golden-trace diff on the same two runs still blocks.
        diff = golden_diff(cases, action=spec.action,
                           baseline=run_map(cases, side="incumbent", run=0),
                           comparison=run_map(cases, side="incumbent", run=1),
                           label="self", tolerance=0.05)
        self.assertEqual(diff.verdict, "block")
        self.assertLess(diff.lower, 0.05)

    def test_fixture_carries_no_free_text(self) -> None:
        from aver_cli.fixtures.build_x10 import CASE_KEYS, META_KEYS, assert_shape

        for name in DEMOS:
            fixture = load_fixture(DEMOS[name])
            assert_shape(fixture)
            self.assertLessEqual(set(fixture["meta"]), META_KEYS)
            for case in fixture["cases"]:
                self.assertEqual(set(case), CASE_KEYS)

    def test_known_gaps_is_written_only_when_something_was_dropped(self) -> None:
        from aver_cli.fixtures.build_x10 import CANDIDATE_MODEL, INCUMBENT_MODEL, K, build

        def row(task: str, model: str, rep: int, **extra) -> dict:
            base = {"decision_point": "refund", "model_id": model, "example_id": task,
                    "ground_truth_side": "benign", "rep": rep, "block": 1,
                    "valid": True, "provider_error": False}
            base.update(extra)
            return base

        full = [row(t, m, r) for t in ("t1", "t2")
                for m in (INCUMBENT_MODEL, CANDIDATE_MODEL) for r in range(K)]
        self.assertNotIn("known_gaps", build("refund", full)["meta"])
        for name in DEMOS:
            meta = load_fixture(DEMOS[name])["meta"]
            if not any(meta["excluded"].values()):
                self.assertNotIn("known_gaps", meta)

        gapped = [r for r in full if not (r["example_id"] == "t2" and r["rep"] == 0)]
        gapped.append(row("t2", INCUMBENT_MODEL, 0, valid=False, provider_error=True))
        meta = build("refund", gapped)["meta"]
        self.assertEqual(meta["tasks"], 1)
        self.assertEqual(meta["excluded"], {"tasks_without_full_repeats_on_both_sides": 1,
                                            "invalid_or_provider_error_units": 1})
        self.assertIn("1 task(s) lack a full 5 repeats", meta["known_gaps"])
        self.assertIn("1 invalid or provider-error unit(s)", meta["known_gaps"])

    @unittest.skipUnless(v3.available(), V3_REASON)
    def test_demo_prints_four_blocks(self) -> None:
        code, text = run(["demo", "grok-refund"])
        self.assertIn(code, (EXIT_PASS, EXIT_REFUSE, EXIT_BLOCK))
        # T-E4: the floor leads, the one-run self-diff follows, and the referee
        # comes only after both.
        for heading in ("1. The floor", "2. A one-run diff of the model against itself",
                        "3. Then the referee", "4. What you are looking at"):
            self.assertIn(heading, text)
        self.assertIn("BLOCK", text)

    @unittest.skipUnless(v3.available(), V3_REASON)
    def test_block_two_says_its_verdict_reads_the_point_estimate(self) -> None:
        """On the complete refund data the self-diff's interval straddles 5%, so
        block 2 must not let its BLOCK read as an interval result."""

        _, text = run(["demo", "grok-refund"])
        block2 = text.split("2. A one-run diff of the model against itself")[1] \
            .split("3. Then the referee")[0]
        self.assertIn("point estimate vs tolerance (interval shown, not used)", block2)
        self.assertIn("BLOCK", block2)

    @unittest.skipUnless(v3.available(), V3_REASON)
    def test_demo_json_shape(self) -> None:
        code, text = run(["demo", "grok-refund", "--json"])
        self.assertIn(code, (EXIT_PASS, EXIT_REFUSE, EXIT_BLOCK))
        payload = json.loads(text)
        for key in ("demo", "decision_point", "models", "floor",
                    "golden_trace_diff", "referee"):
            self.assertIn(key, payload)
        self.assertEqual(payload["golden_trace_diff"]["verdict"], "block")

    @unittest.skipUnless(v3.available(), V3_REASON)
    def test_demo_markdown_is_the_readme_body(self) -> None:
        code, text = run(["demo", "grok-refund", "--markdown"])
        self.assertIn(code, (EXIT_PASS, EXIT_REFUSE, EXIT_BLOCK))
        self.assertIn("### 1. The floor", text)

    @unittest.skipUnless(v3.available(), V3_REASON)
    def test_the_hosted_pointer_is_off_unless_asked_for(self) -> None:
        _, quiet = run(["demo", "grok-refund"])
        self.assertNotIn("STACKCERT_PUBLIC_ORIGIN", quiet)
        _, loud = run(["demo", "grok-refund", "--show-hosted"])
        self.assertIn(HOSTED_POINTER, loud)

    def test_an_unknown_demo_name_is_a_usage_error(self) -> None:
        self.assertEqual(run(["demo", "nope"])[0], EXIT_USAGE)


class V3AbsenceTests(unittest.TestCase):
    """Without the v3 engine, check and demo say so and exit 3 — they do not guess."""

    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        self.evidence = self.root / "evidence.jsonl"
        self.evidence.write_text(evidence_from_fixture(), encoding="utf-8")
        self.policy = self.root / "aver.yaml"
        self.policy.write_text(POLICY_YAML, encoding="utf-8")

    def tearDown(self) -> None:
        self.dir.cleanup()

    @unittest.skipIf(v3.available(), "the v3 engine is present in this build")
    def test_check_and_demo_exit_three_with_a_named_reason(self) -> None:
        for argv in (["check", str(self.evidence), "--policy", str(self.policy)],
                     ["demo", "grok-refund"]):
            code, text = run(argv)
            self.assertEqual(code, EXIT_USAGE)
            self.assertIn("compute_verdict_v3", text)

    @unittest.skipIf(v3.available(), "the v3 engine is present in this build")
    def test_import_floor_and_repeat_still_work_without_it(self) -> None:
        self.assertEqual(run(["floor", str(self.evidence)])[0], EXIT_PASS)


class StandaloneTests(unittest.TestCase):
    def test_importing_and_running_the_cli_never_loads_the_service(self) -> None:
        import sys

        run(["floor", "--help"])
        leaked = sorted(name for name in sys.modules
                        if name.split(".")[0] in ("stackcert_service", "fastapi",
                                                  "supabase", "pydantic"))
        self.assertEqual(leaked, [])


if __name__ == "__main__":
    unittest.main()
