"""The five importers, pinned by fixtures: detection order, sides, no silent drop.

All five fixtures encode the same scenario, so the same assertions apply to each:
four tasks, three runs a side, one unreadable run, one task where the guarded
action fires on one side only, and one task present on the incumbent side alone.
If two importers disagree about that scenario, one of them is wrong.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from aver_cli.cli import _parse_payload
from aver_cli.evidence import FIRE_IS_HARMFUL, WITHHOLD_IS_HARMFUL, group_cases
from aver_cli.importers import FORMATS, detect_format, read_with
from aver_cli.importers.base import ImportContext, SideUndecidable, json_pointer

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ACTIONS = ("issue_refund", "escalate")
DIRECTIONS = (("issue_refund", FIRE_IS_HARMFUL), ("escalate", WITHHOLD_IS_HARMFUL))

INC_MODEL, CAND_MODEL = "acme-m1-2026-04", "acme-m2-2026-09"
INC_PROVIDER, CAND_PROVIDER = "harness:alpha", "harness:beta"

#: fixture file per format
FILES = {
    "otel": "otel/traces.json",
    "langsmith": "langsmith/runs.jsonl",
    "langfuse": "langfuse/traces.json",
    "promptfoo": "promptfoo/results.json",
    "jsonl": "jsonl/records.jsonl",
}


def payload(fmt: str) -> object:
    return _parse_payload((FIXTURES / FILES[fmt]).read_text(encoding="utf-8"))


def context(**overrides) -> ImportContext:
    base = dict(
        actions=ACTIONS,
        incumbent_model=INC_MODEL,
        candidate_model=CAND_MODEL,
        incumbent_provider=INC_PROVIDER,
        candidate_provider=CAND_PROVIDER,
        directions=DIRECTIONS,
    )
    base.update(overrides)
    return ImportContext(**base)  # type: ignore[arg-type]


class DetectionTests(unittest.TestCase):
    def test_each_fixture_detects_as_its_own_format(self) -> None:
        for fmt in FILES:
            with self.subTest(fmt=fmt):
                self.assertEqual(detect_format(payload(fmt)), fmt)

    def test_detector_order_is_the_documented_one(self) -> None:
        self.assertEqual([name for name, _ in FORMATS],
                         ["otel", "langsmith", "langfuse", "promptfoo", "jsonl"])

    def test_an_unrecognised_file_is_refused_not_guessed(self) -> None:
        from aver_cli.importers import UnknownFormat

        with self.assertRaises(UnknownFormat):
            detect_format([{"hello": "world"}])

    def test_already_v1_evidence_detects_as_jsonl_and_passes_through(self) -> None:
        from aver_cli import EVIDENCE_SCHEMA

        row = {"schema": EVIDENCE_SCHEMA, "task_id": "t", "side": "incumbent", "run": 0,
               "facts": {"issue_refund": {"fired": False, "count": 0}}}
        self.assertEqual(detect_format([row]), "jsonl")
        result = read_with("jsonl", [row], context())
        self.assertEqual(len(result.lines), 1)
        self.assertFalse(result.skipped_total)


class ScenarioTests(unittest.TestCase):
    """Every importer must read the same four-task scenario the same way."""

    def scenario(self, fmt: str):
        result = read_with(fmt, payload(fmt), context())
        self.assertFalse(result.skipped_total, f"{fmt} skipped: {result.skipped}")
        return result

    def test_three_runs_per_side_and_the_missing_side(self) -> None:
        for fmt in FILES:
            with self.subTest(fmt=fmt):
                result = self.scenario(fmt)
                _, report = group_cases(result.lines, actions=DIRECTIONS)
                self.assertEqual(report.tasks["incumbent"], 4, fmt)
                self.assertEqual(report.tasks["candidate"], 3, fmt)
                self.assertEqual(report.tasks_one_side_only, ["tkt_d"], fmt)
                self.assertEqual(report.runs_per_task["incumbent"], {"3": 4}, fmt)

    def test_exactly_one_unreadable_run(self) -> None:
        for fmt in FILES:
            with self.subTest(fmt=fmt):
                result = self.scenario(fmt)
                _, report = group_cases(result.lines, actions=DIRECTIONS)
                self.assertEqual(report.unreadable_runs, 1, fmt)
                unreadable = [line for line in result.lines if line.unreadable]
                self.assertEqual([(line.task_id, line.side) for line in unreadable],
                                 [("tkt_c", "incumbent")], fmt)

    def test_the_guarded_action_fires_on_one_side_only(self) -> None:
        for fmt in FILES:
            with self.subTest(fmt=fmt):
                result = self.scenario(fmt)
                cases, _ = group_cases(result.lines, actions=DIRECTIONS)
                refund = {case.id: case for case in cases if case.action == "issue_refund"}
                # tkt_b: incumbent withholds every run, candidate fires every run.
                self.assertEqual(refund["tkt_b"].incumbent, (1, 1, 1), fmt)
                self.assertEqual(refund["tkt_b"].candidate, (0, 0, 0), fmt)
                # tkt_a: both sides withhold.
                self.assertEqual(refund["tkt_a"].incumbent, (1, 1, 1), fmt)
                self.assertEqual(refund["tkt_a"].candidate, (1, 1, 1), fmt)
                # tkt_c: the unreadable run is None on both actions.
                self.assertEqual(refund["tkt_c"].incumbent, (1, None, 1), fmt)

    def test_a_tool_span_that_never_executed_does_not_count_as_fired(self) -> None:
        # otel carries aver.tool.executed=false; langsmith/langfuse carry an errored
        # tool record. Neither may read as the action having fired.
        for fmt in ("otel", "langsmith", "langfuse"):
            with self.subTest(fmt=fmt):
                result = self.scenario(fmt)
                cases, _ = group_cases(result.lines, actions=DIRECTIONS)
                refund = {case.id: case for case in cases if case.action == "issue_refund"}
                self.assertEqual(refund["tkt_a"].incumbent, (1, 1, 1), fmt)

    def test_the_withhold_is_harmful_action_maps_the_other_way(self) -> None:
        for fmt in ("otel", "langsmith", "langfuse", "promptfoo"):
            with self.subTest(fmt=fmt):
                result = self.scenario(fmt)
                cases, _ = group_cases(result.lines, actions=DIRECTIONS)
                escalate = {case.id: case for case in cases if case.action == "escalate"}
                # escalate fires on every readable run, and firing is the safe side.
                self.assertEqual(escalate["tkt_a"].incumbent, (1, 1, 1), fmt)
                self.assertEqual(escalate["tkt_c"].incumbent, (1, None, 1), fmt)

    def test_run_indices_are_zero_based_and_contiguous_per_task_and_side(self) -> None:
        for fmt in FILES:
            with self.subTest(fmt=fmt):
                result = self.scenario(fmt)
                seen: dict[tuple[str, str], list[int]] = {}
                for line in result.lines:
                    seen.setdefault((line.task_id, line.side), []).append(line.run)
                for key, runs in seen.items():
                    self.assertEqual(sorted(runs), [0, 1, 2], f"{fmt} {key}")


class SideResolutionTests(unittest.TestCase):
    def test_ambiguous_sides_refuse_rather_than_guess(self) -> None:
        # promptfoo carries no aver.side attribute, so with no provider or model
        # flags the importer has nothing to split on.
        bare = ImportContext(actions=ACTIONS, directions=DIRECTIONS)
        with self.assertRaises(SideUndecidable) as caught:
            read_with("promptfoo", payload("promptfoo"), bare)
        self.assertIn("pass --side per file", str(caught.exception))

    def test_a_model_matching_both_sides_refuses(self) -> None:
        both = ImportContext(actions=ACTIONS, directions=DIRECTIONS,
                             incumbent_model="acme", candidate_model="acme-m2")
        with self.assertRaises(SideUndecidable):
            both.side_for(model="acme-m2-2026-09")

    def test_explicit_side_flag_overrides_a_model_match(self) -> None:
        ctx = context(side="candidate")
        self.assertEqual(ctx.side_for(model=INC_MODEL), "candidate")

    def test_an_export_attribute_beats_the_side_flag(self) -> None:
        ctx = context(side="candidate")
        self.assertEqual(ctx.side_for(attribute="incumbent", model=INC_MODEL), "incumbent")

    def test_provider_flags_split_the_promptfoo_export(self) -> None:
        result = read_with("promptfoo", payload("promptfoo"),
                           ImportContext(actions=ACTIONS, directions=DIRECTIONS,
                                         incumbent_provider=INC_PROVIDER,
                                         candidate_provider=CAND_PROVIDER))
        self.assertEqual({line.side for line in result.lines},
                         {"incumbent", "candidate"})


class PromptfooToolCallFallbackTests(unittest.TestCase):
    """docs/104 §5.4: with no ``trajectory:tool-used`` assertion, *fired* comes
    from the response's tool-call array. The shared fixture always carries the
    assertion, so this second path is pinned here on its own."""

    @staticmethod
    def result(task: str, provider: object, response: object, **extra) -> dict:
        row = {"vars": {"aver_task_id": task}, "provider": provider,
               "success": True, "response": response}
        row.update(extra)
        return row

    def test_tool_calls_decide_fired_when_no_assertion_is_present(self) -> None:
        rows = [
            # chat-completions shape: {"function": {"name": ...}}, called twice
            self.result("t1", {"id": INC_PROVIDER}, {"toolCalls": [
                {"function": {"name": "issue_refund"}},
                {"function": {"name": "issue_refund"}},
                {"function": {"name": "lookup_order"}},
            ]}),
            # nested under output, flat {"name": ...}, snake_case key
            self.result("t1", {"id": CAND_PROVIDER},
                        {"output": {"tool_calls": [{"name": "escalate"}, {}]}}),
        ]
        result = read_with("promptfoo", {"results": rows}, context())
        by_side = {line.side: line for line in result.lines}
        self.assertEqual(by_side["incumbent"].facts["issue_refund"].count, 2)
        self.assertTrue(by_side["incumbent"].facts["issue_refund"].fired)
        self.assertFalse(by_side["incumbent"].facts["escalate"].fired)
        self.assertTrue(by_side["candidate"].facts["escalate"].fired)
        self.assertFalse(by_side["candidate"].facts["issue_refund"].fired)
        # a tool outside --actions is reported as seen, never recorded as a fact
        self.assertIn("lookup_order", result.actions_seen)
        self.assertNotIn("lookup_order", by_side["incumbent"].facts)

    def test_an_assertion_wins_over_the_tool_call_array(self) -> None:
        row = self.result("t1", {"id": INC_PROVIDER},
                          {"toolCalls": [{"function": {"name": "issue_refund"}}]},
                          gradingResult={"componentResults": [
                              {"pass": False, "assertion": {
                                  "type": "trajectory:tool-used", "value": "issue_refund"}},
                              {"pass": True, "assertion": {"type": "contains", "value": "x"}},
                          ]})
        [line] = read_with("promptfoo", {"results": [row]}, context()).lines
        self.assertFalse(line.facts["issue_refund"].fired)

    def test_a_bare_list_and_a_string_provider_are_read(self) -> None:
        rows = [self.result("t1", INC_PROVIDER, {"toolCalls": []}),
                self.result("t1", INC_PROVIDER, {"toolCalls": []})]
        self.assertEqual(detect_format(rows), "promptfoo")
        result = read_with("promptfoo", rows, context())
        self.assertEqual([(line.side, line.run) for line in result.lines],
                         [("incumbent", 0), ("incumbent", 1)])
        self.assertEqual(result.lines[0].meta["model"], INC_PROVIDER)

    def test_an_errored_result_is_kept_as_unreadable_not_dropped(self) -> None:
        row = self.result("t1", {"id": INC_PROVIDER}, None, error="timeout")
        [line] = read_with("promptfoo", {"results": [row]}, context()).lines
        self.assertTrue(line.unreadable)
        self.assertEqual(line.facts, {})

    def test_results_without_identity_or_at_all_are_counted_with_a_reason(self) -> None:
        nameless = read_with("promptfoo", {"results": [
            {"provider": {"id": INC_PROVIDER}, "response": {"toolCalls": []}},
        ]}, context())
        self.assertEqual((len(nameless.lines), nameless.skipped_total), (0, 1))
        self.assertTrue(any("no task identity" in reason for reason in nameless.skipped))

        empty = read_with("promptfoo", {"results": []}, context())
        self.assertEqual(empty.skipped, {"file contained no results": 1})


class NoSilentDropTests(unittest.TestCase):
    def test_a_record_with_no_task_identity_is_counted_with_its_reason(self) -> None:
        rows = [{"example_id": "", "model_id": INC_MODEL, "decision": 1}]
        result = read_with("jsonl", rows, context())
        self.assertEqual(result.skipped_total, 1)
        self.assertEqual(len(result.lines), 0)
        self.assertTrue(any("task identity" in reason for reason in result.skipped))

    def test_kept_plus_skipped_accounts_for_every_record(self) -> None:
        rows = [
            {"example_id": "t1", "model_id": INC_MODEL, "decision": 1, "run": 0},
            {"example_id": "", "model_id": INC_MODEL, "decision": 1},
            {"example_id": "t1", "model_id": CAND_MODEL, "decision": 0, "run": 0},
        ]
        result = read_with("jsonl", rows, context())
        self.assertEqual(len(result.lines) + result.skipped_total, len(rows))

    def test_actions_seen_but_not_requested_are_reported(self) -> None:
        narrowed = context(actions=("escalate",),
                           directions=(("escalate", WITHHOLD_IS_HARMFUL),))
        result = read_with("otel", payload("otel"), narrowed)
        self.assertIn("issue_refund", result.actions_seen)
        self.assertTrue(all("issue_refund" not in line.facts for line in result.lines))

    def test_an_empty_file_is_reported_not_treated_as_success(self) -> None:
        result = read_with("jsonl", [], context())
        self.assertEqual(result.skipped_total, 1)


class JsonPointerTests(unittest.TestCase):
    def test_pointer_reads_nested_values_and_escapes(self) -> None:
        document = {"a": {"b~c": [{"d": 7}]}}
        self.assertEqual(json_pointer(document, "/a/b~0c/0/d"), 7)
        self.assertIsNone(json_pointer(document, "/a/missing"))

    def test_task_key_pointer_supplies_identity_when_no_attribute_exists(self) -> None:
        rows = [{"model_id": INC_MODEL, "decision": 1, "payload": {"ticket": "abc"}},
                {"model_id": CAND_MODEL, "decision": 1, "payload": {"ticket": "abc"}}]
        result = read_with("jsonl", rows, context(task_key="/payload/ticket"))
        self.assertEqual(len({line.task_id for line in result.lines}), 1)
        self.assertEqual(len(result.lines), 2)


class FixtureHygieneTests(unittest.TestCase):
    def test_fixtures_are_small_enough_to_read(self) -> None:
        for relative in FILES.values():
            path = FIXTURES / relative
            self.assertLess(path.stat().st_size, 200 * 1024, relative)

    def test_fixture_models_are_invented_not_real(self) -> None:
        # docs/104 §8: no provider appears as a shipped integration, so the
        # fixtures' model names are made up. The denylist that enforces this lives
        # in tests/test_aver_cli_public_naming.py, outside the shipped package, so
        # the shipped wheel does not carry a list of provider names.
        for relative in FILES.values():
            text = (FIXTURES / relative).read_text(encoding="utf-8")
            self.assertIn("acme-m", text, f"{relative} lost its invented models")


if __name__ == "__main__":
    unittest.main()
