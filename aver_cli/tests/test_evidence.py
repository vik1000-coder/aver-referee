"""aver-evidence/v1: round-trip, direction mapping, unreadable handling."""

from __future__ import annotations

import unittest

from aver_cli import EVIDENCE_SCHEMA
from aver_cli.evidence import (
    FIRE_IS_HARMFUL,
    WITHHOLD_IS_HARMFUL,
    ActionFacts,
    EvidenceError,
    EvidenceLine,
    evidence_digest,
    group_cases,
    read_evidence_text,
    task_id_from_message,
    write_evidence,
)

ACTIONS = (("issue_refund", FIRE_IS_HARMFUL), ("escalate", WITHHOLD_IS_HARMFUL))


def line(task: str, side: str, run: int, *, refund_fired: bool, escalated: bool,
         unreadable: bool = False) -> EvidenceLine:
    return EvidenceLine(
        task_id=task, side=side, run=run, unreadable=unreadable,
        facts={} if unreadable else {
            "issue_refund": ActionFacts(fired=refund_fired, count=1 if refund_fired else 0),
            "escalate": ActionFacts(fired=escalated, count=1 if escalated else 0),
        },
    )


class SchemaRoundTripTests(unittest.TestCase):
    def test_round_trip_is_lossless(self) -> None:
        original = EvidenceLine(
            task_id="tkt_00017", side="incumbent", run=3, variant="rerun",
            facts={"issue_refund": ActionFacts(False, 0, None),
                   "escalate": ActionFacts(True, 2, True)},
            meta={"model": "m1", "cost_usd": 0.0016, "latency_ms": 4840},
        )
        self.assertEqual(EvidenceLine.from_json_line(original.to_json_line()), original)

    def test_schema_field_is_required_and_named(self) -> None:
        with self.assertRaises(EvidenceError) as caught:
            EvidenceLine.from_json_line('{"task_id":"t","side":"incumbent","run":0}')
        self.assertIn(EVIDENCE_SCHEMA, str(caught.exception))

    def test_bad_side_and_bad_run_are_named_not_guessed(self) -> None:
        base = {"schema": EVIDENCE_SCHEMA, "task_id": "t", "side": "incumbent", "run": 0}
        for patch, needle in (({"side": "baseline"}, "side"), ({"run": -1}, "run"),
                              ({"task_id": ""}, "task_id")):
            with self.assertRaises(EvidenceError) as caught:
                EvidenceLine.from_dict({**base, **patch})
            self.assertIn(needle, str(caught.exception))

    def test_digest_is_order_independent(self) -> None:
        a = line("t1", "incumbent", 0, refund_fired=False, escalated=True)
        b = line("t1", "incumbent", 1, refund_fired=True, escalated=True)
        self.assertEqual(evidence_digest([a, b]), evidence_digest([b, a]))

    def test_write_evidence_is_stably_ordered(self) -> None:
        a = line("t2", "candidate", 1, refund_fired=False, escalated=True)
        b = line("t1", "incumbent", 0, refund_fired=False, escalated=True)
        self.assertEqual(write_evidence([a, b]), write_evidence([b, a]))

    def test_task_id_from_message_is_stable_and_key_order_independent(self) -> None:
        self.assertEqual(task_id_from_message({"a": 1, "b": 2}),
                         task_id_from_message({"b": 2, "a": 1}))


class DirectionMappingTests(unittest.TestCase):
    """1 is always the safe side, whichever way round the action's harm runs."""

    def test_fire_is_harmful_maps_fired_to_zero(self) -> None:
        fired = line("t", "incumbent", 0, refund_fired=True, escalated=False)
        withheld = line("t", "incumbent", 1, refund_fired=False, escalated=False)
        self.assertEqual(fired.decision("issue_refund", FIRE_IS_HARMFUL), 0)
        self.assertEqual(withheld.decision("issue_refund", FIRE_IS_HARMFUL), 1)

    def test_withhold_is_harmful_maps_fired_to_one(self) -> None:
        fired = line("t", "incumbent", 0, refund_fired=False, escalated=True)
        withheld = line("t", "incumbent", 1, refund_fired=False, escalated=False)
        self.assertEqual(fired.decision("escalate", WITHHOLD_IS_HARMFUL), 1)
        self.assertEqual(withheld.decision("escalate", WITHHOLD_IS_HARMFUL), 0)

    def test_an_action_the_export_never_mentioned_is_none_not_a_violation(self) -> None:
        partial = EvidenceLine("t", "incumbent", 0,
                               facts={"escalate": ActionFacts(True, 1, None)})
        self.assertIsNone(partial.decision("issue_refund", FIRE_IS_HARMFUL))

    def test_schema_valid_none_is_preserved_and_never_a_violation(self) -> None:
        facts = ActionFacts.from_dict({"fired": True, "count": 1})
        self.assertIsNone(facts.args_schema_valid)


class UnreadableRunTests(unittest.TestCase):
    def test_unreadable_run_contributes_none_to_every_action(self) -> None:
        broken = line("t", "incumbent", 1, refund_fired=False, escalated=False,
                      unreadable=True)
        self.assertIsNone(broken.decision("issue_refund", FIRE_IS_HARMFUL))
        self.assertIsNone(broken.decision("escalate", WITHHOLD_IS_HARMFUL))

    def test_unreadable_runs_are_kept_and_counted_not_dropped(self) -> None:
        lines = [
            line("t", "incumbent", 0, refund_fired=False, escalated=True),
            line("t", "incumbent", 1, refund_fired=False, escalated=True, unreadable=True),
            line("t", "incumbent", 2, refund_fired=False, escalated=True),
            line("t", "candidate", 0, refund_fired=False, escalated=True),
            line("t", "candidate", 1, refund_fired=False, escalated=True),
            line("t", "candidate", 2, refund_fired=False, escalated=True),
        ]
        cases, report = group_cases(lines, actions=ACTIONS)
        self.assertEqual(report.unreadable_runs, 1)
        refund = next(case for case in cases if case.action == "issue_refund")
        self.assertEqual(refund.incumbent, (1, None, 1))


class GroupingTests(unittest.TestCase):
    def test_one_case_per_task_and_action(self) -> None:
        lines = [line("t1", side, run, refund_fired=False, escalated=True)
                 for side in ("incumbent", "candidate") for run in range(3)]
        cases, _ = group_cases(lines, actions=ACTIONS)
        self.assertEqual(sorted(case.action for case in cases), ["escalate", "issue_refund"])
        self.assertTrue(all(case.id == "t1" for case in cases))

    def test_a_task_on_one_side_only_is_reported_not_silently_paired(self) -> None:
        lines = [line("t1", "incumbent", run, refund_fired=False, escalated=True)
                 for run in range(3)]
        lines += [line("t2", side, run, refund_fired=False, escalated=True)
                  for side in ("incumbent", "candidate") for run in range(3)]
        cases, report = group_cases(lines, actions=ACTIONS)
        self.assertEqual(report.tasks_one_side_only, ["t1"])
        self.assertEqual({case.id for case in cases}, {"t2"})

    def test_duplicate_run_indices_are_counted_not_merged(self) -> None:
        lines = [line("t1", "incumbent", 0, refund_fired=False, escalated=True),
                 line("t1", "incumbent", 0, refund_fired=True, escalated=True),
                 line("t1", "candidate", 0, refund_fired=False, escalated=True)]
        _, report = group_cases(lines, actions=ACTIONS)
        self.assertEqual(report.duplicate_runs, 1)

    def test_runs_are_ordered_by_run_index_not_file_order(self) -> None:
        lines = [line("t1", "incumbent", 2, refund_fired=True, escalated=True),
                 line("t1", "incumbent", 0, refund_fired=False, escalated=True),
                 line("t1", "incumbent", 1, refund_fired=False, escalated=True)]
        lines += [line("t1", "candidate", run, refund_fired=False, escalated=True)
                  for run in range(3)]
        cases, _ = group_cases(lines, actions=ACTIONS)
        refund = next(case for case in cases if case.action == "issue_refund")
        self.assertEqual(refund.incumbent, (1, 1, 0))


class ReaderTests(unittest.TestCase):
    def test_every_unreadable_line_is_counted_with_a_reason(self) -> None:
        good = line("t1", "incumbent", 0, refund_fired=False, escalated=True).to_json_line()
        text = "\n".join([good, "", "{not json", '{"schema":"other/v1"}', good])
        lines, report = read_evidence_text(text, where="f.jsonl")
        self.assertEqual(len(lines), 2)
        self.assertEqual(report.blank_lines, 1)
        self.assertEqual(report.skipped_total, 2)
        self.assertEqual(report.kept + report.skipped_total + report.blank_lines,
                         report.total_lines)
        self.assertTrue(all("f.jsonl:" in item for item in report.first_reasons))


if __name__ == "__main__":
    unittest.main()
