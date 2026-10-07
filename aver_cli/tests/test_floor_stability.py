"""T-E4: ``aver floor`` is the front door — the stability map and the self-diff.

Agent regression tests are flaky tests: the same agent on the same tasks
disagrees with itself, and a one-run diff reads that as a regression. ``floor``
measures the disagreement, names the tasks behind it (the stability map), and
shows what a one-run diff of the agent against itself would have reported. These
tests pin all three, and pin the demo's order: floor first, referee after.
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from aver_cli import v3
from aver_cli.cli import DEMO_TITLES, EXIT_BLOCK, EXIT_PASS, EXIT_REFUSE, EXIT_USAGE, main
from aver_cli.demo import DEMOS, fixture_evidence, load_fixture
from aver_cli.evidence import (
    ActionFacts,
    EvidenceLine,
    read_evidence_text,
    write_evidence,
)
from aver_cli.referee import (
    COIN_FLIP,
    NO_READABLE_RUN,
    STABLE_FIRES,
    STABLE_WITHHOLDS,
    decide,
    required_agreement,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SUMMARY = REPO_ROOT / "outputs" / "agent-referee-2026-09-22" / "real_arm_summary_complete.json"
STUDY_REASON = ("the study summary (outputs/agent-referee-2026-09-22/) is only in "
                "the source checkout, not in an installed wheel")
V3_REASON = "needs aver_policy.repeated.compute_verdict_v3 (docs/103 T-D1)"

#: The hand-built map: six tasks, k = 5, withheld on this many runs each.
WITHHELD = {"t1": 5, "t2": 5, "t3": 4, "t4": 5, "t5": 3, "t6": 5}
K = 5


def run(argv: list[str]) -> tuple[int, str]:
    stream = io.StringIO()
    code = main(argv, stream=stream)
    return code, stream.getvalue()


def hand_built(action: str = "issue_refund", *, side: str = "incumbent",
               withheld: dict[str, int] = WITHHELD, k: int = K) -> str:
    lines = []
    for task, count in withheld.items():
        for index in range(k):
            fired = index >= count
            lines.append(EvidenceLine(
                task_id=task, side=side, run=index,
                facts={action: ActionFacts(fired=fired, count=1 if fired else 0)}))
    return write_evidence(lines)


class _TempDir(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)

    def tearDown(self) -> None:
        self.dir.cleanup()

    def write(self, name: str, text: str) -> Path:
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path


class StabilityMapTests(_TempDir):
    def test_class_counts_and_the_coin_flip_on_a_hand_built_set(self) -> None:
        path = self.write("evidence.jsonl", hand_built())
        code, text = run(["floor", str(path), "--json"])
        self.assertEqual(code, EXIT_PASS)
        smap = json.loads(text)["actions"][0]["stability_map"]
        self.assertEqual(smap["counts"], {STABLE_WITHHOLDS: 5, STABLE_FIRES: 0,
                                          COIN_FLIP: 1, NO_READABLE_RUN: 0})
        self.assertEqual(smap["coin_flip_task_ids"], ["t5"])
        by_id = {task["task_id"]: task for task in smap["tasks"]}
        self.assertEqual((by_id["t5"]["withheld"], by_id["t5"]["readable_runs"]), (3, 5))
        self.assertEqual(by_id["t3"]["class"], STABLE_WITHHOLDS)  # 4 of 5 decides at 0.8

    def test_the_text_lists_every_coin_flip_with_its_count_and_says_why(self) -> None:
        path = self.write("evidence.jsonl", hand_built())
        code, text = run(["floor", str(path)])
        self.assertEqual(code, EXIT_PASS)
        self.assertIn("stability map", text)
        self.assertRegex(text, r"stable-withholds\s+5 task\(s\)")
        self.assertRegex(text, r"coin-flip\s+1 task\(s\)")
        self.assertRegex(text, r"\n  t5\s+3/5\n")
        self.assertIn("under-determined", text)
        self.assertIn("whatever model you run", text)
        # Without --all the decided tasks are counted, not listed.
        self.assertNotRegex(text, r"\n  t1\s")

    def test_all_lists_every_task(self) -> None:
        path = self.write("evidence.jsonl", hand_built())
        _, text = run(["floor", str(path), "--all"])
        for task, count in WITHHELD.items():
            self.assertRegex(text, rf"\n  {task}\s+\S+\s+{count}/{K}\n")

    def test_fires_side_and_a_raised_theta(self) -> None:
        path = self.write("evidence.jsonl", hand_built(withheld={"a": 0, "b": 1, "c": 4}))
        _, text = run(["floor", str(path), "--json", "--theta", "1.0"])
        smap = json.loads(text)["actions"][0]["stability_map"]
        # At theta 1.0 only unanimity decides: a (0/5) fires, b (1/5) and c (4/5) flip.
        self.assertEqual(smap["counts"][STABLE_FIRES], 1)
        self.assertEqual(smap["coin_flip_task_ids"], ["b", "c"])

    def test_a_task_with_no_readable_run_is_not_called_a_coin_flip(self) -> None:
        lines, _ = read_evidence_text(hand_built())
        lines += [EvidenceLine(task_id="t7", side="incumbent", run=index, facts={},
                               unreadable=True) for index in range(K)]
        path = self.write("evidence.jsonl", write_evidence(lines))
        _, text = run(["floor", str(path), "--json"])
        smap = json.loads(text)["actions"][0]["stability_map"]
        self.assertEqual(smap["counts"][NO_READABLE_RUN], 1)
        self.assertEqual(smap["coin_flip_task_ids"], ["t5"])

    def test_the_floor_rule_is_the_referee_rule(self) -> None:
        """floor and check must never disagree about which task is a coin flip."""

        from aver_policy import repeated

        for k in range(1, 11):
            for theta in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
                self.assertEqual(required_agreement(k, theta),
                                 repeated.required_agreement(k, theta), (k, theta))
                for blocks in range(k + 1):
                    decisions = (1,) * blocks + (0,) * (k - blocks)
                    self.assertEqual(
                        decide(decisions, theta),
                        repeated.decide(repeated.RepeatedRecord(decisions=decisions), theta),
                        (k, theta, blocks))

    def test_floor_needs_only_one_side(self) -> None:
        """`aver repeat` then `aver floor`: one side in the file is enough, and the
        side is read from the file when it holds only one."""

        path = self.write("evidence.jsonl", hand_built(side="candidate"))
        code, text = run(["floor", str(path), "--json"])
        self.assertEqual(code, EXIT_PASS)
        payload = json.loads(text)
        self.assertEqual(payload["side"], "candidate")
        self.assertEqual(payload["actions"][0]["tasks"], len(WITHHELD))
        code, text = run(["floor", str(path), "--side", "incumbent"])
        self.assertEqual(code, EXIT_USAGE)
        self.assertIn("no runs on the incumbent side", text)


class SelfDiffTests(_TempDir):
    def fixture_evidence_path(self, name: str = "grok-refund") -> Path:
        spec = DEMOS[name]
        return self.write("evidence.jsonl",
                          write_evidence(fixture_evidence(spec, load_fixture(spec))))

    @unittest.skipUnless(SUMMARY.exists(), STUDY_REASON)
    def test_the_self_diff_line_reproduces_the_study_analysis(self) -> None:
        expected = json.loads(SUMMARY.read_text(encoding="utf-8"))["refund"][
            "inc_self_rep0_vs_rep1"]
        path = self.fixture_evidence_path()
        code, text = run(["floor", str(path), "--json"])
        self.assertEqual(code, EXIT_PASS)
        diff = json.loads(text)["actions"][0]["one_run_self_diff"]
        self.assertEqual(diff["block_to_allow"], expected["u"])
        self.assertEqual(diff["guarded"], expected["m"])
        self.assertEqual(diff["allow_to_block"], expected["f"])
        self.assertAlmostEqual(diff["rate"], expected["rate"], delta=1e-9)
        self.assertAlmostEqual(diff["interval"]["lower"], expected["ci"][0], delta=1e-9)
        self.assertAlmostEqual(diff["interval"]["upper"], expected["ci"][1], delta=1e-9)
        self.assertEqual(diff["verdict"], expected["verdict"])

        _, plain = run(["floor", str(path)])
        headline = (f"a one-run diff of this agent against itself would report: "
                    f"{expected['verdict'].upper()} ({expected['u']}/{expected['m']} = "
                    f"{expected['rate'] * 100:.1f}%)")
        self.assertIn(headline, plain)
        self.assertIn(f"withheld then fired  {expected['u']}/{expected['m']}", plain)

    @unittest.skipUnless(v3.available(), V3_REASON)
    def test_floor_and_the_demo_agree_on_the_fixture(self) -> None:
        path = self.fixture_evidence_path()
        _, floor = run(["floor", str(path), "--json"])
        _, demo = run(["demo", "grok-refund", "--json"])
        floor_entry, demo_payload = json.loads(floor)["actions"][0], json.loads(demo)
        self.assertEqual(floor_entry["one_run_self_diff"]["verdict"],
                         demo_payload["golden_trace_diff"]["verdict"])
        for key in ("block_to_allow", "guarded", "allow_to_block", "rate", "interval"):
            self.assertEqual(floor_entry["one_run_self_diff"][key],
                             demo_payload["golden_trace_diff"][key], key)
        self.assertEqual(floor_entry["stability_map"]["counts"],
                         demo_payload["stability_map"]["counts"])
        self.assertEqual(floor_entry["churn_rate"], demo_payload["floor"]["churn_rate"])

    def test_the_tolerance_flag_moves_the_self_diff_verdict(self) -> None:
        path = self.fixture_evidence_path()
        _, text = run(["floor", str(path), "--json", "--tolerance", "0.10"])
        self.assertEqual(json.loads(text)["actions"][0]["one_run_self_diff"]["verdict"],
                         "pass")

    def test_one_run_per_task_has_no_self_diff(self) -> None:
        path = self.write("evidence.jsonl", hand_built(withheld={"a": 1, "b": 0}, k=1))
        code, text = run(["floor", str(path)])
        self.assertEqual(code, EXIT_PASS)
        self.assertIn("needs two runs of a task", text)
        _, raw = run(["floor", str(path), "--json"])
        self.assertIsNone(json.loads(raw)["actions"][0]["one_run_self_diff"])


class FloorJsonShapeTests(_TempDir):
    def test_json_shape(self) -> None:
        path = self.write("evidence.jsonl", hand_built())
        _, text = run(["floor", str(path), "--json"])
        payload = json.loads(text)
        self.assertEqual(set(payload), {"side", "evidence_digest", "actions"})
        entry = payload["actions"][0]
        for key in ("churn_rate", "undecided_tasks", "stability_map", "one_run_self_diff"):
            self.assertIn(key, entry)
        smap = entry["stability_map"]
        self.assertEqual(set(smap), {"side", "action", "decidability_threshold", "counts",
                                     "coin_flip_task_ids", "tasks"})
        self.assertEqual(set(smap["tasks"][0]),
                         {"task_id", "class", "withheld", "readable_runs"})
        self.assertEqual(set(entry["one_run_self_diff"]),
                         {"label", "action", "block_to_allow", "guarded", "allow_to_block",
                          "rate", "interval", "tolerance", "verdict", "flip_ids"})
        # the coin-flip count is the undecided count floor already reported
        self.assertEqual(smap["counts"][COIN_FLIP], entry["undecided_tasks"])


class AnyBinaryFactTests(_TempDir):
    """A guarded action is any yes/no fact a harness can print, not only a tool call.

    Built without any importer: the evidence format carries an arbitrary action
    name through floor and check unchanged.
    """

    ACTION = "ran_destructive_git_command"

    def test_an_arbitrary_action_name_round_trips_through_floor_and_check(self) -> None:
        lines = []
        for index in range(40):
            for side in ("incumbent", "candidate"):
                for rep in range(5):
                    fired = side == "candidate" and index < 2 and rep < 4
                    lines.append(EvidenceLine(
                        task_id=f"task_{index:03d}", side=side, run=rep,
                        facts={self.ACTION: ActionFacts(fired=fired,
                                                        count=1 if fired else 0)}))
        text = write_evidence(lines)
        back, report = read_evidence_text(text)
        self.assertEqual(report.skipped_total, 0)
        self.assertEqual({name for line in back for name in line.facts}, {self.ACTION})
        path = self.write("evidence.jsonl", text)

        code, out = run(["floor", str(path), "--json"])
        self.assertEqual(code, EXIT_PASS)
        entry = json.loads(out)["actions"][0]
        self.assertEqual(entry["action"], self.ACTION)
        self.assertEqual(entry["stability_map"]["counts"][STABLE_WITHHOLDS], 40)

        if not v3.available():
            return
        policy = self.write("aver.json", json.dumps({
            "schema": "aver-policy/v3",
            "decision_point": "repository maintenance",
            "actions": [{"name": self.ACTION, "direction": "fire_is_harmful",
                         "tolerance": 0.05, "min_evidence": 5}],
        }))
        code, out = run(["check", str(path), "--policy", str(policy), "--json"])
        self.assertIn(code, (EXIT_PASS, EXIT_REFUSE, EXIT_BLOCK))
        payload = json.loads(out)
        self.assertEqual([action["action"] for action in payload["actions"]], [self.ACTION])


class HelpTests(unittest.TestCase):
    def test_the_help_names_the_front_door_and_the_general_fact(self) -> None:
        from aver_cli.cli import build_parser

        text = " ".join(build_parser().format_help().split()).lower()
        self.assertIn("measure how much your agent disagrees with itself, "
                      "then referee a change", text)
        self.assertIn("a guarded action is any yes/no fact about a run", text)
        self.assertIn("a destructive command ran", text)
        # The local decision group is discoverable; floor still precedes check.
        self.assertIn("compare agent pipeline cost and reliability", text)
        self.assertLess(text.index("{decision,floor,check"), text.index("verify}"))


@unittest.skipUnless(v3.available(), V3_REASON)
class DemoOrderTests(unittest.TestCase):
    def test_the_floor_leads_and_the_referee_follows(self) -> None:
        _, text = run(["demo", "grok-refund"])
        positions = [text.index(title) for title in DEMO_TITLES]
        self.assertEqual(positions, sorted(positions))
        self.assertTrue(DEMO_TITLES[0].startswith("1. The floor"))
        self.assertIn("one-run diff", DEMO_TITLES[1])
        self.assertIn("referee", DEMO_TITLES[2])
        block1 = text.split(DEMO_TITLES[0])[1].split(DEMO_TITLES[1])[0]
        self.assertIn("stability map", block1)
        self.assertIn("coin-flip", block1)
        block2 = text.split(DEMO_TITLES[1])[1].split(DEMO_TITLES[2])[0]
        self.assertIn("a one-run diff of this agent against itself would report: BLOCK",
                      block2)
        block3 = text.split(DEMO_TITLES[2])[1].split(DEMO_TITLES[3])[0]
        self.assertIn("what to do next", block3)

    def test_block_four_leads_with_the_flaky_test_framing(self) -> None:
        for name in DEMOS:
            with self.subTest(name=name):
                _, text = run(["demo", name])
                block4 = text.split(DEMO_TITLES[3])[1].lstrip("-\n")
                self.assertTrue(block4.startswith("Agent regression tests are flaky tests."),
                                block4[:120])
                for phrase in ("disagrees with itself", "stability map", "coin-flip",
                               "interval", "what\nevidence is missing"):
                    self.assertIn(phrase, block4)
                self.assertNotIn("proprietary", block4.lower())

    def test_the_demo_evidence_file_reproduces_block_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.jsonl"
            code, text = run(["demo", "grok-refund", "--evidence-out", str(path)])
            self.assertEqual(code, EXIT_PASS)
            self.assertNotIn("grok-4", path.read_text(encoding="utf-8"))
            _, floor = run(["floor", str(path)])
        _, demo = run(["demo", "grok-refund"])
        block1 = demo.split(DEMO_TITLES[0])[1].split(DEMO_TITLES[1])[0]
        body = block1.split("\n", 3)[3].strip()  # drop the rule and the one-line context
        self.assertIn(body, floor)


if __name__ == "__main__":
    unittest.main()
