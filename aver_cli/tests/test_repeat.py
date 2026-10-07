"""``aver repeat``: it runs the customer's command, and it never invents a run.

The runner owns nothing but the loop. The harness decides what "fired" means; a
command that fails produces an ``unreadable`` line that is kept and counted,
because a harness that quietly drops its hard tasks would improve its own verdict.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

from aver_cli import EVIDENCE_SCHEMA
from aver_cli.cli import EXIT_PASS, EXIT_USAGE, main
from aver_cli.evidence import read_evidence_text


def run(argv: list[str]) -> tuple[int, str]:
    stream = io.StringIO()
    code = main(argv, stream=stream)
    return code, stream.getvalue()


ECHO_HARNESS = """\
import json, os, sys
task = os.environ["AVER_TASK_ID"]
run = int(os.environ["AVER_RUN"])
side = os.environ["AVER_SIDE"]
payload = json.loads(os.environ["AVER_INPUT_JSON"])
fired = bool(payload.get("refund")) and run % 2 == 0
print(json.dumps({
    "schema": SCHEMA,
    "task_id": task, "side": side, "run": run,
    "facts": {"issue_refund": {"fired": fired, "count": 1 if fired else 0}},
    "meta": {"harness": "echo"},
}))
""".replace("SCHEMA", repr(EVIDENCE_SCHEMA))

FAILING_HARNESS = "import sys\nsys.exit(7)\n"
SILENT_HARNESS = "pass\n"
GARBAGE_HARNESS = "print('not json at all')\n"


class RepeatRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        self.tasks = self.root / "tasks.jsonl"
        self.tasks.write_text(
            json.dumps({"task_id": "tkt_1", "input": {"refund": True}}) + "\n"
            + json.dumps({"task_id": "tkt_2", "input": {"refund": False}}) + "\n",
            encoding="utf-8")
        self.out = self.root / "evidence.jsonl"

    def tearDown(self) -> None:
        self.dir.cleanup()

    def harness(self, source: str, name: str = "harness.py") -> str:
        path = self.root / name
        path.write_text(source, encoding="utf-8")
        return f"{sys.executable} {path}"

    def lines(self):
        lines, report = read_evidence_text(self.out.read_text(encoding="utf-8"))
        self.assertEqual(report.skipped_total, 0, report.first_reasons)
        return lines

    def test_it_collects_k_runs_per_task_and_passes_the_environment(self) -> None:
        code, text = run(["repeat", "--cmd", self.harness(ECHO_HARNESS),
                          "--tasks", str(self.tasks), "-k", "3",
                          "--side", "candidate", "--out", str(self.out)])
        self.assertEqual(code, EXIT_PASS)
        lines = self.lines()
        self.assertEqual(len(lines), 6)
        self.assertEqual({line.side for line in lines}, {"candidate"})
        self.assertEqual(sorted(line.run for line in lines if line.task_id == "tkt_1"),
                         [0, 1, 2])
        # AVER_INPUT_JSON reached the harness: only tkt_1 has refund=True, and only
        # on even runs.
        fired = {(line.task_id, line.run) for line in lines
                 if line.facts["issue_refund"].fired}
        self.assertEqual(fired, {("tkt_1", 0), ("tkt_1", 2)})
        self.assertIn("collected       6 run(s)", text)

    def test_it_is_resumable_and_skips_runs_it_already_has(self) -> None:
        cmd = self.harness(ECHO_HARNESS)
        run(["repeat", "--cmd", cmd, "--tasks", str(self.tasks), "-k", "2",
             "--side", "candidate", "--out", str(self.out)])
        first = self.out.read_text(encoding="utf-8")
        code, text = run(["repeat", "--cmd", cmd, "--tasks", str(self.tasks), "-k", "2",
                          "--side", "candidate", "--out", str(self.out), "--json"])
        self.assertEqual(code, EXIT_PASS)
        summary = json.loads(text)
        self.assertEqual(summary["runs_collected"], 0)
        self.assertEqual(summary["runs_skipped_already_present"], 4)
        self.assertEqual(self.out.read_text(encoding="utf-8"), first)

    def test_resuming_with_a_larger_k_only_adds_the_new_runs(self) -> None:
        cmd = self.harness(ECHO_HARNESS)
        run(["repeat", "--cmd", cmd, "--tasks", str(self.tasks), "-k", "2",
             "--side", "candidate", "--out", str(self.out)])
        code, text = run(["repeat", "--cmd", cmd, "--tasks", str(self.tasks), "-k", "4",
                          "--side", "candidate", "--out", str(self.out), "--json"])
        self.assertEqual(code, EXIT_PASS)
        summary = json.loads(text)
        self.assertEqual(summary["runs_collected"], 4)
        self.assertEqual(summary["runs_skipped_already_present"], 4)
        self.assertEqual(len(self.lines()), 8)

    def test_a_nonzero_exit_becomes_an_unreadable_line_that_is_kept(self) -> None:
        code, text = run(["repeat", "--cmd", self.harness(FAILING_HARNESS),
                          "--tasks", str(self.tasks), "-k", "2",
                          "--side", "candidate", "--out", str(self.out), "--json"])
        self.assertEqual(code, EXIT_PASS)
        self.assertEqual(json.loads(text)["unreadable_runs"], 4)
        lines = self.lines()
        self.assertEqual(len(lines), 4)
        self.assertTrue(all(line.unreadable for line in lines))
        self.assertIn("exited 7", lines[0].meta["unreadable_reason"])

    def test_no_output_and_garbage_output_are_both_unreadable_with_a_reason(self) -> None:
        for source, needle in ((SILENT_HARNESS, "no output"),
                               (GARBAGE_HARNESS, "not one aver-evidence/v1 line")):
            with self.subTest(needle=needle):
                self.out.unlink(missing_ok=True)
                run(["repeat", "--cmd", self.harness(source, "h.py"),
                     "--tasks", str(self.tasks), "-k", "1",
                     "--side", "candidate", "--out", str(self.out)])
                lines = self.lines()
                self.assertTrue(all(line.unreadable for line in lines))
                self.assertIn(needle, lines[0].meta["unreadable_reason"])

    def test_the_runner_overrides_task_side_and_run_from_its_own_loop(self) -> None:
        # A harness that reports the wrong task, side or run index must not be able
        # to move a run to a different cell.
        liar = self.harness(
            'import json\nprint(json.dumps({"schema": SCHEMA, "task_id": "WRONG",'
            ' "side": "incumbent", "run": 99, "facts": {}}))\n'
            .replace("SCHEMA", repr(EVIDENCE_SCHEMA)),
            "liar.py")
        run(["repeat", "--cmd", liar, "--tasks", str(self.tasks), "-k", "1",
             "--side", "candidate", "--out", str(self.out)])
        lines = self.lines()
        self.assertEqual({line.side for line in lines}, {"candidate"})
        self.assertEqual({line.run for line in lines}, {0})
        self.assertEqual({line.task_id for line in lines}, {"tkt_1", "tkt_2"})

    def test_a_timeout_is_an_unreadable_run_not_a_hang(self) -> None:
        slow = self.harness("import time\ntime.sleep(30)\n", "slow.py")
        code, _ = run(["repeat", "--cmd", slow, "--tasks", str(self.tasks), "-k", "1",
                       "--side", "candidate", "--out", str(self.out),
                       "--timeout", "0.5"])
        self.assertEqual(code, EXIT_PASS)
        lines = self.lines()
        self.assertTrue(all(line.unreadable for line in lines))
        self.assertIn("timed out", lines[0].meta["unreadable_reason"])

    def test_a_task_line_with_no_task_id_is_a_usage_error(self) -> None:
        self.tasks.write_text(json.dumps({"input": {}}) + "\n", encoding="utf-8")
        code, text = run(["repeat", "--cmd", self.harness(ECHO_HARNESS),
                          "--tasks", str(self.tasks), "-k", "1",
                          "--side", "candidate", "--out", str(self.out)])
        self.assertEqual(code, EXIT_USAGE)
        self.assertIn("task_id", text)

    def test_a_malformed_task_file_is_a_usage_error_naming_the_line(self) -> None:
        self.tasks.write_text('{"task_id": "a"}\nnot json\n', encoding="utf-8")
        code, text = run(["repeat", "--cmd", "true", "--tasks", str(self.tasks),
                          "--out", str(self.out)])
        self.assertEqual(code, EXIT_USAGE)
        self.assertIn("line 2", text)

    def test_an_empty_task_file_is_a_usage_error(self) -> None:
        self.tasks.write_text("\n", encoding="utf-8")
        code, _ = run(["repeat", "--cmd", "true", "--tasks", str(self.tasks),
                       "--out", str(self.out)])
        self.assertEqual(code, EXIT_USAGE)

    def test_k_below_one_is_a_usage_error(self) -> None:
        code, _ = run(["repeat", "--cmd", "true", "--tasks", str(self.tasks),
                       "-k", "0", "--out", str(self.out)])
        self.assertEqual(code, EXIT_USAGE)

    def test_the_runner_calls_no_provider_itself(self) -> None:
        # The only way a run happens is the command the customer passed. There is
        # no HTTP client anywhere in the CLI.
        import aver_cli.cli as cli_module

        source = Path(cli_module.__file__).read_text(encoding="utf-8")
        for forbidden in ("urllib.request", "http.client", "requests", "socket"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
