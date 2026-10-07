"""Build the bundled demo fixtures from the X10 repeat study.

Reads ``research_data/x10/decisions.jsonl`` (private research data, not shipped)
and writes the two fixtures that ``aver demo`` reads (committed, and shipped in
the wheel). The fixture carries decision tuples and a ground-truth side and
nothing else: no prompts, no completions, no tokens, no free text. A reader who
wants to know what the tasks said has to go to the study, not to the package.

Task selection matches ``outputs/agent-referee-2026-09-22/analyze_real.py``
exactly -- tasks with a full k=5 on *both* sides -- because ``aver demo``'s
numbers are pinned against that script's output on the same data
(``real_arm_summary_complete.json`` since the owner's 2026-09-22 rerun). Every
task the selection drops is counted in ``meta.excluded`` with its reason, so the
fixture cannot quietly shrink, and ``meta.known_gaps`` is written only when
something was dropped.

Run ``python -m aver_cli.fixtures.build_x10 --check`` to verify the committed
fixtures match the data, which is what CI runs (AGENTS.md: CI runs the check,
not the generator).
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "research_data" / "x10" / "decisions.jsonl"
FIXTURE_DIR = Path(__file__).resolve().parent

INCUMBENT_MODEL = "xai-grok-4-3"
CANDIDATE_MODEL = "xai-grok-4-20-fast"
K = 5

#: One fixture per decision point, named as ``aver demo`` names it.
DECISION_POINTS: dict[str, str] = {"grok-refund": "refund", "grok-escalation": "escalation"}

CASE_KEYS = frozenset({"task_id", "ground_truth_side", "incumbent", "candidate"})
#: ``models`` is what docs/104 §T-E1 requires. The rest is provenance a reader
#: needs in order to trust the numbers: which study, how many tasks the source
#: had, and which ones this fixture could not use. None of it is free text from a
#: task.
META_KEYS = frozenset({
    "models", "decision_point", "source", "reps", "tasks",
    "excluded", "known_gaps",
})


def load_rows(source: Path) -> list[dict]:
    rows = []
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build(decision_point: str, rows: list[dict]) -> dict:
    """One fixture: per-task decision tuples for both sides, plus provenance."""

    runs: dict[str, dict[str, dict[int, int]]] = collections.defaultdict(
        lambda: {INCUMBENT_MODEL: {}, CANDIDATE_MODEL: {}}
    )
    ground_truth: dict[str, str] = {}
    invalid_units = 0
    seen_tasks: set[str] = set()

    for row in rows:
        if row.get("decision_point") != decision_point:
            continue
        model = row.get("model_id")
        if model not in (INCUMBENT_MODEL, CANDIDATE_MODEL):
            continue
        task_id = str(row.get("example_id") or "")
        if not task_id:
            continue
        seen_tasks.add(task_id)
        ground_truth.setdefault(task_id, str(row.get("ground_truth_side") or ""))
        if row.get("provider_error") or not row.get("valid") or row.get("block") is None:
            invalid_units += 1
            continue
        runs[task_id][model][int(row["rep"])] = 1 if int(row["block"]) == 1 else 0

    cases = []
    short_side: list[str] = []
    for task_id in sorted(seen_tasks):
        per_model = runs.get(task_id) or {INCUMBENT_MODEL: {}, CANDIDATE_MODEL: {}}
        incumbent = [per_model[INCUMBENT_MODEL][rep]
                     for rep in sorted(per_model[INCUMBENT_MODEL])]
        candidate = [per_model[CANDIDATE_MODEL][rep]
                     for rep in sorted(per_model[CANDIDATE_MODEL])]
        if len(incumbent) != K or len(candidate) != K:
            short_side.append(task_id)
            continue
        cases.append({
            "task_id": task_id,
            "ground_truth_side": ground_truth.get(task_id, ""),
            "incumbent": incumbent,
            "candidate": candidate,
        })

    meta = {
        "models": {"incumbent": INCUMBENT_MODEL, "candidate": CANDIDATE_MODEL},
        "decision_point": decision_point,
        "source": "research_data/x10/decisions.jsonl (X10 repeat study, 2026-09-22)",
        "reps": K,
        "tasks": len(cases),
        "excluded": {
            "tasks_without_full_repeats_on_both_sides": len(short_side),
            "invalid_or_provider_error_units": invalid_units,
        },
    }
    gap = known_gaps(len(short_side), invalid_units)
    if gap:
        meta["known_gaps"] = gap
    return {"meta": meta, "cases": cases}


def known_gaps(short_tasks: int, invalid_units: int) -> str | None:
    """What the fixture could not use, from the counts; ``None`` when nothing."""

    if not short_tasks and not invalid_units:
        return None
    return (
        f"{short_tasks} task(s) lack a full {K} repeats on one side or both, and the "
        f"source has {invalid_units} invalid or provider-error unit(s). A task missing "
        "any of its repeats is excluded rather than padded, so this fixture is a "
        "subset of the study's tasks; inventing the missing runs would make the "
        "demo's numbers fiction."
    )


def assert_shape(fixture: dict) -> None:
    """The fixture carries decision tuples and nothing a reader could be doxxed by."""

    assert set(fixture) == {"meta", "cases"}, sorted(fixture)
    assert set(fixture["meta"]) <= META_KEYS, sorted(set(fixture["meta"]) - META_KEYS)
    assert set(fixture["meta"]["models"]) == {"incumbent", "candidate"}
    assert fixture["cases"], "fixture has no cases"
    for case in fixture["cases"]:
        assert set(case) == CASE_KEYS, sorted(set(case) ^ CASE_KEYS)
        assert isinstance(case["task_id"], str)
        assert case["ground_truth_side"] in ("benign", "adversarial", "")
        for side in ("incumbent", "candidate"):
            assert len(case[side]) == K, (case["task_id"], side)
            assert all(value in (0, 1) for value in case[side]), case["task_id"]


def fixture_path(name: str) -> Path:
    return FIXTURE_DIR / f"x10_{DECISION_POINTS[name]}.json"


def render(fixture: dict) -> str:
    return json.dumps(fixture, indent=1, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m aver_cli.fixtures.build_x10",
        description="Build or verify the bundled demo fixtures.",
    )
    parser.add_argument("--check", action="store_true",
                        help="verify the committed fixtures match the source data")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    args = parser.parse_args(argv)

    source = Path(args.source)
    if not source.exists():
        print(f"source data not found: {source}", file=sys.stderr)
        if args.check:
            print("cannot verify the fixtures without the study data; "
                  "this check only runs where research_data/ is present.", file=sys.stderr)
        return 3

    rows = load_rows(source)
    drift = 0
    for name, decision_point in DECISION_POINTS.items():
        fixture = build(decision_point, rows)
        assert_shape(fixture)
        text = render(fixture)
        path = fixture_path(name)
        size_kb = len(text.encode("utf-8")) / 1024
        assert size_kb <= 200, f"{path.name} is {size_kb:.0f} KB, over the 200 KB limit"
        if args.check:
            current = path.read_text(encoding="utf-8") if path.exists() else ""
            if current != text:
                print(f"DRIFT: {path.name} does not match the source data", file=sys.stderr)
                drift += 1
            else:
                print(f"ok: {path.name} ({fixture['meta']['tasks']} tasks, {size_kb:.0f} KB)")
        else:
            path.write_text(text, encoding="utf-8")
            print(f"wrote {path.name}: {fixture['meta']['tasks']} tasks, {size_kb:.0f} KB")
    return 1 if drift else 0


if __name__ == "__main__":
    raise SystemExit(main())
