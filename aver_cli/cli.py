"""``aver`` — measure how much an agent disagrees with itself, then referee a change.

``aver floor`` is the front door: the churn floor, the per-task stability map and
what a one-run diff of the agent against itself would report. ``aver check``
follows, and referees a candidate against written acceptance rules.

Plain text with fixed columns, ``--json`` on every command, and four exit codes:
0 pass, 1 refuse, 2 block, 3 usage or parse error. A refusal is not a failed
build; it is the referee saying the evidence cannot answer the question yet, and
it always carries a cure.

No provider is called, no traffic is replayed, no output is judged, and nothing
is enforced at run time.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TextIO

from aver_cli import EVIDENCE_SCHEMA, __version__
from aver_cli import demo as demo_module
from aver_cli import v3
from aver_cli.evidence import (
    SIDE_CANDIDATE,
    SIDE_INCUMBENT,
    SIDES,
    EvidenceLine,
    evidence_digest,
    group_cases,
    read_evidence_text,
    side_cases,
    sides_present,
    write_evidence,
)
from aver_cli.importers import (
    FORMAT_NAMES,
    ImportContext,
    ImportResult,
    SideUndecidable,
    UnknownFormat,
    detect_format,
    read_with,
)
from aver_cli.policy import (
    CliPolicy,
    PolicyError,
    UnsupportedClause,
    load_policy,
)
from aver_cli.referee import (
    COIN_FLIP,
    NO_READABLE_RUN,
    PROPENSITY_BIN_LABELS,
    STABILITY_CLASSES,
    STABLE_FIRES,
    STABLE_WITHHOLDS,
    floor_readout,
    golden_diff,
    run_map,
    self_diff,
    stability_map,
)

EXIT_PASS = 0
EXIT_REFUSE = 1
EXIT_BLOCK = 2
EXIT_USAGE = 3

_STATE_EXIT = {"pass": EXIT_PASS, "review": EXIT_PASS, "refuse": EXIT_REFUSE, "block": EXIT_BLOCK}

RULE = "-" * 72

#: docs/104 §8: one sentence, and the only pointer to the paid product. It is a
#: registry claim that is not production-advertisable, so it is printed only when
#: the reader asks for it.
HOSTED_POINTER = (
    "A signed assurance record can be issued from this evidence at "
    "${STACKCERT_PUBLIC_ORIGIN}."
)


def _out(stream: Optional[TextIO]) -> TextIO:
    return stream or sys.stdout


def _print_json(payload: dict, stream: TextIO) -> None:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _v3_unavailable(stream: TextIO) -> int:
    """The one place that reports a missing v3 engine (docs/103 T-D1)."""

    print(v3.V3_MISSING_MESSAGE, file=stream)
    return EXIT_USAGE


# --------------------------------------------------------------------------- #
# shared loading
# --------------------------------------------------------------------------- #

@dataclass
class LoadedEvidence:
    lines: list[EvidenceLine]
    report: object
    digest: str


def _load_evidence(paths: Sequence[str], stream: TextIO) -> Optional[LoadedEvidence]:
    lines: list[EvidenceLine] = []
    report = None
    for path in paths:
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            print(f"aver: cannot read {path}: {exc.strerror}", file=stream)
            return None
        chunk, report = read_evidence_text(text, where=path, report=report)  # type: ignore[arg-type]
        report.files.append(path)
        lines.extend(chunk)
    if report is None:
        print("aver: no evidence files given", file=stream)
        return None
    if report.skipped_total:
        print(f"aver: {report.skipped_total} line(s) were not readable as {EVIDENCE_SCHEMA}:",
              file=stream)
        for reason, count in sorted(report.skipped.items()):
            print(f"      {count:>6}  {reason}", file=stream)
    if not lines:
        print("aver: no readable evidence lines", file=stream)
        return None
    return LoadedEvidence(lines=lines, report=report, digest=evidence_digest(lines))


def _load_policy(path: str, stream: TextIO) -> tuple[Optional[CliPolicy], int]:
    try:
        return load_policy(path), EXIT_PASS
    except UnsupportedClause as exc:
        print("REFUSE  unsupported_clause", file=stream)
        print(f"        this build does not implement: {', '.join(exc.keys)}", file=stream)
        print("        A clause is enforced or rejected, never silently stored: a rule the",
              file=stream)
        print("        referee ignores would be reported as applied. Remove it, or use a",
              file=stream)
        print("        build that implements it.", file=stream)
        return None, EXIT_REFUSE
    except PolicyError as exc:
        print(f"aver: {path}: {exc}", file=stream)
        return None, EXIT_USAGE


# --------------------------------------------------------------------------- #
# aver import
# --------------------------------------------------------------------------- #

def cmd_import(args: argparse.Namespace, stream: TextIO) -> int:
    actions = tuple(name.strip() for name in args.actions.split(",") if name.strip())
    if not actions:
        print("aver: --actions must name at least one guarded action", file=stream)
        return EXIT_USAGE

    side = None if args.side == "auto" else args.side
    context = ImportContext(
        actions=actions,
        side=side,
        incumbent_model=args.incumbent_model,
        candidate_model=args.candidate_model,
        incumbent_provider=args.incumbent_provider,
        candidate_provider=args.candidate_provider,
        task_key=args.task_key,
        harness=args.harness or "",
        directions=tuple((name, args.direction) for name in actions),
    )

    combined = ImportResult()
    for path in args.files:
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            print(f"aver: cannot read {path}: {exc.strerror}", file=stream)
            return EXIT_USAGE
        payload = _parse_payload(text)
        if payload is None:
            print(f"aver: {path}: not JSON or JSONL", file=stream)
            return EXIT_USAGE
        try:
            fmt = args.format if args.format != "auto" else detect_format(payload)
            result = read_with(fmt, payload, context)
        except UnknownFormat as exc:
            print(f"aver: {path}: {exc}", file=stream)
            return EXIT_USAGE
        except SideUndecidable as exc:
            print(f"aver: {path}: {exc}", file=stream)
            return EXIT_USAGE
        combined.detected_format = combined.detected_format or result.detected_format
        combined.extend(result)

    lines, group = group_cases(combined.lines, actions=context.directions)
    per_side = {side_name: group.tasks.get(side_name, 0) for side_name in SIDES}
    unknown_actions = sorted(combined.actions_seen - set(actions))
    unseen_actions = sorted(set(actions) - combined.actions_seen)

    preview = {
        "schema": EVIDENCE_SCHEMA,
        "detected_format": combined.detected_format,
        "tasks_per_side": per_side,
        "runs_kept": len(combined.lines),
        "actions_requested": list(actions),
        "actions_seen_not_requested": unknown_actions,
        "actions_requested_never_seen": unseen_actions,
        **group.to_dict(),
        "import_skips": combined.to_dict(),
    }

    if args.json:
        _print_json(preview, _out(stream))
    else:
        print(f"format          {combined.detected_format}", file=stream)
        print(f"runs read       {len(combined.lines)}", file=stream)
        for side_name in SIDES:
            hist = group.runs_per_task.get(side_name, {})
            shape = ", ".join(f"{count} task(s) x {runs} run(s)"
                              for runs, count in sorted(hist.items(), key=lambda kv: int(kv[0])))
            print(f"{side_name:<15} {per_side[side_name]:>5} task(s)   {shape or '-'}",
                  file=stream)
        print(f"unreadable runs {group.unreadable_runs}", file=stream)
        if group.duplicate_runs:
            print(f"duplicate runs  {group.duplicate_runs} (same task, side and run index)",
                  file=stream)
        if group.tasks_one_side_only:
            print(f"one side only   {len(group.tasks_one_side_only)} task(s) "
                  "(kept in this report, excluded from paired cases)", file=stream)
        if unknown_actions:
            print(f"seen, not asked {', '.join(unknown_actions)}", file=stream)
        if unseen_actions:
            print(f"asked, unseen   {', '.join(unseen_actions)}", file=stream)
        if combined.skipped_total:
            print(f"records skipped {combined.skipped_total}", file=stream)
            for reason, count in sorted(combined.skipped.items()):
                print(f"                {count:>5}  {reason}", file=stream)

    empty = [side_name for side_name in SIDES if per_side[side_name] == 0]
    if empty:
        print(f"aver: refusing to write: no tasks on {', '.join(empty)}. "
              "A one-sided file cannot be refereed.", file=stream)
        return EXIT_USAGE

    out_path = Path(args.out)
    out_path.write_text(write_evidence(combined.lines), encoding="utf-8")
    if not args.json:
        print(f"wrote {out_path} ({len(combined.lines)} run(s), "
              f"{len(lines)} paired case(s))", file=stream)
    return EXIT_PASS


def _parse_payload(text: str) -> object:
    stripped = text.strip()
    if not stripped:
        return []
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    rows = []
    for line in stripped.splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            return None
    return rows


# --------------------------------------------------------------------------- #
# aver floor
# --------------------------------------------------------------------------- #

def cmd_floor(args: argparse.Namespace, stream: TextIO) -> int:
    loaded = _load_evidence([args.evidence], stream)
    if loaded is None:
        return EXIT_USAGE

    actions = sorted({name for line in loaded.lines for name in line.facts})
    if args.action:
        if args.action not in actions:
            print(f"aver: no action named {args.action!r} in this evidence "
                  f"(found: {', '.join(actions) or 'none'})", file=stream)
            return EXIT_USAGE
        actions = [args.action]
    if not actions:
        print("aver: this evidence carries no guarded actions", file=stream)
        return EXIT_USAGE

    present = sides_present(loaded.lines)
    side = args.side or (present[0] if len(present) == 1 else SIDE_INCUMBENT)
    if side not in present:
        print(f"aver: this evidence has no runs on the {side} side "
              f"(found: {', '.join(present) or 'none'})", file=stream)
        return EXIT_USAGE

    # floor reads every task of ONE side, paired or not: it is a fact about one
    # agent, and a team with only its current agent's repeats has one to read.
    pairs = tuple((name, "fire_is_harmful") for name in actions)
    cases = side_cases(loaded.lines, side=side, actions=pairs)
    profiles = [
        (
            floor_readout(cases, side=side, action=name, theta=args.theta),
            stability_map(cases, side=side, action=name, theta=args.theta),
            self_diff(cases, side=side, action=name, tolerance=args.tolerance),
        )
        for name in actions
    ]

    if args.json:
        entries = []
        for readout, smap, diff in profiles:
            entry = readout.to_dict()
            entry["stability_map"] = smap.to_dict()
            entry["one_run_self_diff"] = diff.to_dict() if diff is not None else None
            entries.append(entry)
        _print_json({
            "side": side,
            "evidence_digest": loaded.digest,
            "actions": entries,
        }, _out(stream))
        return EXIT_PASS

    print(f"repeat-churn floor: {side}", file=stream)
    print(RULE, file=stream)
    for readout, smap, diff in profiles:
        _print_floor_summary(readout, stream)
        print("", file=stream)
        _print_stability_map(smap, stream, show_all=args.all)
        print("", file=stream)
        _print_self_diff(diff, stream)
        print(RULE, file=stream)
    both = set(present) == set(SIDES)
    check = f"aver check {args.evidence} --policy aver.yaml"
    print("next            " + (f"{check} referees a change" if both else
                               f"add a candidate's runs to this file, then {check}"),
          file=stream)
    return EXIT_PASS


def _print_floor_summary(readout, stream: TextIO) -> None:
    print(f"action          {readout.action}", file=stream)
    print(f"tasks / runs    {readout.tasks} / {readout.runs}"
          f"   unreadable {readout.unreadable_runs}", file=stream)
    print(f"churn rate      {_pct(readout.p_noise)}"
          f"   [{_pct(readout.ci_low)}, {_pct(readout.ci_high)}] Wilson",
          file=stream)
    print(f"undecided at {readout.theta:<4.2f} {readout.undecided_tasks} task(s)"
          f"   ({_pct(readout.undecided_fraction)})", file=stream)
    print(f"byte-stable     {readout.byte_stable_tasks} task(s) gave the same "
          "answer every run", file=stream)
    print("propensity      " + "  ".join(
        f"{label} {count:>4}"
        for label, count in zip(PROPENSITY_BIN_LABELS, readout.propensity_bins)
    ), file=stream)
    print(f"regime          {readout.regime}", file=stream)
    if not readout.sufficient:
        print("                not enough repeat-observations to assert a floor yet",
              file=stream)
    print(f"\n{readout.sentence()}.", file=stream)


_CLASS_GLOSS = {
    STABLE_WITHHOLDS: "reliably withholds {action}",
    STABLE_FIRES: "reliably fires {action}",
    COIN_FLIP: "undecided: the runs do not agree",
    NO_READABLE_RUN: "no readable run",
}


def _print_stability_map(smap, stream: TextIO, *, show_all: bool) -> None:
    counts = smap.counts()
    print(f"stability map   per task, runs that withheld {smap.action} (n/k); "
          f"decided at {smap.theta:.2f}", file=stream)
    for name in STABILITY_CLASSES:
        if name == NO_READABLE_RUN and not counts[name]:
            continue
        gloss = _CLASS_GLOSS[name].format(action=smap.action)
        print(f"  {name:<18}{counts[name]:>5} task(s)   {gloss}", file=stream)

    width = max((len(task.task_id) for task in smap.tasks), default=8)
    flips = smap.coin_flips
    if flips:
        print(f"\ncoin-flip tasks: the prompt and tools leave what this agent does "
              f"here\nunder-determined. Fix these whatever model you run.", file=stream)
        for task in flips:
            print(f"  {task.task_id:<{width}}  {task.fraction:>5}", file=stream)
    else:
        print(f"\nno coin-flip tasks at {smap.theta:.2f} agreement.", file=stream)

    if show_all:
        print("\nevery task", file=stream)
        for task in smap.tasks:
            print(f"  {task.task_id:<{width}}  {task.stability:<18}{task.fraction:>5}",
                  file=stream)


def _self_diff_headline(diff) -> str:
    if diff.guarded == 0:
        return ("a one-run diff of this agent against itself would have nothing to "
                f"compare: run 0 was on the safe side of {diff.action} on no task")
    return (f"a one-run diff of this agent against itself would report: "
            f"{diff.verdict.upper()} ({diff.unsafe_flips}/{diff.guarded} = "
            f"{_pct(diff.unsafe_flips / diff.guarded)})")


def _print_self_diff(diff, stream: TextIO) -> None:
    if diff is None:
        print("one-run diff    needs two runs of a task on this side; this evidence "
              "has one", file=stream)
        return
    print("one-run diff    run 0 against run 1 of this same agent, on the same tasks",
          file=stream)
    _print_self_diff_detail(diff, stream)
    print(f"\n{_self_diff_headline(diff)}", file=stream)
    if diff.verdict == "block":
        print("Nothing changed between those two runs: that is this agent's own churn,",
              file=stream)
        print("not a regression.", file=stream)


def _print_self_diff_detail(diff, stream: TextIO,
                            direction: str = "fire_is_harmful") -> None:
    # The diff counts moves off the safe side. For an action whose firing is the
    # harm that is "withheld then fired"; for one whose withholding is the harm
    # (an escalation that must happen) it is the reverse.
    unsafe, friction = ("withheld then fired", "fired then withheld") \
        if direction == "fire_is_harmful" else ("fired then withheld", "withheld then fired")
    print(f"{unsafe}  {diff.unsafe_flips}/{diff.guarded} = {_pct(diff.rate)}"
          f"   [{_pct(diff.lower)}, {_pct(diff.upper)}]", file=stream)
    print(f"{friction}  {diff.friction_flips}", file=stream)
    print(f"tolerance       {_pct(diff.tolerance)}", file=stream)
    # A golden-trace diff reads the point estimate against the tolerance; it has
    # no interval rule. The interval above is printed for reference, and the
    # verdict must not be read as if it came from it (it can straddle).
    print("rule            point estimate vs tolerance (interval shown, not used)",
          file=stream)


# --------------------------------------------------------------------------- #
# aver check
# --------------------------------------------------------------------------- #

def cmd_check(args: argparse.Namespace, stream: TextIO) -> int:
    policy, code = _load_policy(args.policy, stream)
    if policy is None:
        return code
    loaded = _load_evidence([args.evidence], stream)
    if loaded is None:
        return EXIT_USAGE

    module = v3.repeated_module()
    if module is None:
        return _v3_unavailable(stream)

    cases, group = group_cases(
        loaded.lines, actions=policy.action_pairs, decision_point=policy.decision_point
    )
    if not cases:
        print("aver: no task appears on both sides; there is nothing to referee", file=stream)
        return EXIT_USAGE

    return _referee_and_report(
        cases=cases,
        policy=policy,
        module=module,
        stream=stream,
        as_json=args.json,
        json_path=args.json_out,
        evidence_digest_value=loaded.digest,
        show_hosted=args.show_hosted,
    )


def _referee_and_report(
    *,
    cases: list,
    policy: CliPolicy,
    module,
    stream: TextIO,
    as_json: bool,
    json_path: Optional[str],
    evidence_digest_value: str,
    show_hosted: bool,
) -> int:
    repeated_cases = v3.to_repeated_cases(cases, policy, module)
    verdict = module.compute_verdict_v3(repeated_cases, v3.engine_policy(policy, module))
    payload = dict(verdict.to_dict())
    payload["evidence_digest"] = evidence_digest_value
    payload["policy_digest"] = policy.digest()

    if json_path:
        Path(json_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                                   encoding="utf-8")
    if as_json:
        _print_json(payload, _out(stream))
    else:
        _print_verdict(payload, policy, stream, show_hosted=show_hosted)
    state = str(payload.get("state") or "refuse")
    return _STATE_EXIT.get(state, EXIT_REFUSE)


def _clause_counts(clause_key: str, clause: dict) -> str:
    """Clause A counts cases; Clause B is a rate over cases, so they read differently.

    Clause A's numerator is a count of tasks that switched. Clause B's is a sum of
    per-task propensity differences, so printing it raw is 17 digits of noise --
    the fraction it produces is the number that means something.
    """

    denominator = clause.get("denominator")
    if clause_key == "modal":
        numerator = clause.get("numerator")
        if numerator is None or denominator is None:
            return "-"
        return f"{int(numerator)}/{int(denominator)}"
    point = clause.get("point")
    if point is None or denominator is None:
        return "-"
    return f"{_pct(float(point))} of {int(denominator)}"


def _interval_text(interval: object) -> str:
    if isinstance(interval, dict):
        low, high = interval.get("lower"), interval.get("upper")
        if isinstance(low, (int, float)) and isinstance(high, (int, float)):
            return f"[{_pct(float(low))}, {_pct(float(high))}]"
    return "[-, -]"


def _regime_sentence(regime: dict) -> str:
    """The regime readout in the Console's words (docs/103 T-D3 copy rules)."""

    try:
        stable = (1.0 - float(regime["undecided_fraction"])) * 100
        churn = float(regime["p_noise"]) * 100
        low, high = float(regime["ci_low"]) * 100, float(regime["ci_high"]) * 100
    except (KeyError, TypeError, ValueError):
        return str(regime.get("regime", "unknown"))
    return (f"this agent is stable on {stable:.0f}% of cases; repeat churn "
            f"{churn:.1f}% [{low:.1f}, {high:.1f}] — {regime.get('regime', '?')}")


def _print_verdict(payload: dict, policy: CliPolicy, stream: TextIO, *,
                   show_hosted: bool) -> None:
    regime = payload.get("regime")
    if isinstance(regime, dict):
        print(f"regime          {_regime_sentence(regime)}", file=stream)
        print(RULE, file=stream)

    for action in payload.get("actions") or []:
        if not isinstance(action, dict):
            continue
        name = action.get("action", "?")
        print(f"action          {name}", file=stream)
        for clause_key, label in (("modal", "clause A  modal switch"),
                                  ("harm", "clause B  expected harm")):
            clause = action.get(clause_key)
            if not isinstance(clause, dict):
                continue
            print(f"  {label:<24} {_clause_counts(clause_key, clause):>14}"
                  f"   {_interval_text(clause.get('interval')):>18}"
                  f"   {clause.get('state', '-')}", file=stream)
        print("", file=stream)

    state = str(payload.get("state") or "refuse").upper()
    reason = payload.get("reason") or ""
    print(RULE, file=stream)
    print(f"{state}{('  ' + str(reason)) if reason else ''}", file=stream)

    cures = payload.get("cures") or []
    if cures:
        print("\nwhat to do next", file=stream)
        for cure in cures:
            if not isinstance(cure, dict):
                continue
            text = cure.get("text") or cure.get("kind") or ""
            print(f"  - {text}", file=stream)
            case_ids = cure.get("case_ids") or []
            if case_ids:
                print(f"    cases: {', '.join(str(item) for item in case_ids)}", file=stream)
            elif cure.get("count") is not None:
                print(f"    count: {cure['count']}", file=stream)
    elif state == "REFUSE":
        print("\naver: internal error — a refusal with no cure is a contract violation.",
              file=stream)

    if show_hosted:
        print(f"\n{HOSTED_POINTER}", file=stream)


# --------------------------------------------------------------------------- #
# aver repeat
# --------------------------------------------------------------------------- #

def cmd_repeat(args: argparse.Namespace, stream: TextIO) -> int:
    try:
        tasks = _read_tasks(Path(args.tasks))
    except OSError as exc:
        print(f"aver: cannot read {args.tasks}: {exc.strerror}", file=stream)
        return EXIT_USAGE
    except ValueError as exc:
        print(f"aver: {args.tasks}: {exc}", file=stream)
        return EXIT_USAGE
    if not tasks:
        print(f"aver: {args.tasks} contained no tasks", file=stream)
        return EXIT_USAGE
    if args.k < 1:
        print("aver: -k must be at least 1", file=stream)
        return EXIT_USAGE

    out_path = Path(args.out)
    done: set[tuple[str, str, int]] = set()
    existing: list[str] = []
    if out_path.exists():
        text = out_path.read_text(encoding="utf-8")
        lines, _ = read_evidence_text(text, where=str(out_path))
        for line in lines:
            done.add((line.task_id, line.side, line.run))
        existing = [line.to_json_line() for line in lines]

    produced: list[str] = []
    skipped = 0
    unreadable = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for line in existing:
            handle.write(line + "\n")
        for task in tasks:
            task_id = str(task.get("task_id") or "")
            if not task_id:
                print("aver: a task line has no task_id", file=stream)
                return EXIT_USAGE
            for run in range(args.k):
                if (task_id, args.side, run) in done:
                    skipped += 1
                    continue
                line = _run_once(args.cmd, task_id=task_id, run=run, side=args.side,
                                 task=task, timeout=args.timeout)
                if line.unreadable:
                    unreadable += 1
                handle.write(line.to_json_line() + "\n")
                handle.flush()
                produced.append(line.to_json_line())

    summary = {
        "out": str(out_path),
        "tasks": len(tasks),
        "k": args.k,
        "side": args.side,
        "runs_collected": len(produced),
        "runs_skipped_already_present": skipped,
        "unreadable_runs": unreadable,
    }
    if args.json:
        _print_json(summary, _out(stream))
    else:
        print(f"tasks           {len(tasks)}", file=stream)
        print(f"k               {args.k}   side {args.side}", file=stream)
        print(f"collected       {len(produced)} run(s)", file=stream)
        print(f"already present {skipped} run(s) — skipped", file=stream)
        print(f"unreadable      {unreadable} run(s)", file=stream)
        print(f"wrote {out_path}", file=stream)
    return EXIT_PASS


def _read_tasks(path: Path) -> list[dict]:
    tasks: list[dict] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {number} is not JSON: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"line {number} is not a JSON object")
        tasks.append(row)
    return tasks


def _run_once(command: str, *, task_id: str, run: int, side: str, task: dict,
              timeout: Optional[float]) -> EvidenceLine:
    """Run the customer's own command once and read one evidence line from stdout.

    The harness decides what "fired" means; this collects. A non-zero exit or
    stdout that is not one ``aver-evidence/v1`` line becomes an ``unreadable``
    run, which is kept and counted rather than dropped — a harness that fails on
    the hard tasks would otherwise quietly improve its own verdict.
    """

    env = dict(os.environ)
    env.update({
        "AVER_TASK_ID": task_id,
        "AVER_RUN": str(run),
        "AVER_SIDE": side,
        "AVER_INPUT_JSON": json.dumps(task.get("input", {}), sort_keys=True),
    })
    reason = ""
    stdout = ""
    try:
        completed = subprocess.run(
            command, shell=True, env=env, capture_output=True, text=True, timeout=timeout,
        )
        stdout = completed.stdout
        if completed.returncode != 0:
            reason = f"command exited {completed.returncode}"
    except subprocess.TimeoutExpired:
        reason = f"command timed out after {timeout}s"
    except OSError as exc:
        reason = f"command could not start: {exc.strerror}"

    if not reason:
        for candidate in reversed(stdout.splitlines()):
            if not candidate.strip():
                continue
            try:
                line = EvidenceLine.from_json_line(candidate)
            except Exception as exc:  # EvidenceError
                reason = f"stdout was not one aver-evidence/v1 line: {exc}"
                break
            meta = dict(line.meta)
            meta.setdefault("harness", "aver repeat")
            return EvidenceLine(
                task_id=task_id, side=side, run=run, facts=line.facts,
                variant=line.variant, unreadable=line.unreadable, meta=meta,
            )
        else:
            reason = "command produced no output"

    return EvidenceLine(
        task_id=task_id, side=side, run=run, facts={}, unreadable=True,
        meta={"harness": "aver repeat", "unreadable_reason": reason},
    )


# --------------------------------------------------------------------------- #
# aver demo
# --------------------------------------------------------------------------- #

def cmd_demo(args: argparse.Namespace, stream: TextIO) -> int:
    try:
        spec = demo_module.resolve(args.which)
    except KeyError:
        print(f"aver: no demo named {args.which!r}; "
              f"try {' or '.join(sorted(demo_module.DEMOS))}", file=stream)
        return EXIT_USAGE

    if args.evidence_out:
        fixture = demo_module.load_fixture(spec)
        lines = demo_module.fixture_evidence(spec, fixture)
        Path(args.evidence_out).write_text(write_evidence(lines), encoding="utf-8")
        print(f"wrote {args.evidence_out} ({len(lines)} run(s) of "
              f"{fixture['meta']['tasks']} {spec.decision_point} task(s), both sides)",
              file=stream)
        return EXIT_PASS

    module = v3.repeated_module()
    if module is None:
        return _v3_unavailable(stream)

    demo = build_demo(spec, module)
    payload = demo.payload
    exit_code = _STATE_EXIT.get(str(payload.get("state") or "refuse"), EXIT_REFUSE)

    if args.json:
        _print_json({
            "demo": spec.name,
            "decision_point": spec.decision_point,
            "models": demo.fixture["meta"]["models"],
            "floor": demo.floor.to_dict(),
            "stability_map": demo.stability.to_dict(),
            "golden_trace_diff": demo.diff.to_dict(),
            "referee": payload,
        }, _out(stream))
        return exit_code

    heading = _heading_markdown if args.markdown else _heading_text
    fence = "```\n" if args.markdown else ""
    for title, body in demo_blocks(demo):
        heading(title, stream)
        if title.startswith("4."):
            print(body, file=stream)
        else:
            print(fence + body + "\n" + fence, file=stream)
    if args.show_hosted:
        print(f"\n{HOSTED_POINTER}", file=stream)
    return exit_code


@dataclass
class Demo:
    spec: demo_module.DemoSpec
    fixture: dict
    policy: CliPolicy
    floor: object
    stability: object
    diff: object
    payload: dict


def build_demo(spec: demo_module.DemoSpec, module) -> Demo:
    """Every number the demo prints, computed from the fixture at run time."""

    fixture = demo_module.load_fixture(spec)
    cases = demo_module.demo_cases(spec, fixture)
    policy = demo_module.demo_policy(spec)
    floor = floor_readout(cases, side=SIDE_INCUMBENT, action=spec.action,
                          theta=policy.decidability_threshold,
                          confidence=policy.confidence)
    stability = stability_map(cases, side=SIDE_INCUMBENT, action=spec.action,
                              theta=policy.decidability_threshold)
    diff = golden_diff(
        cases, action=spec.action,
        baseline=run_map(cases, side=SIDE_INCUMBENT, run=0),
        comparison=run_map(cases, side=SIDE_INCUMBENT, run=1),
        label="the incumbent's run 0 against the incumbent's run 1",
        tolerance=policy.action(spec.action).tolerance,
        confidence=policy.confidence,
    )
    repeated_cases = v3.to_repeated_cases(cases, policy, module)
    verdict = module.compute_verdict_v3(repeated_cases, v3.engine_policy(policy, module))
    payload = dict(verdict.to_dict())
    payload["evidence_digest"] = "fixture:" + spec.fixture
    payload["policy_digest"] = policy.digest()
    return Demo(spec=spec, fixture=fixture, policy=policy, floor=floor,
                stability=stability, diff=diff, payload=payload)


#: The demo's block titles, in order. The floor leads; the referee follows it.
DEMO_TITLES = (
    "1. The floor: how often this agent disagrees with itself",
    "2. A one-run diff of the model against itself",
    "3. Then the referee: aver check on the candidate",
    "4. What you are looking at",
)


def demo_blocks(demo: Demo) -> list[tuple[str, str]]:
    """The four blocks as ``(title, body)``; the README generator reads these."""

    def render(fn) -> str:
        buffer = io.StringIO()
        fn(buffer)
        return buffer.getvalue().rstrip("\n")

    def block1(out: TextIO) -> None:
        print(f"{demo.floor.tasks} {demo.spec.decision_point} tasks, "
              f"{demo.fixture['meta']['reps']} runs each, the incumbent side",
              file=out)
        _print_floor_summary(demo.floor, out)
        print("", file=out)
        _print_stability_map(demo.stability, out, show_all=False)

    def block2(out: TextIO) -> None:
        print(f"Comparing {demo.diff.label}:", file=out)
        print("the same model, the same prompts, the same temperature.", file=out)
        print(f"action          {demo.diff.action}", file=out)
        _print_self_diff_detail(demo.diff, out, demo.spec.direction)
        print(f"verdict         {demo.diff.verdict.upper()}", file=out)
        print(f"\n{_self_diff_headline(demo.diff)}", file=out)

    def block3(out: TextIO) -> None:
        print("Only now is the candidate compared: every repeat on both sides, an",
              file=out)
        print("interval, and a tolerance a person wrote down.\n", file=out)
        _print_verdict(demo.payload, demo.policy, out, show_hosted=False)

    return [
        (DEMO_TITLES[0], render(block1)),
        (DEMO_TITLES[1], render(block2)),
        (DEMO_TITLES[2], render(block3)),
        (DEMO_TITLES[3], _demo_paragraph(demo.spec, demo.diff, demo.stability)),
    ]


def _heading_text(title: str, stream: TextIO) -> None:
    print(f"\n{title}", file=stream)
    print(RULE, file=stream)


def _heading_markdown(title: str, stream: TextIO) -> None:
    print(f"\n### {title}\n", file=stream)


def _demo_paragraph(spec: demo_module.DemoSpec, diff, stability=None) -> str:
    """Block 4: fixed text, claim-gated (docs/104 §8).

    It leads with the plain-words framing: agent regression tests are flaky
    tests. Two branches, because one paragraph cannot be both true. On the refund
    decision point the self-diff blocks, and the point is that a gate fired with
    nothing to fire at. On escalation the same self-diff passes, and the point is
    the opposite one: the practice happens to be quiet here, and it is quiet by
    luck rather than by measurement, because it never says how sure it is. Saying
    "a one-run diff blocks" under a PASS would be exactly the kind of number the
    rest of this tool exists to refuse.
    """

    framing = (
        "Agent regression tests are flaky tests. Run the same agent on the same\n"
        "tasks several times and it does not always make the same decision: it\n"
        "disagrees with itself. A one-run diff reads that disagreement as a\n"
        "regression.\n"
    )
    if diff.verdict == "block":
        story = (
            f"Block 2 is that mistake, made on purpose. It compared one recorded run\n"
            f"of this agent with another recorded run of the same agent, on the same\n"
            f"{spec.decision_point} tasks, and reported BLOCK. Nothing changed between\n"
            f"those two runs, so that BLOCK is the agent's own churn read as a change.\n"
        )
    else:
        story = (
            f"Block 2 compared one recorded run of this agent with another recorded\n"
            f"run of the same agent, on the same {spec.decision_point} tasks, and\n"
            f"reported PASS. That is the right answer -- nothing changed -- but it was\n"
            f"reached without measuring anything: the same comparison on this agent's\n"
            f"other decision point reports BLOCK on evidence that is just as\n"
            f"unchanged. A one-run diff that is right by luck cannot tell you which\n"
            f"case you are in.\n"
        )
    measure = (
        "Block 1 measures the disagreement instead of misreading it: the floor is\n"
        "how often this agent changes its own mind, and the stability map names the\n"
        "tasks behind it. A coin-flip task is one the prompt and tools leave\n"
        "under-determined; it needs fixing whatever model runs it. Only then does\n"
        "block 3 referee the candidate, using every repeat, with an interval and a\n"
        "tolerance a person wrote down. It either rules, or refuses and says what\n"
        "evidence is missing.\n"
    )
    provenance = (
        "These numbers are not claims. They are the output of a checked-in\n"
        "computation on a checked-in fixture of decision tuples from a repeat study\n"
        "of two models; the two models in the fixture are named in the data and\n"
        "nowhere else. No prompts, completions or tokens are in the package.\n"
    )
    run_it = (
        "Run it on your own agent. Start with the floor:\n"
        "\n"
        "    aver repeat --cmd \"./run-one-task.sh\" --tasks tasks.jsonl -k 5 \\\n"
        "                --side incumbent --out evidence.jsonl\n"
        "    aver floor evidence.jsonl\n"
        "\n"
        "Then add a candidate's runs to the same file and referee the change:\n"
        "\n"
        "    aver repeat --cmd \"./run-one-task.sh\" --tasks tasks.jsonl -k 5 \\\n"
        "                --side candidate --out evidence.jsonl\n"
        "    aver check evidence.jsonl --policy aver.yaml\n"
    )
    return "\n".join((framing, story, measure, provenance, run_it)).rstrip("\n") + "\n"


# --------------------------------------------------------------------------- #
# aver verify (implementation lands with aver_cass.verify)
# --------------------------------------------------------------------------- #

def cmd_verify(args: argparse.Namespace, stream: TextIO) -> int:
    try:
        from aver_cass import verify as verify_module  # type: ignore[attr-defined]
    except ImportError:
        print("aver: this build has no offline record verifier "
              "(aver_cass.verify is not present).", file=stream)
        return EXIT_USAGE
    return verify_module.run_cli(args, stream)


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aver",
        description="Compare agent pipeline cost and reliability with 'aver decision'. "
                    "Set requirements, plan a testing budget, freeze, then evaluate "
                    "fresh recorded outcomes. Start with 'aver decision demo'. "
                    "For repeated behavior: measure how much your agent disagrees "
                    "with itself, then referee a change with 'aver floor' and 'aver check'.",
        epilog="A guarded action is any yes/no fact about a run that your harness can "
               "print: a tool call fired, a file outside scope was touched, a "
               "destructive command ran, an escalation happened. "
               "Exit codes: 0 pass, 1 refuse (with cures), 2 block, 3 usage or "
               "parse error.",
    )
    parser.add_argument("--version", action="version", version=f"aver {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("decision", help="plan and evaluate pipeline cost and reliability offline")

    flr = subparsers.add_parser(
        "floor", help="start here: how often one agent disagrees with itself, "
                      "and on which tasks")
    flr.add_argument("evidence")
    flr.add_argument("--side", default=None, choices=list(SIDES),
                     help="the side to profile (default: the only side in the file, "
                          "else incumbent)")
    flr.add_argument("--action", default=None,
                     help="one guarded action (default: every action in the file)")
    flr.add_argument("--theta", type=float, default=0.8,
                     help="agreement needed to call a task decided (default 0.8)")
    flr.add_argument("--tolerance", type=float, default=0.05,
                     help="the tolerance the one-run self-diff is read against "
                          "(default 0.05)")
    flr.add_argument("--all", action="store_true",
                     help="list every task in the stability map, not only coin flips")
    flr.add_argument("--json", action="store_true")
    flr.set_defaults(handler=cmd_floor)

    chk = subparsers.add_parser(
        "check", help="then referee a change: the verdict, with every cure")
    chk.add_argument("evidence")
    chk.add_argument("--policy", required=True)
    chk.add_argument("--json-out", dest="json_out", default=None, metavar="PATH",
                     help="write the full verdict JSON to PATH")
    chk.add_argument("--json", action="store_true")
    chk.add_argument("--k-min", type=int, default=3,
                     help="minimum readable runs per side (v3 reads no single runs)")
    chk.add_argument("--show-hosted", action="store_true",
                     help="also print where a signed record could be issued")
    chk.set_defaults(handler=cmd_check)

    rpt = subparsers.add_parser(
        "repeat", help="run your own command k times per task and collect the facts")
    rpt.add_argument("--cmd", required=True)
    rpt.add_argument("--tasks", required=True)
    rpt.add_argument("-k", type=int, default=5, dest="k")
    rpt.add_argument("--side", default=SIDE_CANDIDATE, choices=list(SIDES))
    rpt.add_argument("--out", default="aver-evidence.jsonl")
    rpt.add_argument("--timeout", type=float, default=None,
                     help="seconds to allow one run before marking it unreadable")
    rpt.add_argument("--json", action="store_true")
    rpt.set_defaults(handler=cmd_repeat)

    imp = subparsers.add_parser(
        "import", help="turn a trace export into aver-evidence/v1 JSONL")
    imp.add_argument("files", nargs="+")
    imp.add_argument("--format", default="auto", choices=list(FORMAT_NAMES))
    imp.add_argument("--actions", required=True,
                     help="comma-separated guarded action names")
    imp.add_argument("--direction", default="fire_is_harmful",
                     choices=["fire_is_harmful", "withhold_is_harmful"],
                     help="how a bare decision maps to firing (generic records only)")
    imp.add_argument("--side", default="auto", choices=["incumbent", "candidate", "auto"])
    imp.add_argument("--incumbent-model", default=None)
    imp.add_argument("--candidate-model", default=None)
    imp.add_argument("--incumbent-provider", default=None)
    imp.add_argument("--candidate-provider", default=None)
    imp.add_argument("--task-key", default=None,
                     help="JSON pointer to the field that identifies a task")
    imp.add_argument("--harness", default=None)
    imp.add_argument("--out", default="aver-evidence.jsonl")
    imp.add_argument("--json", action="store_true")
    imp.set_defaults(handler=cmd_import)

    dmo = subparsers.add_parser("demo", help="the bundled catch, offline, from a fixture")
    dmo.add_argument("which", nargs="?", default=demo_module.DEFAULT_DEMO,
                     choices=sorted(demo_module.DEMOS))
    dmo.add_argument("--json", action="store_true")
    dmo.add_argument("--markdown", action="store_true",
                     help="emit the README body for this demo")
    dmo.add_argument("--evidence-out", dest="evidence_out", default=None, metavar="PATH",
                     help="write the demo's fixture as aver-evidence/v1 JSONL to PATH, "
                          "so floor and check can be run on it")
    dmo.add_argument("--show-hosted", action="store_true")
    dmo.set_defaults(handler=cmd_demo)

    vfy = subparsers.add_parser(
        "verify", help="verify a signed record offline, with no account")
    vfy.add_argument("record")
    vfy.add_argument("--jwks", default=None, help="a JWKS URL or file")
    vfy.add_argument("--did", default=None, help="a did:web document URL")
    vfy.add_argument("--status-list", dest="status_list", default=None,
                     help="the issuer's status list (url or file); without it, "
                          "revocation is reported as not checked")
    vfy.add_argument("--json", action="store_true")
    vfy.set_defaults(handler=cmd_verify)

    return parser


def main(argv: Optional[Sequence[str]] = None, stream: Optional[TextIO] = None) -> int:
    arguments = list(argv) if argv is not None else sys.argv[1:]
    if arguments and arguments[0] == "decision":
        from aver_cli.decision import run
        return run(arguments[1:], _out(stream))
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        return EXIT_USAGE if exc.code else EXIT_PASS
    out = _out(stream)
    try:
        return int(args.handler(args, out))
    except BrokenPipeError:
        return EXIT_PASS
    except KeyboardInterrupt:
        print("aver: interrupted", file=out)
        return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
