"""Fresh confirmation of a declared operating-cost saving. See docs/107."""
from __future__ import annotations

import copy
import re

from aver_policy.outcomes import (
    evaluate, evidence_groups, integer, mean, number, require,
    task_values, validate_plan,
)

PROTOCOL = "aver-savings-confirmation-v1"


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def reports_agree(a, b):
    """Allow floating-point rounding across the Python/browser implementations."""
    if number(a) and number(b):
        return abs(a - b) <= 1e-12 * (1 + abs(a) + abs(b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(reports_agree(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(reports_agree(x, y) for x, y in zip(a, b))
    return a == b


def validate_study(study):
    require(isinstance(study, dict), "study must be an object")
    plan = study["plan"]
    validate_plan(plan)
    require(len(plan["candidates"]) >= 2, "baseline and alternative required")
    manifests = study["configurations"]
    require(isinstance(manifests, dict) and set(manifests) == set(plan["candidates"])
            and all(valid_hash(v) for v in manifests.values()), "configuration hashes required")
    require(integer(study["shortlist_size"]) and 1 <= study["shortlist_size"] < len(plan["candidates"]),
            "invalid shortlist size")
    require(number(study["minimum_saving_usd"]) and study["minimum_saving_usd"] > 0,
            "positive savings target required")
    for key in ("evaluation_budget_usd", "episode_expense_cap_usd"):
        require(number(study[key]) and study[key] > 0, "invalid expense limit")
    require(integer(study["deployment_episodes"]) and study["deployment_episodes"] >= 1,
            "positive deployment volume required")
    pilot = copy.deepcopy(plan)
    pilot["looks"] = [study["pilot_tasks"]]
    validate_plan(pilot)
    cost = next(m for m in plan["metrics"] if m["name"] == plan["cost_metric"])
    require(cost["kind"] == "bounded", "operating cost must be bounded dollars")
    require(any(m["name"] != plan["cost_metric"] and ("minimum" in m or "maximum" in m)
                for m in plan["metrics"]), "an absolute reliability requirement is required")


def checked_rows(study, plan, rows, source, look):
    require(isinstance(rows, list) and all(isinstance(r, dict) for r in rows), "rows required")
    require(all(r.get("source") == source for r in rows), "incorrect evidence phase")
    contents = {}
    tasks_by_content = {}
    for row in rows:
        candidate = row.get("candidate")
        require(isinstance(candidate, str) and candidate in study["configurations"]
                and row.get("configuration_sha256") == study["configurations"][candidate],
                "configuration changed")
        content = row.get("content_sha256")
        task = row.get("task")
        require(valid_hash(content) and isinstance(task, str) and bool(task), "content and task identities required")
        require(task not in contents or contents[task] == content, "task content changed")
        require(content not in tasks_by_content or tasks_by_content[content] == task,
                "duplicate content under different task identities")
        contents[task] = content
        tasks_by_content[content] = task
        expense = row.get("evaluation_cost_usd")
        require("evaluation_cost_usd" in row and (expense is None or
                (number(expense) and 0 <= expense <= study["episode_expense_cap_usd"])),
                "invalid or above-cap evaluation expense")
    normalized = [{**row, "source": "customer"} for row in rows]
    groups, info = evidence_groups(plan, normalized, look)
    return groups, info, contents


def expense(rows, cap):
    known = sum(r["evaluation_cost_usd"] for r in rows if r["evaluation_cost_usd"] is not None)
    missing = sum(r["evaluation_cost_usd"] is None for r in rows)
    return {"measured_usd": known, "unknown_episodes": missing, "upper_usd": known + missing * cap}


def pilot_rank(study, pilot_rows):
    validate_study(study)
    plan = copy.deepcopy(study["plan"])
    plan["looks"] = [study["pilot_tasks"]]
    groups, info, contents = checked_rows(study, plan, pilot_rows, "pilot", 0)
    summaries = {}
    for candidate in plan["candidates"]:
        metrics = {}
        for metric in plan["metrics"]:
            values = [
                (weight, list(task_values(plan, groups, candidate, info["tasks"][s], metric).values()))
                for s, weight in plan["strata"].items()
            ]
            lo = sum(w * mean([v[0] for v in xs]) for w, xs in values)
            hi = sum(w * mean([v[1] for v in xs]) for w, xs in values)
            metrics[metric["name"]] = lo if lo == hi else None
        summaries[candidate] = {"metrics": metrics, "observed_requirements_met": True}
    for candidate, summary in summaries.items():
        rules = [(m, summary["metrics"][m["name"]]) for m in plan["metrics"]
                 if "minimum" in m or "maximum" in m]
        for rule in plan.get("paired_constraints", []):
            a, b = summary["metrics"][rule["metric"]], summaries[plan["baseline"]]["metrics"][rule["metric"]]
            rules.append((rule, a - b if a is not None and b is not None else None))
        summary["observed_requirements_met"] = all(
            v is not None and v >= rule.get("minimum", float("-inf"))
            and v <= rule.get("maximum", float("inf")) for rule, v in rules)
    alternatives = [c for c in plan["candidates"] if c != plan["baseline"]]
    alternatives.sort(key=lambda c: (
        not summaries[c]["observed_requirements_met"],
        summaries[c]["metrics"][plan["cost_metric"]] if summaries[c]["metrics"][plan["cost_metric"]] is not None else float("inf"),
        plan["candidates"].index(c)))
    return alternatives, summaries, contents


def freeze(study, pilot_rows):
    ranked, summaries, contents = pilot_rank(study, pilot_rows)
    selected = ranked[:int(study["shortlist_size"])]
    plan = copy.deepcopy(study["plan"])
    plan["candidates"] = [plan["baseline"], *selected]
    spent = expense(pilot_rows, study["episode_expense_cap_usd"])
    require(spent["upper_usd"] <= study["evaluation_budget_usd"], "pilot exhausted evaluation budget")
    return {"protocol": PROTOCOL, "study": copy.deepcopy(study), "confirmation_plan": plan,
            "pilot_contents": contents, "pilot_rows": copy.deepcopy(pilot_rows),
            "pilot_ranking": ranked, "pilot_summaries": summaries, "pilot_expense": spent,
            "selected": selected, "dropped": [c for c in ranked if c not in selected]}


def confirm(frozen, rows, look_index=0, previous=None):
    require(isinstance(frozen, dict) and frozen.get("protocol") == PROTOCOL, "unknown protocol")
    # Recompute exploration so corrupted frozen selections or expense cannot pass unnoticed.
    require(freeze(frozen["study"], frozen["pilot_rows"]) == frozen, "frozen study changed")
    study, plan = frozen["study"], frozen["confirmation_plan"]
    _, _, contents = checked_rows(study, plan, rows, "customer", look_index)
    require(not (set(contents) & set(frozen["pilot_contents"])), "pilot task reused in confirmation")
    require(not (set(contents.values()) & set(frozen["pilot_contents"].values())), "pilot content reused in confirmation")
    require((look_index == 0 and previous is None) or (look_index > 0 and isinstance(previous, dict)),
            "successive look receipt required")
    if previous is not None:
        require(previous.get("look_index") == look_index - 1 and previous.get("frozen") == frozen,
                "previous receipt belongs to a different look or study")
        expected = confirm(frozen, previous["rows"], previous["look_index"], previous.get("previous"))
        require(reports_agree(expected["report"], previous["report"]), "previous receipt changed")
        require(previous.get("report", {}).get("next_action") == "collect_next_look", "study already stopped")
        prior = {(r["candidate"], r["task"], r["repeat"]): r for r in previous["rows"]}
        current = {(r["candidate"], r["task"], r["repeat"]): r for r in rows}
        require(all(current.get(k) == row for k, row in prior.items()), "earlier evidence changed or removed")
    result = evaluate(plan, rows, look_index)
    targets = {}
    cost_name = plan["cost_metric"]
    for c in frozen["selected"]:
        candidate = result["candidates"][c]
        delta = candidate["paired"][cost_name]
        lower, upper = -delta["upper"], -delta["lower"]
        saving_state = "supported" if lower >= study["minimum_saving_usd"] else "violated" if upper < study["minimum_saving_usd"] else "unresolved"
        targets[c] = {"reliability_state": candidate["state"], "saving_state": saving_state,
                      "saving_usd_per_episode": {"estimate": -delta["estimate"] if delta["estimate"] is not None else None,
                                                 "lower": lower, "upper": upper},
                      "gross_deployment_saving_usd": {"lower": lower * study["deployment_episodes"],
                                                       "upper": upper * study["deployment_episodes"]},
                      "qualifies": candidate["state"] == "supported" and saving_state == "supported"}
    qualified = [c for c in frozen["selected"] if targets[c]["qualifies"]]
    choice = min(qualified, key=lambda c: result["candidates"][c]["metrics"][cost_name]["upper"]) if qualified else None
    spent = expense(rows, study["episode_expense_cap_usd"])
    upper_spend = spent["upper_usd"] + frozen["pilot_expense"]["upper_usd"]
    next_count = (sum(plan["looks"][look_index + 1].values()) - sum(plan["looks"][look_index].values())) * len(plan["candidates"]) * plan["repeats"] if look_index + 1 < len(plan["looks"]) else 0
    next_cap = next_count * study["episode_expense_cap_usd"]
    optimistic = max([0, *[t["gross_deployment_saving_usd"]["upper"] for t in targets.values()
                           if t["reliability_state"] != "excluded" and t["saving_state"] != "violated"]])
    if upper_spend > study["evaluation_budget_usd"]:
        action, choice = "evaluation_budget_exceeded", None
    elif choice is not None:
        action = "savings_target_met"
    elif result["next_action"]["unknown_measurements"]:
        action = "inspect_missing_measurements"
    elif all(t["reliability_state"] == "excluded" or t["saving_state"] == "violated" for t in targets.values()):
        action = "requirements_not_met"
    elif not next_count:
        action = "confirmation_budget_exhausted_unresolved"
    elif upper_spend + next_cap > study["evaluation_budget_usd"]:
        action = "next_look_exceeds_budget"
    elif next_cap > optimistic:
        action = "next_look_not_economic_under_declared_caps"
    else:
        action = "collect_next_look"
    report = {"protocol": PROTOCOL, "outcomes": result, "targets": targets, "recommendation": choice,
              "recommendation_claim": "declared_savings_and_requirements_supported" if choice else "no_supported_change",
              "next_action": action, "global_cheapest_claim": False, "pilot_expense": frozen["pilot_expense"],
              "confirmation_expense": spent, "total_expense_upper_usd": upper_spend,
              "next_look_expense_upper_usd": next_cap, "optimistic_deployment_saving_usd": optimistic}
    return {"frozen": copy.deepcopy(frozen), "look_index": look_index, "rows": copy.deepcopy(rows), "report": report, "previous": copy.deepcopy(previous)}
