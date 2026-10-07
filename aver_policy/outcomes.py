"""Local outcome/cost evaluator. Contract and assumptions: docs/105.

Independent tasks are the sampling units; repeats never inflate their count.
Benchmark rows may guide a separate design step but never enter inference.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

METHOD = "aver-outcome-cost-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def integer(value: Any) -> bool:
    return number(value) and int(value) == value and abs(value) <= 9007199254740991


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def variance(xs: list[float]) -> float:
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1) if len(xs) > 1 else 0.0


def kl(p: float, q: float) -> float:
    if q <= 0:
        return 0.0 if p == 0 else math.inf
    if q >= 1:
        return 0.0 if p == 1 else math.inf
    return (p * math.log(p / q) if p else 0.0) + ((1-p) * math.log((1-p)/(1-q)) if p < 1 else 0.0)


def mean_interval(xs: list[float], low: float, high: float, alpha: float,
                  binary: bool = False) -> tuple[float, float]:
    """Finite-sample bounded-mean interval; see docs/105 for error splitting."""
    require(0 < alpha < 1 and low < high, "invalid interval parameters")
    require(all(number(x) and low <= x <= high for x in xs), "invalid interval observations")
    if binary:
        require(low == 0 and high == 1 and all(x in (0, 1) for x in xs), "binary interval needs Bernoulli observations")
    n = len(xs)
    if n < 2:
        return low, high
    m = mean(xs)
    if binary:
        target = math.log(2 / alpha) / n
        a, b = 0.0, m
        for _ in range(48):
            mid = (a+b)/2
            if kl(m, mid) > target:
                a = mid
            else:
                b = mid
        lower = (a+b)/2 if m else 0.0
        a, b = m, 1.0
        for _ in range(48):
            mid = (a+b)/2
            if kl(m, mid) > target:
                b = mid
            else:
                a = mid
        return lower, (a+b)/2 if m < 1 else 1.0
    width = high-low
    hoeffding = width * math.sqrt(math.log(4/alpha)/(2*n))
    log = math.log(8/alpha)
    bernstein = math.sqrt(2*variance(xs)*log/n) + 7*width*log/(3*(n-1))
    radius = min(hoeffding, bernstein)
    return max(low, m-radius), min(high, m+radius)


def validate_plan(plan: dict) -> None:
    require(isinstance(plan, dict), "plan must be an object")
    require(isinstance(plan.get("id"), str) and bool(plan["id"]), "plan id required")
    cs = plan.get("candidates", [])
    require(isinstance(cs, list) and len(cs) > 0 and all(isinstance(c,str) and c for c in cs), "candidates required")
    require(len(set(cs)) == len(cs) and plan.get("baseline") in cs, "invalid candidate identities")
    strata = plan.get("strata", {})
    require(isinstance(strata,dict) and bool(strata) and all(isinstance(s,str) and s and number(w) and w>0 for s,w in strata.items()), "invalid strata")
    require(abs(sum(strata.values())-1)<1e-9, "stratum weights must sum to one")
    require(integer(plan.get("repeats")) and plan["repeats"]>=1, "invalid repeats")
    require(number(plan.get("alpha")) and 0<plan["alpha"]<1, "invalid alpha")
    metrics = plan.get("metrics", [])
    require(isinstance(metrics,list) and bool(metrics), "metrics required")
    names=[]
    for m in metrics:
        require(isinstance(m,dict) and isinstance(m.get("name"),str) and bool(m["name"]), "metric name required")
        names.append(m["name"])
        require(number(m.get("low")) and number(m.get("high")) and m["low"]<m["high"], "invalid metric bounds")
        require(m.get("kind") in ("binary","bounded") and m.get("aggregate") in ("mean","all","any"), "invalid metric type")
        require(m["kind"]=="binary" or m["aggregate"]=="mean", "boolean aggregates need binary inputs")
        if m["kind"]=="binary":
            require(m["low"]==0 and m["high"]==1, "binary bounds must be zero and one")
        for key in ("minimum","maximum"):
            if key in m:
                require(number(m[key]) and m["low"]<=m[key]<=m["high"], "invalid metric requirement")
        require(m.get("minimum",m["low"])<=m.get("maximum",m["high"]), "contradictory metric requirements")
    require(len(set(names))==len(names) and plan.get("cost_metric") in names, "invalid metric identities")
    cost = metrics[names.index(plan["cost_metric"])]
    require(cost["low"]>=0 and cost["aggregate"]=="mean", "cost must be nonnegative episode mean")
    if plan.get("completion_metric") is not None:
        require(plan["completion_metric"] in names, "unknown completion metric")
        completion = metrics[names.index(plan["completion_metric"])]
        require(completion["kind"]=="binary" and completion["aggregate"]=="mean", "completion denominator must be a binary episode mean")
    looks=plan.get("looks",[])
    require(isinstance(looks,list) and bool(looks), "looks required")
    previous={s:0 for s in strata}
    for look in looks:
        require(isinstance(look,dict) and set(look)==set(strata), "look strata mismatch")
        require(all(integer(n) and n>=2 and n>=previous[s] for s,n in look.items()) and sum(look.values())>sum(previous.values()), "looks must increase")
        previous=look
    for constraint in plan.get("paired_constraints",[]):
        require(isinstance(constraint,dict) and constraint.get("metric") in names, "unknown paired metric")
        require("minimum" in constraint or "maximum" in constraint, "empty paired constraint")
        for key in ("minimum","maximum"):
            if key in constraint:
                require(number(constraint[key]), "invalid paired requirement")
        require(constraint.get("minimum",-math.inf)<=constraint.get("maximum",math.inf), "contradictory paired requirement")


def aggregate(xs: list[float], mode: str) -> float:
    return min(xs) if mode=="all" else max(xs) if mode=="any" else max(min(xs), min(max(xs), mean(xs)))


def evidence_groups(plan: dict, rows: list[dict], look_index: int) -> tuple[dict,dict]:
    require(integer(look_index) and 0<=look_index<len(plan["looks"]), "undeclared look")
    look_index = int(look_index)
    require(isinstance(rows,list), "evidence must be an array")
    groups: dict = defaultdict(dict)
    task_strata: dict = {}
    task_sources: dict = {}
    customer_tasks: dict = {s:set() for s in plan["strata"]}
    metrics={m["name"]:m for m in plan["metrics"]}
    counts={s:0 for s in ("customer","pilot","benchmark","stress")}
    for row in rows:
        require(isinstance(row,dict), "invalid evidence row")
        source=row.get("source")
        require(isinstance(source,str) and source in counts, "invalid evidence source")
        require(row.get("plan_id")==plan["id"], "wrong plan identity")
        c,t,s,r=row.get("candidate"),row.get("task"),row.get("stratum"),row.get("repeat")
        require(isinstance(c,str) and c in plan["candidates"] and isinstance(t,str) and bool(t) and isinstance(s,str) and s in plan["strata"], "invalid row identity")
        require(integer(r) and 0<=r<plan["repeats"], "invalid repeat identity")
        require(t not in task_sources or task_sources[t]==source, "task leaked across evidence sources")
        require(t not in task_strata or task_strata[t]==s, "task spans strata")
        task_sources[t],task_strata[t]=source,s
        values=row.get("values")
        require(isinstance(values,dict) and set(values)==set(metrics), "metric schema mismatch")
        for name,v in values.items():
            m=metrics[name]
            require(v is None or (number(v) and m["low"]<=v<=m["high"] and (m["kind"]!="binary" or v in (0,1))), "invalid measurement: "+name)
        key=(source,c,t)
        require(r not in groups[key], "duplicate episode")
        groups[key][r]=values
        counts[source]+=1
        if source=="customer":
            customer_tasks[s].add(t)
    for s,tasks in customer_tasks.items():
        require(len(tasks)==plan["looks"][look_index][s], "evidence does not match declared look")
        for t in tasks:
            for c in plan["candidates"]:
                require(set(groups[("customer",c,t)])==set(range(int(plan["repeats"]))), "missing planned episode; use explicit null measurements")
    return groups, {"tasks":customer_tasks,"source_rows":counts}


def task_values(plan:dict, groups:dict, candidate:str, tasks:set, metric:dict) -> dict:
    out={}
    for task in sorted(tasks):
        vals=[groups[("customer",candidate,task)][i][metric["name"]] for i in range(int(plan["repeats"]))]
        lo=aggregate([metric["low"] if v is None else v for v in vals],metric["aggregate"])
        hi=aggregate([metric["high"] if v is None else v for v in vals],metric["aggregate"])
        out[task]=(lo,hi)
    return out


def weighted_estimate(plan:dict, cells:dict, metric:dict, alpha:float, paired:bool=False) -> dict:
    low,high=metric["low"],metric["high"]
    if paired:
        low,high=low-high,high-low
    binary=not paired and metric["kind"]=="binary" and (plan["repeats"]==1 or metric["aggregate"]!="mean")
    observed_low=observed_high=lower=upper=0.0
    strata={}
    for s,w in plan["strata"].items():
        xs=list(cells[s].values())
        ls,us=[v[0] for v in xs],[v[1] for v in xs]
        l=mean_interval(ls,low,high,alpha,binary)[0]
        u=mean_interval(us,low,high,alpha,binary)[1]
        ol,ou=mean(ls),mean(us)
        strata[s]={"tasks":len(xs),"observed_low":ol,"observed_high":ou,"lower":l,"upper":u}
        observed_low+=w*ol; observed_high+=w*ou; lower+=w*l; upper+=w*u
    return {"estimate":observed_low if abs(observed_high-observed_low)<1e-12 else None,
            "observed_low":observed_low,"observed_high":observed_high,"lower":lower,"upper":upper,"strata":strata}


def requirement(est:dict, rule:dict) -> str:
    if ("minimum" in rule and est["upper"]<rule["minimum"]) or ("maximum" in rule and est["lower"]>rule["maximum"]):
        return "violated"
    if ("minimum" in rule and est["lower"]<rule["minimum"]) or ("maximum" in rule and est["upper"]>rule["maximum"]):
        return "unresolved"
    return "supported"


def evaluate(plan:dict, rows:list[dict], look_index:int=0) -> dict:
    validate_plan(plan)
    groups,info=evidence_groups(plan,rows,look_index)
    look_index = int(look_index)
    cs,ms=plan["candidates"],plan["metrics"]
    comparisons=2*len(cs)-1
    endpoint_alpha=plan["alpha"]/(comparisons*len(ms)*len(plan["strata"])*len(plan["looks"])*2)
    values={c:{m["name"]:{s:task_values(plan,groups,c,ts,m) for s,ts in info["tasks"].items()} for m in ms} for c in cs}
    results={}
    for c in cs:
        absolute={}; paired={}; constraints=[]
        for m in ms:
            name=m["name"]
            absolute[name]=weighted_estimate(plan,values[c][name],m,endpoint_alpha)
            if c==plan["baseline"]:
                paired[name]={"estimate":0.0,"observed_low":0.0,"observed_high":0.0,"lower":0.0,"upper":0.0,"strata":{}}
            else:
                pairs={s:{t:(a[0]-values[plan["baseline"]][name][s][t][1], a[1]-values[plan["baseline"]][name][s][t][0]) for t,a in cell.items()} for s,cell in values[c][name].items()}
                paired[name]=weighted_estimate(plan,pairs,m,endpoint_alpha,True)
            if "minimum" in m or "maximum" in m:
                constraints.append({"metric":name,"comparison":"absolute","state":requirement(absolute[name],m)})
        for rule in plan.get("paired_constraints",[]):
            constraints.append({"metric":rule["metric"],"comparison":"paired","state":requirement(paired[rule["metric"]],rule)})
        require(bool(constraints), "at least one reliability or cost requirement is required")
        state="excluded" if any(x["state"]=="violated" for x in constraints) else "unresolved" if any(x["state"]=="unresolved" for x in constraints) else "supported"
        results[c]={"state":state,"metrics":absolute,"paired":paired,"constraints":constraints}
        if plan.get("completion_metric"):
            cost=absolute[plan["cost_metric"]]; completion=absolute[plan["completion_metric"]]
            results[c]["cost_per_completion"]={"lower":cost["lower"]/completion["upper"] if completion["upper"]>0 else None,
                "upper":cost["upper"]/completion["lower"] if completion["lower"]>0 else None,
                "upper_unbounded":completion["lower"]<=0}
    eligible=[c for c in cs if results[c]["state"]=="supported"]
    choice=min(eligible,key=lambda c:results[c]["metrics"][plan["cost_metric"]]["upper"]) if eligible else None
    contenders=[]
    if choice:
        cost_upper=results[choice]["metrics"][plan["cost_metric"]]["upper"]
        contenders=[c for c in cs if c!=choice and results[c]["state"]!="excluded" and results[c]["metrics"][plan["cost_metric"]]["lower"]<=cost_upper]
    else:
        contenders=[c for c in cs if results[c]["state"]!="excluded"]
    missing=sum(v is None for row in rows if row["source"]=="customer" for v in row["values"].values())
    next_action={"kind":"inspect_missing_measurements" if missing else "collect_next_look" if look_index+1<len(plan["looks"]) and (choice is None or contenders) else "review_tradeoff" if contenders else "complete",
        "next_look":plan["looks"][look_index+1] if look_index+1<len(plan["looks"]) else None,
        "unknown_measurements":missing}
    if not missing and all(r["state"]=="excluded" for r in results.values()):
        next_action["kind"]="requirements_violated"
        next_action["next_look"]=None
    elif choice is None and look_index+1==len(plan["looks"]) and not missing:
        next_action["kind"]="requirements_violated" if all(r["state"]=="excluded" for r in results.values()) else "budget_exhausted_unresolved"
    return {"method_version":METHOD,"plan_id":plan["id"],"look_index":look_index,"alpha":plan["alpha"],"endpoint_alpha":endpoint_alpha,
        "independent_tasks":sum(len(ts) for ts in info["tasks"].values()),"repeats":plan["repeats"],"source_rows":info["source_rows"],
        "candidates":results,"recommendation":choice,"recommendation_kind":"conservative_supported_choice" if choice else "no_supported_choice",
        "unresolved_cost_order":contenders,"cost_order_established":choice is not None and not contenders,"next_action":next_action}
