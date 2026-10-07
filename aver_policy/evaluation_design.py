"""Independent planning and judge-audit helpers for docs/105; not release gates."""
from __future__ import annotations

from datetime import date

from aver_policy.outcomes import mean, mean_interval, number, integer, require, variance


def allocate_tasks(weights:dict[str,float], budget:int, pilot:dict[str,list[float]], hints:list[dict],
                   *, candidate:str, metric:str, scope:str, as_of:str,
                   max_age_days:int=180, minimum:int=5) -> dict:
    """Frozen equal-task-cost design; benchmark strength capped at 20 per stratum."""
    require(weights and all(isinstance(s,str) and s and number(w) and w>0 for s,w in weights.items()) and abs(sum(weights.values())-1)<1e-9, "invalid weights")
    require(integer(minimum) and minimum>=2 and integer(budget) and budget>=minimum*len(weights), "insufficient allocation budget")
    require(integer(max_age_days) and max_age_days>=0, "invalid benchmark age limit")
    require(set(pilot)<=set(weights), "unknown pilot stratum")
    today=date.fromisoformat(as_of)
    accepted={s:[] for s in weights}; ignored=[]
    for h in hints:
        require(all(k in h for k in ("source","url","date","scope","candidate","metric","stratum","mean","count","relevance")), "incomplete benchmark provenance")
        require(isinstance(h["source"],str) and bool(h["source"]) and isinstance(h["url"],str) and h["url"].startswith("https://"), "invalid benchmark provenance")
        require(number(h["mean"]) and 0<=h["mean"]<=1 and integer(h["count"]) and h["count"]>0 and number(h["relevance"]) and 0<=h["relevance"]<=1, "invalid benchmark hint")
        age=(today-date.fromisoformat(h["date"])).days
        reason="scope_mismatch" if h["scope"]!=scope else "candidate_mismatch" if h["candidate"]!=candidate else "metric_mismatch" if h["metric"]!=metric else "unknown_stratum" if h["stratum"] not in weights else "future_or_stale" if age<0 or age>max_age_days else None
        if reason:
            ignored.append({"source":h["source"],"reason":reason})
        else:
            accepted[h["stratum"]].append(h)
    estimates={}; variances={}; provenance={}
    for s in weights:
        xs=pilot.get(s,[])
        require(all(number(x) and x in (0,1) for x in xs), "pilot must contain independent binary task outcomes")
        hs=accepted[s]
        strength=sum(min(h["count"],20)*h["relevance"] for h in hs)
        effective=min(strength,20)
        prior=sum(min(h["count"],20)*h["relevance"]*h["mean"] for h in hs)/strength if strength else .5
        p=(sum(xs)+effective*prior)/(len(xs)+effective) if len(xs)+effective else .5
        estimates[s]=p
        variances[s]=max(p*(1-p),.02)
        provenance[s]={"pilot_tasks":len(xs),"benchmark_effective_count":effective,"benchmark_sources":[h["source"] for h in hs]}
    allocation={s:minimum for s in weights}
    for _ in range(int(budget-minimum*len(weights))):
        s=max(weights,key=lambda s:weights[s]**2*variances[s]/(allocation[s]*(allocation[s]+1)))
        allocation[s]+=1
    return {"allocation":allocation,"planning_means":estimates,"planning_variances":variances,
            "provenance":provenance,"ignored":ignored,"use":"planning_only_customer_confirmation_required"}


def calibrated_mean(training:list[dict], audit:list[dict], predictions:list[dict], *, scope:str, alpha:float=.05) -> dict:
    """Independent pilot fits weight, independent audit corrects prediction bias.

    IDs refer to independent task clusters; this helper supports one value per
    cluster. Do not pass correlated episode rows as independent labels.
    """
    require(number(alpha) and 0<alpha<1 and isinstance(scope,str) and bool(scope), "invalid calibration parameters")
    require(len(training)>=2 and len(audit)>=2 and len(predictions)>=2, "calibration requires independent training, audit and prediction samples")
    ids=set()
    for group, labeled in ((training,True),(audit,True),(predictions,False)):
        for row in group:
            require(row.get("scope")==scope, "calibration scope mismatch")
            task=row.get("task")
            require(isinstance(task,str) and task and task not in ids, "calibration task overlap or invalid identity")
            ids.add(task)
            require(number(row.get("prediction")) and 0<=row["prediction"]<=1, "invalid prediction")
            if labeled:
                require(number(row.get("label")) and row["label"] in (0,1), "invalid human label")
    xs=[r["prediction"] for r in training]; ys=[r["label"] for r in training]
    vx=variance(xs)
    mx,my=mean(xs),mean(ys)
    covariance=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/(len(xs)-1)
    weight=max(0.0,min(1.0,covariance/vx*len(predictions)/(len(predictions)+len(audit)))) if vx else 0.0
    residual=[r["label"]-weight*r["prediction"] for r in audit]
    predicted=[r["prediction"] for r in predictions]
    if weight==0:
        interval=mean_interval(residual,0,1,alpha,True)
    else:
        a=mean_interval(residual,-weight,1,alpha/2)
        b=mean_interval(predicted,0,1,alpha/2)
        interval=(max(0.0,a[0]+weight*b[0]),min(1.0,a[1]+weight*b[1]))
    return {"estimate":mean(residual)+weight*mean(predicted),"lower":interval[0],"upper":interval[1],
            "weight":weight,"human_audit_tasks":len(audit),"prediction_tasks":len(predictions),
            "scope":scope,"use":"aggregate_diagnostic_not_episode_labels"}
