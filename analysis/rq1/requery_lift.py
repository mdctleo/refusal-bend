"""RQ1 causal claim on ER-2: do added objects raise a task's bypass rate above the planner's own
non-determinism?

Unit of analysis = task (trials on the same task are correlated). For every task ER-2 refused on its
first clean query (EARDataset/dataset_baseline_er2.csv, Judgement==0):
  object rate  = P(J==1) over its conclusive object trials (J in {0,1}; J==-1 excluded)
                 from EARDataset/robot_constitution_64_automerge_er2_attack_20.csv
  requery rate = its second clean query's J (0/1; J==-1 excluded)
                 from EARDataset/dataset_requery_er2.csv
Tasks lacking either value are dropped. Same test as ER-1.6's stratified test 1
(analysis/stratified_noise_analysis.py), run here on every refused task rather than a 200-task sample.

Writes analysis/output/rq1_er2_requery_lift.json.

Usage:
    ./.venv/bin/python analysis/rq1/requery_lift.py
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from bypass_rates import STRATUM_ORDER as STRATA, assign_stratum, per_task, read_judged  # sibling; shared with RQ2

OBJECT_CSV = "EARDataset/robot_constitution_64_automerge_er2_attack_20.csv"
REQUERY_CSV = "EARDataset/dataset_requery_er2.csv"
OUTPUT_JSON = "analysis/output/rq1_er2_requery_lift.json"


def task_bypass_rate(df):
    """Per-task rate(t) over conclusive trials, only for tasks that have one."""
    return per_task(df)["rate"].dropna()


def stratum(rate):
    return assign_stratum(rate)


def main():
    objects = read_judged(OBJECT_CSV)
    requery = read_judged(REQUERY_CSV)

    r = pd.concat([task_bypass_rate(objects).rename("object"),
                   task_bypass_rate(requery).rename("requery")], axis=1)
    n_tasks = int(objects["Task ID"].nunique())
    dropped = {
        "no_conclusive_object_trial": int(r["object"].isna().sum() + (n_tasks - len(r))),
        "requery_inconclusive_or_missing": int(r["requery"].isna().sum()),
    }
    r = r.dropna()
    n = len(r)

    diff = r["object"] - r["requery"]
    t, p_one = stats.ttest_rel(r["object"], r["requery"], alternative="greater")
    se = diff.std(ddof=1) / np.sqrt(n)
    half = stats.t.ppf(0.975, n - 1) * se
    w = stats.wilcoxon(r["object"], r["requery"], alternative="greater")

    r["stratum"] = stratum(r["object"])
    per_stratum = (r.groupby("stratum")[["object", "requery"]].mean().reindex(STRATA) * 100).round(2)
    per_stratum["n"] = r.groupby("stratum").size().reindex(STRATA)

    rq = requery["Judgement"]
    out = {
        "n_tasks_refused_first_query": n_tasks,
        "n_tasks_analysed": n,
        "dropped": dropped,
        "requery_trial_counts": {"n_rows": int(len(requery)), "J0": int((rq == 0).sum()),
                                 "J1": int((rq == 1).sum()), "J-1": int((rq == -1).sum()),
                                 "unjudged": int(rq.isna().sum())},
        "mean_object_rate": float(r["object"].mean()),
        "mean_requery_rate": float(r["requery"].mean()),
        "mean_diff": float(diff.mean()),
        "ci95": [float(diff.mean() - half), float(diff.mean() + half)],
        "t": float(t),
        "p_one_sided": float(p_one),
        "wilcoxon_p_one_sided": float(w.pvalue),
        "n_tasks_object_higher": int((diff > 0).sum()),
        "n_tasks_requery_higher": int((diff < 0).sum()),
        "n_tasks_equal": int((diff == 0).sum()),
        "per_stratum_pct": per_stratum.reset_index().to_dict(orient="records"),
    }

    print(f"tasks refused on first clean query: {n_tasks} | analysed: {n} | dropped: {dropped}")
    print(f"requery judgements: {out['requery_trial_counts']}")
    print(f"mean per-task object rate  {out['mean_object_rate'] * 100:6.2f}%")
    print(f"mean per-task requery rate {out['mean_requery_rate'] * 100:6.2f}%")
    print(f"paired diff {out['mean_diff'] * 100:+.2f}pp  95% CI [{out['ci95'][0] * 100:.2f}, {out['ci95'][1] * 100:.2f}]  "
          f"t={t:.2f}  p(one-sided)={p_one:.2e}  Wilcoxon p={w.pvalue:.2e}")
    print(f"tasks: object higher {out['n_tasks_object_higher']} | requery higher {out['n_tasks_requery_higher']} | "
          f"equal {out['n_tasks_equal']}")
    print(per_stratum.to_string())

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
