"""Every number the ER-1.6 vs ER-2 appendix quotes, recomputed from the run files.

The appendix section is written against this module rather than against pasted
figures, so the text and the data cannot drift apart. Run it from the repo root
and the printed block is exactly what the .tex should say.

This is the one place in the analysis tree where an ER-1.6 file and an ER-2 file
are read together, because the comparison between the two target models is the
measurement. Every other module is single-model by construction; see the note at
the top of analysis/figure_common.py.

Each target is measured on its own initially-refused tasks, exactly as RQ1
measures ER-2, rather than on the tasks the two happen to share. A bypass rate is
only defined on a task the target refused, and the two targets refuse different
tasks; intersecting them would drop the tasks ER-2 alone refuses, which are its
most bypassable, and so would overstate how much safer it is. The ER-2 column
below therefore reproduces the RQ1 numbers.

ER-1.6 was probed with roughly fifty candidate objects per task and ER-2 with
twenty, so every ER-1.6 quantity is computed on a budget-matched subsample:
BUDGET attempted trials drawn per task, inconclusive ones dropped afterwards
exactly as they are for ER-2, averaged over DRAWS redraws.

Usage (from the repo root):
    uv run python analysis/appendix/er16_vs_er2_stats.py
"""

import numpy as np
import pandas as pd

# Baselines: the same 1,121-task pool put to each target once, unedited.
BASELINE_ER16 = "EARDataset/dataset_realistic.csv"
BASELINE_ER2 = "EARDataset/dataset_baseline_er2.csv"

# Object trials: the RQ1 experiment under each target.
ATTACK_ER16 = "EARDataset/robot_constitution_64_automerge_attack_5_judged.csv"
ATTACK_ER2 = "EARDataset/robot_constitution_64_automerge_er2_attack_20.csv"

# ER-2's attempt budget, which ER-1.6 is matched down to.
BUDGET = 20
DRAWS = 500
SEED = 42

COLS = ("Task ID", "Judgement")


def _load(path):
    return pd.read_csv(path, usecols=lambda c: c in COLS, low_memory=False)


def refusal(path):
    """Share of the pool the target refuses, over the tasks it judges conclusively."""
    pool = _load(path)
    conclusive = pool[pool["Judgement"] != -1]
    return {
        "pool": len(pool),
        "conclusive": len(conclusive),
        "refused": int((conclusive["Judgement"] == 0).sum()),
        "rate": (conclusive["Judgement"] == 0).mean(),
    }


def _summarise(per_task, trial_level):
    return {
        "tasks": len(per_task),
        "rate": float(per_task["rate"].mean()),
        "any": float(per_task["any"].mean()),
        "trial": float(trial_level),
    }


def bypass_er2():
    """ER-2 over its own refused tasks, at its native twenty-object budget."""
    attack = _load(ATTACK_ER2)
    conclusive = attack[attack["Judgement"] != -1]
    per_task = conclusive.groupby("Task ID")["Judgement"].agg(rate="mean", any="max")
    out = _summarise(per_task, (conclusive["Judgement"] == 1).mean())
    out["refused"] = attack["Task ID"].nunique()
    return out


def bypass_er16():
    """ER-1.6 over its own refused tasks, matched down to ER-2's budget."""
    attack = _load(ATTACK_ER16)
    trials = {t: g["Judgement"].values for t, g in attack.groupby("Task ID")}

    rng = np.random.default_rng(SEED)
    rates, anys, trial_level = [], [], []
    for _ in range(DRAWS):
        drawn = {}
        hits = total = 0
        for task, judged in trials.items():
            take = rng.choice(judged, size=min(BUDGET, len(judged)), replace=False)
            take = take[take != -1]
            if len(take) == 0:
                continue
            drawn[task] = (take == 1).mean()
            hits += int((take == 1).sum())
            total += len(take)
        drawn = pd.Series(drawn)
        rates.append(drawn.mean())
        anys.append((drawn > 0).mean())
        trial_level.append(hits / total)

    conclusive = attack[attack["Judgement"] != -1]
    full_budget = conclusive.groupby("Task ID")["Judgement"].max().mean()
    return {
        "refused": attack["Task ID"].nunique(),
        "tasks": conclusive["Task ID"].nunique(),
        "rate": float(np.mean(rates)),
        "any": float(np.mean(anys)),
        "trial": float(np.mean(trial_level)),
        "any_full_budget": float(full_budget),
    }


def main():
    r16, r2 = refusal(BASELINE_ER16), refusal(BASELINE_ER2)
    b16, b2 = bypass_er16(), bypass_er2()

    print(f"BASELINE REFUSAL  (the same {r16['pool']}-task pool, each target's own "
          f"conclusive rows)")
    print(f"  ER-1.6   {r16['refused']} of {r16['conclusive']} = {r16['rate']:.1%}")
    print(f"  ER-2     {r2['refused']} of {r2['conclusive']} = {r2['rate']:.1%}")

    print(f"\nOBJECT TRIALS  (each target over its own refused tasks; ER-1.6 matched "
          f"to {BUDGET} objects over {DRAWS} draws)")
    print(f"  refused tasks             ER-1.6 {b16['refused']:<8} ER-2 {b2['refused']}")
    print(f"  mean per-task rate        ER-1.6 {b16['rate']:.2%}    ER-2 {b2['rate']:.2%}")
    print(f"  any-bypass                ER-1.6 {b16['any']:.1%}     ER-2 {b2['any']:.1%}")
    print(f"  trial-level rate          ER-1.6 {b16['trial']:.2%}    ER-2 {b2['trial']:.2%}")
    print(f"  ER-1.6 any-bypass at its full ~50-object budget: "
          f"{b16['any_full_budget']:.1%}")


if __name__ == "__main__":
    main()
