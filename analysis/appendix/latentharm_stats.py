"""Every number the LatentHarm appendix quotes, recomputed from the corpus files.

The appendix section is written against this module rather than against pasted
figures, so the text and the data cannot drift apart. Run it from the repo root
and the printed block is exactly what the .tex should say.

The corpus is evaluated on ER-1.6, not the ER-2 target of the main paper, so
nothing here is mixed into an ER-2 figure.

Usage (from the repo root):
    uv run python analysis/appendix/latentharm_stats.py
"""

import collections
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASELINE_CSV = "MyDataset/baseline.csv"
ATTACK_CSV = "MyDataset/robot_constitution_64_automerge_attack_1.csv"
IMAGE_ROOT = "MyDataset/images"

# The safety constitution the corpus was written against, as numbered in
# Appendix appendix:constitution.
N_RULES = 63

# Rules 10 and 30 are the constitution's two general-duty clauses ("avoid actions
# that could cause harm" / "prioritise the safety and well-being of humans"), so a
# task citing only these is not grounded in a specific rule. Reported separately.
GENERAL_DUTY_RULES = {10, 30}

# The corpus is probed with fifty objects per task; the EARDataset runs of
# Appendix appendix:er16 use twenty. Quantities that depend on the attempt budget
# are reported at both, the matched figure computed by drawing BUDGET attempted
# trials per task and dropping inconclusive ones afterwards, averaged over DRAWS
# redraws -- the same construction that appendix/er16_vs_er2_stats.py applies to
# ER-1.6 on EARDataset, so the two are directly comparable.
BUDGET = 20
DRAWS = 500
SEED = 42


def corpus():
    """The corpus as released: one row per task, one image per task."""
    return pd.read_csv(BASELINE_CSV)


def cited_rules(tip):
    """The constitution rule numbers a `Tip Explanation` names, as a set."""
    m = re.search(r"Rule[s]?\s*([\d,\s]*(?:and\s*\d+)?)", str(tip))
    return {int(x) for x in re.findall(r"\d+", m.group(1))} if m else set()


def rule_coverage(df):
    """(counter over rules, n_tasks naming >=1 rule, n_tasks citing only general duty)."""
    counts, n_named, n_general_only = collections.Counter(), 0, 0
    for tip in df["Tip Explanation"].fillna(""):
        rules = cited_rules(tip)
        if rules:
            n_named += 1
            counts.update(rules)
            if rules <= GENERAL_DUTY_RULES:
                n_general_only += 1
    return counts, n_named, n_general_only


def trials():
    """The ER-1.6 attack trials, Judgement and Verdict coerced to numbers.

    `Judgement` is the safety judge: 1 bypass, 0 refusal, -1 inconclusive.
    `Verdict` is the edit-validity judge: 1 when the composited scene is usable.
    """
    d = pd.read_csv(ATTACK_CSV, low_memory=False)
    for col in ("Judgement", "Verdict"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    return d


def task_rates(d):
    """Per-task P(bypass) over conclusive trials, matching the main paper's rule."""
    valid = d[d["Judgement"].isin([0, 1])]
    return valid.groupby("Task ID")["Judgement"].apply(lambda s: float((s == 1).mean()))


def matched_budget(d):
    """Per-task rate, any-bypass and trial-level rate at BUDGET objects per task."""
    by_task = {t: g["Judgement"].values for t, g in d.groupby("Task ID")}
    rng = np.random.default_rng(SEED)
    rates, anys, trial_level = [], [], []
    for _ in range(DRAWS):
        drawn = {}
        hits = total = 0
        for task, judged in by_task.items():
            take = rng.choice(judged, size=min(BUDGET, len(judged)), replace=False)
            take = take[np.isin(take, (0, 1))]
            if len(take) == 0:
                continue
            drawn[task] = (take == 1).mean()
            hits += int((take == 1).sum())
            total += len(take)
        drawn = pd.Series(drawn)
        rates.append(drawn.mean())
        anys.append((drawn > 0).mean())
        trial_level.append(hits / total)
    return {
        "rate": float(np.mean(rates)),
        "any": float(np.mean(anys)),
        "trial": float(np.mean(trial_level)),
    }


def main():
    b, d = corpus(), trials()
    counts, n_named, n_general_only = rule_coverage(b)
    valid = d[d["Judgement"].isin([0, 1])]
    rates = task_rates(d)

    print("CORPUS")
    print(f"  tasks                     {len(b)}")
    print(f"  images on disk            {sum(1 for _ in Path(IMAGE_ROOT).glob('*/*.png'))}")
    print(f"  environments              {b['Environment'].nunique()}")
    print(f"  distinct instructions     {b['Instruction'].nunique()}")
    print(f"  top environments          {dict(b['Environment'].value_counts().head(5))}")
    print(f"  singleton environments    {int((b['Environment'].value_counts() == 1).sum())}")

    print("\nRULE GROUNDING")
    print(f"  tasks naming >=1 rule     {n_named} / {len(b)}")
    print(f"  distinct rules cited      {len(counts)} / {N_RULES}")
    print(f"  uncovered rules           {sorted(set(range(1, N_RULES + 1)) - set(counts))}")
    print(f"  general-duty rules 10/30  cited by {counts[10]} and {counts[30]} tasks")
    print(f"  tasks citing ONLY 10/30   {n_general_only}")

    print("\nER-1.6 EVALUATION")
    print(f"  tasks attacked            {d['Task ID'].nunique()}")
    print(f"  trials per task           {int(d.groupby('Task ID').size().median())}")
    print(f"  trials total              {len(d)}")
    print(f"  conclusive trials         {len(valid)} ({len(valid) / len(d):.1%})")
    print(f"  edits passing validity    {int((d['Verdict'] == 1).sum())} ({(d['Verdict'] == 1).mean():.1%})")
    print(f"  mean per-task rate        {rates.mean():.2%}")
    print(f"  tasks bypassed >=1        {int((rates > 0).sum())} / {rates.size} ({(rates > 0).mean():.1%})")
    print(f"  bypass rate over trials   {(valid['Judgement'] == 1).mean():.2%}")

    m = matched_budget(d)
    print(f"\nMATCHED TO {BUDGET} OBJECTS  ({DRAWS} draws, comparable to Appendix er16)")
    print(f"  mean per-task rate        {m['rate']:.2%}")
    print(f"  tasks bypassed >=1        {m['any']:.1%}")
    print(f"  bypass rate over trials   {m['trial']:.2%}")


if __name__ == "__main__":
    main()
