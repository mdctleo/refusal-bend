"""The target planners RQ1 was run on, and their per-task results, for the cross-planner figures.

Each planner went through the RQ1 chain from its own clean baseline (scripts/run_rq1.sh with
TARGET=<name>), so each has its own refused set, its own attack CSV and its own clean re-query.
This module resolves a planner name to those files and reduces them to the per-task quantities
the figures draw: the object bypass rate of every refused task, the clean re-query rate, and the
tasks refused by every planner under comparison (the shared set).

The rate definitions are the paper's (analysis/figure_common.py): a task's rate is P(J == 1) over
its conclusive trials, J == -1 excluded; the corpus size is every task in the attack CSV, so a
task whose trials all came back inconclusive counts as not bypassed in survival denominators.

Usage: imported by rq1_targets_strata.py and rq1_targets_survival.py; both take
    --targets er2 gpt54 ...     planners to draw (default: every planner in TARGETS)
and fail on a planner whose judged attack CSV does not exist yet. A planner whose re-query CSV
does not exist yet is drawn without its floor, with a printed notice.
"""

import argparse
import glob
import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from figure_common import BLUE, GOLD, MAGENTA, ORANGE, VIOLET, read_judged, task_bypass_rate

GREEN = "#009E73"   # sixth planner color, outside the five-color palette of figure_common

# name used by scripts/run_rq1.sh -> (figure label, color)
TARGETS = {
    "er2": ("ER-2", BLUE),
    "gpt54": ("GPT-5.4", MAGENTA),
    "opus55": ("Claude Opus 5.5", ORANGE),
    "kimi26": ("Kimi K2.6", VIOLET),
    "cosmosr2": ("Cosmos-Reason2", GOLD),
    "robobrain25": ("RoboBrain2.5", GREEN),
}


def attack_csv(name):
    return f"EARDataset/robot_constitution_64_automerge_{name}_attack_20.csv"


def requery_csv(name):
    return f"EARDataset/dataset_requery_{name}.csv"


def requery_passes(name):
    """Every judged re-query pass of a planner: the main CSV and any dataset_requery_<name>_<k>.csv."""
    return [p for p in [requery_csv(name), *sorted(glob.glob(f"EARDataset/dataset_requery_{name}_[0-9]*.csv"))]
            if os.path.exists(p)]


@dataclass
class Target:
    name: str
    label: str
    color: str
    rates: pd.Series          # per-task object bypass rate, tasks with >= 1 conclusive trial
    n_tasks: int              # every task in the attack CSV (the planner's refused set)
    n_trials: int             # conclusive object trials
    trial_rate: float         # pooled per-trial bypass rate
    requery: pd.Series | None  # per-task clean re-query rate over all passes, None until that stage has run

    @property
    def floor(self):
        """Mean per-task re-query rate, the planner's non-determinism floor; None if not run."""
        return None if self.requery is None else float(self.requery.mean())


def load_target(name):
    label, color = TARGETS[name]
    path = attack_csv(name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name}: {path} does not exist; run RQ1 for TARGET={name} first.")
    d = read_judged(path, usecols=["Task ID", "Judgement"])
    if d["Judgement"].isna().any():
        raise ValueError(f"{name}: {path} has unjudged rows; run the judge stage first.")
    valid = d[d["Judgement"].isin([0, 1])]
    requery = None
    passes = requery_passes(name)
    if passes:
        # a task's re-query rate is over its conclusive answers in every pass
        requery = task_bypass_rate(pd.concat(read_judged(p, usecols=["Task ID", "Judgement"]) for p in passes))
        if len(passes) > 1:
            print(f"  {label}: re-query floor pooled over {len(passes)} passes")
    if requery is not None and requery.empty:
        requery = None
        print(f"  note: {requery_csv(name)} is not judged yet; {label} is drawn without its re-query floor")
    elif requery is None:
        print(f"  note: {requery_csv(name)} not found; {label} is drawn without its re-query floor")
    return Target(name, label, color, task_bypass_rate(d), d["Task ID"].nunique(), len(valid),
                  float((valid["Judgement"] == 1).mean()), requery)


def load_targets(names):
    return [load_target(n) for n in names]


def shared_tasks(targets):
    """Tasks refused by every planner and carrying a conclusive rate under each."""
    ids = None
    for t in targets:
        ids = t.rates.index if ids is None else ids.intersection(t.rates.index)
    return ids


def targets_arg(description):
    """The --outdir and --targets arguments both figure modules take."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--outdir", required=True, help="Directory to write the figure into")
    ap.add_argument("--targets", nargs="+", default=list(TARGETS), choices=list(TARGETS),
                    help="Planners to draw, in legend order")
    args = ap.parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    return args
