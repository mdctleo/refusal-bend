"""RQ1 across planners: mean object bypass rate by stratum, each planner binned by its own rate.

Every planner is taken over its own refused tasks (the tasks of its attack CSV with at least one
conclusive trial), and each task is assigned to one of the paper's four strata by that planner's
own per-task bypass rate: zero, (0, 10%], (10%, 25%], > 25%. A stratum shows each planner's mean
per-task rate over its tasks in that bin, with the 95% t-interval across tasks, and the number of
tasks above each bar. The gradient within a planner is by construction, since the bins are cut on
the rate being averaged; what the figure compares across planners is how the refused tasks
distribute over the strata (the n above the bars) and how far the rate climbs inside each one.

Alongside the figure it writes analysis/output/rq1_targets_strata.json: the per-planner stratum
table, and on the tasks every planner refused (the shared set) the pairwise Spearman correlation
of per-task rates, how many tasks bend on k planners, and each planner's rate on the shared tasks
against its exclusively refused ones.

Usage (from the repo root):
    uv run python analysis/rq1/rq1_targets_strata.py --outdir paper/<project>/figures \
        --targets er2 gpt54
"""

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from bypass_rates import STRATUM_ORDER, assign_stratum  # sibling; the paper's stratum cuts
from figure_common import save
from targets import TARGETS, load_targets, shared_tasks, targets_arg

STEM = "rq1_targets_strata"
OUTPUT_JSON = "analysis/output/rq1_targets_strata.json"


def _ci95(x):
    """Half-width of the t-based 95% CI of the mean; 0 for a constant or single value."""
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return 0.0
    return float(stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n))


def _own_strata(target):
    """Per stratum of the planner's own rate: n, mean and CI over its refused tasks."""
    strata = assign_stratum(target.rates)
    out = {}
    for s in STRATUM_ORDER:
        r = target.rates[strata == s]
        out[s] = {"n": int(len(r)), "mean": float(r.mean()) if len(r) else 0.0, "ci95": _ci95(r)}
    return out


def _cross_stats(targets, shared):
    """Pairwise Spearman, bend-on-k counts, and shared vs exclusive rates, on the shared set."""
    table = pd.DataFrame({t.name: t.rates.loc[shared] for t in targets})
    spearman = {f"{a}:{b}": float(stats.spearmanr(table[a], table[b]).statistic)
                for a, b in itertools.combinations(table.columns, 2)}
    k = (table > 0).sum(axis=1)
    bend_on_k = {int(i): int((k == i).sum()) for i in range(len(targets) + 1)}
    by_planner = {}
    for t in targets:
        excl = t.rates.index.difference(shared)
        by_planner[t.name] = {"n_refused": t.n_tasks, "n_shared": int(len(shared)),
                              "n_exclusive": int(len(excl)),
                              "shared_mean_rate": float(t.rates.loc[shared].mean()),
                              "exclusive_mean_rate": float(t.rates.loc[excl].mean()) if len(excl) else None,
                              "floor": t.floor}
    return {"n_shared": int(len(shared)), "spearman": spearman,
            "bend_on_k_planners": bend_on_k, "by_planner": by_planner}


def build(outdir, names=None):
    targets = load_targets(names or list(TARGETS))
    table = {t.name: _own_strata(t) for t in targets}
    cross = _cross_stats(targets, shared_tasks(targets))

    # Same panel as the RQ2 strata figure: printed at 0.6-0.7 \linewidth, 13pt/12pt type.
    fig, ax = plt.subplots(figsize=(6.0, 3.7))
    xs = np.arange(len(STRATUM_ORDER))
    width = 0.8 / len(targets)
    err_kw = dict(ecolor="#333333", elinewidth=1.2, capsize=2.5, capthick=1.2)
    top = 0.0
    for i, t in enumerate(targets):
        offset = (i - len(targets) / 2 + 0.5) * width
        means = np.array([table[t.name][s]["mean"] for s in STRATUM_ORDER]) * 100
        errs = np.array([table[t.name][s]["ci95"] for s in STRATUM_ORDER]) * 100
        ax.bar(xs + offset, means, width, yerr=errs, label=t.label, color=t.color, alpha=0.9,
               edgecolor="white", linewidth=0.6, error_kw=err_kw, zorder=3)
        for x, m, e, s in zip(xs + offset, means, errs, STRATUM_ORDER):
            ax.annotate(f"$n$={table[t.name][s]['n']}", (x, m + e), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8.5,
                        color=t.color)
        top = max(top, (means + errs).max())

    ax.set_xticks(xs)
    ax.set_xticklabels(STRATUM_ORDER, fontsize=12)
    ax.set_xlabel("Stratum (by each planner's own bypass rate)", fontsize=13)
    ax.set_ylabel("Mean bypass rate (%)", fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(0, top + 8)
    ax.grid(True, axis="y", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", framealpha=0.92, fontsize=11, ncol=2,
              handlelength=1.2, columnspacing=1.0)
    fig.tight_layout()
    save(fig, outdir, STEM)

    out = {"targets": [t.name for t in targets], "strata": table, "shared": cross}
    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {OUTPUT_JSON}")

    for t in targets:
        print(f"  {t.label} ({t.n_tasks} refused tasks, {len(t.rates)} with a rate): "
              + "  ".join(f"{s}={100 * table[t.name][s]['mean']:.2f}+/-{100 * table[t.name][s]['ci95']:.2f}"
                          f" (n={table[t.name][s]['n']})" for s in STRATUM_ORDER))
    print(f"  shared refused tasks: {cross['n_shared']}; spearman "
          + ", ".join(f"{k} {v:.3f}" for k, v in cross["spearman"].items())
          + f"; bending on k planners {cross['bend_on_k_planners']}")
    for t in targets:
        m = cross["by_planner"][t.name]
        excl = "n/a" if m["exclusive_mean_rate"] is None else f"{100 * m['exclusive_mean_rate']:.2f}%"
        floor = "not run" if m["floor"] is None else f"{100 * m['floor']:.2f}%"
        print(f"    {t.label}: shared rate {100 * m['shared_mean_rate']:.2f}%, "
              f"exclusive ({m['n_exclusive']} tasks) {excl}, floor {floor}")


if __name__ == "__main__":
    args = targets_arg(__doc__.splitlines()[0])
    build(args.outdir, args.targets)
