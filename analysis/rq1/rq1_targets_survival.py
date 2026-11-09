"""RQ1 across planners: the bypass survival curve of every planner, each against its own floor.

One curve per planner over that planner's own refused tasks: the fraction of tasks whose object
bypass rate is strictly above x, as in the ER-2 figure of the strata appendix. Each curve starts
at the share of tasks bypassed by at least one object and ends at the tasks bypassed on every
object. A planner's clean re-query rate, its non-determinism floor, is the dashed vertical in its
color, so every bypassed task of that planner sits to the right of its own floor. Populations
differ between planners (each refuses its own set of tasks), which is the point: the figure shows
that the shape of the RQ1 result repeats, planner by planner, not that the planners agree task by
task (that is rq1_targets_strata.py).

Usage (from the repo root):
    uv run python analysis/rq1/rq1_targets_survival.py --outdir paper/<project>/figures \
        --targets er2 gpt54
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from figure_common import save
from targets import TARGETS, load_targets, targets_arg

STEM = "rq1_targets_survival"


def build(outdir, names=None):
    targets = load_targets(names or list(TARGETS))

    # Same panel as the single-planner survival figure (0.55 \linewidth, 13pt/12pt type).
    fig, ax = plt.subplots(figsize=(6.0, 3.7))
    y_top = 0.0
    for t in targets:
        x = t.rates.to_numpy()
        vs = np.unique(np.concatenate([[0.0], x]))
        surv = np.array([float((x > v).sum()) / t.n_tasks for v in vs])
        n_any = int((x > 0).sum())
        legend = f"{t.label} ($n={t.n_tasks}$"
        if t.floor is not None:
            ax.axvline(t.floor * 100, color=t.color, linestyle="--", linewidth=1.6, alpha=0.9)
            legend += f", floor {100 * t.floor:.2f}%"
        ax.step(vs * 100, surv, where="post", color=t.color, linewidth=2.4, label=legend + ")")
        y_top = max(y_top, surv.max())
        print(f"  {t.label}: {t.n_tasks} tasks, {t.n_trials:,} conclusive trials, pooled trial rate "
              f"{t.trial_rate:.4f}; bypassed at all {n_any} ({100 * surv[0]:.1f}%); "
              + ("floor not run" if t.floor is None
                 else f"floor {t.floor:.4f} over {len(t.requery)} re-queried tasks"))

    ax.set_xlabel("Per-task bypass rate (%)", fontsize=13)
    ax.set_ylabel("Fraction of tasks above rate", fontsize=13)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, max(y_top * 1.12, 0.05))
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper right", framealpha=0.92, fontsize=11, handlelength=1.6)
    fig.tight_layout()
    save(fig, outdir, STEM)


if __name__ == "__main__":
    args = targets_arg(__doc__.splitlines()[0])
    build(args.outdir, args.targets)
