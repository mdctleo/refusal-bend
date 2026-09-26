"""Appendix figure: per-task bypass counts against a shared-rate null.

Supporting evidence for RQ3 and RQ4 rather than for RQ1 itself. It shows the
per-task counts are far more dispersed than a single shared flip probability
allows, so susceptibility varies systematically across tasks and there is a
latent per-task propensity to root-cause and to predict. Each task keeps its own
trial count n_t, which makes the curve robust to the tasks probed with only a
handful of objects.

It establishes heterogeneity only. It cannot separate an object effect from a
task-specific instability under re-query, since both yield a per-task propensity;
the clean re-query control in the RQ1 figure is what settles that.

Usage (from the repo root):
    uv run python analysis/rq1/rq1_overdispersion.py --outdir paper/<project>/figures
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from figure_common import GRAY, MAGENTA, OBJECT_CSV, outdir_arg, read_judged, save

STEM = "rq1_overdispersion"


def build(outdir):
    d = read_judged(OBJECT_CSV, usecols=["Task ID", "Judgement"])
    valid = d[d["Judgement"].isin([0, 1])].groupby("Task ID")["Judgement"]
    n = valid.size().to_numpy()
    k = valid.apply(lambda s: int((s == 1).sum())).to_numpy()
    p = k.sum() / n.sum()
    n_tasks = len(n)

    ks = np.arange(0, k.max() + 1)
    observed = np.array([(k >= x).mean() for x in ks])

    # Null: every task tossed n_t independent coins at the pooled rate p. Drawn
    # rather than evaluated in closed form so the band is a sampling envelope on
    # the same footing as the observed curve.
    rng = np.random.default_rng(42)
    draws = rng.binomial(n[None, :], p, size=(2000, n_tasks))
    null_curves = np.array([[(row >= x).mean() for x in ks] for row in draws])
    null_mean = null_curves.mean(axis=0)
    lo, hi = np.percentile(null_curves, [2.5, 97.5], axis=0)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.fill_between(ks, lo, hi, color=GRAY, alpha=0.25, linewidth=0, step="post")
    ax.step(ks, null_mean, where="post", color=GRAY, linestyle="--", linewidth=2.0,
            label=f"Shared-rate null ($p={p:.3f}$, per-task $n_t$)")
    ax.step(ks, observed, where="post", color=MAGENTA, linewidth=2.4,
            label="Observed")

    # The two curves cross: too many tasks are never bypassed at all, and too many
    # are bypassed far more often than one shared rate would allow.
    n_zero = int((k == 0).sum())
    null_zero = (1.0 - null_mean[1]) * n_tasks
    tail_k = 10
    n_tail = int((k >= tail_k).sum())
    null_tail = null_mean[tail_k] * n_tasks

    arrow = dict(arrowstyle="->", color="black", alpha=0.55, lw=1.1,
                 connectionstyle="arc3,rad=-0.15")
    box = dict(boxstyle="round", facecolor="wheat", alpha=0.55, linewidth=0.6)
    ax.annotate(f"{n_zero} tasks never bypass\n(null expects {null_zero:.0f})",
                xy=(1.0, observed[1]), xytext=(5.5, 0.70), fontsize=12,
                arrowprops=arrow, bbox=box)
    ax.annotate(f"{n_tail} tasks bypass $\\geq {tail_k}$ objects\n"
                f"(null expects {null_tail:.2f})",
                xy=(tail_k, observed[tail_k]), xytext=(12.5, 0.45), fontsize=12,
                arrowprops=arrow, bbox=box)

    # Tie the picture to the dispersion test reported in the text.
    chi2 = (((k - n * p) ** 2) / (n * p * (1 - p))).sum()
    ax.text(0.975, 0.21, f"variance ${chi2 / (n_tasks - 1):.1f}\\times$ Binomial\n"
            f"($\\chi^2$ = {chi2:,.1f}, {n_tasks - 1} df)",
            transform=ax.transAxes, ha="right", va="top", fontsize=12)

    ax.set_xlabel("Bypasses per task $k$")
    ax.set_ylabel("Fraction of tasks with $\\geq k$ bypasses")
    ax.set_xlim(0, k.max())
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", framealpha=0.92, fontsize=12)
    save(fig, outdir, STEM)

    crossing = next((int(x) for x in ks if observed[x] > null_mean[x]), None)
    print(f"  chi2 = {chi2:,.1f} on {n_tasks - 1} df, "
          f"dispersion {chi2 / (n_tasks - 1):.1f}x; pooled p = {p:.4f}")
    print(f"  {n_zero} zeros vs {null_zero:.0f} expected; "
          f"{n_tail} tasks >= {tail_k} vs {null_tail:.2f} expected; "
          f"curves cross at k = {crossing}")


if __name__ == "__main__":
    build(outdir_arg(__doc__.splitlines()[0]))
