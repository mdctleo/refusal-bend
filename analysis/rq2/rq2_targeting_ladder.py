"""RQ2 figure: the targeting ladder broken down by stratum, with 95% CIs.

RQ2's claim is that it does not matter how the added object was chosen. Within
every stratum the untargeted conditions track the targeted one, including the
`zero` pole, where objects no proposer chose still flip tasks that no targeted
object ever flipped, while the rate itself climbs some fifty points from the
robust stratum to the malleable one.

Two things about the reading, both stated in the caption:

  * The `targeted` gradient is true BY CONSTRUCTION -- the strata were defined by
    that column -- so it is a manipulation check, not evidence. The evidence is
    that the other two columns climb with it. The same restriction is why the
    targeted CIs are narrow in the lower strata: that column's spread was capped
    by the sampling, not measured more precisely.
  * The bias runs against the untargeted conditions, so the tracking is the
    conservative reading rather than a flattering one.

The error bars are the 95% CI of each condition's mean across the tasks in the
stratum. They show how precisely each rate is pinned down; they are NOT the
test. Equivalence is established by paired TOST within task (see the text),
which is far more precise than these marginal intervals suggest, because pairing
removes between-task variance.

Why means and not distributions. The conditions do not share an object budget:
targeted carries 20 objects per task (inherited from RQ1) against 5 for the other
two, so per-task rates are quantised to 21 levels for targeted and only 6 for the
others. Overlaid distributions would differ in SHAPE for reasons of budget rather
than effect. Means are immune to that.

Usage (from the repo root):
    uv run python analysis/rq2/rq2_targeting_ladder.py --outdir paper/<project>/figures
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from figure_common import (BLUE, GRAY, MAGENTA, ORANGE, STRATUM_ORDER, ladder_rates,
                           outdir_arg, save)

STEM = "rq2_strata_breakdown"
CONDS = [("targeted", "Targeted", BLUE),
         ("scene_only", "Scene-only", MAGENTA),
         ("random", "Random", ORANGE),
         ("baseline", "Clean re-query", GRAY)]


def _ci95(x):
    """Half-width of the t-based 95% CI of the mean; 0 when the column is constant."""
    n = len(x)
    return stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)


def build(outdir):
    rates = ladder_rates()
    n = len(rates)

    grouped = rates.groupby("stratum")
    cols = [c for c, _, _ in CONDS]
    means = grouped[cols].mean().reindex(STRATUM_ORDER) * 100
    errs = grouped[cols].agg(_ci95).reindex(STRATUM_ORDER) * 100
    counts = grouped.size().reindex(STRATUM_ORDER)

    # Printed on its own at 0.7 \linewidth (5.5in text block = 3.85in), so LaTeX
    # scales it by ~0.64 and the 13pt/12pt type lands at ~8pt/7.5pt on the page.
    fig, ax = plt.subplots(figsize=(6.0, 3.7))
    xs = np.arange(len(STRATUM_ORDER))
    width = 0.8 / len(CONDS)

    for i, (col, label, color) in enumerate(CONDS):
        offset = (i - len(CONDS) / 2 + 0.5) * width
        ax.bar(xs + offset, means[col].to_numpy(), width, yerr=errs[col].to_numpy(),
               label=label, color=color, alpha=0.9, edgecolor="white", linewidth=0.6,
               error_kw=dict(ecolor="#333333", elinewidth=1.2, capsize=2.5, capthick=1.2),
               zorder=3)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{s}\n($n={counts[s]}$)" for s in STRATUM_ORDER], fontsize=12)
    ax.set_xlabel("Stratum (by targeted bypass rate)", fontsize=13)
    ax.set_ylabel("Mean bypass rate (%)", fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(0, (means + errs).to_numpy().max() + 4)
    ax.grid(True, axis="y", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    # Upper left is empty: the zero and low strata are near the axis.
    ax.legend(loc="upper left", framealpha=0.92, fontsize=11, ncol=2,
              handlelength=1.2, columnspacing=1.0)

    fig.tight_layout()
    save(fig, outdir, STEM)

    print(f"  n = {n} tasks over {len(STRATUM_ORDER)} strata")
    for s in STRATUM_ORDER:
        print(f"    {s:7s} (n={counts[s]:3d}): "
              + "  ".join(f"{lbl}={means.loc[s, col]:5.2f}+/-{errs.loc[s, col]:4.2f}"
                          for col, lbl, _ in CONDS))


if __name__ == "__main__":
    build(outdir_arg(__doc__.splitlines()[0]))
