"""Appendix figure (strata): how many tasks are bypassed at least this often, against the baseline.

RQ1 has to carry two claims, and this figure carries both in one panel.

  1. Coherent objects bypass refusals, and unevenly. The curve starts at the
     fifth of the corpus bypassed by at least one object, and decays across
     every rate up to 100%, so the tasks that give way are spread over the whole
     range rather than clustered at one susceptibility.

  2. Those bypasses are not the planner re-rolling its own refusal. The dashed
     line is the bypass rate the planner produces on *unmodified* images, so the
     tasks to the right of it flip more often under object insertion than
     non-determinism alone accounts for.

Plotted as the survival function, the fraction of tasks with a rate strictly
above x, rather than the CDF. On a CDF the curve reaches 1.0 at x = 100 for any
dataset at all -- every task has a rate of at most 100% by definition -- which
reads as though every task were fully bypassed. The survival form has no such
vacuous endpoint: both ends carry information, and the two numbers the text
quotes can be read straight off the y-axis, at x = 0 and at the dashed line.

On ER-2 the floor is measured on EVERY refused task rather than on the stratified
subsample the ER-1.6 run used, so it is a corpus-wide quantity and carries no
over-representation of malleable tasks to apologise for. It is also very low
(ER-2 returns the identical plan on a repeated input): 0.24% against the 5.0% the
ER-1.6 figure showed, which leaves the band too thin to read at full scale. It is
left that way on purpose. The floor stays a vertical line because it is a RATE and
the claim about it is a threshold -- every bypassed task sits to its right -- and a
band drawn any wider than the 0.24% it actually spans would overstate the planner's
own instability. The legend carries the number the band is too thin to show.

Usage (from the repo root):
    uv run python analysis/rq1/rq1_bypass_survival.py --outdir paper/<project>/figures
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from figure_common import BLUE, GRAY, corpus_rates, outdir_arg, requery_floor, save

STEM = "rq1_bypass_survival"


def build(outdir):
    rates, n_tasks, n_trials, trial_rate = corpus_rates()
    x = rates.to_numpy()

    # Survival function: S(v) is the fraction of tasks with rate strictly above
    # v. S(0) is therefore the fraction bypassed at all, and the curve ends at
    # zero rather than at a value forced by the definition. The denominator is
    # the full corpus, so the one task with no conclusive trial counts as not
    # bypassed and the figure quotes the same 846 the text does.
    vs = np.unique(x)
    surv = np.array([float((x > v).sum()) / n_tasks for v in vs])

    floor, n_floor = requery_floor()
    s_floor = float((x > floor).sum()) / n_tasks
    n_above = int((x > floor).sum())
    n_bypassed = int((x > 0).sum())

    # Printed on its own at 0.55 \linewidth (5.5in text block = 3.03in), so LaTeX
    # scales it by ~0.5: the 13pt/12pt type below lands at ~6.5pt/6pt on the page,
    # comfortably below the 10pt caption, and the axes get most of the panel.
    fig, ax = plt.subplots(figsize=(6.0, 3.7))

    # The floor is a RATE, so it belongs on the rate axis: everything at or below it
    # is within reach of the planner's own non-determinism, and the claim the text
    # makes is that every bypassed task sits to its RIGHT. Drawing the control as its
    # own survival curve would answer a different question -- how the two
    # distributions compare -- and would make "above the control" read in the wrong
    # direction.
    ax.axvspan(0, floor * 100, color=GRAY, alpha=0.18, linewidth=0)
    ax.axvline(floor * 100, color=GRAY, linestyle="--", linewidth=1.8)

    ax.step(vs * 100, surv, where="post", color=BLUE, linewidth=2.6)

    ax.set_xlabel("Per-task bypass rate (%)", fontsize=13)
    ax.set_ylabel("Fraction of tasks above rate", fontsize=13)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, max(surv.max() * 1.12, 0.05))
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)

    # No legend at panel size: the shared caption names both the curve and the
    # dashed control line, and a legend box would crowd the curve's start.
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    save(fig, outdir, STEM)

    print(f"  corpus: {n_tasks} tasks, {n_trials:,} conclusive trials, "
          f"pooled trial rate {trial_rate:.4f}")
    print(f"  bypassed at all: {n_bypassed} ({100 * surv[0]:.1f}%); "
          f"floor {floor:.4f} over {n_floor} re-queried tasks; "
          f"above it: {n_above} ({100 * s_floor:.1f}%)")


if __name__ == "__main__":
    build(outdir_arg(__doc__.splitlines()[0]))
