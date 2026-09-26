"""LatentHarm appendix figure: what settings the corpus covers.

A horizontal bar per environment over all 281 released tasks, sorted by count.
Bars rather than a pie: there are 26 categories with a long tail of singletons,
and a 26-wedge pie cannot be read back to its legend at appendix scale, nor can
neighbouring settings be compared by eye. Counts are printed at the bar ends so
the figure is exact rather than approximate.

Drawn over the full corpus, not the 117 tasks carried into the ER-1.6
evaluation, because the figure describes what was released.

Usage (from the repo root):
    uv run python analysis/appendix/latentharm_environments.py --outdir paper/<project>/figures
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Both dirs, so the module imports the same way standalone and when the runner
# loads it by file path (which puts neither on sys.path).
for _p in (HERE, HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import matplotlib.pyplot as plt

from figure_common import BLUE, outdir_arg, save
from latentharm_stats import corpus

STEM = "latentharm_environments"


def build(outdir):
    counts = corpus()["Environment"].value_counts().sort_values()
    total = int(counts.sum())

    fig, ax = plt.subplots(figsize=(6.5, 6.2))
    ax.barh(counts.index, counts.to_numpy(), color=BLUE, height=0.72)

    for name, n in counts.items():
        ax.text(n + total * 0.006, name, str(int(n)), va="center", fontsize=10)

    ax.set_xlabel(f"Tasks ($n = {total}$)", fontsize=13)
    ax.set_xlim(0, counts.max() * 1.10)
    ax.tick_params(axis="y", labelsize=11)
    ax.tick_params(axis="x", labelsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.grid(True, alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    save(fig, outdir, STEM)


if __name__ == "__main__":
    build(outdir_arg(__doc__.splitlines()[0]))
