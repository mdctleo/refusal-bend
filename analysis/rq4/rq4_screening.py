"""Figure 3 (RQ4): two readings of one malleability ranking, one panel each.

Every proxy assigns each of the 846 tasks a continuous score, and the figure
gives the two readings of that ranking their own panel, so the picture follows
the argument of Section 4.5 rather than restating one column of the scoreboard
table:

  (a) Separation. ROC curves against Bypass(t); the diagonal is chance. The
      composite is the curve of interest and the single signals it fuses sit
      behind it, so the gain from fusing them is visible rather than asserted.
  (b) Recovery at a budget. Precision@k as the audit budget k sweeps from 10 to
      the whole pool, with the base rate as the floor: the share of the k
      highest-scored tasks that are genuinely malleable, which is what a
      screener who can afford to probe only k tasks would see. Table 8's
      precision@50 and @100 are two points on the composite curve, marked.

Not drawn: a mean-bypass-rate-by-score-decile panel for the Spearman reading.
Four in five tasks have a rate of exactly zero, so a decile's mean rate is
mostly its share of bypassed tasks and the panel restates (b) per bin; within
the 171 bypassed tasks the composite orders by rate only weakly (rho 0.16) and
trace length not at all, so there is no degree-ordering picture to show.

Precision@k follows analysis/rq4/evaluate.py exactly (sort by score, take the
top k, mean of the labels), so the marked values agree with the table. Signals
are listed in ER-2 strength order, which is not the ER-1.6 order: A2 overtook
A3 as the strongest single signal when the target changed.

Placement. Printed at \\linewidth (5.5in), two panels in a row with the legend
under them; authored at 9in wide so LaTeX scales it by ~0.61, which puts the
15pt labels at ~9pt and the 14pt ticks at ~8.5pt on the page, the caption's
own size. The row is ~2.4in tall on the page.

Usage (from the repo root):
    uv run python analysis/rq4/rq4_screening.py --outdir paper/<project>/figures
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from figure_common import (BLUE, GOLD, GRAY, GT_CSV, MAGENTA, ORANGE, VIOLET,
                           outdir_arg, save)

STEM = "rq4_screening"

# (score CSV, score column, legend label, colour, linestyle, linewidth)
SOURCES = [
    ("analysis/output/rq4_er2_approach5_scores.csv", "score_curated_nolen",
     "A5 composite", MAGENTA, "-", 2.6),
    ("analysis/output/rq4_er2_approach1_features.csv", "word_count",
     "A1 trace length", ORANGE, "-.", 1.7),
    ("analysis/output/rq4_er2_approach2_scores.csv", "std_shift",
     "A2 perturbation spread", VIOLET, ":", 1.9),
    ("analysis/output/rq4_er2_approach3_scores.csv", "constitution_reliance",
     "A3 constitution reliance", BLUE, "--", 1.6),
    ("analysis/output/rq4_er2_approach4_scores.csv", "std_comply",
     "A4 decoding instability", GOLD, (0, (3, 1, 1, 1)), 1.6),
]
# Drawn at full strength in both panels; the rest are faded in (b) so the
# composite and the two strongest singles stay readable.
FOREGROUND = ("A5 composite", "A1 trace length", "A2 perturbation spread")
BUDGETS = (50, 100)          # the two precision@k columns of the scoreboard table
K_MIN = 10                   # below this the curve is a handful of tasks and pure noise
FS = 15                      # label size; ~9pt on the page


def paper_round(x, places=3):
    """Round the way the paper's tables do: to four places first (the precision
    the results summary records), then half-up to three. Formatting the raw
    float with :.3f disagrees with the text at the boundary (A5 0.76846 prints
    0.768 raw but 0.769 in Table 8 and Section 4.5; A2 0.70347 prints 0.703
    raw but 0.704), so the legend follows the tables rather than the float."""
    q = Decimal(1).scaleb(-places)
    return float(Decimal(f"{x:.4f}").quantize(q, rounding=ROUND_HALF_UP))


def precision_at_k(y_sorted):
    """Precision@k for every k, given the labels in descending score order."""
    k = np.arange(1, len(y_sorted) + 1)
    return k, np.cumsum(y_sorted) / k


def build(outdir):
    import matplotlib.pyplot as plt

    gt = pd.read_csv(GT_CSV)
    base = gt["bypassed_any"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9))
    ax_roc, ax_pk = axes
    handles = []
    for path, col, label, color, ls, lw in SOURCES:
        s = pd.read_csv(path, usecols=["Task ID", col])
        m = gt[["Task ID", "bypassed_any"]].merge(s, on="Task ID", how="inner").dropna()
        y = m["bypassed_any"].to_numpy(int)
        score = m[col].to_numpy(float)
        strong = label in FOREGROUND

        # (a) separation
        auc = roc_auc_score(y, score)
        fpr, tpr, _ = roc_curve(y, score)
        h, = ax_roc.plot(fpr, tpr, color=color, linestyle=ls, linewidth=lw,
                         zorder=3 if strong else 2,
                         label=f"{label} ({paper_round(auc):.3f})")
        handles.append(h)

        # (b) recovery at a budget; same sort as evaluate.py
        order = np.argsort(-score)
        k, pk = precision_at_k(y[order])
        ax_pk.plot(k[K_MIN - 1:], pk[K_MIN - 1:], color=color, linestyle=ls, linewidth=lw,
                   alpha=1.0 if strong else 0.45, zorder=3 if strong else 1)
        at_budget = {b: float(pk[b - 1]) for b in BUDGETS}
        if label == FOREGROUND[0]:
            for b, p in at_budget.items():
                ax_pk.plot(b, p, "o", color=color, markersize=6, zorder=4)
                ax_pk.annotate(f"{p:.2f}", (b, p), (-4, 9), textcoords="offset points",
                               fontsize=FS - 2, color=color)

        print(f"  {label}: n={len(m)}, ROC-AUC {auc:.5f} -> {paper_round(auc):.3f}, "
              + ", ".join(f"P@{b} {p:.2f}" for b, p in at_budget.items()))

    h, = ax_roc.plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=1.2, alpha=0.8,
                     label="chance (0.500)")
    handles.append(h)
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1)
    ax_roc.set_xticks([0, 0.5, 1.0])
    ax_roc.set_yticks([0, 0.5, 1.0])
    ax_roc.set_xlabel("False positive rate", fontsize=FS)
    ax_roc.set_ylabel("True positive rate", fontsize=FS)

    n = len(gt)
    ax_pk.axhline(base, color=GRAY, linestyle="--", linewidth=1.2)
    ax_pk.text(K_MIN * 1.15, base + 0.03, f"base rate {base:.3f}", fontsize=FS - 2, color=GRAY)
    ax_pk.set_xscale("log")
    ax_pk.set_xlim(K_MIN, n)
    ax_pk.set_ylim(0, 1)
    ax_pk.set_xticks([10, 50, 100, 300, n])
    ax_pk.set_xticklabels(["10", "50", "100", "300", str(n)])
    ax_pk.minorticks_off()
    ax_pk.set_yticks([0, 0.5, 1.0])
    ax_pk.set_xlabel("Audit budget $k$ (tasks probed)", fontsize=FS)
    ax_pk.set_ylabel("Precision@$k$", fontsize=FS)

    for ax, tag in zip(axes, ("(a)", "(b)")):
        ax.set_title(tag, loc="left", fontsize=FS, pad=4)
        ax.tick_params(axis="both", labelsize=FS - 1)
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=FS - 1, frameon=False,
               columnspacing=1.6, handlelength=2.6, bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout(w_pad=1.6)
    save(fig, outdir, STEM)
    print(f"  base rate {base:.4f} over {n} tasks")


if __name__ == "__main__":
    build(outdir_arg(__doc__.splitlines()[0]))
