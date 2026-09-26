"""Shared style, data paths and helpers for the paper's figure modules.

Every figure in the paper is produced by a module inside the `analysis/rqN/`
folder of the research question it belongs to, one module per figure, so that
the code behind any given figure sits next to the analysis it illustrates.
Statistics that appear inside a figure are recomputed here from the canonical
result CSVs rather than pasted in, so a figure and the text quoting it cannot
drift apart.

All paths are relative to the repository root; run the modules from there.

Target model: `gemini-robotics-er-2-preview` (ER-2). The earlier ER-1.6 inputs
are deliberately NOT referenced here -- that run is preserved in the
`*_ER16.md` summaries, and mixing the two in one figure would compare across
target models. Every CSV below is ER-2.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

# IBM colorblind-safe palette.
BLUE, VIOLET, MAGENTA, ORANGE, GOLD = "#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000"
GRAY = "#808080"

# RQ1: the object trials over the full study population, and the clean re-query
# control. On ER-2 the control covers EVERY refused task rather than a stratified
# subsample, so the floor is a corpus-wide measurement and needs no correction for
# the subsample over-representing malleable tasks.
OBJECT_CSV = "EARDataset/robot_constitution_64_automerge_er2_attack_20.csv"
REQUERY_CSV = "EARDataset/dataset_requery_er2.csv"

# RQ2: the targeting ladder, all four arms on the same 194 stratified tasks.
BASELINE_CSV = "EARDataset/stratified_baseline_er2.csv"
SCENE_ONLY_CSV = "EARDataset/stratified_scene_only_er2.csv"
RANDOM_CSV = "EARDataset/stratified_random_er2.csv"

# RQ4: per-task labels used by every proxy signal.
GT_CSV = "analysis/output/rq4_er2_ground_truth.csv"

# Image assets for the worked example.
IMAGE_ROOT = "EARDataset/images"
ATTACK_IMAGE_ROOT = "EARDataset/robot_constitution_64_automerge_er2_attacked_20"
RANDOM_IMAGE_ROOT = "EARDataset/stratified_random_er2_images"

STRATUM_ORDER = ["zero", "low", "medium", "high"]


def read_judged(path, usecols=None):
    """Read a judged trial CSV with Judgement coerced to a number.

    Several of these files carry a stray non-numeric Judgement, so the coercion
    is not optional: without it `!= -1` silently keeps unjudged rows.
    """
    d = pd.read_csv(path, usecols=usecols, low_memory=False)
    d["Judgement"] = pd.to_numeric(d["Judgement"], errors="coerce")
    return d


def task_bypass_rate(df):
    """Per-task P(J == 1), counting conclusive trials only (J == -1 excluded)."""
    valid = df[df["Judgement"].isin([0, 1])]
    return valid.groupby("Task ID")["Judgement"].apply(lambda s: float((s == 1).mean()))


def corpus_rates():
    """Per-task bypass rate over the full ER-2 study population.

    Returns (rates, n_corpus, n_valid_trials, pooled_trial_rate). n_corpus is the
    846-task study population, one larger than len(rates): a single task (the
    injection-needles task) had every one of its trials come back empty and judged
    inconclusive, so it has no measurable rate. It stays in the denominator,
    counted as not bypassed, both to keep every figure on the same corpus the text
    quotes and because that is the conservative direction.
    """
    d = read_judged(OBJECT_CSV, usecols=["Task ID", "Judgement"])
    valid = d[d["Judgement"].isin([0, 1])]
    return (task_bypass_rate(d), d["Task ID"].nunique(), len(valid),
            float((valid["Judgement"] == 1).mean()))


def requery_rates():
    """Per-task bypass rate of the clean re-query, one row per re-queried task.

    This is the control for RQ1: the same unmodified image queried a second time.
    ER-2 is deterministic on a repeated input (845 of 846 responses come back
    word-for-word identical to the first query), so what little this measures is
    judge variability rather than planner re-rolling. Exactly two tasks move, and
    both carry a single trial, so both sit at a rate of 1.0.
    """
    d = read_judged(REQUERY_CSV, usecols=["Task ID", "Judgement"])
    return task_bypass_rate(d)


def requery_floor():
    """Mean per-task re-query rate -- the non-determinism floor. (mean, n_tasks)."""
    rates = requery_rates()
    return float(rates.mean()), int(len(rates))


def ladder_rates():
    """One row per task: targeted, scene_only, random, baseline, stratum.

    The four RQ2 arms intersected and NaN-dropped, so every claim about this
    subsample is made over an identical set of tasks.
    """
    baseline = read_judged(BASELINE_CSV)
    ids = baseline["Task ID"].unique()
    targeted = read_judged(OBJECT_CSV, usecols=["Task ID", "Judgement"])
    targeted = targeted[targeted["Task ID"].isin(ids)]
    rates = pd.concat([task_bypass_rate(targeted).rename("targeted"),
                       task_bypass_rate(read_judged(SCENE_ONLY_CSV)).rename("scene_only"),
                       task_bypass_rate(read_judged(RANDOM_CSV)).rename("random"),
                       task_bypass_rate(baseline).rename("baseline")], axis=1)
    rates["stratum"] = baseline.drop_duplicates("Task ID").set_index("Task ID")["stratum"]
    return rates.dropna(subset=["targeted", "scene_only", "random", "baseline"])


def outdir_arg(description):
    """The --outdir argument every figure module takes."""
    import argparse

    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--outdir", required=True, help="Directory to write the figure into")
    args = ap.parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    return args.outdir


def save(fig, outdir, stem, formats=("pdf", "png")):
    for ext in formats:
        fig.savefig(Path(outdir) / f"{stem}.{ext}", format=ext)
    plt.close(fig)
    print(f"  wrote {stem}." + "/.".join(formats))
