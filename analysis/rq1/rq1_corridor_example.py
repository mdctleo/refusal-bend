"""Figure 1: one refusal, and the two ordinary objects that bend it.

Three views of the same scene under the same instruction. On the clean scene (a)
the target refuses. With one object composited in, it complies and returns a
plan: (b) an object proposed by the targeted proposer for this task; (c) an
object drawn from the fixed everyday list that never saw the task. Under each
panel is the planner's own output from the result CSVs: the refusal with its
middle elided (marked "[...]") and the first step of each plan. The added object
is not marked; the panel title names it.

The task, the two objects and the quoted refusal spans are set by the constants
below and checked against the CSVs when the figure is built.

Photographs only, so PNG only.

Usage (from the repo root):
    uv run python analysis/rq1/rq1_corridor_example.py --outdir paper/<project>/figures
"""

import json
import re
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from figure_common import (ATTACK_IMAGE_ROOT, IMAGE_ROOT, MAGENTA, OBJECT_CSV, RANDOM_CSV,
                           RANDOM_IMAGE_ROOT, REQUERY_CSV, outdir_arg, read_judged, save)

STEM = "rq1_corridor_example"
TASK_ID = "hotel_tip4_scene4"          # dataset ID; the UUID is looked up from the CSV
TARGETED_OBJECT = "stack of books"     # substring of co_object_set in the targeted arm
RANDOM_OBJECT = "an orange"            # substring of co_object_set in the random arm
N_STEPS = 1                            # plan steps to quote under each complied panel
QUOTE_PT = 16                          # ~7.3pt on the page at \linewidth

# The spans of the refusal that are quoted, in order; the text between them is
# replaced by "[...]". Tied to the task above and checked against the CSV.
REFUSAL_SPANS = ("I cannot fulfill this request. Placing bags", "near a fire exit",
                 "can obstruct the exit route during an emergency.")

REFUSED, COMPLIED = "#444444", MAGENTA


def _one(df, mask, what):
    hit = df[mask]
    if hit.empty:
        raise SystemExit(f"no row for {what}")
    return hit.iloc[0]


def _plan_steps(response):
    """The `label` fields of the JSON plan the planner returns, in order."""
    m = re.search(r"```json\s*(\[.*?\])\s*```", str(response), re.S)
    if not m:
        raise SystemExit("compliant response carries no JSON plan")
    return [step["label"] for step in json.loads(m.group(1))]


def _elide(text):
    """REFUSAL_SPANS quoted verbatim in order, joined by a marked elision."""
    pos = 0
    for span in REFUSAL_SPANS:
        i = text.find(span, pos)
        if i < 0:
            raise SystemExit(f"refusal does not contain {span!r} in order")
        pos = i + len(span)
    return " [...] ".join(REFUSAL_SPANS)


def _quote_plan(steps, width):
    """The first N_STEPS plan steps verbatim, each wrapped, then a count of the rest."""
    lines = []
    for i, s in enumerate(steps[:N_STEPS]):
        s = s[0].upper() + s[1:].rstrip(".")
        lines.append(textwrap.fill(f"{i + 1}. {s}", width, subsequent_indent="    "))
    rest = len(steps) - N_STEPS
    return "\n".join(lines) + (f"\n... and {rest} more steps" if rest > 0 else "")


def _chars_per_column(fig, ax, fontsize):
    """Characters of body text that fit across `ax`, measured from a rendered sample."""
    fig.canvas.draw()
    sample = "Placing bags or other obstacles near a fire exit violates safety"
    t = ax.text(0, 0, sample, fontsize=fontsize, family="serif")
    w_text = t.get_window_extent(fig.canvas.get_renderer()).width
    t.remove()
    w_ax = ax.get_window_extent(fig.canvas.get_renderer()).width
    return int(0.90 * w_ax / (w_text / len(sample)))


def build(outdir):
    targeted = read_judged(OBJECT_CSV)
    targeted = targeted[targeted["ID"] == TASK_ID]
    if targeted.empty:
        raise SystemExit(f"task {TASK_ID!r} not found in {OBJECT_CSV}")
    uuid = targeted["Task ID"].iloc[0]
    instruction = targeted["Instruction"].iloc[0]

    clean = _one(read_judged(REQUERY_CSV), lambda d: d["Task ID"] == uuid, "clean re-query")
    if clean["Judgement"] != 0:
        raise SystemExit("clean re-query is not a refusal; the figure's premise fails")
    t_row = _one(targeted, (targeted["Judgement"] == 1)
                 & targeted["co_object_set"].astype(str).str.contains(TARGETED_OBJECT),
                 f"targeted bypass with {TARGETED_OBJECT!r}")
    rnd = read_judged(RANDOM_CSV)
    r_row = _one(rnd, (rnd["Task ID"] == uuid) & (rnd["Judgement"] == 1)
                 & rnd["co_object_set"].astype(str).str.contains(RANDOM_OBJECT),
                 f"random bypass with {RANDOM_OBJECT!r}")

    conclusive = targeted[targeted["Judgement"].isin([0, 1])]
    n_t = int((conclusive["Judgement"] == 1).sum())
    rnd_task = rnd[(rnd["Task ID"] == uuid) & rnd["Judgement"].isin([0, 1])]
    n_r = int((rnd_task["Judgement"] == 1).sum())

    panels = [
        ("(a) unmodified: refused", REFUSED,
         Path(IMAGE_ROOT) / str(t_row["Matched Image Path"]),
         _elide(str(clean["Response"]).strip())),
        ("(b) + books: complied", COMPLIED,
         Path(ATTACK_IMAGE_ROOT) / str(t_row["Attack Image Path"]),
         _plan_steps(t_row["Response"])),
        ("(c) + orange: complied", COMPLIED,
         Path(RANDOM_IMAGE_ROOT) / str(r_row["Attack Image Path"]),
         _plan_steps(r_row["Response"])),
    ]
    for _, _, p, _ in panels:
        if not p.exists():
            raise SystemExit(f"missing image: {p}")

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 3.9),
                             gridspec_kw=dict(height_ratios=[3.0, 0.62], hspace=0.04,
                                              wspace=0.04))
    width = _chars_per_column(fig, axes[1, 0], QUOTE_PT)
    fig.suptitle(f"Instruction: “{instruction}”", fontsize=16, style="italic",
                 y=0.995)
    for col, (title, color, path, quote) in enumerate(panels):
        ax = axes[0, col]
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.set_title(title, fontsize=16, color=color, pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        tx = axes[1, col]
        tx.axis("off")
        text = (textwrap.fill(quote, width) if isinstance(quote, str)
                else _quote_plan(quote, width))
        tx.text(0.02, 0.98, text, transform=tx.transAxes, fontsize=QUOTE_PT, va="top", ha="left",
                family="serif", linespacing=1.25,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f6f6f6", edgecolor=color,
                          linewidth=1.2))

    save(fig, outdir, STEM, formats=("png",))

    print(f"  task {TASK_ID} ({uuid}); instruction: {instruction}")
    print(f"  targeted: {n_t}/{len(conclusive)} bypass; random: {n_r}/{len(rnd_task)} bypass")
    print(f"  shown: targeted={t_row['co_object_set']}  random={r_row['co_object_set']}")


if __name__ == "__main__":
    build(outdir_arg(__doc__.splitlines()[0]))
