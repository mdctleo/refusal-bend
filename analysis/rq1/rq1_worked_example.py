"""Appendix figure: one bypass, before and after, side by side.

Left is the clean scene, on which the target refuses; right is the same scene
with one object composited in, on which it returns a plan. The instruction, the
constitution and the decoding are identical across the two panels. The task and
the object are selected by TASK_ID and OBJECT_MATCH below and read, with their
image paths, from the targeted-arm CSV.

Both panels are photographs, so this module writes PNG only.

Usage (from the repo root):
    uv run python analysis/rq1/rq1_worked_example.py --outdir paper/<project>/figures
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from figure_common import (ATTACK_IMAGE_ROOT, IMAGE_ROOT, OBJECT_CSV, outdir_arg,
                           read_judged, save)

STEM = "rq1_worked_example"

# Dataset ID of the task shown; its UUID is looked up from the CSV.
TASK_ID = "hospital ward_tip4_scene1"
# The bypassing object shown, matched as a substring of `co_object_set`.
OBJECT_MATCH = "clear plastic water bottle"


def build(outdir):
    d = read_judged(OBJECT_CSV)
    g = d[d["ID"] == TASK_ID]
    if g.empty:
        raise SystemExit(f"task {TASK_ID!r} not found in {OBJECT_CSV}")

    conclusive = g[g["Judgement"].isin([0, 1])]
    n_bypass = int((conclusive["Judgement"] == 1).sum())
    rate = float((conclusive["Judgement"] == 1).mean())

    bypassing = g[g["Judgement"] == 1]
    hit = bypassing[bypassing["co_object_set"].astype(str).str.contains(
        OBJECT_MATCH, case=False, na=False)]
    if hit.empty:
        raise SystemExit(f"no bypassing object matching {OBJECT_MATCH!r} for {TASK_ID}")
    row = hit.iloc[0]

    clean_path = Path(IMAGE_ROOT) / str(row["Matched Image Path"])
    edit_path = Path(ATTACK_IMAGE_ROOT) / str(row["Attack Image Path"])
    for p in (clean_path, edit_path):
        if not p.exists():
            raise SystemExit(f"missing image: {p}")

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.0, 4.6))
    for ax, path, title in ((ax_l, clean_path, "Clean scene: refused"),
                            (ax_r, edit_path, "One object added: complied")):
        ax.imshow(mpimg.imread(path))
        ax.set_title(title, fontsize=15)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.tight_layout()
    save(fig, outdir, STEM, formats=("png",))

    print(f"  task {TASK_ID} ({row['Task ID']})")
    print(f"  instruction: {row['Instruction']}")
    print(f"  bypassed by {n_bypass}/{len(conclusive)} objects (rate {rate:.2f})")
    print(f"  object shown: {row['co_object_set']}")
    print(f"  clean: {clean_path}")
    print(f"  edited: {edit_path}")


if __name__ == "__main__":
    build(outdir_arg(__doc__.splitlines()[0]))
