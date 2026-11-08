"""Regenerate every figure in the paper, from the canonical ER-2 result files.

Each figure has its own module inside the `analysis/rqN/` folder of the research
question it belongs to, and can be run on its own; this runner exists so that a
single command refreshes the whole paper after a data or scoring change. No
figure anywhere in the paper should come from a scratchpad script -- if a number
appears in a figure, it is computed in one of these modules from the CSVs named
in analysis/figure_common.py.

Usage (from the repo root):
    uv run python analysis/generate_figures.py \
        --outdir paper/<project>/figures [--only rq1_bypass_survival]
"""

import argparse
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (module path relative to analysis/, output stem, what it appears as)
FIGURES = [
    ("rq1/rq1_corridor_example.py", "rq1_corridor_example", "Figure 1, teaser"),
    ("rq1/rq1_bypass_survival.py", "rq1_bypass_survival", "Appendix, strata distribution"),
    ("rq1/rq1_worked_example.py", "rq1_worked_example", "Appendix, worked bypass"),
    ("rq2/rq2_targeting_ladder.py", "rq2_strata_breakdown", "Figure 2, RQ2"),
    ("rq4/rq4_screening.py", "rq4_screening", "Figure 3, RQ4"),
    ("appendix/latentharm_environments.py", "latentharm_environments",
     "Appendix, corpus composition"),
    ("appendix/latentharm_example.py", "latentharm_example", "Appendix, corpus worked example"),
]


def load(rel_path, stem):
    spec = importlib.util.spec_from_file_location(f"_fig_{stem}", HERE / rel_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    stems = [s for _, s, _ in FIGURES]
    ap = argparse.ArgumentParser(description="Regenerate the paper's figures")
    ap.add_argument("--outdir", required=True, help="Directory to write the figures into")
    ap.add_argument("--only", default=None, choices=stems,
                    help="Regenerate a single figure by its output stem")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    for rel_path, stem, appears_as in FIGURES:
        if args.only in (None, stem):
            print(f"[{stem}]  ({appears_as})")
            load(rel_path, stem).build(args.outdir)


if __name__ == "__main__":
    main()
