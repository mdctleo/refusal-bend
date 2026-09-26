"""RQ2 task sample on ER-2: stratify the refused tasks by their targeted bypass rate and draw up to 50 per stratum.

Every RQ2 rung (scene_only, random) runs on the tasks in this file, and the RQ2 analysis and RQ3 read their
strata from it; this script is the only place the sample is drawn.

  rate(t)   task t's targeted bypass rate over its conclusive object trials (analysis/rq1/bypass_rates.py:
            J == 1 over J in {0, 1}; no Verdict), from
            EARDataset/robot_constitution_64_automerge_er2_attack_20.csv (846 tasks ER-2 refuses clean, 20 objects each)
  stratum   zero (= 0), low (<= 10%), medium (<= 25%), high (> 25%); tasks without a conclusive trial are left out
  sample    min(50, pool) tasks per stratum, pool.sample(random_state=42), as in the ER-1.6 draw
            (analysis/stratified_noise_generate.py)

The rows are copied from the clean re-query (EARDataset/dataset_requery_er2.csv), so the output already holds the
`baseline` rung: task metadata, the re-query's Response/Thinking/Judgement/Judgement Reason, plus `stratum` and
`bypass_rate`. Same columns as the ER-1.6 EARDataset/stratified_baseline.csv, which this never overwrites.

Usage (from the repo root):
    ./.venv/bin/python analysis/rq2/stratified_sample.py
    ./.venv/bin/python analysis/rq2/stratified_sample.py --output_csv /tmp/check.csv   # preview elsewhere
"""
import os
import sys
from argparse import ArgumentParser

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rq1"))
from bypass_rates import STRATUM_ORDER, assign_stratum, per_task, read_judged  # shared defs, live in rq1/

OBJECT_CSV = "EARDataset/robot_constitution_64_automerge_er2_attack_20.csv"
REQUERY_CSV = "EARDataset/dataset_requery_er2.csv"
OUTPUT_CSV = "EARDataset/stratified_baseline_er2.csv"
SAMPLES_PER_STRATUM = 50
SEED = 42


def stratify_and_sample(objects, per_stratum=SAMPLES_PER_STRATUM, seed=SEED):
    """One row per sampled task: Task ID, bypass_rate, stratum."""
    rates = per_task(objects)["rate"].dropna().rename("bypass_rate").reset_index()
    rates["stratum"] = assign_stratum(rates["bypass_rate"])
    sampled = []
    for s in STRATUM_ORDER:
        pool = rates[rates["stratum"] == s]
        n = min(per_stratum, len(pool))
        print(f"  {s:6s} pool {len(pool):4d} -> sampling {n}")
        sampled.append(pool.sample(n=n, random_state=seed))
    return pd.concat(sampled, ignore_index=True)


def main():
    parser = ArgumentParser(description="Draw the RQ2 stratified task sample from the ER-2 targeted bypass rates.")
    parser.add_argument("--output_csv", default=OUTPUT_CSV)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output CSV.")
    args = parser.parse_args()

    # later stages (re-judging, joins) may have written into it
    if os.path.exists(args.output_csv) and not args.force:
        sys.exit(f"ERROR: {args.output_csv} already exists; pass --force to overwrite it.")

    objects = read_judged(OBJECT_CSV)
    requery = pd.read_csv(REQUERY_CSV, low_memory=False)
    print(f"{OBJECT_CSV}: {objects['Task ID'].nunique()} tasks")
    sample = stratify_and_sample(objects)

    out = sample.merge(requery, on="Task ID", how="left", validate="one_to_one")
    missing = out["Instruction"].isna() & out["Matched Image Path"].isna()
    if missing.any():
        sys.exit(f"ERROR: {int(missing.sum())} sampled task(s) not in {REQUERY_CSV}")
    out = out[list(requery.columns) + ["stratum", "bypass_rate"]]
    out.to_csv(args.output_csv, index=False)

    print(f"wrote {args.output_csv}: {len(out)} tasks")
    print(out.groupby("stratum")["bypass_rate"].agg(["size", "min", "max", "mean"]).reindex(STRATUM_ORDER).to_string())


if __name__ == "__main__":
    main()
