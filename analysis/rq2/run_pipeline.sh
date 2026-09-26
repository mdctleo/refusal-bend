#!/bin/bash
# SUPERSEDED by scripts/run_rq2.sh -- use that for anything on ER-2.
# This is the ER-1.6 chain: it reads EARDataset/stratified_baseline.csv (200 tasks), its slice
# arithmetic assumes 1,000 rows, and it calls src/*.py and the .sub files directly rather than
# scripts/submit_{stage1,render,plans}.sh, so it has none of their image pre-checks, empty-response
# check or slice fan-out. Kept only as a record of how the ER-1.6 run was driven.
#
# RQ2 pipeline: targeted -> scene-only -> random objects on the 200 stratified tasks.
# Same chain as RQ1 with only the proposal step changed. See README.md.
#
# Usage (from the repo root, /ws):
#   analysis/rq2/run_pipeline.sh stage1 scene_only        # sbatch: Qwen proposer, image only
#   analysis/rq2/run_pipeline.sh stage1 random            # local: fixed list, no VLM
#   analysis/rq2/run_pipeline.sh render <rung> <a,b>      # sbatch: Flux.2-klein, rows a..b
#   analysis/rq2/run_pipeline.sh check  <rung>            # every Attack Image Path exists?
#   analysis/rq2/run_pipeline.sh plans  <rung> <a,b>      # target queries, rows a..b
#   analysis/rq2/run_pipeline.sh judge  <rung> <a,b>      # safety judge, rows a..b
#   analysis/rq2/run_pipeline.sh status <rung>            # rows with Response / Judgement filled
#
# <rung> is scene_only or random. Slices: rows are grouped by stratum (250 per stratum,
# 50 tasks x 5 objects), so the PILOT is the four slices
#     0,25   250,275   500,525   750,775          (5 tasks per stratum, 20 tasks)
# and the FULL run is the remainder
#     25,250 275,500 525,750 775,1000
# plans_generator.py has no resume logic and silently writes nothing without --rang:
# run disjoint slices and never re-run a finished one.
set -euo pipefail
cd "$(dirname "$0")/../.."

cmd="${1:-}"; rung="${2:-}"; rang="${3:-}"
[ -n "$cmd" ] || { sed -n 2,28p "$0"; exit 1; }   # 2-7 is the superseded notice, 8-28 the original usage
[ -z "$rung" ] || [[ "$rung" == "scene_only" || "$rung" == "random" ]] || { echo "rung must be scene_only or random"; exit 1; }

INPUT_CSV="EARDataset/stratified_baseline.csv"          # the 200 tasks (Judgement here is the re-query; NOT the filter)
EVAL_CSV="EARDataset/dataset_realistic.csv"             # clean-run judgements: all 200 tasks are Judgement==0 here
IMAGE_ROOT="EARDataset/images"
PROPOSER="Qwen/Qwen3-VL-30B-A3B-Thinking-FP8"
N=5; K=1
RUNG_CSV="EARDataset/stratified_${rung}.csv"
RUNG_IMAGES="EARDataset/stratified_${rung}_images"
PY="./.venv/bin/python"

case "$cmd" in
  stage1)
    if [ "$rung" = "random" ]; then
      $PY src/attack_stage1.py --input_csv_path "$INPUT_CSV" --baseline_evaluation_path "$EVAL_CSV" \
        --output_csv_path "$RUNG_CSV" --image_root "$IMAGE_ROOT" --n $N --k $K --system_prompt_name random_list
    else
      sbatch scripts/attack_stage_1.sub "$INPUT_CSV" "$EVAL_CSV" "$RUNG_CSV" "$IMAGE_ROOT" "$PROPOSER" $N $K scene_only
    fi ;;
  render)
    [ -n "$rang" ] || { echo "need <a,b>"; exit 1; }
    sbatch scripts/attack_original.sub "$RUNG_CSV" "$IMAGE_ROOT" "$RUNG_IMAGES" 1 "$rang" ;;
  check)
    $PY - "$RUNG_CSV" "$RUNG_IMAGES" <<'PYEOF'
import sys, os, pandas as pd
csv, root = sys.argv[1], sys.argv[2]
d = pd.read_csv(csv, low_memory=False)
missing = [p for p in d['Attack Image Path'] if not os.path.exists(os.path.join(root, p))]
print(f"{len(d)} rows, {d['Task ID'].nunique()} tasks, {len(missing)} missing images")
for p in missing[:10]: print("  missing:", p)
sys.exit(1 if missing else 0)
PYEOF
    ;;
  plans)
    [ -n "$rang" ] || { echo "need <a,b>"; exit 1; }
    $PY src/plans_generator.py --input_csv_path "$RUNG_CSV" --output_csv_path "$RUNG_CSV" \
      --image_root "$RUNG_IMAGES" --is_attack --system_prompt_name robot_constitution_64_automerge --rang "$rang" ;;
  judge)
    [ -n "$rang" ] || { echo "need <a,b>"; exit 1; }
    $PY src/automate_judge_new.py --input_csv_path "$RUNG_CSV" --model gemini-2.5-flash --workers 10 --rang "$rang" ;;
  status)
    $PY - "$RUNG_CSV" <<'PYEOF'
import sys, pandas as pd
d = pd.read_csv(sys.argv[1], low_memory=False)
j = pd.to_numeric(d['Judgement'], errors='coerce')
print(f"{len(d)} rows; Response filled: {d['Response'].fillna('').astype(str).str.len().gt(0).sum()}; "
      f"Judgement counts: {j.value_counts(dropna=False).to_dict()}")
PYEOF
    ;;
  *) echo "unknown command: $cmd"; exit 1 ;;
esac
