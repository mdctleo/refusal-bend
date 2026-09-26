#!/bin/bash
# FILENAME: submit_plans.sh
# Target queries, run LOCALLY (like automate_judge_new.sh): NSHARDS parallel src/plans_generator.py
# processes, one per --rang slice of a CSV. Gemini-Robotics-ER plans on every image under the robot
# constitution and writes Response/Thinking back into the same CSV via filelock. Blocks until every
# slice finishes. See analysis/rq2/README.md.
#
# plans_generator.py has NO resume: a process re-queries its whole slice and writes only at the end. Never
# rerun a slice that finished. If any image in a slice is missing, it skips that row, fails its length assert
# and discards the whole slice's responses -- so slices with missing images are NOT started here.
#
# Run from repo root:
#   bash scripts/submit_plans.sh random                         # 4 parallel slices over all rows
#   bash scripts/submit_plans.sh random_er2                     # RQ2 rung on the 194 ER-2 stratified tasks
#   bash scripts/submit_plans.sh scene_only_er2 4               # the other ER-2 rung
#   bash scripts/submit_plans.sh scene_only 8                   # 8 parallel slices
#   bash scripts/submit_plans.sh random --rang 250,500          # one slice (e.g. an interrupted one)
#   bash scripts/submit_plans.sh baseline                       # clean images, EARDataset/dataset_baseline_er2.csv
#   bash scripts/submit_plans.sh requery                        # clean re-query, EARDataset/dataset_requery_er2.csv
# Then judge:
#   bash scripts/automate_judge_new.sh --input_csv_path EARDataset/stratified_random.csv
set -euo pipefail

RUNG="${1:-}"
IMAGE_COL="Attack Image Path"
ATTACK_FLAG=(--is_attack)
case "$RUNG" in
    random|scene_only)
        CSV="EARDataset/stratified_${RUNG}.csv"
        IMAGE_ROOT="EARDataset/stratified_${RUNG}_images" ;;
    random_er2|scene_only_er2)
        # RQ2 rungs on the 194 ER-2 stratified tasks, 5 objects each (submit_stage1.sh / submit_render.sh <rung>_er2).
        # These CSVs still carry the clean re-query's Response/Thinking until this step overwrites them.
        CSV="EARDataset/stratified_${RUNG%_er2}_er2.csv"
        IMAGE_ROOT="EARDataset/stratified_${RUNG%_er2}_er2_images" ;;
    targeted_er2)
        # RQ1 targeted proposer on the 846 tasks ER-2 refuses clean, 20 objects each (scripts/submit_stage1.sh targeted_er2).
        # Images and "Attack Image Path" come from scripts/submit_render.sh targeted_er2 (src/attack_original.py).
        CSV="EARDataset/robot_constitution_64_automerge_er2_attack_20.csv"
        IMAGE_ROOT="EARDataset/robot_constitution_64_automerge_er2_attacked_20" ;;
    baseline)
        # no object: plan on the unedited image of every task in dataset_realistic.csv.
        CSV="EARDataset/dataset_baseline_er2.csv"
        IMAGE_ROOT="EARDataset/images"
        IMAGE_COL="Matched Image Path"
        ATTACK_FLAG=() ;;
    requery)
        # RQ1 non-determinism control: a second clean query of the 846 tasks ER-2 refused in dataset_baseline_er2.csv.
        # Its own CSV, so the first clean query in dataset_baseline_er2.csv is never overwritten.
        CSV="EARDataset/dataset_requery_er2.csv"
        IMAGE_ROOT="EARDataset/images"
        IMAGE_COL="Matched Image Path"
        ATTACK_FLAG=() ;;
    *)
        sed -n 2,21p "$0"; exit 1 ;;
esac
shift

NSHARDS="${NSHARDS:-4}"
RANG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --rang) RANG="$2"; shift 2 ;;
        *) NSHARDS="$1"; shift ;;
    esac
done
LABEL="$RUNG"

SYSTEM_PROMPT_NAME="${SYSTEM_PROMPT_NAME:-robot_constitution_64_automerge}"
# gemini-robotics-er-1.6-preview (RQ1's target) was removed from the Gemini API; new runs use ER-2.
MODEL="${MODEL:-gemini-robotics-er-2-preview}"
# --- Preconditions -----------------------------------------------------------
for f in "$CSV" src/plans_generator.py; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found." >&2
        exit 1
    fi
done
if [ ! -d "$IMAGE_ROOT" ]; then
    echo "ERROR: $IMAGE_ROOT not found." >&2
    exit 1
fi
# plans_generator.py reads the key from the environment only.
if [ -z "${GENAI_API_KEY:-}" ]; then
    echo "ERROR: GENAI_API_KEY is not set in this shell." >&2
    exit 1
fi

total=$(uv run python -c "import pandas as pd; print(len(pd.read_csv('$CSV', low_memory=False)))")

# --- Slices: one explicit --rang, or NSHARDS contiguous row slices -----------
ranges=()
if [ -n "$RANG" ]; then
    ranges+=("$RANG")
else
    chunk=$(( (total + NSHARDS - 1) / NSHARDS ))
    for i in $(seq 0 $((NSHARDS - 1))); do
        start=$(( i * chunk ))
        end=$(( start + chunk ))
        if [ "$start" -ge "$total" ]; then
            echo "shard $i: start $start >= total $total, nothing left -- skipping"
            continue
        fi
        if [ "$end" -gt "$total" ]; then end=$total; fi
        ranges+=("${start},${end}")
    done
fi
echo "=== Plans ($LABEL) ==="
echo "  CSV:     $CSV"
echo "  Images:  $IMAGE_ROOT ($IMAGE_COL)"
echo "  Prompt:  $SYSTEM_PROMPT_NAME"
echo "  Model:   $MODEL"
echo "  Rows:    $total | slices: ${#ranges[@]}"
echo ""

failed=0
pids=()
labels=()
for i in "${!ranges[@]}"; do
    r="${ranges[$i]}"
    missing=$(uv run python - "$CSV" "$IMAGE_ROOT" "$r" "$IMAGE_COL" <<'PYEOF'
import os, sys
import pandas as pd
csv, root, rang, col = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
start, end = map(int, rang.split(","))
paths = pd.read_csv(csv, low_memory=False)[col].iloc[start:end]
print(sum(not os.path.exists(os.path.join(root, p)) for p in paths))
PYEOF
)
    if [ "$missing" -ne 0 ]; then
        echo "shard $i -> rows [${r}) -> NOT STARTED: $missing image(s) missing under $IMAGE_ROOT"
        failed=1
        continue
    fi
    log="plans_${LABEL}_${i}.log"
    # --rang is required: without it plans_generator.py queries every row but writes nothing back.
    uv run python src/plans_generator.py \
        --input_csv_path "$CSV" \
        --output_csv_path "$CSV" \
        --image_root "$IMAGE_ROOT" \
        --system_prompt_name "$SYSTEM_PROMPT_NAME" \
        --model "$MODEL" \
        --rang "$r" \
        "${ATTACK_FLAG[@]}" > "$log" 2>&1 &
    pids+=("$!")
    labels+=("shard $i -> rows [${r}) -> $log")
    echo "shard $i -> rows [${r}) -> pid $! -> $log"
done

echo ""
echo "Waiting for ${#pids[@]} slice(s)..."
for j in "${!pids[@]}"; do
    if wait "${pids[$j]}"; then
        echo "done:   ${labels[$j]}"
    else
        echo "FAILED: ${labels[$j]} (see log; rerun with --rang)"
        failed=1
    fi
done

# Rows whose API calls all failed come back as an empty Response rather than an error: don't judge yet.
if ! uv run python - "$CSV" <<'PYEOF'
import sys
import pandas as pd
df = pd.read_csv(sys.argv[1], low_memory=False)
empty = df["Response"].fillna("").astype(str).str.strip().eq("")
print(f"Empty Response rows: {int(empty.sum())}/{len(df)}")
if empty.any():
    print(f"  rows: {empty[empty].index.tolist()[:20]} -- rerun the slices containing them with --rang before judging")
sys.exit(1 if empty.any() else 0)
PYEOF
then
    failed=1
fi

exit $failed
