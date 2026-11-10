#!/bin/bash
# FILENAME: submit_plans.sh
# Target queries, run LOCALLY (like automate_judge_new.sh): NSHARDS parallel src/plans_generator.py
# processes, one per --rang slice of a CSV. The target planner plans on every image under the robot
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
#   MODEL=azure_openai/gpt-5.4-1 bash scripts/submit_plans.sh random_gpt54   # an RQ2 rung of another planner
#   bash scripts/submit_plans.sh null_edit_er2                  # null-edit control: same editor, nothing added
#   bash scripts/submit_plans.sh scene_only 8                   # 8 parallel slices
#   bash scripts/submit_plans.sh random --rang 250,500          # one slice (e.g. an interrupted one)
#   bash scripts/submit_plans.sh baseline                       # clean images, EARDataset/dataset_baseline_<TARGET>.csv
#   bash scripts/submit_plans.sh targeted 8                     # RQ1 attacked images of <TARGET>
#   bash scripts/submit_plans.sh requery                        # clean re-query, EARDataset/dataset_requery_<TARGET>.csv
#   CLUSTER=1 bash scripts/submit_plans.sh targeted             # cluster clone: 10 jobs, one slice each (scripts/plan_api.sub)
# Then judge:
#   bash scripts/automate_judge_new.sh --input_csv_path EARDataset/stratified_random.csv
#
# TARGET names the planner whose RQ1 CSVs 'baseline', 'targeted' and 'requery' resolve to (default er2;
# 'targeted_er2' is the same as TARGET=er2 targeted). scripts/run_rq1.sh sets it from its target table.
# MODEL is a {provider}/{model} string; the prefix picks the route and its credentials (see
# src/query_litellm.py): gemini/ azure_openai/ azure_anthropic/ hosted_vllm/ anthropic/ openai/ litellm_proxy/
#   MODEL=azure_anthropic/claude-opus-5-5 bash scripts/submit_plans.sh baseline
# A hosted_vllm/ target is served on the GPU node that runs this script: scripts/run_rq1.sh submits
# scripts/serve_plan.sub, which sets HOSTED_VLLM_API_BASE and PYTHON=python before calling this script.
# CLUSTER=1 (cluster clone) submits every slice as its own job through scripts/plan_api.sub instead of
# starting local processes, then returns: NSHARDS defaults to 10, QOS/MEM/TIME override the job's
# standby / 256G / 4:00:00. Each job runs this script on its single --rang, logs to
# plans_<label>_<start>-<end>.log and checks its own rows; a slice must finish inside the job's time.
set -euo pipefail

RUNG="${1:-}"
TARGET="${TARGET:-er2}"
IMAGE_COL="Attack Image Path"
ATTACK_FLAG=(--is_attack)
case "$RUNG" in
    random|scene_only)
        CSV="EARDataset/stratified_${RUNG}.csv"
        IMAGE_ROOT="EARDataset/stratified_${RUNG}_images" ;;
    random_*|scene_only_*|null_edit_er2)
        # RQ2 rungs on a planner's stratified tasks (submit_stage1.sh / submit_render.sh <rung>_<planner>).
        # The suffix names the CSV only; MODEL and SYSTEM_PROMPT_NAME select the planner that is queried
        # (scripts/run_rq2.sh sets both from TARGET).
        # These CSVs still carry the clean re-query's Response/Thinking until this step overwrites them.
        CSV="EARDataset/stratified_${RUNG}.csv"
        IMAGE_ROOT="EARDataset/stratified_${RUNG}_images" ;;
    targeted_er2|targeted)
        # RQ1 targeted proposer on the tasks TARGET refuses clean, 20 objects each (scripts/submit_stage1.sh targeted).
        # Images and "Attack Image Path" come from scripts/submit_render.sh targeted (src/attack_original.py).
        [ "$RUNG" = "targeted_er2" ] && TARGET="er2"
        CSV="EARDataset/robot_constitution_64_automerge_${TARGET}_attack_20.csv"
        IMAGE_ROOT="EARDataset/robot_constitution_64_automerge_${TARGET}_attacked_20" ;;
    baseline)
        # no object: plan on the unedited image of every realistic task in dataset.csv.
        CSV="EARDataset/dataset_baseline_${TARGET}.csv"
        IMAGE_ROOT="EARDataset/images"
        IMAGE_COL="Matched Image Path"
        ATTACK_FLAG=() ;;
    requery)
        # RQ1 non-determinism control: a second clean query of the tasks TARGET refused in its baseline CSV.
        # Its own CSV, so the first clean query in the baseline CSV is never overwritten.
        CSV="EARDataset/dataset_requery_${TARGET}.csv"
        IMAGE_ROOT="EARDataset/images"
        IMAGE_COL="Matched Image Path"
        ATTACK_FLAG=() ;;
    *)
        awk 'NR == 1 { next } /^set -euo pipefail/ { exit } { print }' "$0"; exit 1 ;;
esac
shift

# CLUSTER=1 submits each slice as its own job (scripts/plan_api.sub) instead of a local process; 10 by default.
CLUSTER="${CLUSTER:-0}"
if [ "$CLUSTER" = "1" ]; then NSHARDS="${NSHARDS:-10}"; else NSHARDS="${NSHARDS:-4}"; fi
RANG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --rang) RANG="$2"; shift 2 ;;
        *) NSHARDS="$1"; shift ;;
    esac
done
LABEL="$RUNG"
[ "$RUNG" = "targeted" ] && LABEL="targeted_${TARGET}"

SYSTEM_PROMPT_NAME="${SYSTEM_PROMPT_NAME:-robot_constitution_64_automerge}"
# gemini-robotics-er-1.6-preview (RQ1's target) was removed from the Gemini API; new runs use ER-2.
MODEL="${MODEL:-gemini/gemini-robotics-er-2-preview}"
# Interpreter for the planner and the row checks; the GPU job passes its conda env's python.
PYTHON="${PYTHON:-uv run python}"
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
# Credentials are read by LiteLLM from the environment or the repo-root .env, named by provider.
need_env() {
    if [ -z "${!1:-}" ] && ! grep -qE "^$1=" .env 2>/dev/null; then
        echo "ERROR: $1 is not set in this shell or in .env (required for MODEL=$MODEL)." >&2
        exit 1
    fi
}
case "$MODEL" in
    gemini/*)          need_env GEMINI_API_KEY ;;
    azure_openai/*)    need_env AZURE_OPENAI_API_BASE;    need_env AZURE_API_KEY ;;
    azure_anthropic/*) need_env AZURE_ANTHROPIC_API_BASE; need_env AZURE_API_KEY ;;
    hosted_vllm/*)     need_env HOSTED_VLLM_API_BASE ;;
    anthropic/*)       need_env ANTHROPIC_API_KEY ;;
    openai/*)          need_env OPENAI_API_KEY ;;
    litellm_proxy/*)   need_env LITELLM_PROXY_API_BASE ;;
    *)
        echo "ERROR: MODEL must carry a route prefix (gemini/ azure_openai/ azure_anthropic/ hosted_vllm/ anthropic/ openai/ litellm_proxy/): $MODEL" >&2
        exit 1 ;;
esac

total=$($PYTHON -c "import pandas as pd; print(len(pd.read_csv('$CSV', low_memory=False)))")

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
    missing=$($PYTHON - "$CSV" "$IMAGE_ROOT" "$r" "$IMAGE_COL" <<'PYEOF'
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
    if [ "$CLUSTER" = "1" ]; then
        jid=$(sbatch --parsable --qos="${QOS:-standby}" --mem="${MEM:-256G}" --time="${TIME:-4:00:00}" \
                     -J "plan_${LABEL}_${i}" -o "plan_${LABEL}_${i}_%j.out" scripts/plan_api.sub "$RUNG" --rang "$r")
        echo "shard $i -> rows [${r}) -> job $jid -> plan_${LABEL}_${i}_${jid}.out"
        continue
    fi
    log="plans_${LABEL}_${i}.log"
    # A single explicit slice is logged under its row range, so concurrent slices never share a log.
    [ -n "$RANG" ] && log="plans_${LABEL}_${RANG/,/-}.log"
    # --rang is required: without it plans_generator.py queries every row but writes nothing back.
    $PYTHON src/plans_generator.py \
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

if [ "$CLUSTER" = "1" ]; then
    echo ""
    echo "${#ranges[@]} job(s) submitted. Each job checks its own rows when it ends; judge once every job has finished."
    exit $failed
fi

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
# A single explicit slice checks its own rows only.
if ! $PYTHON - "$CSV" "$RANG" <<'PYEOF'
import sys
import pandas as pd
df = pd.read_csv(sys.argv[1], low_memory=False)
rang = sys.argv[2]
if rang:
    start, end = map(int, rang.split(","))
    df = df.iloc[start:end]
empty = df["Response"].fillna("").astype(str).str.strip().eq("")
print(f"Empty Response rows: {int(empty.sum())}/{len(df)}" + (f" in [{rang})" if rang else ""))
if empty.any():
    print(f"  rows: {empty[empty].index.tolist()[:20]} -- rerun the slices containing them with --rang before judging")
sys.exit(1 if empty.any() else 0)
PYEOF
then
    failed=1
fi

exit $failed
