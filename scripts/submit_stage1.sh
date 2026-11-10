#!/bin/bash
# FILENAME: submit_stage1.sh
# Stage 1: propose the added objects for one target (N sets x K objects per task) and write its CSV.
#   targeted      RQ1 red-team proposer (image + instruction) on every task the planner TARGET refuses on the clean
#                 image (EARDataset/dataset_baseline_<TARGET>.csv, Judgement==0): ONE GPU job via scripts/attack_stage_1.sub
#                 -> EARDataset/robot_constitution_64_automerge_<TARGET>_attack_20.csv. TARGET= defaults to er2 and is
#                 set by scripts/run_rq1.sh; 'targeted_er2' is the same as TARGET=er2 targeted.
#   scene_only    RQ2 (ER-1.6 sample): Qwen proposer sees the image only, 200 stratified tasks: ONE GPU job
#   random        RQ2 (ER-1.6 sample): no VLM, draws from RANDOM_OBJECT_LIST, seconds -- FINE TO RUN ON THE LOGIN NODE
#   scene_only_<planner>, random_<planner>
#                 the same two rungs on a planner's stratified tasks (EARDataset/stratified_baseline_<planner>.csv,
#                 analysis/rq2/stratified_sample.py --target <planner>; ER-2: 194 tasks)
#                 -> EARDataset/stratified_<rung>_<planner>.csv, e.g. scene_only_er2, random_gpt54
#   null_edit_<planner>
#                 RQ2 null-edit control on the same stratified tasks: no VLM, 20 empty object sets per task, so the
#                 editor re-renders the clean image with nothing added -- FINE TO RUN ON THE LOGIN NODE
#   (attack_stage1.py has no --rang, so this stage is not sharded)
#
# Run from repo root:
#   bash scripts/submit_stage1.sh targeted_er2
#   bash scripts/submit_stage1.sh scene_only_er2
#   bash scripts/submit_stage1.sh random_er2
#   bash scripts/submit_stage1.sh null_edit_er2
#   QOS=normal TIME=12:00:00 bash scripts/submit_stage1.sh scene_only   # override qos/wall time/mem
#   N=50 bash scripts/submit_stage1.sh targeted_er2                     # override objects per task
#   FORCE=1 bash scripts/submit_stage1.sh random                        # overwrite an existing output CSV
#   PYTHON="uv run python" bash scripts/submit_stage1.sh random         # interpreter for the local target
set -euo pipefail

PLANNER="${TARGET:-er2}"   # the planner whose baseline is the filter (env TARGET); the positional below is the rung
TARGET="${1:-}"
IMAGE_ROOT="EARDataset/images"
case "$TARGET" in
    targeted_er2|targeted)
        # stage 1 keeps the Judgement==0 tasks of EVAL_CSV: the tasks the planner refused on the clean image (ER-2: 846).
        [ "$TARGET" = "targeted_er2" ] && PLANNER="er2"
        INPUT_CSV="EARDataset/dataset_baseline_${PLANNER}.csv"
        EVAL_CSV="EARDataset/dataset_baseline_${PLANNER}.csv"
        OUT_CSV="EARDataset/robot_constitution_64_automerge_${PLANNER}_attack_20.csv"
        PROMPT="image_and_task_instruction"
        DEFAULT_N=20
        # no resume: a preempted standby job loses every proposal, so default to the .sub's normal/12h.
        DEFAULT_QOS="normal"; DEFAULT_TIME="12:00:00" ;;
    scene_only|random)
        INPUT_CSV="EARDataset/stratified_baseline.csv"   # the 200 tasks (its Judgement is the re-query; NOT the filter)
        EVAL_CSV="EARDataset/dataset_realistic.csv"      # clean-run judgements: all 200 tasks are Judgement==0 here
        OUT_CSV="EARDataset/stratified_${TARGET}.csv"
        if [ "$TARGET" = "random" ]; then PROMPT="random_list"; else PROMPT="scene_only"; fi
        DEFAULT_N=5
        DEFAULT_QOS="standby"; DEFAULT_TIME="4:00:00" ;;
    scene_only_*|random_*)
        PLANNER="${TARGET##*_}"                                      # the rung's suffix names the planner
        INPUT_CSV="EARDataset/stratified_baseline_${PLANNER}.csv"   # the sampled tasks (its Judgement is the re-query; NOT the filter)
        EVAL_CSV="EARDataset/dataset_baseline_${PLANNER}.csv"       # first clean query: every sampled task is Judgement==0 here
        OUT_CSV="EARDataset/stratified_${TARGET}.csv"
        if [ "${TARGET%_*}" = "random" ]; then PROMPT="random_list"; else PROMPT="scene_only"; fi
        DEFAULT_N=5
        DEFAULT_QOS="standby"; DEFAULT_TIME="4:00:00" ;;
    null_edit_*)
        PLANNER="${TARGET##*_}"                                      # the rung's suffix names the planner
        INPUT_CSV="EARDataset/stratified_baseline_${PLANNER}.csv"   # the sampled tasks, as for the planner's other rungs
        EVAL_CSV="EARDataset/dataset_baseline_${PLANNER}.csv"
        OUT_CSV="EARDataset/stratified_${TARGET}.csv"
        PROMPT="null_edit"
        DEFAULT_N=20                                                 # renders per task, matching the rendered rungs
        DEFAULT_QOS="standby"; DEFAULT_TIME="4:00:00" ;;
    *)
        awk 'NR == 1 { next } /^set -euo pipefail/ { exit } { print }' "$0"; exit 1 ;;
esac
[ "$TARGET" = "targeted" ] && TARGET="targeted_${PLANNER}"   # job and log names carry the planner

QOS="${QOS:-$DEFAULT_QOS}"
MEM="${MEM:-256G}"
TIME="${TIME:-$DEFAULT_TIME}"
PYTHON="${PYTHON:-python}"
PROPOSER="${PROPOSER:-Qwen/Qwen3-VL-30B-A3B-Thinking-FP8}"
N="${N:-$DEFAULT_N}"
K="${K:-1}"

# --- Preconditions -----------------------------------------------------------
for f in "$INPUT_CSV" "$EVAL_CSV" src/attack_stage1.py src/system_prompts_2.py scripts/attack_stage_1.sub; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found." >&2
        exit 1
    fi
done
# An older attack_stage1.py accepts --system_prompt_name but ignores it and runs the RQ1
# red-team prompt, so a stale copy on the cluster fails silently. Refuse to run on one.
if ! grep -q "scene_only_object_user" src/attack_stage1.py || ! grep -q "null_edit" src/attack_stage1.py \
   || ! grep -q "RANDOM_OBJECT_LIST" src/system_prompts_2.py || ! grep -q "system_prompt_name" scripts/attack_stage_1.sub; then
    echo "ERROR: src/attack_stage1.py, src/system_prompts_2.py or scripts/attack_stage_1.sub predates RQ2 -- copy the current ones over." >&2
    exit 1
fi
if [ -f "$OUT_CSV" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "ERROR: $OUT_CSV already exists (it may already hold renders' paths, responses and judgements)." >&2
    echo "       Move it aside, or rerun with FORCE=1 to overwrite it." >&2
    exit 1
fi

echo "target: $TARGET | prompt: $PROMPT | n=$N k=$K | input: $INPUT_CSV | filter: $EVAL_CSV | output: $OUT_CSV"

if [ "$PROMPT" = "random_list" ] || [ "$PROMPT" = "null_edit" ]; then
    $PYTHON src/attack_stage1.py \
        --input_csv_path "$INPUT_CSV" \
        --baseline_evaluation_path "$EVAL_CSV" \
        --output_csv_path "$OUT_CSV" \
        --image_root "$IMAGE_ROOT" \
        --n "$N" \
        --k "$K" \
        --system_prompt_name "$PROMPT"
    exit 0
fi

jid=$(sbatch --parsable \
             --qos="$QOS" --mem="$MEM" --time="$TIME" \
             -J "stage1_${TARGET}" \
             -o "stage1_${TARGET}_%j.out" \
             scripts/attack_stage_1.sub \
             "$INPUT_CSV" "$EVAL_CSV" "$OUT_CSV" "$IMAGE_ROOT" "$PROPOSER" "$N" "$K" "$PROMPT")
echo "stage 1 ($TARGET, qos=$QOS mem=$MEM time=$TIME, model=$PROPOSER) -> job $jid"
echo "When it finishes, check:"
if [ "$PROMPT" = "scene_only" ]; then
    echo "  grep -A2 'First user prompt' stage1_${TARGET}_${jid}.out    # expect: Analyze the provided image."
    echo "  grep 'Length of filtered input DataFrame' stage1_${TARGET}_${jid}.out    # expect the task count of $INPUT_CSV (ER-2: 194)"
    echo "Then render: bash scripts/submit_render.sh $TARGET"
else
    echo "  grep -A2 'First user prompt' stage1_${TARGET}_${jid}.out    # expect: Analyze the provided image with the following task instruction: ..."
    echo "  grep 'Length of filtered input DataFrame' stage1_${TARGET}_${jid}.out    # expect the Judgement==0 count of $EVAL_CSV (ER-2: 846)"
    echo "  $PYTHON -c \"import pandas as pd; d=pd.read_csv('$OUT_CSV'); print(len(d), d['Task ID'].nunique())\"    # expect <= $N rows per refused task, one Task ID each"
    echo "Then render: bash scripts/submit_render.sh $TARGET   (src/attack_original.py writes the images and 'Attack Image Path')"
fi
