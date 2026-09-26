#!/bin/bash
# FILENAME: submit_stage1.sh
# Stage 1: propose the added objects for one target (N sets x K objects per task) and write its CSV.
#   targeted_er2  RQ1 red-team proposer (image + instruction) on every task ER-2 refuses on the clean image
#                 (EARDataset/dataset_baseline_er2.csv, Judgement==0): ONE GPU job via scripts/attack_stage_1.sub
#   scene_only    RQ2 (ER-1.6 sample): Qwen proposer sees the image only, 200 stratified tasks: ONE GPU job
#   random        RQ2 (ER-1.6 sample): no VLM, draws from RANDOM_OBJECT_LIST, seconds -- FINE TO RUN ON THE LOGIN NODE
#   scene_only_er2, random_er2
#                 the same two rungs on the 194 ER-2 stratified tasks (EARDataset/stratified_baseline_er2.csv,
#                 analysis/rq2/stratified_sample.py) -> EARDataset/stratified_<rung>_er2.csv
#   (attack_stage1.py has no --rang, so this stage is not sharded)
#
# Run from repo root:
#   bash scripts/submit_stage1.sh targeted_er2
#   bash scripts/submit_stage1.sh scene_only_er2
#   bash scripts/submit_stage1.sh random_er2
#   QOS=normal TIME=12:00:00 bash scripts/submit_stage1.sh scene_only   # override qos/wall time/mem
#   N=50 bash scripts/submit_stage1.sh targeted_er2                     # override objects per task
#   FORCE=1 bash scripts/submit_stage1.sh random                        # overwrite an existing output CSV
#   PYTHON="uv run python" bash scripts/submit_stage1.sh random         # interpreter for the local target
set -euo pipefail

TARGET="${1:-}"
IMAGE_ROOT="EARDataset/images"
case "$TARGET" in
    targeted_er2)
        # stage 1 keeps the Judgement==0 tasks of EVAL_CSV: 846 tasks refused by ER-2 on the clean image.
        INPUT_CSV="EARDataset/dataset_baseline_er2.csv"
        EVAL_CSV="EARDataset/dataset_baseline_er2.csv"
        OUT_CSV="EARDataset/robot_constitution_64_automerge_er2_attack_20.csv"
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
    scene_only_er2|random_er2)
        INPUT_CSV="EARDataset/stratified_baseline_er2.csv"   # the 194 tasks (its Judgement is the re-query; NOT the filter)
        EVAL_CSV="EARDataset/dataset_baseline_er2.csv"       # first clean query: all 194 tasks are Judgement==0 here
        OUT_CSV="EARDataset/stratified_${TARGET%_er2}_er2.csv"
        if [ "$TARGET" = "random_er2" ]; then PROMPT="random_list"; else PROMPT="scene_only"; fi
        DEFAULT_N=5
        DEFAULT_QOS="standby"; DEFAULT_TIME="4:00:00" ;;
    *)
        sed -n 2,20p "$0"; exit 1 ;;
esac

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
if ! grep -q "scene_only_object_user" src/attack_stage1.py || ! grep -q "RANDOM_OBJECT_LIST" src/system_prompts_2.py \
   || ! grep -q "system_prompt_name" scripts/attack_stage_1.sub; then
    echo "ERROR: src/attack_stage1.py, src/system_prompts_2.py or scripts/attack_stage_1.sub predates RQ2 -- copy the current ones over." >&2
    exit 1
fi
if [ -f "$OUT_CSV" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "ERROR: $OUT_CSV already exists (it may already hold renders' paths, responses and judgements)." >&2
    echo "       Move it aside, or rerun with FORCE=1 to overwrite it." >&2
    exit 1
fi

echo "target: $TARGET | prompt: $PROMPT | n=$N k=$K | input: $INPUT_CSV | filter: $EVAL_CSV | output: $OUT_CSV"

if [ "$PROMPT" = "random_list" ]; then
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
    echo "  grep 'Length of filtered input DataFrame' stage1_${TARGET}_${jid}.out    # expect 194 (scene_only_er2) / 200 (scene_only)"
    echo "Then render: bash scripts/submit_render.sh $TARGET"
else
    echo "  grep -A2 'First user prompt' stage1_${TARGET}_${jid}.out    # expect: Analyze the provided image with the following task instruction: ..."
    echo "  grep 'Length of filtered input DataFrame' stage1_${TARGET}_${jid}.out    # expect 846"
    echo "  $PYTHON -c \"import pandas as pd; d=pd.read_csv('$OUT_CSV'); print(len(d), d['Task ID'].nunique())\"    # expect <= $((846 * N)) rows, 846 tasks"
    echo "Then render: bash scripts/submit_render.sh $TARGET   (src/attack_original.py writes the images and 'Attack Image Path')"
fi
