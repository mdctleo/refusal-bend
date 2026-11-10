#!/bin/bash
# FILENAME: submit_render.sh
# Render: submit N INDEPENDENT attack_original.sub jobs (no SLURM array). Each job = one GPU + one --rang
# slice of the target CSV; Flux.2-klein adds that row's object to the task image and writes
# <OUT_ROOT>/{Task ID}/{Object ID}/final_0.png. Before rendering, each job overwrites "Attack Image Path"
# for its slice with that relative path (src/attack_original.py, filelock on the shared CSV).
#   targeted      EARDataset/robot_constitution_64_automerge_<TARGET>_attack_20.csv -> EARDataset/robot_constitution_64_automerge_<TARGET>_attacked_20
#                 (TARGET= names the planner, default er2; scripts/run_rq1.sh sets it. 'targeted_er2' is the same as TARGET=er2.)
#   random        EARDataset/stratified_random.csv     -> EARDataset/stratified_random_images
#   scene_only    EARDataset/stratified_scene_only.csv -> EARDataset/stratified_scene_only_images
#   random_<planner>     EARDataset/stratified_random_<planner>.csv     -> EARDataset/stratified_random_<planner>_images
#   scene_only_<planner> EARDataset/stratified_scene_only_<planner>.csv -> EARDataset/stratified_scene_only_<planner>_images
#                  (the RQ2 rungs on a planner's stratified tasks, e.g. random_er2, scene_only_gpt54)
#   null_edit_<planner>  EARDataset/stratified_null_edit_<planner>.csv -> EARDataset/stratified_null_edit_<planner>_images
#                  (empty object set: the clean image through the same editor with nothing added, 20 per task)
# Re-rendering a slice overwrites its images and paths. attack_original.py does not skip existing images.
#
# Run from repo root, AFTER scripts/submit_stage1.sh has written the target CSV:
#   bash scripts/submit_render.sh targeted_er2                     # default shards for the target
#   bash scripts/submit_render.sh random_er2                       # 4 shards (970 rows)
#   bash scripts/submit_render.sh scene_only 8                     # 8 shards
#   bash scripts/submit_render.sh random --rang 250,500            # one slice (e.g. a preempted shard)
#   QOS=normal TIME=8:00:00 bash scripts/submit_render.sh random   # override qos/wall time/mem
set -euo pipefail

PLANNER="${TARGET:-er2}"   # the planner whose attack CSV is rendered (env TARGET); the positional below is the rung
TARGET="${1:-}"
case "$TARGET" in
    targeted_er2|targeted)
        [ "$TARGET" = "targeted_er2" ] && PLANNER="er2"
        CSV="EARDataset/robot_constitution_64_automerge_${PLANNER}_attack_20.csv"
        OUT_ROOT="EARDataset/robot_constitution_64_automerge_${PLANNER}_attacked_20"
        # RQ2 rung shards rendered 250 rows in ~16 min (<=3.9 s/image incl. model load), so ~1,700 rows per job
        # is ~1.8 h, well inside the 4 h standby limit; the default shard count is set from the row count below
        # (10 for ER-2's 16,788 rows).
        DEFAULT_NSHARDS="" ;;
    random|scene_only)
        CSV="EARDataset/stratified_${TARGET}.csv"
        OUT_ROOT="EARDataset/stratified_${TARGET}_images"
        DEFAULT_NSHARDS=4 ;;
    random_*|scene_only_*|null_edit_*)
        # RQ2 rungs on a planner's stratified tasks (scripts/submit_stage1.sh <rung>_<planner>).
        CSV="EARDataset/stratified_${TARGET}.csv"
        OUT_ROOT="EARDataset/stratified_${TARGET}_images"
        DEFAULT_NSHARDS=4 ;;
    *)
        awk 'NR == 1 { next } /^set -euo pipefail/ { exit } { print }' "$0"; exit 1 ;;
esac
shift
LABEL="$TARGET"
[ "$TARGET" = "targeted" ] && LABEL="targeted_${PLANNER}"

NSHARDS="${NSHARDS:-}"
RANG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --rang) RANG="$2"; shift 2 ;;
        *) NSHARDS="$1"; shift ;;
    esac
done

QOS="${QOS:-standby}"
MEM="${MEM:-256G}"
TIME="${TIME:-4:00:00}"
PYTHON="${PYTHON:-python}"
IMAGE_ROOT="EARDataset/images"
NUM_SAMPLES=1   # one image per object: attack_original.py writes Attack Image Path only at num_samples 1

# --- Preconditions -----------------------------------------------------------
for f in "$CSV" src/attack_original.py src/diffusion_2.py scripts/attack_original.sub; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found." >&2
        exit 1
    fi
done
# An older attack_original.py renders but never writes Attack Image Path. Refuse to run on one.
if ! grep -q "write_attack_image_paths" src/attack_original.py; then
    echo "ERROR: src/attack_original.py predates the Attack Image Path write -- copy the current one over." >&2
    exit 1
fi

total=$($PYTHON -c "import pandas as pd; print(len(pd.read_csv('$CSV', low_memory=False)))")
[ -n "$DEFAULT_NSHARDS" ] || DEFAULT_NSHARDS=$(( (total + 1699) / 1700 ))
NSHARDS="${NSHARDS:-$DEFAULT_NSHARDS}"

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
echo "Rows: $total | target: $LABEL | out: $OUT_ROOT | jobs: ${#ranges[@]} | qos=$QOS mem=$MEM time=$TIME"

for i in "${!ranges[@]}"; do
    r="${ranges[$i]}"
    jid=$(sbatch --parsable \
                 --qos="$QOS" --mem="$MEM" --time="$TIME" \
                 -J "render_${LABEL}_${i}" \
                 -o "render_${LABEL}_${i}_%j.out" \
                 scripts/attack_original.sub "$CSV" "$IMAGE_ROOT" "$OUT_ROOT" "$NUM_SAMPLES" "$r")
    echo "shard $i -> rows [${r}) -> job $jid"
done
