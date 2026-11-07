#!/bin/bash
# FILENAME: submit_render.sh
# Render: submit N INDEPENDENT attack_original.sub jobs (no SLURM array). Each job = one GPU + one --rang
# slice of the target CSV; Flux.2-klein adds that row's object to the task image and writes
# <OUT_ROOT>/{Task ID}/{Object ID}/final_0.png. Before rendering, each job overwrites "Attack Image Path"
# for its slice with that relative path (src/attack_original.py, filelock on the shared CSV).
#   targeted_er2  EARDataset/robot_constitution_64_automerge_er2_attack_20.csv -> EARDataset/robot_constitution_64_automerge_er2_attacked_20
#   random        EARDataset/stratified_random.csv     -> EARDataset/stratified_random_images
#   scene_only    EARDataset/stratified_scene_only.csv -> EARDataset/stratified_scene_only_images
#   random_er2    EARDataset/stratified_random_er2.csv     -> EARDataset/stratified_random_er2_images
#   scene_only_er2 EARDataset/stratified_scene_only_er2.csv -> EARDataset/stratified_scene_only_er2_images
#   null_edit_er2  EARDataset/stratified_null_edit_er2.csv  -> EARDataset/stratified_null_edit_er2_images
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

TARGET="${1:-}"
case "$TARGET" in
    targeted_er2)
        CSV="EARDataset/robot_constitution_64_automerge_er2_attack_20.csv"
        OUT_ROOT="EARDataset/robot_constitution_64_automerge_er2_attacked_20"
        # ~16.8k rows. RQ2 rung shards rendered 250 rows in ~16 min (<=3.9 s/image incl. model load), so 10 shards
        # is ~1,680 rows / ~1.8 h per job, well inside the 4 h standby limit.
        DEFAULT_NSHARDS=10 ;;
    random|scene_only)
        CSV="EARDataset/stratified_${TARGET}.csv"
        OUT_ROOT="EARDataset/stratified_${TARGET}_images"
        DEFAULT_NSHARDS=4 ;;
    random_er2|scene_only_er2|null_edit_er2)
        # RQ2 rungs on the 194 ER-2 stratified tasks (scripts/submit_stage1.sh <rung>_er2).
        CSV="EARDataset/stratified_${TARGET%_er2}_er2.csv"
        OUT_ROOT="EARDataset/stratified_${TARGET%_er2}_er2_images"
        DEFAULT_NSHARDS=4 ;;
    *)
        sed -n 2,19p "$0"; exit 1 ;;
esac
shift

NSHARDS="${NSHARDS:-$DEFAULT_NSHARDS}"
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
echo "Rows: $total | target: $TARGET | out: $OUT_ROOT | jobs: ${#ranges[@]} | qos=$QOS mem=$MEM time=$TIME"

for i in "${!ranges[@]}"; do
    r="${ranges[$i]}"
    jid=$(sbatch --parsable \
                 --qos="$QOS" --mem="$MEM" --time="$TIME" \
                 -J "render_${TARGET}_${i}" \
                 -o "render_${TARGET}_${i}_%j.out" \
                 scripts/attack_original.sub "$CSV" "$IMAGE_ROOT" "$OUT_ROOT" "$NUM_SAMPLES" "$r")
    echo "shard $i -> rows [${r}) -> job $jid"
done
