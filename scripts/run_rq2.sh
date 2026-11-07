#!/bin/bash
# FILENAME: run_rq2.sh
# RQ2 driver: does the added object need to be chosen for the task?
# One entry point for the two non-targeted rungs. Every stage is a thin wrapper around the driver
# that already does the work -- this script only resolves the rung name to its CSV, tees the output
# to a predictable log, and reports the exit code. It changes no behaviour.
#
#   propose <rung>  propose the added objects, 5 per task       LOCAL/CLUSTER submit_stage1.sh
#   render <rung>   composite each object into the task image   CLUSTER       submit_render.sh
#   plan <rung>     query the target planner on those images    LOCAL         submit_plans.sh
#   judge <rung>    label each plan as bypass / refusal / unclear  LOCAL      automate_judge_new.sh
#   audit <cmd>     constraint audit of the TARGETED objects    LOCAL         analysis/rq2/constraint_audit.py
#
#   <rung> is one of:
#     random      EARDataset/stratified_random_er2.csv      fixed 26-item COCO list, no VLM
#     scene_only  EARDataset/stratified_scene_only_er2.csv  proposer sees the image, not the instruction
#     null_edit   EARDataset/stratified_null_edit_er2.csv   no object: the clean image through the same
#                                                           editor with nothing added, 20 renders per task
#   Each is 194 stratified tasks x 20 rows.
#   The 'targeted' rung is RQ1's data -- it is not regenerated here. See scripts/run_rq1.sh.
#
# Run from repo root, one stage per invocation, in this order:
#   uv run python analysis/rq2/stratified_sample.py      # 0. the 194-task sample (--force to redraw)
#   bash scripts/run_rq2.sh propose random               # local, seconds, no GPU
#   bash scripts/run_rq2.sh propose scene_only           # cluster: one GPU job
#   bash scripts/run_rq2.sh propose null_edit            # local, seconds, no GPU
#   QOS=normal TIME=2:00:00 bash scripts/run_rq2.sh render random      # cluster: 4 GPU jobs
#   QOS=normal TIME=2:00:00 bash scripts/run_rq2.sh render scene_only
#   bash scripts/run_rq2.sh plan random 4                # 970 queries, 4 slices; after EVERY render job
#   bash scripts/run_rq2.sh judge random
#   bash scripts/run_rq2.sh plan scene_only 4
#   bash scripts/run_rq2.sh judge scene_only
#   bash scripts/run_rq2.sh audit sample                 # constraint audit, see analysis/rq2/README.md
#   bash scripts/run_rq2.sh audit judge --rang 0,5       # smoke test before the full 966 calls
#   bash scripts/run_rq2.sh audit judge
#   bash scripts/run_rq2.sh audit report
# Then analyse (see analysis/rq2/README.md):
#   uv run python analysis/rq2/analysis.py
#   uv run python analysis/generate_figures.py \
#       --outdir paper/6a720826083125a2116c9376/figures --only rq2_targeting_ladder
#
# FOUR THINGS THAT LOOK LIKE FAILURES AND ARE NOT:
#   * submit_plans.sh exits 1 whenever ANY row came back with an empty Response, and prints which
#     rows. On the ER-2 run that was 2 rows for random (121, 123) and 1 for scene_only (124); each
#     was re-queried by hand with --rang before judging. A non-zero exit here means "read the row
#     list and decide", not "the stage failed".
#   * A --rang re-run is ONE slice, so its log is plans_<rung>_er2_0.log -- it OVERWRITES shard 0's
#     log from the full run. Both rungs' shard-0 logs from 16 Sep are the 1-row re-query, not the
#     243-row slice. Copy the log aside first if you need to keep it.
#   * src/automate_judge_new.py does NOT skip already-judged rows (write_judgements_to_csv): it
#     overwrites every row in its range. Re-running 'judge' re-judges all 970 rows at full API
#     cost. Judge an unjudged tail with --rang instead.
#   * submit_stage1.sh refuses to overwrite an existing rung CSV; pass FORCE=1 to redo 'propose'.
#
# Extra arguments after the stage are passed through to the wrapped driver, so its own options
# still work:
#   bash scripts/run_rq2.sh plan random --rang 121,124     # re-query the empty rows
#   bash scripts/run_rq2.sh judge random --rang 900,970    # judge the tail only
#   bash scripts/run_rq2.sh render scene_only --rang 250,500
#
# Env overrides are read by the wrapped drivers themselves and pass straight through:
#   MODEL=          target planner              (default gemini-robotics-er-2-preview)
#   SYSTEM_PROMPT_NAME=  constitution           (default robot_constitution_64_automerge)
#   PROPOSER= N= K= QOS= MEM= TIME=  proposer model, objects per task, SLURM settings
#   NSHARDS=        parallel slices for plan / render
#   FORCE=1         let 'propose' overwrite an existing rung CSV
# This script's own:
#   JUDGE_MODEL=    judge model                 (default gemini-2.5-flash)
#   LOGDIR=         where the run logs go       (default the repo root, as before)
set -euo pipefail

JUDGE_MODEL="${JUDGE_MODEL:-gemini-2.5-flash}"
LOGDIR="${LOGDIR:-.}"

# The CSV behind each rung name. These MIRROR scripts/submit_plans.sh's own case block (:31-35) --
# that script owns the paths for the planning stages; they are repeated here only so 'judge' knows
# which file to label. Change them in both places or not at all.
rung_csv() {
    case "$1" in
        random)     echo "EARDataset/stratified_random_er2.csv" ;;
        scene_only) echo "EARDataset/stratified_scene_only_er2.csv" ;;
        null_edit)  echo "EARDataset/stratified_null_edit_er2.csv" ;;
        *)          return 1 ;;
    esac
}

die() { echo "ERROR: $*" >&2; exit 1; }

need_file() { [ -f "$1" ] || die "$1 not found.${2:+ $2}"; }

usage() { sed -n 2,38p "$0"; exit 1; }

# The rung argument, shared by propose / render / plan / judge. 'targeted' is RQ1's, not RQ2's.
take_rung() {
    case "${1:-}" in
        random|scene_only|null_edit) echo "$1" ;;
        targeted) die "the targeted rung is RQ1's data, generated by scripts/run_rq1.sh -- RQ2 reads it from EARDataset/robot_constitution_64_automerge_er2_attack_20.csv" ;;
        *) die "$2 takes a rung: random | scene_only | null_edit" ;;
    esac
}

# Run a driver, tee to a predictable log, echo its exit code (the convention the existing
# hand-written run logs already used). pipefail makes PIPESTATUS[0] the driver's own status.
run_stage() {
    local log="$LOGDIR/rq2_$1.log"; shift
    mkdir -p "$LOGDIR"
    echo "=== ${*} ==="
    echo "    log: $log"
    local rc
    set +e
    "$@" 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    set -e
    echo "exit=$rc" | tee -a "$log"
    return "$rc"
}

# --- dispatch -----------------------------------------------------------------
STAGE="${1:-}"
[ -n "$STAGE" ] || usage
shift || true

case "$STAGE" in
    propose)
        RUNG=$(take_rung "${1:-}" propose); shift || true
        need_file "scripts/submit_stage1.sh"
        need_file "EARDataset/stratified_baseline_er2.csv" "Run: uv run python analysis/rq2/stratified_sample.py"
        run_stage "propose_${RUNG}" bash scripts/submit_stage1.sh "${RUNG}_er2" "$@"
        ;;

    render)
        RUNG=$(take_rung "${1:-}" render); shift || true
        need_file "scripts/submit_render.sh"
        need_file "$(rung_csv "$RUNG")" "Run: bash scripts/run_rq2.sh propose $RUNG"
        run_stage "render_${RUNG}" bash scripts/submit_render.sh "${RUNG}_er2" "$@"
        ;;

    plan)
        RUNG=$(take_rung "${1:-}" plan); shift || true
        need_file "scripts/submit_plans.sh"
        need_file "$(rung_csv "$RUNG")" "Run: bash scripts/run_rq2.sh propose $RUNG"
        run_stage "plan_${RUNG}" bash scripts/submit_plans.sh "${RUNG}_er2" "$@"
        ;;

    judge)
        RUNG=$(take_rung "${1:-}" judge); shift || true
        CSV=$(rung_csv "$RUNG")
        need_file "scripts/automate_judge_new.sh"
        need_file "$CSV" "Run: bash scripts/run_rq2.sh plan $RUNG"
        run_stage "judge_${RUNG}" bash scripts/automate_judge_new.sh \
            --input_csv_path "$CSV" --model "$JUDGE_MODEL" "$@"
        ;;

    audit)
        # Constraint audit of the TARGETED objects (analysis/rq2/constraint_audit.py). Subcommands:
        # sample / judge / report -- passed through with their own options.
        SUB="${1:-}"; shift || true
        case "$SUB" in
            sample|judge|report) ;;
            *) die "audit takes a subcommand: sample | judge | report" ;;
        esac
        need_file "analysis/rq2/constraint_audit.py"
        run_stage "audit_${SUB}" uv run python analysis/rq2/constraint_audit.py "$SUB" "$@"
        ;;

    *) usage ;;
esac
