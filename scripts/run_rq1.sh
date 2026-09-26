#!/bin/bash
# FILENAME: run_rq1.sh
# RQ1 driver: does adding a coherent object bypass the target's safety constitution?
# One entry point for the generation chain. Every stage is a thin wrapper around the driver
# that already does the work -- this script only resolves the target name to its CSV, tees the
# output to a predictable log, and reports the exit code. It changes no behaviour.
#
#   plan <target>   query the target planner on that CSV's images   LOCAL   submit_plans.sh
#   propose         propose the added objects, 20 per task          CLUSTER submit_stage1.sh
#   render          composite each object into the task image       CLUSTER submit_render.sh
#   judge <target>  label each plan as bypass / refusal / unclear   LOCAL   automate_judge_new.sh
#   requery         second clean query of the refused tasks         LOCAL   submit_plans.sh
#
#   <target> is one of:
#     baseline   EARDataset/dataset_baseline_er2.csv                   (1,121 tasks, clean image)
#     targeted   EARDataset/robot_constitution_64_automerge_er2_attack_20.csv (16,788 rows)
#     requery    EARDataset/dataset_requery_er2.csv                    (846 tasks, clean image)
#   'plan' takes baseline or targeted; the re-query arm has its own stage.
#
# Run from repo root, one stage per invocation, in this order:
#   bash scripts/run_rq1.sh plan baseline        # 1,121 clean queries
#   bash scripts/run_rq1.sh judge baseline       # -> 846 tasks refused (Judgement == 0)
#   bash scripts/run_rq1.sh propose              # cluster: one GPU job
#   bash scripts/run_rq1.sh render               # cluster: 10 GPU jobs
#   bash scripts/run_rq1.sh plan targeted 8      # 16,788 attacked-image queries, 8 slices
#   bash scripts/run_rq1.sh judge targeted
#   bash scripts/run_rq1.sh requery              # the non-determinism control
#   bash scripts/run_rq1.sh judge requery
# Then analyse (see analysis/rq1/README.md):
#   uv run python analysis/rq1/bypass_rates.py --input_csv <targeted csv> --output_json ...
#   uv run python analysis/rq1/requery_lift.py
#
# TWO THINGS THAT LOOK LIKE FAILURES AND ARE NOT:
#   * submit_plans.sh exits 1 whenever ANY row came back with an empty Response, and prints
#     which rows. On ER-2 that is expected: 23 rows across 2 tasks are permanently empty (the
#     model returns nothing for them, on the attacked images AND on the clean re-query), and
#     they are judged -1, which removes them from every rate. A non-zero exit here means "read
#     the row list and decide", not "the stage failed". Re-run a genuinely crashed slice with
#     --rang; leave the permanently-empty rows alone.
#   * src/automate_judge_new.py does NOT skip already-judged rows (write_judgements_to_csv,
#     :204-243): it overwrites every row in its range. Re-running 'judge targeted' re-judges
#     all 16,788 rows at full API cost. Judge an unjudged tail with --rang instead.
#
# Extra arguments after the stage are passed through to the wrapped driver, so its own options
# still work:
#   bash scripts/run_rq1.sh plan targeted --rang 14693,16265   # one slice
#   bash scripts/run_rq1.sh judge targeted --rang 16000,16788  # judge the tail only
#   bash scripts/run_rq1.sh judge baseline --workers 20
#
# Env overrides are read by the wrapped drivers themselves and pass straight through:
#   MODEL=          target planner              (default gemini-robotics-er-2-preview)
#   SYSTEM_PROMPT_NAME=  constitution           (default robot_constitution_64_automerge)
#   PROPOSER= N= QOS= MEM= TIME=  proposer model, objects per task, SLURM settings
#   NSHARDS=        parallel slices for plan / requery / render
#   FORCE=1         let 'propose' overwrite an existing attack CSV
# This script's own:
#   JUDGE_MODEL=    judge model                 (default gemini-2.5-flash)
#   LOGDIR=         where the run logs go       (default the repo root, as before)
set -euo pipefail

JUDGE_MODEL="${JUDGE_MODEL:-gemini-2.5-flash}"
LOGDIR="${LOGDIR:-.}"

# The CSV behind each target name. These MIRROR scripts/submit_plans.sh's own case block
# (:29-53) -- that script owns the paths for the planning stages; they are repeated here only
# so 'judge' knows which file to label. Change them in both places or not at all.
target_csv() {
    case "$1" in
        baseline) echo "EARDataset/dataset_baseline_er2.csv" ;;
        targeted) echo "EARDataset/robot_constitution_64_automerge_er2_attack_20.csv" ;;
        requery)  echo "EARDataset/dataset_requery_er2.csv" ;;
        *)        return 1 ;;
    esac
}

die() { echo "ERROR: $*" >&2; exit 1; }

need_file() { [ -f "$1" ] || die "$1 not found.${2:+ $2}"; }

usage() { sed -n 2,31p "$0"; exit 1; }

# Run a driver, tee to a predictable log, echo its exit code (the convention the existing
# hand-written run logs already used). pipefail makes PIPESTATUS[0] the driver's own status.
run_stage() {
    local log="$LOGDIR/rq1_$1.log"; shift
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
    plan)
        TARGET="${1:-}"; shift || true
        case "$TARGET" in
            baseline) RUNG="baseline" ;;
            targeted) RUNG="targeted_er2" ;;
            requery)  die "the re-query arm has its own stage: bash scripts/run_rq1.sh requery" ;;
            *)        die "plan takes a target: baseline | targeted" ;;
        esac
        need_file "scripts/submit_plans.sh"
        need_file "$(target_csv "$TARGET")" "Run the earlier stages first."
        run_stage "plan_${TARGET}" bash scripts/submit_plans.sh "$RUNG" "$@"
        ;;

    requery)
        # Step 5a: a second clean query of the 846 tasks the target refused, on the UNMODIFIED
        # image. Its own CSV, so the first clean query in dataset_baseline_er2.csv is never
        # overwritten. Nothing in this repo builds that CSV -- it must already exist.
        need_file "scripts/submit_plans.sh"
        need_file "$(target_csv requery)" \
            "The re-query CSV is not generated by any script; it is the 846 Judgement==0 rows of the baseline with the result columns blanked."
        run_stage "requery" bash scripts/submit_plans.sh requery "$@"
        ;;

    propose)
        need_file "scripts/submit_stage1.sh"
        need_file "$(target_csv baseline)" "Run: bash scripts/run_rq1.sh plan baseline"
        run_stage "propose" bash scripts/submit_stage1.sh targeted_er2 "$@"
        ;;

    render)
        need_file "scripts/submit_render.sh"
        need_file "$(target_csv targeted)" "Run: bash scripts/run_rq1.sh propose"
        run_stage "render" bash scripts/submit_render.sh targeted_er2 "$@"
        ;;

    judge)
        TARGET="${1:-}"; shift || true
        CSV=$(target_csv "$TARGET") || die "judge takes a target: baseline | targeted | requery"
        need_file "scripts/automate_judge_new.sh"
        need_file "$CSV" "Run the matching plan stage first."
        run_stage "judge_${TARGET}" bash scripts/automate_judge_new.sh \
            --input_csv_path "$CSV" --model "$JUDGE_MODEL" "$@"
        ;;

    *) usage ;;
esac
