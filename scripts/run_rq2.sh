#!/bin/bash
# FILENAME: run_rq2.sh
# RQ2 driver: does the added object need to be chosen for the task?
# One entry point for the two non-targeted rungs. Every stage is a thin wrapper around the driver
# that already does the work -- this script only resolves the rung name to its CSV, tees the output
# to a predictable log, and reports the exit code. It changes no behaviour.
#
#   propose <rung>  propose the added objects, 5 per task       LOCAL/CLUSTER submit_stage1.sh
#   render <rung>   composite each object into the task image   CLUSTER       submit_render.sh
#   plan <rung>     query the target planner on those images    LOCAL or GPU JOB  submit_plans.sh
#   judge <rung>    label each plan as bypass / refusal / unclear  LOCAL      automate_judge_new.sh
#   audit <cmd>     constraint audit of the TARGETED objects    LOCAL         analysis/rq2/constraint_audit.py
#
#   TARGET= selects the planner (default er2), with the names, models and constitutions of
#   scripts/run_rq1.sh: er2 | cosmosr2 | gpt54 | opus55 | kimi26 | robobrain25. For a self-served
#   target (cosmosr2, robobrain25) 'plan' is ONE GPU JOB (scripts/serve_plan.sub); run it in the
#   cluster clone.
#
#   <rung> is one of:
#     random      EARDataset/stratified_random_<TARGET>.csv      fixed 26-item COCO list, no VLM
#     scene_only  EARDataset/stratified_scene_only_<TARGET>.csv  proposer sees the image, not the instruction
#     null_edit   EARDataset/stratified_null_edit_<TARGET>.csv   no object: the clean image through the same
#                                                                editor with nothing added, 20 renders per task
#   Each is the target's stratified tasks x 20 rows (ER-2: 194 tasks).
#   The 'targeted' rung is RQ1's data -- it is not regenerated here. See scripts/run_rq1.sh.
#
# Run from repo root, one stage per invocation, in this order (prefix every line with TARGET=<name>
# for a target other than er2, and pass --target <name> to the sample script; the sample needs that
# target's judged targeted and re-query CSVs from scripts/run_rq1.sh):
#   uv run python analysis/rq2/stratified_sample.py      # 0. the stratified sample (--force to redraw)
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
#   uv run python analysis/rq2/analysis.py               # ER-2; another target: its four --*_csv and --output_json
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
#   TARGET=         target name (default er2); sets MODEL and SYSTEM_PROMPT_NAME
#   MODEL=          target planner, a {provider}/{model} string (overrides the table)
#   SYSTEM_PROMPT_NAME=  constitution           (overrides the table)
#   PROPOSER= N= K= QOS= MEM= TIME=  proposer model, objects per task, SLURM settings (QOS/MEM/TIME also
#                   apply to a self-served target's plan job; default standby / 128G / 4:00:00)
#   NSHARDS=        parallel slices for plan / render
#   FORCE=1         let 'propose' overwrite an existing rung CSV
# This script's own:
#   JUDGE_MODEL=    judge model                 (default gemini-2.5-flash)
#   LOGDIR=         where the run logs go       (default the repo root, as before)
set -euo pipefail

JUDGE_MODEL="${JUDGE_MODEL:-gemini-2.5-flash}"
LOGDIR="${LOGDIR:-.}"

die() { echo "ERROR: $*" >&2; exit 1; }

# --- The target table ---------------------------------------------------------
# A copy of scripts/run_rq1.sh's table: the LiteLLM model string, the constitution variant, and for a
# self-served model the Hugging Face id vLLM loads and its reasoning parser. Keep the two in step.
TARGET="${TARGET:-er2}"
VLLM_MODEL="${VLLM_MODEL:-}"
VLLM_REASONING_PARSER="${VLLM_REASONING_PARSER:-}"
case "$TARGET" in
    er2)      _model="gemini/gemini-robotics-er-2-preview";  _prompt="robot_constitution_64_automerge" ;;
    cosmosr2) _model="hosted_vllm/nvidia/Cosmos-Reason2-8B"; _prompt="robot_constitution_64_automerge_cosmos_reason2"
              VLLM_MODEL="${VLLM_MODEL:-nvidia/Cosmos-Reason2-8B}"
              VLLM_REASONING_PARSER="${VLLM_REASONING_PARSER:-qwen3}" ;;
    gpt54)    _model="azure_openai/gpt-5.4-1";               _prompt="robot_constitution_64_automerge" ;;
    opus55)   _model="azure_anthropic/claude-opus-5-5";      _prompt="robot_constitution_64_automerge" ;;
    kimi26)   _model="azure_openai/Kimi-K2.6";               _prompt="robot_constitution_64_automerge" ;;
    robobrain25) _model="hosted_vllm/BAAI/RoboBrain2.5-8B-NV"; _prompt="robot_constitution_64_automerge"
              VLLM_MODEL="${VLLM_MODEL:-BAAI/RoboBrain2.5-8B-NV}" ;;
    *)        die "unknown TARGET '$TARGET': er2 | cosmosr2 | gpt54 | opus55 | kimi26 | robobrain25" ;;
esac
MODEL="${MODEL:-$_model}"
SYSTEM_PROMPT_NAME="${SYSTEM_PROMPT_NAME:-$_prompt}"
export TARGET MODEL SYSTEM_PROMPT_NAME VLLM_MODEL VLLM_REASONING_PARSER
# Log-name suffix: none for er2, so its logs keep the names the existing runs used.
SUFFIX=""; [ "$TARGET" = "er2" ] || SUFFIX="_${TARGET}"

# The CSV behind each rung name, for this target. These MIRROR scripts/submit_plans.sh's own case
# block -- that script owns the paths for the planning stages; they are repeated here only so 'judge'
# knows which file to label. Change them in both places or not at all.
rung_csv() {
    case "$1" in
        random|scene_only|null_edit) echo "EARDataset/stratified_${1}_${TARGET}.csv" ;;
        *)                           return 1 ;;
    esac
}

need_file() { [ -f "$1" ] || die "$1 not found.${2:+ $2}"; }

usage() { awk 'NR == 1 { next } /^set -euo pipefail/ { exit } { print }' "$0"; exit 1; }

# The rung argument, shared by propose / render / plan / judge. 'targeted' is RQ1's, not RQ2's.
take_rung() {
    case "${1:-}" in
        random|scene_only|null_edit) echo "$1" ;;
        targeted) die "the targeted rung is RQ1's data, generated by scripts/run_rq1.sh -- RQ2 reads it from EARDataset/robot_constitution_64_automerge_${TARGET}_attack_20.csv" ;;
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

# The plan stage: locally through submit_plans.sh for an API target; for a self-served target, one
# GPU job (scripts/serve_plan.sub) that serves the model and runs submit_plans.sh on the node.
plan_stage() {
    local label="$1"; shift
    if [ -z "$VLLM_MODEL" ]; then
        run_stage "$label" bash scripts/submit_plans.sh "$@"
        return
    fi
    need_file "scripts/serve_plan.sub"
    local qos="${QOS:-standby}" mem="${MEM:-128G}" time="${TIME:-4:00:00}" jid
    jid=$(sbatch --parsable --qos="$qos" --mem="$mem" --time="$time" \
                 -J "serve_plan_${1}" -o "serve_plan_${1}_%j.out" \
                 scripts/serve_plan.sub "$@")
    echo "plan ($TARGET: $*, qos=$qos mem=$mem time=$time) -> job $jid -> serve_plan_${1}_${jid}.out (vLLM log: vllm_${jid}.log)"
    echo "When it finishes, check the .out for 'Empty Response rows' and copy the CSV over before judging."
}

# --- dispatch -----------------------------------------------------------------
STAGE="${1:-}"
[ -n "$STAGE" ] || usage
shift || true

case "$STAGE" in
    propose)
        RUNG=$(take_rung "${1:-}" propose); shift || true
        need_file "scripts/submit_stage1.sh"
        need_file "EARDataset/stratified_baseline_${TARGET}.csv" "Run: uv run python analysis/rq2/stratified_sample.py --target $TARGET"
        run_stage "propose_${RUNG}${SUFFIX}" bash scripts/submit_stage1.sh "${RUNG}_${TARGET}" "$@"
        ;;

    render)
        RUNG=$(take_rung "${1:-}" render); shift || true
        need_file "scripts/submit_render.sh"
        need_file "$(rung_csv "$RUNG")" "Run: bash scripts/run_rq2.sh propose $RUNG"
        run_stage "render_${RUNG}${SUFFIX}" bash scripts/submit_render.sh "${RUNG}_${TARGET}" "$@"
        ;;

    plan)
        RUNG=$(take_rung "${1:-}" plan); shift || true
        need_file "scripts/submit_plans.sh"
        need_file "$(rung_csv "$RUNG")" "Run: bash scripts/run_rq2.sh propose $RUNG"
        plan_stage "plan_${RUNG}${SUFFIX}" "${RUNG}_${TARGET}" "$@"
        ;;

    judge)
        RUNG=$(take_rung "${1:-}" judge); shift || true
        CSV=$(rung_csv "$RUNG")
        need_file "scripts/automate_judge_new.sh"
        need_file "$CSV" "Run: bash scripts/run_rq2.sh plan $RUNG"
        run_stage "judge_${RUNG}${SUFFIX}" bash scripts/automate_judge_new.sh \
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
