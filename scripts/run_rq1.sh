#!/bin/bash
# FILENAME: run_rq1.sh
# RQ1 driver: does adding a coherent object bypass the target's safety constitution?
# One entry point for the generation chain, for any target planner. Every stage is a thin wrapper
# around the driver that already does the work -- this script resolves the target to its model and
# CSVs, tees the output to a predictable log, and reports the exit code. It changes no behaviour.
#
#   plan <csv>      query the target planner on that CSV's images   LOCAL or GPU JOB  submit_plans.sh
#   propose         propose the added objects, 20 per task          CLUSTER           submit_stage1.sh
#   render          composite each object into the task image       CLUSTER           submit_render.sh
#   judge <csv>     label each plan as bypass / refusal / unclear   LOCAL             automate_judge_new.sh
#   requery         second clean query of the refused tasks         LOCAL or GPU JOB  submit_plans.sh
#
#   TARGET= selects the planner (default er2). Each target has its own CSVs, named after it:
#     er2       gemini/gemini-robotics-er-2-preview   API; every stage below runs where listed
#     cosmosr2  nvidia/Cosmos-Reason2-8B via vLLM     self-served: plan and requery are each ONE GPU JOB
#                                                     (scripts/serve_plan.sub serves the model on the node
#                                                     and runs submit_plans.sh there); run them in the cluster clone
#     gpt54     azure_openai/gpt-5.4-1                Azure Foundry /openai/v1 (the name is the deployment)
#     opus55    azure_anthropic/claude-opus-5-5       Azure Foundry /anthropic
#     kimi26    azure_openai/Kimi-K2.6                Azure Foundry /openai/v1 (Moonshot AI's open-weight model)
#     robobrain25  BAAI/RoboBrain2.5-8B-NV via vLLM   self-served like cosmosr2; the model has no thinking mode,
#                                                     so it takes the plain constitution, is served without a
#                                                     reasoning parser, and Thinking stays empty
#   <csv> is one of:
#     baseline   EARDataset/dataset_baseline_<TARGET>.csv                        (1,121 tasks, clean image)
#     targeted   EARDataset/robot_constitution_64_automerge_<TARGET>_attack_20.csv (20 objects per refused task)
#     requery    EARDataset/dataset_requery_<TARGET>.csv                         (the refused tasks, clean image)
#   'plan' takes baseline or targeted; the re-query arm has its own stage.
#
# The baseline CSV is created by 'plan baseline' when it does not exist yet: the realistic tasks of
# EARDataset/dataset.csv (isRealistic == 1, 1,121 rows), task columns only, so every target starts
# from the same clean task list. An existing baseline CSV is never rewritten by this step.
# The re-query CSV is created by 'requery' the same way when it does not exist yet: the refused rows
# (Judgement == 0) of the JUDGED baseline, with Response / Thinking / Judgement / Judgement Reason
# blanked. A self-served target runs this in the cluster clone, so copy the judged baseline there first.
#
# Run from repo root, one stage per invocation, in this order (prefix every line with TARGET=<name>
# for a target other than er2):
#   bash scripts/run_rq1.sh plan baseline        # 1,121 clean queries (cosmosr2: --rang 0,2 first as a smoke test)
#   bash scripts/run_rq1.sh judge baseline       # -> the refused tasks (Judgement == 0); ER-2: 846
#   bash scripts/run_rq1.sh propose              # cluster: one GPU job
#   bash scripts/run_rq1.sh render               # cluster: ~1,700 rows per GPU job
#   bash scripts/run_rq1.sh plan targeted 8      # attacked-image queries, 8 slices (cosmosr2: one job per --rang slice)
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
#   TARGET=         target name from the table above (default er2); sets MODEL and SYSTEM_PROMPT_NAME
#   MODEL=          target planner, a {provider}/{model} string (overrides the table; see src/query_litellm.py)
#   SYSTEM_PROMPT_NAME=  constitution           (overrides the table)
#   PROPOSER= N= QOS= MEM= TIME=  proposer model, objects per task, SLURM settings (QOS/MEM/TIME also
#                   apply to a self-served target's plan / requery job; default standby / 128G / 4:00:00)
#   NSHARDS=        parallel slices for plan / requery / render
#   CLUSTER=1       cluster clone: run an API target's plan / requery as one job per slice (scripts/plan_api.sub,
#                   10 slices by default, standby / 256G / 4:00:00) instead of local processes
#   FORCE=1         let 'propose' overwrite an existing attack CSV
# This script's own:
#   JUDGE_MODEL=    judge model                 (default gemini-2.5-flash)
#   PYTHON=         interpreter for the baseline-CSV step (default: uv run python if uv exists, else python)
#   LOGDIR=         where the run logs go       (default the repo root, as before)
set -euo pipefail

JUDGE_MODEL="${JUDGE_MODEL:-gemini-2.5-flash}"
LOGDIR="${LOGDIR:-.}"
# Interpreter for this script's own pandas step: the uv environment where there is one, else python.
if [ -z "${PYTHON:-}" ]; then
    if command -v uv > /dev/null 2>&1; then PYTHON="uv run python"; else PYTHON="python"; fi
fi

die() { echo "ERROR: $*" >&2; exit 1; }

# --- The target table ---------------------------------------------------------
# One line per planner: the LiteLLM model string, the constitution variant, and for a self-served
# model the Hugging Face id vLLM loads. MODEL / SYSTEM_PROMPT_NAME already set in the environment win.
TARGET="${TARGET:-er2}"
VLLM_MODEL="${VLLM_MODEL:-}"
# vLLM --reasoning-parser for a self-served model that emits thinking; empty serves without one.
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

# The CSV behind each name, for this target. These MIRROR scripts/submit_plans.sh's own case block
# -- that script owns the paths for the planning stages; they are repeated here only so 'judge'
# knows which file to label. Change them in both places or not at all.
target_csv() {
    case "$1" in
        baseline) echo "EARDataset/dataset_baseline_${TARGET}.csv" ;;
        targeted) echo "EARDataset/robot_constitution_64_automerge_${TARGET}_attack_20.csv" ;;
        requery)  echo "EARDataset/dataset_requery_${TARGET}.csv" ;;
        *)        return 1 ;;
    esac
}

need_file() { [ -f "$1" ] || die "$1 not found.${2:+ $2}"; }

usage() { awk 'NR == 1 { next } /^set -euo pipefail/ { exit } { print }' "$0"; exit 1; }

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

# A planning stage: locally through submit_plans.sh for an API target; for a self-served target,
# one GPU job (scripts/serve_plan.sub) that serves the model and runs submit_plans.sh on the node.
plan_stage() {
    local label="$1"; shift
    if [ -z "$VLLM_MODEL" ]; then
        run_stage "$label" bash scripts/submit_plans.sh "$@"
        return
    fi
    need_file "scripts/serve_plan.sub"
    # QOS= MEM= TIME= override the .sub's SLURM settings (standby / 128G / 4h), as for propose and render.
    local qos="${QOS:-standby}" mem="${MEM:-128G}" time="${TIME:-4:00:00}" jid
    jid=$(sbatch --parsable --qos="$qos" --mem="$mem" --time="$time" \
                 -J "serve_plan_${TARGET}_${1}" -o "serve_plan_${TARGET}_${1}_%j.out" \
                 scripts/serve_plan.sub "$@")
    echo "plan ($TARGET: $*, qos=$qos mem=$mem time=$time) -> job $jid -> serve_plan_${TARGET}_${1}_${jid}.out (vLLM log: vllm_${jid}.log)"
    echo "When it finishes, check the .out for 'Empty Response rows' and copy the CSV over before judging."
}

# --- dispatch -----------------------------------------------------------------
STAGE="${1:-}"
[ -n "$STAGE" ] || usage
shift || true

case "$STAGE" in
    plan)
        CSV_NAME="${1:-}"; shift || true
        case "$CSV_NAME" in
            baseline) RUNG="baseline" ;;
            targeted) RUNG="targeted" ;;
            requery)  die "the re-query arm has its own stage: bash scripts/run_rq1.sh requery" ;;
            *)        die "plan takes a CSV name: baseline | targeted" ;;
        esac
        need_file "scripts/submit_plans.sh"
        if [ "$CSV_NAME" = "baseline" ] && [ ! -f "$(target_csv baseline)" ]; then
            # First query of this target: start from the clean task list, no result columns.
            need_file "EARDataset/dataset.csv"
            $PYTHON - "$(target_csv baseline)" <<'PYEOF'
import sys
import pandas as pd
d = pd.read_csv("EARDataset/dataset.csv", low_memory=False)
d = d[d["isRealistic"] == 1]
d.to_csv(sys.argv[1], index=False)
print(f"wrote {sys.argv[1]}: {len(d)} realistic tasks from EARDataset/dataset.csv")
PYEOF
        fi
        need_file "$(target_csv "$CSV_NAME")" "Run the earlier stages first."
        plan_stage "plan_${CSV_NAME}${SUFFIX}" "$RUNG" "$@"
        ;;

    requery)
        need_file "scripts/submit_plans.sh"
        if [ ! -f "$(target_csv requery)" ]; then
            # The refused tasks of the judged baseline, result columns blanked for the second query.
            need_file "$(target_csv baseline)" "Run: bash scripts/run_rq1.sh plan baseline, then judge baseline"
            $PYTHON - "$(target_csv baseline)" "$(target_csv requery)" <<'PYEOF'
import sys
import pandas as pd
src, dst = sys.argv[1], sys.argv[2]
d = pd.read_csv(src, low_memory=False)
if "Judgement Reason" not in d.columns or d["Judgement Reason"].isna().all():
    sys.exit(f"ERROR: {src} is not judged yet. Run: bash scripts/run_rq1.sh judge baseline")
d = d[d["Judgement"] == 0].copy()
d[["Response", "Thinking", "Judgement", "Judgement Reason"]] = ""
d.to_csv(dst, index=False)
print(f"wrote {dst}: {len(d)} refused tasks from {src}")
PYEOF
        fi
        plan_stage "requery${SUFFIX}" requery "$@"
        ;;

    propose)
        need_file "scripts/submit_stage1.sh"
        need_file "$(target_csv baseline)" "Run: bash scripts/run_rq1.sh plan baseline"
        run_stage "propose${SUFFIX}" bash scripts/submit_stage1.sh targeted "$@"
        ;;

    render)
        need_file "scripts/submit_render.sh"
        need_file "$(target_csv targeted)" "Run: bash scripts/run_rq1.sh propose"
        run_stage "render${SUFFIX}" bash scripts/submit_render.sh targeted "$@"
        ;;

    judge)
        CSV_NAME="${1:-}"; shift || true
        CSV=$(target_csv "$CSV_NAME") || die "judge takes a CSV name: baseline | targeted | requery"
        need_file "scripts/automate_judge_new.sh"
        need_file "$CSV" "Run the matching plan stage first."
        run_stage "judge_${CSV_NAME}${SUFFIX}" bash scripts/automate_judge_new.sh \
            --input_csv_path "$CSV" --model "$JUDGE_MODEL" "$@"
        ;;

    *) usage ;;
esac
