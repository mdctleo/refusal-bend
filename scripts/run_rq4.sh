#!/bin/bash
# FILENAME: run_rq4.sh
# RQ4 driver: predict task malleability from cheap LOCAL proxy signals, then score those
# predictions against the target's real bypass rate. One entry point for every stage.
#
# Each approach has up to three stages (analysis/rq4/approachN.py --stage ...):
#   prepare   build model INPUTS          SUBMITTED  (only A2: it writes ~31 GB of noise
#                                                 images, which must land on the
#                                                 filesystem the infer shards read. The
#                                                 work is pure CPU, but this cluster has
#                                                 no CPU queue, so the job takes a GPU it
#                                                 never uses just to be admitted.)
#   infer     query the proxy VLM         GPU, SUBMITTED  -> N independent jobs
#   score     raw output -> one number/task  CPU, LOCAL   (re-runnable with no GPU: the
#                                                 aggregator has changed twice; see README)
#
#   ground_truth  per-task bypass labels from the judged attack CSV   CPU, LOCAL
#   evaluate      sweep every approach's headline score vs ground truth   CPU, LOCAL
#
# So: everything that touches a model or writes model inputs goes to the cluster;
# ground_truth / score / evaluate are seconds-to-minutes of local CPU.
#
# Run from repo root:
#   bash scripts/run_rq4.sh ground_truth
#   bash scripts/run_rq4.sh a2 prepare            # submits a CPU job; LOCAL=1 to run inline
#   bash scripts/run_rq4.sh a2 infer 10           # 10 GPU shards
#   bash scripts/run_rq4.sh a4 infer              # default shards for that approach
#   bash scripts/run_rq4.sh a3 infer --rang 0,200 # one slice (e.g. a preempted shard)
#   bash scripts/run_rq4.sh a1 score
#   bash scripts/run_rq4.sh a5 score
#   bash scripts/run_rq4.sh evaluate
#
# One stage per invocation -- submit them yourself, in the order above.
#
# Env overrides:
#   PREFIX=analysis/output/rq4_er2   output path prefix (every CSV derives from it)
#   ATTACK=... BASELINE=...          input CSVs (default ER-2)
#   VLM=...                          proxy model (same string for all four approaches)
#   MANIFEST=... NOISE_DIR=...       A2's images + manifest
#   QOS=... MEM=... TIME=...         SLURM overrides, applied to whichever stage you run.
#                                    Defaults differ per stage: infer is standby/256G/
#                                    2-4h (preemption costs one shard, retried via
#                                    --rang); prepare is normal/64G/4h, NOT standby,
#                                    because it is not resumable.
#   LOCAL=1                          run a2 prepare inline here instead of submitting
#   FORCE=1                          allow overwriting an existing CPU-stage output
#
# Defaults target ER-2. The ER-1.6 artifacts live at analysis/output/proxy_* and are
# unreachable unless you pass PREFIX=analysis/output/proxy explicitly.
set -euo pipefail

PREFIX="${PREFIX:-analysis/output/rq4_er2}"
ATTACK="${ATTACK:-EARDataset/robot_constitution_64_automerge_er2_attack_20.csv}"
BASELINE="${BASELINE:-EARDataset/dataset_baseline_er2.csv}"
VLM="${VLM:-Qwen/Qwen3-VL-30B-A3B-Thinking-FP8}"
MANIFEST="${MANIFEST:-EARDataset/proxy_noise_er2_manifest.csv}"
NOISE_DIR="${NOISE_DIR:-EARDataset/proxy_noise_er2}"
PY=(uv run python)
SUB="scripts/rq4_proxy.sub"          # GPU: one infer shard
PRESUB="scripts/rq4_prepare.sub"     # CPU: A2 prepare (images + manifest)

# Per-approach: module, shard unit, default shard count, default wall time.
# A2 shards over MANIFEST ROWS; A1/A3/A4 over unique Task IDs.
approach_module() { echo "analysis/rq4/approach${1#a}.py"; }
approach_shards()  { case "$1" in a1) echo 1 ;; a2) echo 10 ;; a3) echo 1 ;; a4) echo 4 ;; esac; }
approach_time()    { case "$1" in a1|a3) echo "2:00:00" ;; a2|a4) echo "4:00:00" ;; esac; }
# headline score column each approach is judged on (see RQ4_RESULTS_SUMMARY_ER16.md)
approach_score()   { case "$1" in
                       a1) echo "object_ambiguity_density" ;;
                       a2) echo "std_shift" ;;
                       a3) echo "constitution_reliance" ;;
                       a4) echo "std_comply" ;;
                       a5) echo "score_curated_nolen" ;;
                     esac; }
approach_input()   { case "$1" in
                       a1) echo "${PREFIX}_approach1_features.csv" ;;
                       a2) echo "${PREFIX}_approach2_scores.csv" ;;
                       a3) echo "${PREFIX}_approach3_scores.csv" ;;
                       a4) echo "${PREFIX}_approach4_scores.csv" ;;
                       a5) echo "${PREFIX}_approach5_scores.csv" ;;
                     esac; }

die() { echo "ERROR: $*" >&2; exit 1; }

need_file() { [ -f "$1" ] || die "$1 not found.${2:+ $2}"; }

guard_output() {   # refuse to clobber a CPU-stage output unless FORCE=1
    if [ -e "$1" ] && [ "${FORCE:-0}" != "1" ]; then
        die "$1 already exists; move it aside or rerun with FORCE=1."
    fi
}

# --- GPU fan-out: N independent jobs, one --rang slice each (no SLURM array; arrays
# --- dispatch slower on this cluster, which is why the old scripts fanned out too).
submit_infer() {
    local a="$1" nshards="$2" rang="$3"
    local module; module=$(approach_module "$a")
    # Per-stage defaults; one set of env names (QOS/MEM/TIME) steers whichever you run.
    local qos="${QOS:-standby}"   # preemption costs one shard, retried with --rang
    local mem="${MEM:-256G}"      # holds the model weights
    local time="${TIME:-$(approach_time "$a")}"
    need_file "$module"
    need_file "$SUB"

    local extra=()
    local total
    if [ "$a" = "a2" ]; then
        extra=(--manifest "$MANIFEST" --noise_dir "$NOISE_DIR")
        need_file "$MANIFEST" "Run: bash scripts/run_rq4.sh a2 prepare"
        [ -d "$NOISE_DIR" ] || die "$NOISE_DIR not found. Run: bash scripts/run_rq4.sh a2 prepare"
        total=$("${PY[@]}" -c "import pandas as pd; print(len(pd.read_csv('$MANIFEST')))")
        echo "  shard unit: manifest rows"
    else
        need_file "$ATTACK"
        need_file "$BASELINE"
        total=$("${PY[@]}" -c "import pandas as pd; print(pd.read_csv('$ATTACK', low_memory=False)['Task ID'].nunique())")
        echo "  shard unit: unique Task IDs"
    fi

    local ranges=()
    if [ -n "$rang" ]; then
        ranges+=("$rang")
    else
        local chunk=$(( (total + nshards - 1) / nshards ))
        for i in $(seq 0 $((nshards - 1))); do
            local start=$(( i * chunk )) end=$(( i * chunk + chunk ))
            [ "$start" -ge "$total" ] && continue
            [ "$end" -gt "$total" ] && end=$total
            ranges+=("${start},${end}")
        done
    fi

    echo "  total: $total | jobs: ${#ranges[@]} | qos=$qos mem=$mem time=$time"
    echo "  model: $VLM"
    for i in "${!ranges[@]}"; do
        local r="${ranges[$i]}"
        local jid
        jid=$(sbatch --parsable --qos="$qos" --mem="$mem" --time="$time" \
                     -J "rq4_${a}_${i}" -o "rq4_${a}_${i}_%j.out" \
                     "$SUB" "$module" "$r" "$VLM" "$PREFIX" "$ATTACK" "$BASELINE" "${extra[@]}")
        echo "  shard $i -> [${r}) -> job $jid"
    done
}

run_ground_truth() {
    need_file "$ATTACK"
    guard_output "${PREFIX}_ground_truth.csv"
    "${PY[@]}" analysis/rq4/ground_truth.py --attack_csv "$ATTACK" --out_prefix "$PREFIX"
}

# A2 only: writes the perturbed images + manifest. The work is pure CPU, but it is
# SUBMITTED TO THE CLUSTER -- the ~31 GB of images has to land on the filesystem the A2
# infer shards read, and generating it locally only works if the local checkout is that
# same mount.
run_prepare() {
    need_file "$ATTACK"; need_file "$BASELINE"
    need_file "$PRESUB"
    guard_output "$MANIFEST"
    if [ "${LOCAL:-0}" = "1" ]; then
        echo "LOCAL=1: running prepare inline instead of submitting." >&2
        "${PY[@]}" analysis/rq4/approach2.py --stage prepare \
            --attack_csv "$ATTACK" --baseline_csv "$BASELINE" \
            --manifest "$MANIFEST" --noise_dir "$NOISE_DIR"
        return
    fi
    # NOT RESUMABLE: a re-run regenerates every task from scratch, so a preempted job
    # throws away the whole pass (~20 min and ~40 GB of writes) rather than picking up
    # where it stopped -- hence the qos=normal default, NOT standby.
    # Partition/gres live in the .sub: this cluster has no CPU queue, so prepare takes a
    # GPU it never uses simply to be admitted.
    local qos="${QOS:-normal}"    # see above: standby preemption costs the whole pass
    local mem="${MEM:-64G}"       # PIL image work, no model weights
    local time="${TIME:-4:00:00}"
    local jid
    jid=$(sbatch --parsable --qos="$qos" --mem="$mem" --time="$time" \
                 "$PRESUB" "$ATTACK" "$BASELINE" "$MANIFEST" "$NOISE_DIR")
    echo "  a2 prepare -> job $jid  (qos=$qos mem=$mem time=$time, not resumable)"
}

run_score() {
    local a="$1"
    local module; module=$(approach_module "$a")
    case "$a" in
        a1) need_file "${PREFIX}_approach1_traces.csv" "Run the A1 infer stage first."
            need_file "${PREFIX}_ground_truth.csv" "Run: bash scripts/run_rq4.sh ground_truth" ;;
        a2) need_file "${PREFIX}_approach2_results.csv" "Run the A2 infer stage first." ;;
        a3) need_file "${PREFIX}_approach3_results.csv" "Run the A3 infer stage first." ;;
        a4) need_file "${PREFIX}_approach4_results.csv" "Run the A4 infer stage first." ;;
        a5) for x in a1 a2 a3 a4; do
                need_file "$(approach_input "$x")" "Run the $x score stage first."
            done
            need_file "${PREFIX}_ground_truth.csv" "Run: bash scripts/run_rq4.sh ground_truth" ;;
    esac
    # A3's score rewrites its results CSV in place (idempotent), so it is not guarded.
    "${PY[@]}" "$module" --stage score --out_prefix "$PREFIX"
}

run_evaluate() {
    need_file "${PREFIX}_ground_truth.csv" "Run: bash scripts/run_rq4.sh ground_truth"
    local missing=0
    for a in a1 a2 a3 a4 a5; do
        [ -f "$(approach_input "$a")" ] || { echo "  [skip] $a: $(approach_input "$a") not found"; missing=1; }
    done
    for a in a1 a2 a3 a4 a5; do
        local in; in=$(approach_input "$a")
        [ -f "$in" ] || continue
        "${PY[@]}" analysis/rq4/evaluate.py \
            --input "$in" --score_col "$(approach_score "$a")" \
            --gt "${PREFIX}_ground_truth.csv" \
            --label "$a: $(approach_score "$a")"
    done
    [ $missing -eq 0 ] || echo "(some approaches were skipped -- see above)"
}

# --- dispatch -----------------------------------------------------------------
CMD="${1:-}"
case "$CMD" in
    ground_truth) run_ground_truth ;;
    evaluate)     run_evaluate ;;
    a1|a2|a3|a4|a5)
        STAGE="${2:-}"
        shift 2 || true
        NSHARDS_ARG=""; RANG=""
        while [ $# -gt 0 ]; do
            case "$1" in
                --rang) RANG="$2"; shift 2 ;;
                *) NSHARDS_ARG="$1"; shift ;;
            esac
        done
        case "$STAGE" in
            prepare)
                [ "$CMD" = "a2" ] || die "only a2 has a prepare stage (A1/A3/A4 use the existing clean images)."
                run_prepare ;;
            infer)
                [ "$CMD" = "a5" ] && die "a5 never queries a model -- it has only a score stage."
                echo "=== RQ4 $CMD infer ==="
                submit_infer "$CMD" "${NSHARDS_ARG:-${NSHARDS:-$(approach_shards "$CMD")}}" "$RANG" ;;
            score)
                echo "=== RQ4 $CMD score ==="
                run_score "$CMD" ;;
            *) die "unknown stage '$STAGE' for $CMD (prepare|infer|score)" ;;
        esac ;;
    *) sed -n 2,42p "$0"; exit 1 ;;
esac
