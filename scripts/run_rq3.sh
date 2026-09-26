#!/bin/bash
# FILENAME: run_rq3.sh
# RQ3 analyses, run LOCALLY: contrast the robust (zero) and malleable (high) poles of a stratified sample.
# Run one analysis at a time, or all four:
#   instruction   LLM, 1 call: instruction patterns              -> <PREFIX>_instruction_patterns.json
#   scene         LLM, 1 call: scene patterns (Text Observation)  -> <PREFIX>_scene_patterns.json
#   interaction   LLM, 1 call: instruction x scene interaction    -> <PREFIX>_interaction_patterns.json
#   proximity     CLIP + SBERT, no API: proximity to the constitution
#                 -> <PREFIX>_constitution_analysis.json + <PREFIX>_constitution_proximity.csv
#   all           the four above, in that order
# Defaults: ER-2 sample EARDataset/stratified_baseline_er2.csv, PREFIX analysis/output/rq3_er2.
# Never overwrites an existing output unless FORCE=1 (the ER-1.6 results are analysis/output/rq3_*.json/csv).
#
# Run from repo root:
#   bash scripts/run_rq3.sh instruction
#   bash scripts/run_rq3.sh proximity
#   bash scripts/run_rq3.sh all
#   STRATA=zero,low,medium,high PREFIX=analysis/output/rq3_er2_allstrata bash scripts/run_rq3.sh proximity
#   MAX_PER_POLE=5 PREFIX=/tmp/rq3_smoke bash scripts/run_rq3.sh scene     # capped smoke test, elsewhere
#   BASELINE=EARDataset/stratified_baseline.csv PREFIX=analysis/output/rq3_er16_rerun bash scripts/run_rq3.sh all
#   MODEL=gemini/gemini-2.5-flash FORCE=1 bash scripts/run_rq3.sh interaction
set -euo pipefail

ANALYSIS="${1:-}"
case "$ANALYSIS" in
    instruction|scene|interaction|proximity) SELECTED=("$ANALYSIS") ;;
    all) SELECTED=(instruction scene interaction proximity) ;;
    *) sed -n 2,21p "$0"; exit 1 ;;
esac

BASELINE="${BASELINE:-EARDataset/stratified_baseline_er2.csv}"
PREFIX="${PREFIX:-analysis/output/rq3_er2}"
MODEL="${MODEL:-gemini/gemini-2.5-flash}"
STRATA="${STRATA:-zero,high}"
MAX_PER_POLE="${MAX_PER_POLE:-}"
IMAGE_ROOT="EARDataset/images"
PY=(uv run python)

CAP=()
if [ -n "$MAX_PER_POLE" ]; then CAP=(--max_per_pole "$MAX_PER_POLE"); fi

outputs_for() {
    case "$1" in
        instruction) echo "${PREFIX}_instruction_patterns.json" ;;
        scene)       echo "${PREFIX}_scene_patterns.json" ;;
        interaction) echo "${PREFIX}_interaction_patterns.json" ;;
        proximity)   echo "${PREFIX}_constitution_analysis.json ${PREFIX}_constitution_proximity.csv" ;;
    esac
}

# --- Preconditions (all checked before anything runs) -------------------------
if [ ! -f "$BASELINE" ]; then
    echo "ERROR: $BASELINE not found." >&2
    exit 1
fi
for a in "${SELECTED[@]}"; do
    for f in $(outputs_for "$a"); do
        if [ -e "$f" ] && [ "${FORCE:-0}" != "1" ]; then
            echo "ERROR: $f already exists; move it aside or rerun with FORCE=1." >&2
            exit 1
        fi
    done
    if [ "$a" = "proximity" ]; then
        if [ ! -d "$IMAGE_ROOT" ]; then
            echo "ERROR: $IMAGE_ROOT not found (proximity embeds the scene images)." >&2
            exit 1
        fi
    elif [ -z "${GEMINI_API_KEY:-}${LITELLM_API_KEY:-}" ] && ! grep -qE '^(GEMINI_API_KEY|LITELLM_API_KEY)=' .env 2>/dev/null; then
        # src/query_litellm.py reads the key from the environment or the repo-root .env
        echo "ERROR: no GEMINI_API_KEY / LITELLM_API_KEY in the environment or .env." >&2
        exit 1
    fi
done
mkdir -p "$(dirname "$PREFIX")"

echo "=== RQ3 (${SELECTED[*]}) ==="
echo "  Sample:  $BASELINE"
echo "  Outputs: ${PREFIX}_*"
echo "  Model:   $MODEL | cap per pole: ${MAX_PER_POLE:-all} | proximity strata: $STRATA"
"${PY[@]}" - "$BASELINE" <<'PYEOF'
import sys
import pandas as pd
c = pd.read_csv(sys.argv[1])["stratum"].value_counts()
print("  Strata:  " + " | ".join(f"{s} {int(c.get(s, 0))}" for s in ["zero", "low", "medium", "high"]))
PYEOF
echo ""

failed=0
for a in "${SELECTED[@]}"; do
    out=$(outputs_for "$a")
    echo "== $a -> $out"
    case "$a" in
        instruction|scene|interaction)
            "${PY[@]}" "analysis/rq3/${a}_features.py" --baseline "$BASELINE" --model "$MODEL" \
                --output "$out" "${CAP[@]}"
            # an LLM reply that is not valid JSON is saved as raw_response + parse_error
            if ! "${PY[@]}" - "$out" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
p = d.get("patterns", d.get("interaction_patterns", {}))
if isinstance(p, dict) and p.get("parse_error"):
    print(f"WARNING: {sys.argv[1]}: the LLM reply did not parse as JSON (kept as raw_response)")
    sys.exit(1)
PYEOF
            then
                failed=1
            fi ;;
        proximity)
            read -r json csv <<< "$out"
            "${PY[@]}" analysis/rq3/constitution_proximity.py --baseline "$BASELINE" --strata "$STRATA" \
                --image_base_path "$IMAGE_ROOT" --output "$json" --per_task_csv "$csv" ;;
    esac
    echo ""
done

exit $failed
