#!/bin/bash
# Driver script for automate_judge_new.py
#
# Usage:
#   ./scripts/automate_judge_new.sh                                          # full CSV, 10 workers
#   ./scripts/automate_judge_new.sh --rang 0,10000                           # first 10k rows
#   ./scripts/automate_judge_new.sh --rang 10000,20000 --workers 20          # custom range and workers
#   ./scripts/automate_judge_new.sh --input_csv_path path/to/other.csv       # different CSV

CSV_PATH="EARDataset/robot_constitution_64_automerge_attack_5.csv"
MODEL="gemini-2.5-flash"
WORKERS=10
RANG=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_csv_path) CSV_PATH="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --rang) RANG="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

CMD="uv run python src/automate_judge_new.py --input_csv_path $CSV_PATH --model $MODEL --workers $WORKERS"
if [ -n "$RANG" ]; then
    CMD="$CMD --rang $RANG"
fi

echo "=== Running Judge ==="
echo "  CSV:     $CSV_PATH"
echo "  Model:   $MODEL"
echo "  Workers: $WORKERS"
echo "  Range:   ${RANG:-all}"
echo ""
echo "  $CMD"
echo ""

exec $CMD
