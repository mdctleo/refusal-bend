"""
Analyze instruction-scene interactions for malleable vs robust tasks.

Groups come from the `stratum` column of stratified_baseline.csv (zero = robust,
high = malleable). We pair each task's `Instruction` with its `Text Observation`
scene prose and ask an LLM whether malleability is driven by the instruction, the
scene, or their specific combination.
"""

import pandas as pd
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from query_litellm import query_text

BASELINE_CSV = "EARDataset/stratified_baseline_er2.csv"  # ER-1.6 sample: EARDataset/stratified_baseline.csv
SCENE_COL = "Text Observation"
INSTRUCTION_COL = "Instruction"


def compare_interaction_groups(
    high_mall_pairs: list[dict],
    low_mall_pairs: list[dict],
    model: str = "gemini/gemini-2.5-flash"
) -> dict:
    """
    Ask LLM to identify patterns in instruction-scene interactions.

    Args:
        high_mall_pairs: List of {"instruction": ..., "scene": ...} from malleable (high) tasks
        low_mall_pairs: List of {"instruction": ..., "scene": ...} from robust (zero) tasks
        model: LLM model to use

    Returns:
        Dictionary with identified interaction patterns
    """
    high_text = chr(10).join(
        f"- Instruction: {p['instruction']}\n  Scene: {p['scene'][:200]}..."
        if len(p['scene']) > 200 else f"- Instruction: {p['instruction']}\n  Scene: {p['scene']}"
        for p in high_mall_pairs
    )

    low_text = chr(10).join(
        f"- Instruction: {p['instruction']}\n  Scene: {p['scene'][:200]}..."
        if len(p['scene']) > 200 else f"- Instruction: {p['instruction']}\n  Scene: {p['scene']}"
        for p in low_mall_pairs
    )

    prompt = f"""You are analyzing two groups of robot task (instruction, scene) pairs. Your goal is to find patterns that distinguish Group A from Group B.

=== GROUP A (instruction + scene) ===
{high_text}

=== GROUP B (instruction + scene) ===
{low_text}

Analyze the relationship between instruction and scene in each group:

1. **Patterns** - What distinguishes Group A pairs from Group B pairs?
2. **Instruction-scene relationship** - How does the instruction relate to the scene in each group?
3. **Primary factor** - Are the differences mainly in the instructions, scenes, or how they combine?

Respond in JSON:
{{
    "interaction_patterns": [
        {{"pattern": "description", "group_a": "how it appears", "group_b": "how it appears"}}
    ],
    "primary_driver": "instruction" or "scene" or "interaction",
    "driver_explanation": "why you think this is the primary factor distinguishing the groups",
    "hypotheses": [
        "hypothesis about what distinguishes these groups"
    ],
    "summary": "one paragraph on the key distinctions between the groups"
}}"""

    response = query_text(prompt, model=model)

    try:
        start = response.find('{')
        end = response.rfind('}') + 1
        if start != -1 and end > start:
            return json.loads(response[start:end])
    except json.JSONDecodeError:
        pass

    return {"raw_response": response, "parse_error": True}


def _pairs_for_stratum(baseline_df: pd.DataFrame, stratum: str,
                       instruction_col: str, scene_col: str,
                       max_per_pole: int = None) -> list[dict]:
    sub = baseline_df[baseline_df["stratum"] == stratum]
    if max_per_pole is not None:
        sub = sub.head(max_per_pole)
    return [
        {"instruction": str(row[instruction_col]), "scene": str(row[scene_col])}
        for _, row in sub.iterrows()
        if pd.notna(row[instruction_col]) and pd.notna(row[scene_col])
    ]


def run_interaction_analysis(
    baseline_df: pd.DataFrame,
    max_per_pole: int = None,
    instruction_col: str = INSTRUCTION_COL,
    scene_col: str = SCENE_COL,
    model: str = "gemini/gemini-2.5-flash",
) -> dict:
    """
    Compare instruction-scene pairs of malleable (high) vs robust (zero) tasks.

    Args:
        baseline_df: stratified_baseline.csv loaded as a DataFrame
        max_per_pole: cap pairs per pole (None = use all)
        instruction_col: column with instructions
        scene_col: column with scene text (default 'Text Observation')
        model: LLM model to use

    Returns:
        Dictionary with interaction analysis results
    """
    high_pairs = _pairs_for_stratum(baseline_df, "high", instruction_col, scene_col, max_per_pole)
    zero_pairs = _pairs_for_stratum(baseline_df, "zero", instruction_col, scene_col, max_per_pole)

    print(f"Analyzing {len(high_pairs)} high (malleable) vs {len(zero_pairs)} zero (robust) pairs...")
    interaction_patterns = compare_interaction_groups(high_pairs, zero_pairs, model=model)

    return {
        "interaction_patterns": interaction_patterns,
        "high_pairs": high_pairs,
        "zero_pairs": zero_pairs,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze instruction-scene interactions (zero vs high stratum)")
    parser.add_argument("--baseline", type=str, default=BASELINE_CSV, help="Path to stratified_baseline.csv")
    parser.add_argument("--output", type=str, default=None, help="Output path for results JSON")
    parser.add_argument("--max_per_pole", type=int, default=None, help="Cap pairs per pole (default: all)")
    parser.add_argument("--instruction_col", type=str, default=INSTRUCTION_COL)
    parser.add_argument("--scene_col", type=str, default=SCENE_COL)
    parser.add_argument("--model", type=str, default="gemini/gemini-2.5-flash", help="LLM model to use")
    args = parser.parse_args()

    df = pd.read_csv(args.baseline)
    results = {"input_csv": args.baseline, "model": args.model,
               **run_interaction_analysis(df, args.max_per_pole, args.instruction_col, args.scene_col, args.model)}

    print("\n=== Interaction Analysis Results ===")
    patterns = results["interaction_patterns"]
    if "summary" in patterns:
        print(patterns["summary"])
    if "primary_driver" in patterns:
        print(f"\nPrimary driver: {patterns['primary_driver']}")
        print(f"Explanation: {patterns.get('driver_explanation', 'N/A')}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
