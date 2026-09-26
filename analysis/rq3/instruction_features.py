"""
Discover instruction-level patterns that distinguish malleable from robust tasks.

Groups come straight from the `stratum` column of stratified_baseline.csv:
  - "zero" stratum = robust tasks (never bypassed)
  - "high" stratum = malleable tasks (frequently bypassed)

We sample the instructions from each pole and ask an LLM to name the distinguishing
patterns. This replaces the old proxy-variance quartile (Q1 vs Q4) grouping.
"""

import pandas as pd
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from query_litellm import query_text

BASELINE_CSV = "EARDataset/stratified_baseline_er2.csv"  # ER-1.6 sample: EARDataset/stratified_baseline.csv


def compare_instruction_groups(
    high_mall_instructions: list[str],
    low_mall_instructions: list[str],
    model: str = "gemini/gemini-2.5-flash"
) -> dict:
    """
    Ask LLM to identify patterns that distinguish high vs low malleability instructions.

    Args:
        high_mall_instructions: Instructions from malleable tasks (high stratum)
        low_mall_instructions: Instructions from robust tasks (zero stratum)
        model: LLM model to use

    Returns:
        Dictionary with identified patterns and hypotheses
    """
    prompt = f"""You are analyzing two groups of robot task instructions. Your goal is to find patterns that distinguish Group A from Group B.

=== GROUP A INSTRUCTIONS ===
{chr(10).join(f"- {instr}" for instr in high_mall_instructions)}

=== GROUP B INSTRUCTIONS ===
{chr(10).join(f"- {instr}" for instr in low_mall_instructions)}

Analyze these two groups and identify:

1. **Patterns** that distinguish Group A from Group B
2. **Linguistic features** (verbs, objects, phrasing) that differ between groups
3. **Hypotheses** about what might explain these differences

Respond in JSON format:
{{
    "distinguishing_patterns": [
        {{"pattern": "description", "group_a_tendency": "how it appears in A", "group_b_tendency": "how it appears in B"}}
    ],
    "linguistic_features": {{
        "verbs": {{"group_a": ["common verbs"], "group_b": ["common verbs"]}},
        "objects": {{"group_a": "description", "group_b": "description"}},
        "phrasing": {{"group_a": "description", "group_b": "description"}}
    }},
    "hypotheses": [
        "hypothesis 1",
        "hypothesis 2"
    ],
    "summary": "one paragraph summary of key distinctions between the groups"
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


def run_instruction_analysis(
    baseline_df: pd.DataFrame,
    max_per_pole: int = None,
    model: str = "gemini/gemini-2.5-flash",
) -> dict:
    """
    Compare instructions of malleable (high) vs robust (zero) tasks.

    Args:
        baseline_df: stratified_baseline.csv loaded as a DataFrame (has 'stratum', 'Instruction')
        max_per_pole: cap instructions per pole (None = use all, the full stratified draw)
        model: LLM model to use

    Returns:
        Dictionary with patterns and the sampled instructions
    """
    high = baseline_df[baseline_df["stratum"] == "high"]["Instruction"].dropna().tolist()
    zero = baseline_df[baseline_df["stratum"] == "zero"]["Instruction"].dropna().tolist()

    if max_per_pole is not None:
        high = high[:max_per_pole]
        zero = zero[:max_per_pole]

    print(f"Comparing {len(high)} high (malleable) vs {len(zero)} zero (robust) instructions...")
    patterns = compare_instruction_groups(high, zero, model=model)

    return {
        "patterns": patterns,
        "high_samples": high,
        "zero_samples": zero,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze instruction patterns (zero vs high stratum)")
    parser.add_argument("--baseline", type=str, default=BASELINE_CSV, help="Path to stratified_baseline.csv")
    parser.add_argument("--output", type=str, default=None, help="Output path for results JSON")
    parser.add_argument("--max_per_pole", type=int, default=None, help="Cap instructions per pole (default: all)")
    parser.add_argument("--model", type=str, default="gemini/gemini-2.5-flash", help="LLM model to use")
    args = parser.parse_args()

    df = pd.read_csv(args.baseline)
    results = {"input_csv": args.baseline, "model": args.model,
               **run_instruction_analysis(df, args.max_per_pole, args.model)}

    print("\n=== Discovered Patterns ===")
    if "summary" in results["patterns"]:
        print(results["patterns"]["summary"])
    else:
        print(json.dumps(results["patterns"], indent=2))

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
