"""
Discover scene-level patterns that distinguish malleable from robust tasks.

Groups come from the `stratum` column of stratified_baseline.csv (zero = robust,
high = malleable). Scene text is the `Text Observation` column (free-text scene prose
already present in the baseline) — no image captioning required.
"""

import pandas as pd
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from query_litellm import query_text

BASELINE_CSV = "EARDataset/stratified_baseline_er2.csv"  # ER-1.6 sample: EARDataset/stratified_baseline.csv
SCENE_COL = "Text Observation"


def compare_scene_groups(
    high_mall_scenes: list[str],
    low_mall_scenes: list[str],
    model: str = "gemini/gemini-2.5-flash"
) -> dict:
    """
    Ask LLM to identify patterns that distinguish high vs low malleability scenes.

    Args:
        high_mall_scenes: Scene descriptions from malleable tasks (high stratum)
        low_mall_scenes: Scene descriptions from robust tasks (zero stratum)
        model: LLM model to use

    Returns:
        Dictionary with identified patterns and hypotheses
    """
    prompt = f"""You are analyzing two groups of robot task scene descriptions. Your goal is to find patterns that distinguish Group A from Group B.

=== GROUP A SCENES ===
{chr(10).join(f"- {scene[:500]}..." if len(scene) > 500 else f"- {scene}" for scene in high_mall_scenes)}

=== GROUP B SCENES ===
{chr(10).join(f"- {scene[:500]}..." if len(scene) > 500 else f"- {scene}" for scene in low_mall_scenes)}

Analyze these two groups and identify:

1. **Patterns** that distinguish Group A from Group B
2. **Scene properties** (environment type, objects, layout) that differ
3. **Hypotheses** about what might explain these differences

Respond in JSON format:
{{
    "distinguishing_patterns": [
        {{"pattern": "description", "group_a_tendency": "how it appears in A", "group_b_tendency": "how it appears in B"}}
    ],
    "scene_properties": {{
        "environments": {{"group_a": ["common environments"], "group_b": ["common environments"]}},
        "hazards": {{"group_a": "description", "group_b": "description"}},
        "object_types": {{"group_a": "description", "group_b": "description"}}
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


def run_scene_analysis(
    baseline_df: pd.DataFrame,
    max_per_pole: int = None,
    scene_col: str = SCENE_COL,
    model: str = "gemini/gemini-2.5-flash",
) -> dict:
    """
    Compare scenes of malleable (high) vs robust (zero) tasks.

    Args:
        baseline_df: stratified_baseline.csv loaded as a DataFrame (has 'stratum', scene_col)
        max_per_pole: cap scenes per pole (None = use all)
        scene_col: column holding scene text (default 'Text Observation')
        model: LLM model to use

    Returns:
        Dictionary with patterns and the sampled scenes
    """
    high = baseline_df[baseline_df["stratum"] == "high"][scene_col].dropna().astype(str).tolist()
    zero = baseline_df[baseline_df["stratum"] == "zero"][scene_col].dropna().astype(str).tolist()

    if max_per_pole is not None:
        high = high[:max_per_pole]
        zero = zero[:max_per_pole]

    print(f"Comparing {len(high)} high (malleable) vs {len(zero)} zero (robust) scenes...")
    patterns = compare_scene_groups(high, zero, model=model)

    return {
        "patterns": patterns,
        "high_samples": high,
        "zero_samples": zero,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze scene patterns (zero vs high stratum)")
    parser.add_argument("--baseline", type=str, default=BASELINE_CSV, help="Path to stratified_baseline.csv")
    parser.add_argument("--output", type=str, default=None, help="Output path for results JSON")
    parser.add_argument("--max_per_pole", type=int, default=None, help="Cap scenes per pole (default: all)")
    parser.add_argument("--scene_col", type=str, default=SCENE_COL, help="Column with scene descriptions")
    parser.add_argument("--model", type=str, default="gemini/gemini-2.5-flash", help="LLM model to use")
    args = parser.parse_args()

    df = pd.read_csv(args.baseline)
    results = {"input_csv": args.baseline, "model": args.model,
               **run_scene_analysis(df, args.max_per_pole, args.scene_col, args.model)}

    print("\n=== Discovered Patterns ===")
    if "summary" in results["patterns"]:
        print(results["patterns"]["summary"])
    else:
        print(json.dumps(results["patterns"], indent=2))

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
