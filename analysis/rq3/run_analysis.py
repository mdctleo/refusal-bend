"""
RQ3 root-cause pipeline: why are some tasks more malleable than others?

Grouping is read directly from the `stratum` column of the stratified sample
(zero = robust / never bypassed, high = malleable / frequently bypassed). There is
no proxy-variance quartile step and no intermediate malleability.csv anymore.

All four analyses now run on the same zero-vs-high contrast.

Runs four analyses (default --prefix rq3_er2, so the ER-1.6 rq3_*.json are never overwritten):
  1. Instruction patterns (LLM)   -> <prefix>_instruction_patterns.json
  2. Scene patterns (LLM)         -> <prefix>_scene_patterns.json
  3. Interaction patterns (LLM)   -> <prefix>_interaction_patterns.json
  4. Constitution proximity (CLIP + SBERT embeddings) -> <prefix>_constitution_analysis.json
Plus a combined <prefix>_full_analysis.json.

To run one analysis at a time, use scripts/run_rq3.sh.
"""

import pandas as pd
from pathlib import Path
import argparse
import json

BASELINE_CSV = "EARDataset/stratified_baseline_er2.csv"  # ER-1.6 sample: EARDataset/stratified_baseline.csv
PREFIX = "rq3_er2"                                        # ER-1.6 outputs: rq3_*
POLE_STRATA = ["zero", "high"]


def run_root_cause_analysis(
    baseline_df: pd.DataFrame,
    output_dir: Path,
    max_per_pole: int = None,
    model: str = "gemini/gemini-2.5-flash",
    skip_llm: bool = False,
    skip_embeddings: bool = False,
    strata: list = None,
    prefix: str = PREFIX,
    input_csv: str = None,
) -> dict:
    """Run the four RQ3 analyses on the baseline DataFrame."""
    results = {}
    provenance = {"input_csv": input_csv, "model": model}

    if not skip_llm:
        print("\n" + "=" * 60)
        print("INSTRUCTION ANALYSIS (LLM)")
        print("=" * 60)
        from instruction_features import run_instruction_analysis
        try:
            instr_results = {**provenance, **run_instruction_analysis(baseline_df, max_per_pole, model)}
            results["instruction_analysis"] = instr_results
            out = output_dir / f"{prefix}_instruction_patterns.json"
            with open(out, 'w') as f:
                json.dump(instr_results, f, indent=2)
            print(f"Saved to {out}")
            if "summary" in instr_results.get("patterns", {}):
                print(f"\nSummary: {instr_results['patterns']['summary']}")
        except Exception as e:
            print(f"Error in instruction analysis: {e}")
            results["instruction_analysis"] = {"error": str(e)}

        print("\n" + "=" * 60)
        print("SCENE ANALYSIS (LLM)")
        print("=" * 60)
        from scene_features import run_scene_analysis
        try:
            scene_results = {**provenance, **run_scene_analysis(baseline_df, max_per_pole, model=model)}
            results["scene_analysis"] = scene_results
            out = output_dir / f"{prefix}_scene_patterns.json"
            with open(out, 'w') as f:
                json.dump(scene_results, f, indent=2)
            print(f"Saved to {out}")
            if "summary" in scene_results.get("patterns", {}):
                print(f"\nSummary: {scene_results['patterns']['summary']}")
        except Exception as e:
            print(f"Error in scene analysis: {e}")
            results["scene_analysis"] = {"error": str(e)}

        print("\n" + "=" * 60)
        print("INTERACTION ANALYSIS (LLM)")
        print("=" * 60)
        from interaction_features import run_interaction_analysis
        try:
            interaction_results = {**provenance, **run_interaction_analysis(baseline_df, max_per_pole, model=model)}
            results["interaction_analysis"] = interaction_results
            out = output_dir / f"{prefix}_interaction_patterns.json"
            with open(out, 'w') as f:
                json.dump(interaction_results, f, indent=2)
            print(f"Saved to {out}")
            patterns = interaction_results.get("interaction_patterns", {})
            if "primary_driver" in patterns:
                print(f"\nPrimary driver: {patterns['primary_driver']}")
                print(f"Explanation: {patterns.get('driver_explanation', 'N/A')}")
            if "summary" in patterns:
                print(f"\nSummary: {patterns['summary']}")
        except Exception as e:
            print(f"Error in interaction analysis: {e}")
            results["interaction_analysis"] = {"error": str(e)}

    if not skip_embeddings:
        print("\n" + "=" * 60)
        print("CONSTITUTION PROXIMITY ANALYSIS (Embeddings)")
        print("=" * 60)
        from constitution_proximity import analyze_constitution_proximity
        try:
            constitution_results = analyze_constitution_proximity(
                baseline_df, strata=strata or POLE_STRATA, input_csv=input_csv
            )
            results["constitution_analysis"] = constitution_results["analysis"]
            out = output_dir / f"{prefix}_constitution_analysis.json"
            with open(out, 'w') as f:
                json.dump(constitution_results["analysis"], f, indent=2, default=str)
            print(f"Saved to {out}")
        except ImportError as e:
            print(f"Skipping constitution analysis (missing dependency): {e}")
            print("Install with: pip install transformers torch sentence-transformers")
            results["constitution_analysis"] = {"error": str(e)}
        except Exception as e:
            print(f"Error in constitution analysis: {e}")
            results["constitution_analysis"] = {"error": str(e)}

    return results


def print_stratum_summary(baseline_df: pd.DataFrame):
    """Print the group sizes RQ3 is contrasting."""
    print("\n" + "=" * 60)
    print("RQ3 GROUPS (from the sample's `stratum` column)")
    print("=" * 60)
    counts = baseline_df["stratum"].value_counts()
    for s in ["zero", "low", "medium", "high"]:
        if s in counts:
            print(f"  {s:<7}: {counts[s]} tasks")
    print(f"\nContrast: zero (robust) vs high (malleable)")


def main():
    parser = argparse.ArgumentParser(description="Run RQ3 root cause analysis (zero vs high stratum)")
    parser.add_argument("--baseline", type=str, default=BASELINE_CSV, help="Stratified sample CSV (has `stratum`)")
    parser.add_argument("--output_dir", type=str, default="analysis/output", help="Output directory")
    parser.add_argument("--prefix", type=str, default=PREFIX, help="Output file prefix (ER-1.6 used rq3)")
    parser.add_argument("--max_per_pole", type=int, default=None, help="Cap samples per pole (default: all)")
    parser.add_argument("--model", type=str, default="gemini/gemini-2.5-flash", help="LLM model for analysis")
    parser.add_argument("--skip_llm", action="store_true", help="Skip LLM-based analysis")
    parser.add_argument("--skip_embeddings", action="store_true", help="Skip embedding-based analysis")
    parser.add_argument("--strata", type=str, default=",".join(POLE_STRATA),
                        help="Strata for the constitution proximity contrast (default: zero,high)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading baseline: {args.baseline}")
    baseline_df = pd.read_csv(args.baseline)
    print(f"Loaded {len(baseline_df)} tasks")

    print_stratum_summary(baseline_df)

    root_cause_results = run_root_cause_analysis(
        baseline_df, output_dir,
        max_per_pole=args.max_per_pole,
        model=args.model,
        skip_llm=args.skip_llm,
        skip_embeddings=args.skip_embeddings,
        strata=[s.strip() for s in args.strata.split(",") if s.strip()],
        prefix=args.prefix,
        input_csv=args.baseline,
    )

    all_results = {
        "grouping": "`stratum` column of input_csv (zero=robust, high=malleable)",
        "input_csv": args.baseline,
        "root_cause": {k: v for k, v in root_cause_results.items() if "error" not in v},
    }
    with open(output_dir / f"{args.prefix}_full_analysis.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"Analysis complete. Results saved to {output_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
