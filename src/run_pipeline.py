"""
Run Pipeline Module

Executes the complete AEI analysis pipeline:
1. Data Ingestion
2. Data Cleaning
3. Exploratory Analysis
4. Regression Analysis
5. Visualization Outputs
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import data_ingestion
from src import data_cleaning
from src import exploratory_analysis
from src import regression_analysis
from src import visualization_outputs


def run_full_pipeline(skip_download: bool = False) -> dict:
    """
    Run the complete analysis pipeline.

    Args:
        skip_download: If True, skip data ingestion (assumes data exists)

    Returns:
        Dictionary with results from each stage
    """
    results = {}

    print("\n" + "=" * 80)
    print("AEI ANALYSIS PIPELINE")
    print("=" * 80)

    # Stage 1: Data Ingestion
    if not skip_download:
        print("\n" + "-" * 80)
        print("STAGE 1: DATA INGESTION")
        print("-" * 80)
        try:
            df_raw = data_ingestion.main()
            results["ingestion"] = "success"
            print("✓ Data ingestion complete")
        except Exception as e:
            print(f"✗ Data ingestion failed: {e}")
            results["ingestion"] = f"failed: {e}"
            return results
    else:
        print("\n[Skipping data ingestion - using existing data]")
        results["ingestion"] = "skipped"

    # Stage 2: Data Cleaning
    print("\n" + "-" * 80)
    print("STAGE 2: DATA CLEANING")
    print("-" * 80)
    try:
        df_cleaned = data_cleaning.main()
        if df_cleaned is not None:
            results["cleaning"] = "success"
            print("✓ Data cleaning complete")
        else:
            results["cleaning"] = "failed: no data returned"
            return results
    except Exception as e:
        print(f"✗ Data cleaning failed: {e}")
        results["cleaning"] = f"failed: {e}"
        return results

    # Stage 3: Exploratory Analysis
    print("\n" + "-" * 80)
    print("STAGE 3: EXPLORATORY ANALYSIS")
    print("-" * 80)
    try:
        eda_results = exploratory_analysis.main()
        results["exploratory"] = "success"
        print("✓ Exploratory analysis complete")
    except Exception as e:
        print(f"✗ Exploratory analysis failed: {e}")
        results["exploratory"] = f"failed: {e}"

    # Stage 4: Regression Analysis
    print("\n" + "-" * 80)
    print("STAGE 4: REGRESSION ANALYSIS")
    print("-" * 80)
    try:
        reg_results = regression_analysis.main()
        results["regression"] = "success"
        print("✓ Regression analysis complete")
    except Exception as e:
        print(f"✗ Regression analysis failed: {e}")
        results["regression"] = f"failed: {e}"

    # Stage 5: Visualization Outputs
    print("\n" + "-" * 80)
    print("STAGE 5: VISUALIZATION OUTPUTS")
    print("-" * 80)
    try:
        viz_outputs = visualization_outputs.main()
        results["visualization"] = "success"
        print("✓ Visualization outputs complete")
    except Exception as e:
        print(f"✗ Visualization outputs failed: {e}")
        results["visualization"] = f"failed: {e}"

    # Summary
    print("\n" + "=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)
    for stage, status in results.items():
        symbol = "✓" if status == "success" else ("○" if status == "skipped" else "✗")
        print(f"  {symbol} {stage}: {status}")
    print("=" * 80)

    return results


def run_stage(stage: str) -> dict:
    """
    Run a specific pipeline stage.

    Args:
        stage: Stage name ('ingestion', 'cleaning', 'exploratory',
               'regression', 'visualization')

    Returns:
        Dictionary with stage result
    """
    stages = {
        "ingestion": data_ingestion.main,
        "cleaning": data_cleaning.main,
        "exploratory": exploratory_analysis.main,
        "regression": regression_analysis.main,
        "visualization": visualization_outputs.main,
    }

    if stage not in stages:
        print(f"Unknown stage: {stage}")
        print(f"Available stages: {list(stages.keys())}")
        return {"error": f"Unknown stage: {stage}"}

    print(f"\nRunning stage: {stage}")
    print("-" * 40)

    try:
        result = stages[stage]()
        return {stage: "success", "result": result}
    except Exception as e:
        return {stage: f"failed: {e}"}


def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Run the AEI Analysis Pipeline"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["ingestion", "cleaning", "exploratory", "regression", "visualization"],
        help="Run a specific stage only"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip data ingestion (use existing data)"
    )

    args = parser.parse_args()

    if args.stage:
        results = run_stage(args.stage)
    else:
        results = run_full_pipeline(skip_download=args.skip_download)

    return results


if __name__ == "__main__":
    main()
