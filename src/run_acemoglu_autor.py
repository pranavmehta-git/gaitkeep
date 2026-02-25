"""
Run Acemoglu-Autor Toy Model Pipeline

Executes the full analysis pipeline:
1. Load/generate AEI v4 data
2. Prepare analysis dataset
3. Compute all three indices (ECI, PWAS, Pipeline Fragility)
4. Compute net exposure scores
5. Generate all visualizations
6. Export results

Usage:
    python -m src.run_acemoglu_autor [--synthetic] [--n-tasks N] [--seed S]

Or from project root:
    python src/run_acemoglu_autor.py
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running as script or module
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.acemoglu_autor_data import (
    load_aei_v4_data,
    prepare_analysis_dataset,
    save_analysis_data,
    get_data_summary,
    RESULTS_DIR,
)
from src.acemoglu_autor_models import (
    compute_combined_results,
    print_model_summary,
)
from src.acemoglu_autor_visualizations import (
    generate_all_visualizations,
)


def export_results(results: dict, summary: dict) -> None:
    """Export model results to CSV and JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save each result DataFrame
    for name, df in results.items():
        csv_path = RESULTS_DIR / f"{name}.csv"
        df.to_csv(csv_path, index=False)

    # Save summary JSON
    summary_path = RESULTS_DIR / "analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Save top-level findings
    findings = generate_findings(results)
    findings_path = RESULTS_DIR / "key_findings.json"
    with open(findings_path, "w") as f:
        json.dump(findings, f, indent=2, default=str)

    print(f"\nResults exported to {RESULTS_DIR}")


def generate_findings(results: dict) -> dict:
    """Generate structured key findings from model results."""
    eci = results["eci_occupation"]
    pwas = results["pwas_occupation"]
    frag = results["fragility_occupation"]
    net = results["net_exposure"]

    findings = {
        "model_a_eci": {
            "description": "Expertise Commodification Index",
            "mean_eci": float(eci["ECI"].mean()),
            "top_5_commodified": eci.nlargest(5, "ECI")[
                ["title", "ECI", "soc_major", "median_annual_wage"]
            ].to_dict("records"),
            "bottom_5_commodified": eci.nsmallest(5, "ECI")[
                ["title", "ECI", "soc_major", "median_annual_wage"]
            ].to_dict("records"),
            "eci_wage_correlation": float(eci["ECI"].corr(eci["median_annual_wage"])),
        },
        "model_b_pwas": {
            "description": "Pro-Worker AI Score",
            "mean_pwas": float(pwas["PWAS"].mean()),
            "top_5_pro_worker": pwas.nlargest(5, "PWAS")[
                ["title", "PWAS", "soc_major", "median_annual_wage"]
            ].to_dict("records"),
            "pwas_higher_for_blue_collar": bool(
                pwas[pwas["soc_major"].isin(["47", "49", "51", "53"])]["PWAS"].mean()
                > pwas[pwas["soc_major"].isin(["13", "15", "23"])]["PWAS"].mean()
            ),
        },
        "model_c_fragility": {
            "description": "Junior Pipeline Fragility Index",
            "mean_fragility": float(frag["pipeline_fragility"].mean()),
            "zone_distribution": frag["zone"].value_counts().to_dict(),
            "top_5_fragile": frag.nlargest(5, "pipeline_fragility")[
                ["title", "pipeline_fragility", "zone", "soc_major"]
            ].to_dict("records"),
            "most_fragile_soc_groups": (
                frag.groupby("soc_major")["pipeline_fragility"]
                .mean()
                .nlargest(5)
                .to_dict()
            ),
        },
        "net_exposure": {
            "description": "Net Exposure (ECI - PWAS)",
            "class_distribution": net["exposure_class"].value_counts().to_dict(),
            "most_exposed": net.nlargest(5, "net_exposure")[
                ["title", "net_exposure", "soc_major"]
            ].to_dict("records"),
            "most_protected": net.nsmallest(5, "net_exposure")[
                ["title", "net_exposure", "soc_major"]
            ].to_dict("records"),
        },
        "acemoglu_autor_predictions": {
            "prediction_1": {
                "claim": "High-ECI occupations should correlate with lower wages",
                "result": float(eci["ECI"].corr(eci["median_annual_wage"])),
                "note": "Positive/negative correlation with wage level",
            },
            "prediction_2": {
                "claim": "PWAS should be higher for blue-collar than white-collar",
                "blue_collar_pwas": float(
                    pwas[pwas["soc_major"].isin(["47", "49", "51", "53"])]["PWAS"].mean()
                ),
                "white_collar_pwas": float(
                    pwas[pwas["soc_major"].isin(["13", "15", "23"])]["PWAS"].mean()
                ),
            },
            "prediction_3": {
                "claim": "Pipeline fragility highest in professional services",
                "professional_fragility": float(
                    frag[frag["soc_major"].isin(["13", "23"])]["pipeline_fragility"].mean()
                ),
                "trades_fragility": float(
                    frag[frag["soc_major"].isin(["47", "49"])]["pipeline_fragility"].mean()
                ),
            },
        },
    }

    return findings


def run_pipeline(
    use_synthetic: bool = True,
    n_tasks: int = 5000,
    seed: int = 42,
    skip_viz: bool = False,
) -> dict:
    """
    Run the full Acemoglu-Autor toy model pipeline.

    Args:
        use_synthetic: Use synthetic data if real data unavailable.
        n_tasks: Number of synthetic task records.
        seed: Random seed.
        skip_viz: Skip visualization generation.

    Returns:
        Dictionary with all results.
    """
    print("=" * 70)
    print("ACEMOGLU-AUTOR x ANTHROPIC ECONOMIC INDEX: TOY MODEL PIPELINE")
    print("=" * 70)

    # Step 1: Load data
    print("\n[Step 1/5] Loading AEI v4 data...")
    raw_df = load_aei_v4_data(
        use_synthetic=use_synthetic,
        synthetic_n_tasks=n_tasks,
        synthetic_seed=seed,
    )

    # Step 2: Prepare dataset
    print("\n[Step 2/5] Preparing analysis dataset...")
    df = prepare_analysis_dataset(raw_df)
    save_analysis_data(df)

    # Data summary
    summary = get_data_summary(df)
    print(f"  Records: {summary['n_records']:,}")
    print(f"  Occupations: {summary['n_occupations']}")
    print(f"  SOC groups: {summary['n_soc_major_groups']}")
    print(f"  Collaboration modes:")
    for mode, val in summary["collaboration_modes"].items():
        print(f"    {mode}: {val:.3f}")

    # Step 3: Compute models
    print("\n[Step 3/5] Computing Acemoglu-Autor indices...")
    results = compute_combined_results(df)
    print_model_summary(results)

    # Step 4: Export results
    print("\n[Step 4/5] Exporting results...")
    export_results(results, summary)

    # Step 5: Visualizations
    if not skip_viz:
        print("\n[Step 5/5] Generating visualizations...")
        viz_paths = generate_all_visualizations(df, results)
    else:
        print("\n[Step 5/5] Skipping visualizations (--skip-viz)")
        viz_paths = {}

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nResults directory: {RESULTS_DIR}")
    if viz_paths:
        print(f"Figures: {len(viz_paths)} plots generated")
    print(f"CSVs: {len(results)} result tables exported")

    return {"data": df, "results": results, "summary": summary, "viz_paths": viz_paths}


def main():
    parser = argparse.ArgumentParser(
        description="Run Acemoglu-Autor x AEI Toy Model Pipeline"
    )
    parser.add_argument(
        "--synthetic", action="store_true", default=True,
        help="Use synthetic data if real AEI v4 data unavailable (default: True)",
    )
    parser.add_argument(
        "--n-tasks", type=int, default=5000,
        help="Number of synthetic task records (default: 5000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--skip-viz", action="store_true",
        help="Skip visualization generation",
    )

    args = parser.parse_args()
    run_pipeline(
        use_synthetic=args.synthetic,
        n_tasks=args.n_tasks,
        seed=args.seed,
        skip_viz=args.skip_viz,
    )


if __name__ == "__main__":
    main()
