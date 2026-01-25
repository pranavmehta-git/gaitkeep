"""
Generate optimized data for the Competition Inequality Dashboard.
"""

import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DASHBOARD_DIR = PROJECT_ROOT / "results" / "dashboard"


def calculate_lorenz_curve(values):
    """Calculate Lorenz curve points from a list of values."""
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    total = np.sum(sorted_vals)

    # Create cumulative sums
    cumulative = np.cumsum(sorted_vals)

    # Normalize to percentages
    x_values = np.arange(1, n + 1) / n
    y_values = cumulative / total

    # Add origin point
    x_values = np.insert(x_values, 0, 0)
    y_values = np.insert(y_values, 0, 0)

    # Downsample to ~200 points for smoother rendering
    indices = np.linspace(0, len(x_values) - 1, 200, dtype=int)

    return [
        {"x": round(float(x_values[i]), 4), "y": round(float(y_values[i]), 4)}
        for i in indices
    ]


def calculate_gini(values):
    """Calculate Gini coefficient."""
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    cumulative = np.cumsum(sorted_vals)

    # Gini = 1 - 2 * area under Lorenz curve
    # Area under Lorenz = sum of trapezoids
    total = np.sum(sorted_vals)
    lorenz_y = cumulative / total

    # Trapezoidal integration
    area = np.trapezoid(lorenz_y, dx=1/n)
    gini = 1 - 2 * area

    return round(float(gini), 4)


def calculate_hhi(values):
    """Calculate Herfindahl-Hirschman Index."""
    total = np.sum(values)
    shares = values / total
    hhi = np.sum(shares ** 2) * 10000
    return round(float(hhi), 2)


def calculate_percentiles(values):
    """Calculate key percentiles."""
    return {
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99))
    }


def create_distribution_bins(values, n_bins=100):
    """Create histogram bins for the heat strip."""
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    bin_size = n // n_bins

    bins = []
    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = start_idx + bin_size if i < n_bins - 1 else n
        bin_values = sorted_vals[start_idx:end_idx]
        bins.append({
            "bin": i,
            "mean": round(float(np.mean(bin_values)), 2),
            "min": float(np.min(bin_values)),
            "max": float(np.max(bin_values)),
            "count": len(bin_values),
            "percentile_start": round(start_idx / n * 100, 1),
            "percentile_end": round(end_idx / n * 100, 1)
        })

    return bins


def main():
    """Generate optimized dashboard data."""
    print("Generating optimized dashboard data...")

    # Load the full dashboard data
    with open(DASHBOARD_DIR / "dashboard_data.json", "r") as f:
        data = json.load(f)

    # Extract incumbents_responding values
    incumbents = np.array([
        record["incumbents_responding"]
        for record in data
        if record.get("incumbents_responding") is not None
    ])

    print(f"Loaded {len(incumbents):,} records")

    # Calculate all metrics
    lorenz_curve = calculate_lorenz_curve(incumbents)
    gini = calculate_gini(incumbents)
    hhi = calculate_hhi(incumbents)
    percentiles = calculate_percentiles(incumbents)
    distribution_bins = create_distribution_bins(incumbents)

    # Create optimized data structure
    optimized_data = {
        "metadata": {
            "n_observations": len(incumbents),
            "generated_at": "2026-01-25",
            "source": "Platform Task Competition Study (AEI Dataset)",
            "description": "Optimized data for Competition Inequality Dashboard"
        },
        "regression": {
            "r_squared": 0.765,
            "adj_r_squared": 0.765,
            "n_obs": 18872,
            "coefficients": {
                "intercept": -24360,
                "incumbents_responding": 28.8757,
                "log_task_id": 5065.2238,
                "log_incumbents_responding": -3054.2842
            }
        },
        "summary_stats": {
            "n": len(incumbents),
            "mean": round(float(np.mean(incumbents)), 2),
            "median": round(float(np.median(incumbents)), 2),
            "std": round(float(np.std(incumbents)), 2),
            "min": float(np.min(incumbents)),
            "max": float(np.max(incumbents)),
            "cv": round(float(np.std(incumbents) / np.mean(incumbents)), 3),
            "gini": gini,
            "hhi": hhi,
            "percentiles": percentiles
        },
        "task_type_breakdown": {
            "Core": {
                "count": 13487,
                "mean": 75.83,
                "median": 77.0,
                "std": 39.64
            },
            "Supplemental": {
                "count": 5385,
                "mean": 88.44,
                "median": 87.0,
                "std": 37.50
            }
        },
        "lorenz_curve": lorenz_curve,
        "distribution_bins": distribution_bins,
        "raw_distribution": sorted(incumbents.tolist())
    }

    # Save optimized data
    output_path = DASHBOARD_DIR / "competition_optimized.json"
    with open(output_path, "w") as f:
        json.dump(optimized_data, f, indent=2)

    print(f"Saved optimized data to: {output_path}")
    print(f"Gini coefficient: {gini}")
    print(f"HHI: {hhi}")
    print(f"Mean: {optimized_data['summary_stats']['mean']}")
    print(f"Median: {optimized_data['summary_stats']['median']}")

    return optimized_data


if __name__ == "__main__":
    main()
