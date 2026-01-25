"""
Generate compact data for GitHub Pages dashboard embedding.
"""

import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DASHBOARD_DIR = PROJECT_ROOT / "results" / "dashboard"


def calculate_lorenz_curve(values, n_points=100):
    """Calculate Lorenz curve points from a list of values."""
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    total = np.sum(sorted_vals)

    cumulative = np.cumsum(sorted_vals)

    x_values = np.arange(1, n + 1) / n
    y_values = cumulative / total

    x_values = np.insert(x_values, 0, 0)
    y_values = np.insert(y_values, 0, 0)

    # Downsample to n_points for compact size
    indices = np.linspace(0, len(x_values) - 1, n_points, dtype=int)

    return [
        {"x": round(float(x_values[i]), 4), "y": round(float(y_values[i]), 4)}
        for i in indices
    ]


def calculate_gini(values):
    """Calculate Gini coefficient."""
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    cumulative = np.cumsum(sorted_vals)
    total = np.sum(sorted_vals)
    lorenz_y = cumulative / total
    area = np.trapezoid(lorenz_y, dx=1/n)
    return round(float(1 - 2 * area), 4)


def calculate_hhi(values):
    """Calculate Herfindahl-Hirschman Index."""
    total = np.sum(values)
    shares = values / total
    hhi = np.sum(shares ** 2) * 10000
    return round(float(hhi), 2)


def create_distribution_bins(values, n_bins=50):
    """Create histogram bins for the heat strip (compact version)."""
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    bin_size = n // n_bins

    bins = []
    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = start_idx + bin_size if i < n_bins - 1 else n
        bin_values = sorted_vals[start_idx:end_idx]
        bins.append({
            "m": round(float(np.mean(bin_values)), 1),  # mean
            "n": float(np.min(bin_values)),  # min
            "x": float(np.max(bin_values)),  # max
            "p": round(start_idx / n * 100, 0)  # percentile start
        })

    return bins


def main():
    """Generate compact dashboard data for embedding."""
    print("Generating compact dashboard data...")

    # Load the full dashboard data
    with open(DASHBOARD_DIR / "dashboard_data.json", "r") as f:
        data = json.load(f)

    # Extract incumbents_responding values
    incumbents = np.array([
        record["incumbents_responding"]
        for record in data
        if record.get("incumbents_responding") is not None
    ])

    print(f"Processing {len(incumbents):,} records")

    # Calculate all metrics
    lorenz_curve = calculate_lorenz_curve(incumbents, n_points=100)
    gini = calculate_gini(incumbents)
    hhi = calculate_hhi(incumbents)
    distribution_bins = create_distribution_bins(incumbents, n_bins=50)

    # Create compact data structure
    compact_data = {
        "meta": {
            "n": len(incumbents),
            "src": "AEI Dataset"
        },
        "reg": {
            "r2": 0.765,
            "nobs": 18872,
            "coef": {
                "b0": -24360,
                "b1": 28.8757,
                "b2": -3054.2842
            }
        },
        "stats": {
            "mean": round(float(np.mean(incumbents)), 1),
            "med": round(float(np.median(incumbents)), 0),
            "std": round(float(np.std(incumbents)), 1),
            "min": float(np.min(incumbents)),
            "max": float(np.max(incumbents)),
            "cv": round(float(np.std(incumbents) / np.mean(incumbents)), 3),
            "gini": gini,
            "hhi": hhi,
            "p10": float(np.percentile(incumbents, 10)),
            "p25": float(np.percentile(incumbents, 25)),
            "p50": float(np.percentile(incumbents, 50)),
            "p75": float(np.percentile(incumbents, 75)),
            "p90": float(np.percentile(incumbents, 90)),
            "p99": round(float(np.percentile(incumbents, 99)), 1)
        },
        "types": {
            "Core": {"n": 13487, "m": 75.8},
            "Supp": {"n": 5385, "m": 88.4}
        },
        "lorenz": lorenz_curve,
        "bins": distribution_bins
    }

    # Save compact data
    output_path = DASHBOARD_DIR / "data.json"
    with open(output_path, "w") as f:
        json.dump(compact_data, f, separators=(',', ':'))

    # Print size info
    import os
    size_kb = os.path.getsize(output_path) / 1024
    print(f"Saved compact data to: {output_path}")
    print(f"File size: {size_kb:.1f} KB")
    print(f"Gini: {gini}, HHI: {hhi}")

    # Also output as JS variable for embedding
    js_output = f"const DATA={json.dumps(compact_data, separators=(',', ':'))};"
    with open(DASHBOARD_DIR / "data.js", "w") as f:
        f.write(js_output)

    print(f"Saved JS data to: {DASHBOARD_DIR / 'data.js'}")

    return compact_data


if __name__ == "__main__":
    main()
