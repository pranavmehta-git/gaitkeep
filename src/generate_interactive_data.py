"""
Generate Interactive Data for the AEI Explorer Tool

Creates comprehensive aggregated data for interactive visualization,
supporting dimension selection (X/Y axes) and metric exploration.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
DASHBOARD_DIR = RESULTS_DIR / "dashboard"


def compute_gini(values: np.ndarray) -> float:
    """Compute Gini coefficient for a distribution."""
    values = np.array(values).flatten()
    values = values[~np.isnan(values)]
    values = values[values >= 0]
    if len(values) < 2:
        return np.nan
    sorted_values = np.sort(values)
    n = len(sorted_values)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n


def compute_percentiles(values: np.ndarray) -> Dict[str, float]:
    """Compute percentile values."""
    values = np.array(values).flatten()
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {}
    return {
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
    }


def compute_group_stats(df: pd.DataFrame, group_col: str, value_col: str = "incumbents_responding") -> List[Dict]:
    """Compute statistics for each group."""
    results = []
    for name, group in df.groupby(group_col, dropna=False):
        values = group[value_col].dropna().values
        if len(values) == 0:
            continue
        stats = {
            "name": str(name) if name is not None else "Unknown",
            "count": int(len(values)),
            "total": float(np.sum(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)) if len(values) > 1 else 0,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "gini": float(compute_gini(values)) if len(values) > 1 else 0,
            "cv": float(np.std(values) / np.mean(values)) if np.mean(values) > 0 else 0,
        }
        stats.update({f"pct_{k}": v for k, v in compute_percentiles(values).items()})
        results.append(stats)
    return sorted(results, key=lambda x: x["mean"], reverse=True)


def compute_cross_stats(df: pd.DataFrame, dim1: str, dim2: str, value_col: str = "incumbents_responding") -> List[Dict]:
    """Compute cross-dimensional statistics."""
    results = []
    for (d1, d2), group in df.groupby([dim1, dim2], dropna=False):
        values = group[value_col].dropna().values
        if len(values) == 0:
            continue
        results.append({
            "dim1": str(d1) if d1 is not None else "Unknown",
            "dim2": str(d2) if d2 is not None else "Unknown",
            "count": int(len(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "total": float(np.sum(values)),
            "gini": float(compute_gini(values)) if len(values) > 1 else 0,
        })
    return results


def compute_temporal_stats(df: pd.DataFrame, date_col: str = "date", value_col: str = "incumbents_responding") -> List[Dict]:
    """Compute temporal statistics with proper date parsing."""
    results = []

    # Parse dates and sort
    df_temp = df.copy()
    df_temp["parsed_date"] = pd.to_datetime(df_temp[date_col], format="%m/%Y", errors="coerce")
    df_temp = df_temp.dropna(subset=["parsed_date"])

    for date, group in df_temp.groupby("parsed_date"):
        values = group[value_col].dropna().values
        if len(values) == 0:
            continue

        # Task type breakdown
        core_count = len(group[group["task_type"] == "Core"])
        supp_count = len(group[group["task_type"] == "Supplemental"])

        results.append({
            "date": date.strftime("%Y-%m"),
            "year": int(date.year),
            "month": int(date.month),
            "count": int(len(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)) if len(values) > 1 else 0,
            "gini": float(compute_gini(values)) if len(values) > 1 else 0,
            "core_count": core_count,
            "supp_count": supp_count,
            "core_pct": core_count / len(values) if len(values) > 0 else 0,
        })

    return sorted(results, key=lambda x: x["date"])


def compute_occupation_categories(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Group occupations into meaningful categories based on SOC codes."""
    # Extract 2-digit major group from SOC code
    df_temp = df.copy()
    df_temp["soc_major"] = df_temp["onet_soc_code"].str[:2]

    # SOC major group mapping
    soc_groups = {
        "11": "Management",
        "13": "Business & Financial",
        "15": "Computer & Mathematical",
        "17": "Architecture & Engineering",
        "19": "Life, Physical & Social Science",
        "21": "Community & Social Service",
        "23": "Legal",
        "25": "Education & Library",
        "27": "Arts, Design, Entertainment",
        "29": "Healthcare Practitioners",
        "31": "Healthcare Support",
        "33": "Protective Service",
        "35": "Food Preparation",
        "37": "Building & Grounds",
        "39": "Personal Care",
        "41": "Sales",
        "43": "Office & Administrative",
        "45": "Farming, Fishing, Forestry",
        "47": "Construction",
        "49": "Installation & Maintenance",
        "51": "Production",
        "53": "Transportation",
    }

    categories = {}
    for soc_code, group_name in soc_groups.items():
        titles = df_temp[df_temp["soc_major"] == soc_code]["title"].unique().tolist()
        if titles:
            categories[group_name] = sorted(titles)

    return categories


def load_data() -> pd.DataFrame:
    """Load the cleaned AEI data from available sources."""
    # Try processed data first
    parquet_path = PROCESSED_DATA_DIR / "aei_cleaned.parquet"
    csv_path = PROCESSED_DATA_DIR / "aei_cleaned.csv"

    # Fall back to dashboard data (JSON)
    dashboard_json = DASHBOARD_DIR / "dashboard_data.json"

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    elif csv_path.exists():
        return pd.read_csv(csv_path)
    elif dashboard_json.exists():
        print(f"Loading from dashboard JSON: {dashboard_json}")
        return pd.read_json(dashboard_json)
    else:
        raise FileNotFoundError("No data found. Run data pipeline first.")


def main():
    """Generate all interactive data files."""
    print("Generating Interactive Data for AEI Explorer...")
    print("=" * 60)

    # Ensure output directory exists
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_data()
    print(f"Loaded {len(df):,} records with {len(df.columns)} columns")

    # 1. Overall statistics
    print("\n1. Computing overall statistics...")
    values = df["incumbents_responding"].dropna().values
    overall_stats = {
        "total_records": int(len(df)),
        "unique_occupations": int(df["title"].nunique()),
        "unique_tasks": int(df["task"].nunique()) if "task" in df.columns else 0,
        "date_range": {
            "min": df["date"].min() if "date" in df.columns else None,
            "max": df["date"].max() if "date" in df.columns else None,
        },
        "incumbents": {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "gini": float(compute_gini(values)),
            "cv": float(np.std(values) / np.mean(values)),
        },
        "task_types": {
            "Core": int((df["task_type"] == "Core").sum()),
            "Supplemental": int((df["task_type"] == "Supplemental").sum()),
        }
    }
    overall_stats["incumbents"].update(compute_percentiles(values))

    # 2. By occupation (title)
    print("2. Computing occupation statistics...")
    by_occupation = compute_group_stats(df, "title", "incumbents_responding")

    # 3. By task type
    print("3. Computing task type statistics...")
    by_task_type = compute_group_stats(df, "task_type", "incumbents_responding")

    # 4. Temporal statistics
    print("4. Computing temporal statistics...")
    temporal = compute_temporal_stats(df, "date", "incumbents_responding")

    # 5. Cross-dimensional: occupation x task_type
    print("5. Computing occupation × task_type statistics...")
    occupation_x_tasktype = compute_cross_stats(df, "title", "task_type", "incumbents_responding")

    # 6. Occupation categories
    print("6. Building occupation categories...")
    occupation_categories = compute_occupation_categories(df)

    # 7. By SOC major group
    print("7. Computing SOC major group statistics...")
    df_temp = df.copy()
    df_temp["soc_major"] = df_temp["onet_soc_code"].str[:2]
    soc_groups = {
        "11": "Management", "13": "Business & Financial", "15": "Computer & Mathematical",
        "17": "Architecture & Engineering", "19": "Life, Physical & Social Science",
        "21": "Community & Social Service", "23": "Legal", "25": "Education & Library",
        "27": "Arts, Design, Entertainment", "29": "Healthcare Practitioners",
        "31": "Healthcare Support", "33": "Protective Service", "35": "Food Preparation",
        "37": "Building & Grounds", "39": "Personal Care", "41": "Sales",
        "43": "Office & Administrative", "45": "Farming, Fishing, Forestry",
        "47": "Construction", "49": "Installation & Maintenance", "51": "Production",
        "53": "Transportation",
    }
    df_temp["soc_group_name"] = df_temp["soc_major"].map(soc_groups).fillna("Other")
    by_soc_group = compute_group_stats(df_temp, "soc_group_name", "incumbents_responding")

    # 8. Gini by occupation (for inequality comparison)
    print("8. Computing Gini by occupation...")
    gini_by_occupation = []
    for title, group in df.groupby("title"):
        values = group["incumbents_responding"].dropna().values
        if len(values) >= 5:  # Need enough data points for meaningful Gini
            gini_by_occupation.append({
                "name": str(title),
                "gini": float(compute_gini(values)),
                "count": int(len(values)),
                "mean": float(np.mean(values)),
            })
    gini_by_occupation = sorted(gini_by_occupation, key=lambda x: x["gini"], reverse=True)

    # 9. Yearly aggregates for cleaner temporal view
    print("9. Computing yearly aggregates...")
    yearly = []
    for year_data in temporal:
        year = year_data["year"]
        existing = next((y for y in yearly if y["year"] == year), None)
        if existing:
            # Aggregate
            existing["count"] += year_data["count"]
            existing["months"].append(year_data)
        else:
            yearly.append({
                "year": year,
                "count": year_data["count"],
                "months": [year_data]
            })

    # Compute yearly averages
    for y in yearly:
        y["mean"] = np.mean([m["mean"] for m in y["months"]])
        y["gini"] = np.mean([m["gini"] for m in y["months"]])
        y["core_pct"] = np.mean([m["core_pct"] for m in y["months"]])
        del y["months"]  # Remove detailed monthly data for compactness
    yearly = sorted(yearly, key=lambda x: x["year"])

    # 10. Sample raw data for scatter plots (limit to prevent huge files)
    print("10. Sampling raw data for scatter plots...")
    sample_size = min(2000, len(df))
    raw_sample = df.sample(n=sample_size, random_state=42)[
        ["title", "task_type", "incumbents_responding", "date", "onet_soc_code"]
    ].to_dict(orient="records")

    # Compile interactive data
    interactive_data = {
        "meta": {
            "generated": pd.Timestamp.now().isoformat(),
            "source": "Anthropic Economic Index (AEI)",
            "version": "2.0-interactive",
        },
        "overall": overall_stats,
        "dimensions": {
            "occupation": by_occupation[:100],  # Top 100 by mean
            "task_type": by_task_type,
            "soc_group": by_soc_group,
            "temporal": temporal,
            "yearly": yearly,
        },
        "cross": {
            "occupation_x_tasktype": occupation_x_tasktype[:500],  # Limit size
        },
        "inequality": {
            "gini_by_occupation": gini_by_occupation[:100],  # Top 100 by Gini
        },
        "categories": occupation_categories,
        "sample": raw_sample,
    }

    # Save interactive data
    output_path = DASHBOARD_DIR / "interactive_data.json"
    with open(output_path, "w") as f:
        json.dump(interactive_data, f, indent=2, default=str)

    file_size = output_path.stat().st_size / 1024
    print(f"\nGenerated: {output_path} ({file_size:.1f} KB)")

    # Also save a compact version for quick loading
    compact_data = {
        "meta": interactive_data["meta"],
        "overall": overall_stats,
        "occupation": by_occupation[:50],
        "task_type": by_task_type,
        "soc_group": by_soc_group,
        "yearly": yearly,
        "gini_top": gini_by_occupation[:20],
    }

    compact_path = DASHBOARD_DIR / "explorer_data.json"
    with open(compact_path, "w") as f:
        json.dump(compact_data, f, default=str)

    compact_size = compact_path.stat().st_size / 1024
    print(f"Generated: {compact_path} ({compact_size:.1f} KB)")

    print("\n" + "=" * 60)
    print("Interactive data generation complete!")

    return interactive_data


if __name__ == "__main__":
    main()
