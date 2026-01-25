"""
Visualization Outputs Module

Structures data to feed into dashboards and story-based visuals.
Creates summarized tables and exports for visualization tools.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
DASHBOARD_DIR = RESULTS_DIR / "dashboard"


def setup_output_directories():
    """Create output directories if they don't exist."""
    for directory in [RESULTS_DIR, FIGURES_DIR, TABLES_DIR, DASHBOARD_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def load_cleaned_data(filename: str = "aei_cleaned.parquet") -> pd.DataFrame:
    """Load cleaned data from processed directory."""
    parquet_path = PROCESSED_DATA_DIR / filename
    csv_path = PROCESSED_DATA_DIR / filename.replace(".parquet", ".csv")

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    elif csv_path.exists():
        return pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            "No cleaned data found. Run data_cleaning.py first."
        )


# ============================================================================
# SUMMARY TABLE GENERATORS
# ============================================================================

def create_summary_by_group(df: pd.DataFrame,
                            group_column: str,
                            value_columns: List[str],
                            agg_functions: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Create summary statistics by group.

    Args:
        df: Input DataFrame
        group_column: Column to group by
        value_columns: Columns to aggregate
        agg_functions: Aggregation functions (default: count, mean, median, std)

    Returns:
        Summary DataFrame
    """
    if agg_functions is None:
        agg_functions = ['count', 'mean', 'median', 'std']

    # Filter to existing columns
    existing_value_cols = [c for c in value_columns if c in df.columns]

    if not existing_value_cols:
        return pd.DataFrame()

    summary = df.groupby(group_column)[existing_value_cols].agg(agg_functions)

    # Flatten multi-index columns
    summary.columns = ['_'.join(col).strip() for col in summary.columns]
    summary = summary.reset_index()

    return summary


def create_occupation_summary(df: pd.DataFrame,
                              occupation_col: str = "occupation_code",
                              value_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Create summary statistics by occupation group."""
    if value_columns is None:
        value_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    return create_summary_by_group(df, occupation_col, value_columns)


def create_region_summary(df: pd.DataFrame,
                          region_col: str = "geo",
                          value_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Create summary statistics by region/country."""
    if value_columns is None:
        value_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    return create_summary_by_group(df, region_col, value_columns)


def create_collaboration_summary(df: pd.DataFrame,
                                 collab_col: str = "collaboration_mode",
                                 value_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Create summary statistics by collaboration type."""
    if value_columns is None:
        value_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    return create_summary_by_group(df, collab_col, value_columns)


def create_task_success_bands(df: pd.DataFrame,
                              success_col: str = "task_success",
                              n_bands: int = 5) -> pd.DataFrame:
    """
    Create task success bands and summarize.

    Args:
        df: Input DataFrame
        success_col: Task success column
        n_bands: Number of bands to create

    Returns:
        Summary by success bands
    """
    df = df.copy()

    if success_col not in df.columns:
        return pd.DataFrame()

    # Create success bands
    df['success_band'] = pd.cut(
        df[success_col],
        bins=n_bands,
        labels=[f"Band_{i+1}" for i in range(n_bands)]
    )

    # Summarize by band
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != success_col]

    return create_summary_by_group(df, 'success_band', numeric_cols)


# ============================================================================
# DASHBOARD DATA EXPORTS
# ============================================================================

def export_for_dashboard(df: pd.DataFrame,
                         name: str,
                         format: str = "json") -> Path:
    """
    Export data in dashboard-friendly format.

    Args:
        df: DataFrame to export
        name: Output filename (without extension)
        format: Output format ('json', 'csv', 'both')

    Returns:
        Path to exported file
    """
    setup_output_directories()

    if format in ["json", "both"]:
        json_path = DASHBOARD_DIR / f"{name}.json"
        df.to_json(json_path, orient='records', indent=2)
        print(f"Exported to JSON: {json_path}")

    if format in ["csv", "both"]:
        csv_path = DASHBOARD_DIR / f"{name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Exported to CSV: {csv_path}")

    return DASHBOARD_DIR / name


def create_dashboard_metadata(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create metadata about the dataset for dashboard configuration.

    Args:
        df: Input DataFrame

    Returns:
        Metadata dictionary
    """
    metadata = {
        "total_records": len(df),
        "columns": {},
        "numeric_ranges": {},
        "categorical_values": {},
    }

    for col in df.columns:
        dtype = str(df[col].dtype)
        metadata["columns"][col] = dtype

        if df[col].dtype in ['float64', 'int64']:
            metadata["numeric_ranges"][col] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
            }
        elif df[col].dtype in ['object', 'category']:
            unique_vals = df[col].unique().tolist()
            if len(unique_vals) <= 50:  # Only include if not too many
                metadata["categorical_values"][col] = unique_vals

    return metadata


def export_regression_for_display(model_results: Dict,
                                  name: str = "regression_results") -> Path:
    """
    Export regression results in display-friendly format.

    Args:
        model_results: Dictionary with regression results
        name: Output filename

    Returns:
        Path to exported file
    """
    setup_output_directories()

    # Convert any numpy types to Python native types
    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    clean_results = convert_types(model_results)

    output_path = DASHBOARD_DIR / f"{name}.json"
    with open(output_path, 'w') as f:
        json.dump(clean_results, f, indent=2, default=str)

    print(f"Exported regression results to: {output_path}")
    return output_path


# ============================================================================
# VISUALIZATION GENERATION
# ============================================================================

def create_heatmap(df: pd.DataFrame,
                   title: str = "Correlation Heatmap",
                   save_path: Optional[Path] = None):
    """Create correlation heatmap."""
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty or len(numeric_df.columns) < 2:
        print("Not enough numeric columns for heatmap")
        return

    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0,
                fmt='.2f', square=True, ax=ax)
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap: {save_path}")

    plt.close()


def create_bar_chart(data: pd.DataFrame,
                     x_col: str,
                     y_col: str,
                     title: str = "Bar Chart",
                     save_path: Optional[Path] = None):
    """Create bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(data[x_col].astype(str), data[y_col], color='steelblue', edgecolor='black')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)

    # Rotate labels if many categories
    if len(data) > 5:
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved bar chart: {save_path}")

    plt.close()


def create_scatter_plot(df: pd.DataFrame,
                        x_col: str,
                        y_col: str,
                        hue_col: Optional[str] = None,
                        title: str = "Scatter Plot",
                        save_path: Optional[Path] = None):
    """Create scatter plot."""
    fig, ax = plt.subplots(figsize=(10, 8))

    if hue_col and hue_col in df.columns:
        for group in df[hue_col].unique():
            mask = df[hue_col] == group
            ax.scatter(df.loc[mask, x_col], df.loc[mask, y_col],
                       label=group, alpha=0.6)
        ax.legend()
    else:
        ax.scatter(df[x_col], df[y_col], alpha=0.6)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved scatter plot: {save_path}")

    plt.close()


def generate_summary_visualizations(df: pd.DataFrame):
    """Generate all summary visualizations."""
    setup_output_directories()

    # Correlation heatmap
    create_heatmap(df, "AEI Data Correlations",
                   FIGURES_DIR / "correlation_heatmap.png")

    # Distribution plots for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(10, 6))
        df[col].hist(bins=50, ax=ax, edgecolor='black', alpha=0.7)
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {col}')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"dist_{col}.png", dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Generated visualizations in: {FIGURES_DIR}")


# ============================================================================
# MAIN OUTPUT PIPELINE
# ============================================================================

def generate_all_outputs(df: pd.DataFrame) -> Dict[str, Path]:
    """
    Generate all outputs for visualization and dashboards.

    Args:
        df: Cleaned DataFrame

    Returns:
        Dictionary mapping output names to paths
    """
    setup_output_directories()
    outputs = {}

    print("Generating Visualization Outputs...")
    print("=" * 60)

    # Get column info for adaptive processing
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # 1. Summary tables by category
    for cat_col in categorical_cols[:5]:  # Limit to first 5
        print(f"\nCreating summary by {cat_col}...")
        summary = create_summary_by_group(df, cat_col, numeric_cols[:5])
        if not summary.empty:
            name = f"summary_by_{cat_col}"
            export_for_dashboard(summary, name, format="both")
            summary.to_csv(TABLES_DIR / f"{name}.csv", index=False)
            outputs[name] = TABLES_DIR / f"{name}.csv"

    # 2. Dashboard metadata
    print("\nCreating dashboard metadata...")
    metadata = create_dashboard_metadata(df)
    metadata_path = DASHBOARD_DIR / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    outputs["metadata"] = metadata_path

    # 3. Full dataset for dashboard (sampled if large)
    print("\nExporting dashboard data...")
    if len(df) > 10000:
        dashboard_df = df.sample(n=10000, random_state=42)
    else:
        dashboard_df = df
    export_for_dashboard(dashboard_df, "dashboard_data", format="both")
    outputs["dashboard_data"] = DASHBOARD_DIR / "dashboard_data.json"

    # 4. Generate visualizations
    print("\nGenerating visualizations...")
    generate_summary_visualizations(df)
    outputs["visualizations"] = FIGURES_DIR

    # 5. Overall statistics summary
    print("\nCreating overall statistics...")
    stats_summary = {
        "total_records": len(df),
        "columns": len(df.columns),
        "numeric_columns": len(numeric_cols),
        "categorical_columns": len(categorical_cols),
        "memory_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
    }

    # Add basic stats for numeric columns
    stats_summary["column_stats"] = {}
    for col in numeric_cols:
        stats_summary["column_stats"][col] = {
            "mean": float(df[col].mean()),
            "median": float(df[col].median()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
        }

    stats_path = DASHBOARD_DIR / "statistics_summary.json"
    with open(stats_path, 'w') as f:
        json.dump(stats_summary, f, indent=2)
    outputs["statistics"] = stats_path

    print("\n" + "=" * 60)
    print("Output Generation Complete!")
    print(f"\nOutputs saved to:")
    print(f"  - Figures: {FIGURES_DIR}")
    print(f"  - Tables: {TABLES_DIR}")
    print(f"  - Dashboard: {DASHBOARD_DIR}")

    return outputs


def main():
    """Main function to generate all outputs."""
    print("Starting AEI Visualization Output Generation")
    print("=" * 60)

    # Load cleaned data
    try:
        df = load_cleaned_data()
        print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

    # Generate all outputs
    outputs = generate_all_outputs(df)

    return outputs


if __name__ == "__main__":
    main()
