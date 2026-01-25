"""
Exploratory Analysis Module

Uncovers inequality-related patterns in AI usage from the AEI dataset.
Includes distribution analysis, inequality metrics, and visualization.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"


def setup_output_directories():
    """Create output directories if they don't exist."""
    for directory in [RESULTS_DIR, FIGURES_DIR, TABLES_DIR]:
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
# INEQUALITY METRICS
# ============================================================================

def compute_gini_coefficient(values: np.ndarray) -> float:
    """
    Compute the Gini coefficient for a distribution.

    The Gini coefficient measures inequality, where:
    - 0 = perfect equality (everyone has the same)
    - 1 = perfect inequality (one person has everything)

    Args:
        values: Array of values (e.g., AI usage counts)

    Returns:
        Gini coefficient (0 to 1)
    """
    values = np.array(values).flatten()
    values = values[~np.isnan(values)]
    values = values[values >= 0]

    if len(values) == 0:
        return np.nan

    # Sort values
    sorted_values = np.sort(values)
    n = len(sorted_values)
    index = np.arange(1, n + 1)

    # Gini formula
    gini = (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n

    return gini


def compute_theil_index(values: np.ndarray) -> float:
    """
    Compute the Theil index (generalized entropy measure).

    The Theil index measures inequality and is decomposable
    into between-group and within-group components.

    Args:
        values: Array of values (e.g., AI usage counts)

    Returns:
        Theil index (0 = equality, higher = more inequality)
    """
    values = np.array(values).flatten()
    values = values[~np.isnan(values)]
    values = values[values > 0]  # Theil requires positive values

    if len(values) == 0:
        return np.nan

    mean_val = np.mean(values)
    n = len(values)

    # Theil index formula
    theil = np.sum((values / mean_val) * np.log(values / mean_val)) / n

    return theil


def compute_percentile_ratios(values: np.ndarray) -> Dict[str, float]:
    """
    Compute various percentile ratios (e.g., 90/10, 80/20).

    Args:
        values: Array of values

    Returns:
        Dictionary with percentile ratios
    """
    values = np.array(values).flatten()
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return {}

    p10 = np.percentile(values, 10)
    p20 = np.percentile(values, 20)
    p50 = np.percentile(values, 50)
    p80 = np.percentile(values, 80)
    p90 = np.percentile(values, 90)

    ratios = {
        "p90_p10": p90 / p10 if p10 > 0 else np.inf,
        "p80_p20": p80 / p20 if p20 > 0 else np.inf,
        "p90_p50": p90 / p50 if p50 > 0 else np.inf,
        "p50_p10": p50 / p10 if p10 > 0 else np.inf,
    }

    return ratios


def compute_inequality_metrics(df: pd.DataFrame,
                               value_column: str,
                               group_column: Optional[str] = None) -> Dict:
    """
    Compute comprehensive inequality metrics.

    Args:
        df: DataFrame with the data
        value_column: Column containing values to measure inequality
        group_column: Optional column for group-wise analysis

    Returns:
        Dictionary with inequality metrics
    """
    results = {}

    # Overall metrics
    values = df[value_column].dropna().values
    results["overall"] = {
        "gini": compute_gini_coefficient(values),
        "theil": compute_theil_index(values),
        "percentile_ratios": compute_percentile_ratios(values),
        "mean": np.mean(values),
        "median": np.median(values),
        "std": np.std(values),
        "cv": np.std(values) / np.mean(values) if np.mean(values) > 0 else np.nan,
    }

    # Group-wise metrics
    if group_column and group_column in df.columns:
        results["by_group"] = {}
        for group in df[group_column].unique():
            group_values = df[df[group_column] == group][value_column].dropna().values
            if len(group_values) > 0:
                results["by_group"][group] = {
                    "gini": compute_gini_coefficient(group_values),
                    "theil": compute_theil_index(group_values),
                    "mean": np.mean(group_values),
                    "median": np.median(group_values),
                    "n": len(group_values),
                }

    return results


# ============================================================================
# DISTRIBUTION ANALYSIS
# ============================================================================

def analyze_distribution_by_group(df: pd.DataFrame,
                                  value_column: str,
                                  group_column: str) -> pd.DataFrame:
    """
    Analyze value distribution across groups.

    Args:
        df: DataFrame with the data
        value_column: Column to analyze
        group_column: Column to group by

    Returns:
        Summary statistics by group
    """
    summary = df.groupby(group_column)[value_column].agg([
        'count',
        'mean',
        'median',
        'std',
        lambda x: x.quantile(0.25),
        lambda x: x.quantile(0.75),
        'min',
        'max'
    ]).round(4)

    summary.columns = ['count', 'mean', 'median', 'std', 'q25', 'q75', 'min', 'max']
    summary = summary.sort_values('mean', ascending=False)

    return summary


def analyze_usage_by_income_decile(df: pd.DataFrame,
                                   usage_column: str,
                                   income_decile_column: str = "income_decile") -> pd.DataFrame:
    """
    Analyze AI usage by income decile.

    Args:
        df: DataFrame with the data
        usage_column: Column containing usage metric
        income_decile_column: Column with income decile

    Returns:
        Usage statistics by income decile
    """
    if income_decile_column not in df.columns:
        print(f"Warning: '{income_decile_column}' not found. Creating deciles...")
        # Try to create deciles from numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df = df.copy()
            df[income_decile_column] = pd.qcut(
                df[numeric_cols[0]],
                q=10,
                labels=range(1, 11),
                duplicates='drop'
            )

    return analyze_distribution_by_group(df, usage_column, income_decile_column)


def analyze_success_rate(df: pd.DataFrame,
                         success_column: str,
                         group_columns: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Analyze AI task success rates across different groups.

    Args:
        df: DataFrame with the data
        success_column: Column containing success indicator
        group_columns: List of columns to group by

    Returns:
        Dictionary with success rate analysis per group
    """
    results = {}

    for group_col in group_columns:
        if group_col in df.columns:
            success_rates = df.groupby(group_col)[success_column].agg([
                'count',
                'mean',
                'std'
            ]).round(4)
            success_rates.columns = ['n_tasks', 'success_rate', 'std']
            success_rates = success_rates.sort_values('success_rate', ascending=False)
            results[group_col] = success_rates

    return results


def analyze_collaboration_modes(df: pd.DataFrame,
                                mode_column: str,
                                group_column: Optional[str] = None) -> pd.DataFrame:
    """
    Analyze collaboration mode distribution (automation vs augmentation).

    Args:
        df: DataFrame with the data
        mode_column: Column containing collaboration mode
        group_column: Optional column to group by

    Returns:
        Mode distribution analysis
    """
    if group_column and group_column in df.columns:
        # Cross-tabulation
        crosstab = pd.crosstab(
            df[group_column],
            df[mode_column],
            normalize='index'
        ).round(4)
        return crosstab
    else:
        # Overall distribution
        return df[mode_column].value_counts(normalize=True).round(4).to_frame()


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_distribution_histogram(df: pd.DataFrame,
                                column: str,
                                title: str = None,
                                save_path: Path = None):
    """Plot histogram of a distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(df[column].dropna(), bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')
    ax.set_title(title or f'Distribution of {column}')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {save_path}")

    plt.close()


def plot_boxplot_by_group(df: pd.DataFrame,
                          value_column: str,
                          group_column: str,
                          title: str = None,
                          save_path: Path = None):
    """Plot boxplot of values by group."""
    fig, ax = plt.subplots(figsize=(12, 6))

    df.boxplot(column=value_column, by=group_column, ax=ax)
    ax.set_xlabel(group_column)
    ax.set_ylabel(value_column)
    ax.set_title(title or f'{value_column} by {group_column}')
    plt.suptitle('')  # Remove automatic title

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {save_path}")

    plt.close()


def plot_gini_comparison(gini_by_group: Dict[str, float],
                         title: str = "Gini Coefficient by Group",
                         save_path: Path = None):
    """Plot bar chart comparing Gini coefficients."""
    fig, ax = plt.subplots(figsize=(10, 6))

    groups = list(gini_by_group.keys())
    values = list(gini_by_group.values())

    bars = ax.bar(groups, values, color='steelblue', edgecolor='black')
    ax.set_xlabel('Group')
    ax.set_ylabel('Gini Coefficient')
    ax.set_title(title)
    ax.axhline(y=np.mean(values), color='red', linestyle='--', label='Mean')
    ax.legend()

    # Rotate x labels if many groups
    if len(groups) > 5:
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {save_path}")

    plt.close()


def plot_lorenz_curve(values: np.ndarray,
                      title: str = "Lorenz Curve",
                      save_path: Path = None):
    """Plot Lorenz curve showing inequality."""
    values = np.array(values).flatten()
    values = values[~np.isnan(values)]
    values = np.sort(values)

    # Compute cumulative shares
    n = len(values)
    cum_pop = np.arange(1, n + 1) / n
    cum_wealth = np.cumsum(values) / np.sum(values)

    # Add origin point
    cum_pop = np.insert(cum_pop, 0, 0)
    cum_wealth = np.insert(cum_wealth, 0, 0)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot Lorenz curve
    ax.plot(cum_pop, cum_wealth, 'b-', linewidth=2, label='Lorenz Curve')

    # Plot line of equality
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Line of Equality')

    # Fill area between curves
    ax.fill_between(cum_pop, cum_wealth, cum_pop, alpha=0.2)

    ax.set_xlabel('Cumulative Share of Population')
    ax.set_ylabel('Cumulative Share of Value')
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {save_path}")

    plt.close()


def generate_all_visualizations(df: pd.DataFrame,
                                numeric_columns: List[str],
                                group_columns: List[str]):
    """Generate all standard visualizations."""
    setup_output_directories()

    for num_col in numeric_columns:
        if num_col in df.columns:
            # Distribution histogram
            plot_distribution_histogram(
                df, num_col,
                save_path=FIGURES_DIR / f"dist_{num_col}.png"
            )

            # Lorenz curve
            plot_lorenz_curve(
                df[num_col].dropna().values,
                title=f"Lorenz Curve: {num_col}",
                save_path=FIGURES_DIR / f"lorenz_{num_col}.png"
            )

            # Boxplots by group
            for grp_col in group_columns:
                if grp_col in df.columns:
                    plot_boxplot_by_group(
                        df, num_col, grp_col,
                        save_path=FIGURES_DIR / f"box_{num_col}_by_{grp_col}.png"
                    )


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run_exploratory_analysis(df: pd.DataFrame) -> Dict:
    """
    Run complete exploratory analysis.

    Args:
        df: Cleaned DataFrame

    Returns:
        Dictionary with all analysis results
    """
    setup_output_directories()
    results = {}

    print("Running Exploratory Analysis...")
    print("=" * 60)

    # Identify columns by type
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"Numeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")

    # 1. Basic statistics
    results["basic_stats"] = df.describe()
    print("\nBasic Statistics:")
    print(results["basic_stats"])

    # 2. Compute inequality metrics for numeric columns
    print("\nInequality Metrics:")
    results["inequality"] = {}
    for col in numeric_cols[:5]:  # Limit to first 5 for demo
        print(f"\n  {col}:")
        metrics = compute_inequality_metrics(df, col)
        results["inequality"][col] = metrics
        print(f"    Gini: {metrics['overall']['gini']:.4f}")
        print(f"    Theil: {metrics['overall']['theil']:.4f}")

    # 3. Distribution analysis by groups
    results["distributions"] = {}
    for num_col in numeric_cols[:3]:
        for cat_col in categorical_cols[:2]:
            if num_col in df.columns and cat_col in df.columns:
                key = f"{num_col}_by_{cat_col}"
                results["distributions"][key] = analyze_distribution_by_group(
                    df, num_col, cat_col
                )

    # 4. Save summary tables
    for name, table in results.get("distributions", {}).items():
        if isinstance(table, pd.DataFrame):
            table.to_csv(TABLES_DIR / f"{name}.csv")

    print("\n" + "=" * 60)
    print("Exploratory Analysis Complete!")
    print(f"Results saved to: {RESULTS_DIR}")

    return results


def main():
    """Main function to run exploratory analysis."""
    print("Starting AEI Exploratory Analysis")
    print("=" * 60)

    # Load cleaned data
    try:
        df = load_cleaned_data()
        print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

    # Run analysis
    results = run_exploratory_analysis(df)

    return results


if __name__ == "__main__":
    main()
