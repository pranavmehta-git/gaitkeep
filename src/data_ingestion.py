"""
Data Ingestion Module

Downloads and loads the Anthropic Economic Index (AEI) dataset
from Hugging Face and exports to various formats.
"""

import os
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


def setup_directories() -> None:
    """Create necessary data directories if they don't exist."""
    for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Directory ready: {directory}")


def load_aei_dataset(split: str = "train") -> pd.DataFrame:
    """
    Load the AEI dataset from Hugging Face.

    Args:
        split: Dataset split to load (default: "train")

    Returns:
        DataFrame containing the AEI data
    """
    print(f"Loading AEI dataset (split: {split}) from Hugging Face...")

    try:
        aei = load_dataset("Anthropic/EconomicIndex", split=split)
        df = aei.to_pandas()
        print(f"Successfully loaded {len(df):,} records")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise


def export_to_csv(df: pd.DataFrame, filename: str = "aei.csv",
                  output_dir: Optional[Path] = None) -> Path:
    """
    Export DataFrame to CSV format.

    Args:
        df: DataFrame to export
        filename: Output filename
        output_dir: Output directory (default: raw data dir)

    Returns:
        Path to the exported file
    """
    if output_dir is None:
        output_dir = RAW_DATA_DIR

    output_path = output_dir / filename
    df.to_csv(output_path, index=False)
    print(f"Exported to CSV: {output_path}")
    return output_path


def export_to_parquet(df: pd.DataFrame, filename: str = "aei.parquet",
                      output_dir: Optional[Path] = None) -> Path:
    """
    Export DataFrame to Parquet format (more efficient for large datasets).

    Args:
        df: DataFrame to export
        filename: Output filename
        output_dir: Output directory (default: raw data dir)

    Returns:
        Path to the exported file
    """
    if output_dir is None:
        output_dir = RAW_DATA_DIR

    output_path = output_dir / filename
    df.to_parquet(output_path, index=False)
    print(f"Exported to Parquet: {output_path}")
    return output_path


def get_dataset_info(df: pd.DataFrame) -> dict:
    """
    Get basic information about the dataset.

    Args:
        df: DataFrame to analyze

    Returns:
        Dictionary with dataset metadata
    """
    info = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
        "missing_values": df.isnull().sum().to_dict(),
    }
    return info


def print_dataset_summary(df: pd.DataFrame) -> None:
    """Print a summary of the dataset."""
    info = get_dataset_info(df)

    print("\n" + "=" * 60)
    print("AEI DATASET SUMMARY")
    print("=" * 60)
    print(f"Total records: {info['n_rows']:,}")
    print(f"Total columns: {info['n_columns']}")
    print(f"Memory usage: {info['memory_usage_mb']:.2f} MB")

    print("\nColumns:")
    for col in info['columns']:
        dtype = info['dtypes'][col]
        missing = info['missing_values'][col]
        missing_pct = (missing / info['n_rows']) * 100
        print(f"  - {col}: {dtype} (missing: {missing:,} / {missing_pct:.1f}%)")

    print("\nSample data:")
    print(df.head())
    print("=" * 60)


def main():
    """Main function to run data ingestion pipeline."""
    print("Starting AEI Data Ingestion Pipeline")
    print("-" * 40)

    # Setup directories
    setup_directories()

    # Load dataset
    df = load_aei_dataset()

    # Print summary
    print_dataset_summary(df)

    # Export to both formats
    export_to_csv(df)
    export_to_parquet(df)

    print("\nData ingestion complete!")
    return df


if __name__ == "__main__":
    main()
