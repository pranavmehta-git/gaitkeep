"""
Data Ingestion Module

Downloads and loads the Anthropic Economic Index (AEI) dataset
from Hugging Face and exports to various formats.

The AEI dataset is stored as individual CSV files in release folders,
not as a standard HuggingFace dataset.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files, HfApi


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Dataset configuration
REPO_ID = "Anthropic/EconomicIndex"
DEFAULT_RELEASE = "release_2025_09_15"  # Most recent release with full data

# Key data files in the AEI dataset
AEI_DATA_FILES = {
    "cluster_level": "data/output/cluster_level_dataset.csv",
    "onet_tasks": "onet_task_statements.csv",
    "soc_structure": "SOC_Structure.csv",
    "task_pct_v1": "data/output/task_pct_v1.csv",
    "task_pct_v2": "data/output/task_pct_v2.csv",
    "automation_augmentation": "data/output/automation_vs_augmentation_by_task.csv",
}


def setup_directories() -> None:
    """Create necessary data directories if they don't exist."""
    for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Directory ready: {directory}")


def list_available_releases() -> List[str]:
    """
    List available release versions in the AEI repository.

    Returns:
        List of release folder names
    """
    try:
        files = list_repo_files(REPO_ID, repo_type="dataset")
        releases = set()
        for f in files:
            if f.startswith("release_"):
                release = f.split("/")[0]
                releases.add(release)
        return sorted(releases)
    except Exception as e:
        print(f"Error listing releases: {e}")
        return [DEFAULT_RELEASE]


def list_release_files(release: str = DEFAULT_RELEASE) -> List[str]:
    """
    List all files in a specific release.

    Args:
        release: Release folder name

    Returns:
        List of file paths within the release
    """
    try:
        all_files = list_repo_files(REPO_ID, repo_type="dataset")
        release_files = [f for f in all_files if f.startswith(f"{release}/")]
        return release_files
    except Exception as e:
        print(f"Error listing files: {e}")
        return []


def download_file(filename: str, release: str = DEFAULT_RELEASE,
                  local_dir: Optional[Path] = None) -> Path:
    """
    Download a single file from the AEI repository.

    Args:
        filename: Path to file within the release folder
        release: Release version
        local_dir: Local directory to save file

    Returns:
        Path to downloaded file
    """
    if local_dir is None:
        local_dir = RAW_DATA_DIR

    # Construct the full path in the repo
    repo_path = f"{release}/{filename}"

    try:
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=repo_path,
            repo_type="dataset",
            local_dir=local_dir,
        )
        print(f"Downloaded: {repo_path}")
        return Path(local_path)
    except Exception as e:
        print(f"Error downloading {repo_path}: {e}")
        raise


def download_all_data_files(release: str = DEFAULT_RELEASE) -> Dict[str, Path]:
    """
    Download all key data files from a release.

    Args:
        release: Release version to download

    Returns:
        Dictionary mapping file keys to local paths
    """
    downloaded = {}

    print(f"\nDownloading AEI data files from {release}...")
    print("-" * 50)

    for key, filename in AEI_DATA_FILES.items():
        try:
            path = download_file(filename, release)
            downloaded[key] = path
        except Exception as e:
            print(f"  Skipping {key}: {e}")

    return downloaded


def load_cluster_data(release: str = DEFAULT_RELEASE) -> pd.DataFrame:
    """
    Load the main cluster-level dataset.

    This is the primary dataset containing:
    - O*NET task mappings
    - Automation vs augmentation classifications
    - Usage statistics by occupation/task

    Args:
        release: Release version

    Returns:
        DataFrame with cluster-level data
    """
    # Try to load from local first
    local_path = RAW_DATA_DIR / release / "data" / "output" / "cluster_level_dataset.csv"

    if not local_path.exists():
        # Download if not present
        download_file("data/output/cluster_level_dataset.csv", release)

    df = pd.read_csv(local_path)
    print(f"Loaded cluster_level_dataset: {len(df):,} rows")
    return df


def load_task_percentages(version: str = "v2",
                          release: str = DEFAULT_RELEASE) -> pd.DataFrame:
    """
    Load task percentage data.

    Args:
        version: "v1" or "v2"
        release: Release version

    Returns:
        DataFrame with task percentages
    """
    filename = f"data/output/task_pct_{version}.csv"
    local_path = RAW_DATA_DIR / release / filename

    if not local_path.exists():
        download_file(filename, release)

    df = pd.read_csv(local_path)
    print(f"Loaded task_pct_{version}: {len(df):,} rows")
    return df


def load_automation_augmentation(release: str = DEFAULT_RELEASE) -> pd.DataFrame:
    """
    Load automation vs augmentation data by task.

    Args:
        release: Release version

    Returns:
        DataFrame with automation/augmentation classifications
    """
    filename = "data/output/automation_vs_augmentation_by_task.csv"
    local_path = RAW_DATA_DIR / release / filename

    if not local_path.exists():
        download_file(filename, release)

    df = pd.read_csv(local_path)
    print(f"Loaded automation_vs_augmentation: {len(df):,} rows")
    return df


def load_onet_tasks(release: str = DEFAULT_RELEASE) -> pd.DataFrame:
    """
    Load O*NET task statements.

    Args:
        release: Release version

    Returns:
        DataFrame with O*NET task definitions
    """
    local_path = RAW_DATA_DIR / release / "onet_task_statements.csv"

    if not local_path.exists():
        download_file("onet_task_statements.csv", release)

    df = pd.read_csv(local_path)
    print(f"Loaded onet_task_statements: {len(df):,} rows")
    return df


def load_soc_structure(release: str = DEFAULT_RELEASE) -> pd.DataFrame:
    """
    Load Standard Occupational Classification structure.

    Args:
        release: Release version

    Returns:
        DataFrame with SOC codes and titles
    """
    local_path = RAW_DATA_DIR / release / "SOC_Structure.csv"

    if not local_path.exists():
        download_file("SOC_Structure.csv", release)

    df = pd.read_csv(local_path)
    print(f"Loaded SOC_Structure: {len(df):,} rows")
    return df


def load_all_aei_data(release: str = DEFAULT_RELEASE) -> Dict[str, pd.DataFrame]:
    """
    Load all AEI datasets into a dictionary.

    Args:
        release: Release version

    Returns:
        Dictionary with all DataFrames
    """
    print(f"\nLoading all AEI data from {release}...")
    print("=" * 50)

    data = {}

    try:
        data["cluster_level"] = load_cluster_data(release)
    except Exception as e:
        print(f"Could not load cluster_level: {e}")

    try:
        data["task_pct"] = load_task_percentages("v2", release)
    except Exception as e:
        print(f"Could not load task_pct: {e}")

    try:
        data["automation_augmentation"] = load_automation_augmentation(release)
    except Exception as e:
        print(f"Could not load automation_augmentation: {e}")

    try:
        data["onet_tasks"] = load_onet_tasks(release)
    except Exception as e:
        print(f"Could not load onet_tasks: {e}")

    try:
        data["soc_structure"] = load_soc_structure(release)
    except Exception as e:
        print(f"Could not load soc_structure: {e}")

    return data


def create_merged_dataset(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Create a merged analysis-ready dataset from individual AEI files.

    Args:
        data: Dictionary of loaded DataFrames

    Returns:
        Merged DataFrame for analysis
    """
    # Start with cluster-level data as the base
    if "cluster_level" not in data:
        raise ValueError("cluster_level data required for merging")

    df = data["cluster_level"].copy()

    # Merge with automation/augmentation if available
    if "automation_augmentation" in data:
        aa_df = data["automation_augmentation"]
        # Find common merge columns
        common_cols = set(df.columns) & set(aa_df.columns)
        if common_cols:
            merge_col = list(common_cols)[0]
            df = df.merge(aa_df, on=merge_col, how="left", suffixes=("", "_aa"))

    print(f"Created merged dataset: {len(df):,} rows, {len(df.columns)} columns")
    return df


def export_to_csv(df: pd.DataFrame, filename: str = "aei.csv",
                  output_dir: Optional[Path] = None) -> Path:
    """Export DataFrame to CSV format."""
    if output_dir is None:
        output_dir = RAW_DATA_DIR

    output_path = output_dir / filename
    df.to_csv(output_path, index=False)
    print(f"Exported to CSV: {output_path}")
    return output_path


def export_to_parquet(df: pd.DataFrame, filename: str = "aei.parquet",
                      output_dir: Optional[Path] = None) -> Path:
    """Export DataFrame to Parquet format."""
    if output_dir is None:
        output_dir = RAW_DATA_DIR

    output_path = output_dir / filename
    df.to_parquet(output_path, index=False)
    print(f"Exported to Parquet: {output_path}")
    return output_path


def get_dataset_info(df: pd.DataFrame) -> dict:
    """Get basic information about a dataset."""
    info = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
        "missing_values": df.isnull().sum().to_dict(),
    }
    return info


def print_dataset_summary(df: pd.DataFrame, name: str = "Dataset") -> None:
    """Print a summary of a dataset."""
    info = get_dataset_info(df)

    print("\n" + "=" * 60)
    print(f"{name.upper()} SUMMARY")
    print("=" * 60)
    print(f"Total records: {info['n_rows']:,}")
    print(f"Total columns: {info['n_columns']}")
    print(f"Memory usage: {info['memory_usage_mb']:.2f} MB")

    print("\nColumns:")
    for col in info['columns'][:15]:  # Show first 15 columns
        dtype = info['dtypes'][col]
        missing = info['missing_values'][col]
        missing_pct = (missing / info['n_rows']) * 100 if info['n_rows'] > 0 else 0
        print(f"  - {col}: {dtype} (missing: {missing:,} / {missing_pct:.1f}%)")

    if len(info['columns']) > 15:
        print(f"  ... and {len(info['columns']) - 15} more columns")

    print("\nSample data:")
    print(df.head(3))
    print("=" * 60)


def main(release: str = DEFAULT_RELEASE):
    """Main function to run data ingestion pipeline."""
    print("Starting AEI Data Ingestion Pipeline")
    print("-" * 40)

    # Setup directories
    setup_directories()

    # List available releases
    print("\nAvailable releases:")
    releases = list_available_releases()
    for r in releases:
        marker = " <-- using" if r == release else ""
        print(f"  - {r}{marker}")

    # Load all data
    data = load_all_aei_data(release)

    if not data:
        print("No data loaded!")
        return None

    # Print summaries for each dataset
    for name, df in data.items():
        print_dataset_summary(df, name)

    # Create and export merged dataset
    try:
        df_merged = create_merged_dataset(data)
        export_to_csv(df_merged)
        export_to_parquet(df_merged)
    except Exception as e:
        print(f"Could not create merged dataset: {e}")
        # Export the main cluster-level data instead
        if "cluster_level" in data:
            export_to_csv(data["cluster_level"])
            export_to_parquet(data["cluster_level"])
            df_merged = data["cluster_level"]

    print("\nData ingestion complete!")
    return df_merged


if __name__ == "__main__":
    main()
