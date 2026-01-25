"""
Data Ingestion Module

Downloads and loads the Anthropic Economic Index (AEI) dataset
from Hugging Face and exports to various formats.

The AEI dataset is stored as individual CSV/JSON files in release folders,
not as a standard HuggingFace dataset.

Release structure (2025_09_15):
- data/intermediate/aei_raw_*.csv - Raw usage data
- data/output/aei_enriched_*.csv - Enriched data with metrics
- request_hierarchy_tree_*.json - Request cluster hierarchies
"""

import json
import os
from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd

try:
    from huggingface_hub import hf_hub_download, list_repo_files, HfApi
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: huggingface_hub not installed. Using direct download.")

import requests


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Dataset configuration
REPO_ID = "Anthropic/EconomicIndex"

# Available releases (newest first)
RELEASES = [
    "release_2025_09_15",  # Geographic + first-party API data
    "release_2025_03_27",  # Claude 3.7 Sonnet cluster-level data
    "release_2025_02_10",  # Initial release with O*NET mappings
]

DEFAULT_RELEASE = "release_2025_03_27"  # Most stable with cluster-level data

# File patterns for different releases
RELEASE_FILES = {
    "release_2025_03_27": {
        "cluster_level": "cluster_level_data/cluster_level_dataset.csv",
        "onet_tasks": "onet_task_statements.csv",
        "soc_structure": "SOC_Structure.csv",
        "task_pct_v1": "task_pct_v1.csv",
        "task_pct_v2": "task_pct_v2.csv",
        "automation_augmentation": "automation_vs_augmentation_by_task.csv",
    },
    "release_2025_02_10": {
        "cluster_level": "cluster_level_data/cluster_level_dataset.csv",
        "onet_tasks": "onet_task_statements.csv",
        "soc_structure": "SOC_Structure.csv",
        "automation_augmentation": "automation_vs_augmentation_by_task.csv",
    },
    "release_2025_09_15": {
        # This release has date-stamped files - we'll discover them dynamically
        "hierarchy_claude": "request_hierarchy_tree_claude_ai.json",
        "hierarchy_api": "request_hierarchy_tree_1p_api.json",
    },
}


def setup_directories() -> None:
    """Create necessary data directories if they don't exist."""
    for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Directory ready: {directory}")


def download_file_hf(filename: str, release: str = DEFAULT_RELEASE,
                     local_dir: Optional[Path] = None) -> Path:
    """
    Download a single file from the AEI repository using huggingface_hub.

    Args:
        filename: Path to file within the release folder
        release: Release version
        local_dir: Local directory to save file

    Returns:
        Path to downloaded file
    """
    if not HF_AVAILABLE:
        raise ImportError("huggingface_hub required for download")

    if local_dir is None:
        local_dir = RAW_DATA_DIR

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


def download_file_direct(filename: str, release: str = DEFAULT_RELEASE,
                         local_dir: Optional[Path] = None) -> Path:
    """
    Download a file directly via HTTPS from Hugging Face.

    Args:
        filename: Path to file within the release folder
        release: Release version
        local_dir: Local directory to save file

    Returns:
        Path to downloaded file
    """
    if local_dir is None:
        local_dir = RAW_DATA_DIR

    # Construct URL
    url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{release}/{filename}"

    # Create local path
    local_path = local_dir / release / filename
    local_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Downloading: {url}")
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        with open(local_path, 'wb') as f:
            f.write(response.content)

        print(f"Saved to: {local_path}")
        return local_path
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        raise


def download_file(filename: str, release: str = DEFAULT_RELEASE,
                  local_dir: Optional[Path] = None) -> Path:
    """
    Download a file from the AEI repository.

    Args:
        filename: Path to file within the release folder
        release: Release version
        local_dir: Local directory to save file

    Returns:
        Path to downloaded file
    """
    if HF_AVAILABLE:
        try:
            return download_file_hf(filename, release, local_dir)
        except Exception:
            print("Falling back to direct download...")

    return download_file_direct(filename, release, local_dir)


def list_available_releases() -> List[str]:
    """List available release versions."""
    return RELEASES


def get_release_files(release: str) -> Dict[str, str]:
    """Get the file mapping for a specific release."""
    return RELEASE_FILES.get(release, RELEASE_FILES[DEFAULT_RELEASE])


def load_csv_file(filename: str, release: str = DEFAULT_RELEASE) -> pd.DataFrame:
    """
    Load a CSV file from a release, downloading if necessary.

    Args:
        filename: Path to CSV within release folder
        release: Release version

    Returns:
        DataFrame with loaded data
    """
    local_path = RAW_DATA_DIR / release / filename

    if not local_path.exists():
        download_file(filename, release)

    df = pd.read_csv(local_path)
    print(f"Loaded {filename}: {len(df):,} rows, {len(df.columns)} columns")
    return df


def load_json_file(filename: str, release: str = DEFAULT_RELEASE) -> dict:
    """
    Load a JSON file from a release, downloading if necessary.

    Args:
        filename: Path to JSON within release folder
        release: Release version

    Returns:
        Dictionary with loaded data
    """
    local_path = RAW_DATA_DIR / release / filename

    if not local_path.exists():
        download_file(filename, release)

    with open(local_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {filename}")
    return data


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
    files = get_release_files(release)
    if "cluster_level" not in files:
        raise ValueError(f"No cluster_level data in {release}")

    return load_csv_file(files["cluster_level"], release)


def load_onet_tasks(release: str = DEFAULT_RELEASE) -> pd.DataFrame:
    """Load O*NET task statements."""
    files = get_release_files(release)
    if "onet_tasks" not in files:
        raise ValueError(f"No onet_tasks data in {release}")

    return load_csv_file(files["onet_tasks"], release)


def load_soc_structure(release: str = DEFAULT_RELEASE) -> pd.DataFrame:
    """Load Standard Occupational Classification structure."""
    files = get_release_files(release)
    if "soc_structure" not in files:
        raise ValueError(f"No soc_structure data in {release}")

    return load_csv_file(files["soc_structure"], release)


def load_automation_augmentation(release: str = DEFAULT_RELEASE) -> pd.DataFrame:
    """Load automation vs augmentation data by task."""
    files = get_release_files(release)
    if "automation_augmentation" not in files:
        raise ValueError(f"No automation_augmentation data in {release}")

    return load_csv_file(files["automation_augmentation"], release)


def load_task_percentages(version: str = "v2",
                          release: str = DEFAULT_RELEASE) -> pd.DataFrame:
    """Load task percentage data."""
    files = get_release_files(release)
    key = f"task_pct_{version}"
    if key not in files:
        raise ValueError(f"No {key} data in {release}")

    return load_csv_file(files[key], release)


def load_all_aei_data(release: str = DEFAULT_RELEASE) -> Dict[str, pd.DataFrame]:
    """
    Load all available AEI datasets into a dictionary.

    Args:
        release: Release version

    Returns:
        Dictionary with all DataFrames
    """
    print(f"\nLoading all AEI data from {release}...")
    print("=" * 60)

    data = {}
    files = get_release_files(release)

    for key, filename in files.items():
        try:
            if filename.endswith('.csv'):
                data[key] = load_csv_file(filename, release)
            elif filename.endswith('.json'):
                data[key] = load_json_file(filename, release)
        except Exception as e:
            print(f"Could not load {key}: {e}")

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
        # Return the largest available dataset
        if data:
            largest_key = max(data.keys(),
                            key=lambda k: len(data[k]) if isinstance(data[k], pd.DataFrame) else 0)
            if isinstance(data[largest_key], pd.DataFrame):
                return data[largest_key].copy()
        raise ValueError("No suitable data for merging")

    df = data["cluster_level"].copy()

    # Merge with automation/augmentation if available
    if "automation_augmentation" in data:
        aa_df = data["automation_augmentation"]
        common_cols = list(set(df.columns) & set(aa_df.columns))
        if common_cols:
            # Use the first common column as merge key
            merge_col = common_cols[0]
            df = df.merge(aa_df, on=merge_col, how="left", suffixes=("", "_aa"))

    print(f"Created merged dataset: {len(df):,} rows, {len(df.columns)} columns")
    return df


def export_to_csv(df: pd.DataFrame, filename: str = "aei.csv",
                  output_dir: Optional[Path] = None) -> Path:
    """Export DataFrame to CSV format."""
    if output_dir is None:
        output_dir = RAW_DATA_DIR

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    df.to_csv(output_path, index=False)
    print(f"Exported to CSV: {output_path}")
    return output_path


def export_to_parquet(df: pd.DataFrame, filename: str = "aei.parquet",
                      output_dir: Optional[Path] = None) -> Path:
    """Export DataFrame to Parquet format."""
    if output_dir is None:
        output_dir = RAW_DATA_DIR

    output_dir.mkdir(parents=True, exist_ok=True)
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
    for col in info['columns'][:15]:
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
    for r in list_available_releases():
        marker = " <-- using" if r == release else ""
        print(f"  - {r}{marker}")

    # Load all data
    data = load_all_aei_data(release)

    if not data:
        print("No data loaded!")
        return None

    # Print summaries for DataFrames
    for name, item in data.items():
        if isinstance(item, pd.DataFrame):
            print_dataset_summary(item, name)

    # Create and export merged dataset
    try:
        df_merged = create_merged_dataset(data)
        export_to_csv(df_merged)
        export_to_parquet(df_merged)
    except Exception as e:
        print(f"Could not create merged dataset: {e}")
        # Export the first available DataFrame
        for name, item in data.items():
            if isinstance(item, pd.DataFrame):
                export_to_csv(item)
                export_to_parquet(item)
                df_merged = item
                break

    print("\nData ingestion complete!")
    return df_merged


if __name__ == "__main__":
    main()
