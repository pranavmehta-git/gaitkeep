"""
Data Cleaning & Preprocessing Module

Prepares a tidy, analysis-ready dataframe from the raw AEI data.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


def load_raw_data(filename: str = "aei.parquet") -> pd.DataFrame:
    """Load raw data from parquet or CSV file."""
    parquet_path = RAW_DATA_DIR / filename
    csv_path = RAW_DATA_DIR / filename.replace(".parquet", ".csv")

    if parquet_path.exists():
        print(f"Loading from parquet: {parquet_path}")
        return pd.read_parquet(parquet_path)
    elif csv_path.exists():
        print(f"Loading from CSV: {csv_path}")
        return pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            f"No data file found. Run data_ingestion.py first."
        )


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to snake_case.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with standardized column names
    """
    df = df.copy()

    # Convert to lowercase and replace spaces/hyphens with underscores
    df.columns = (
        df.columns
        .str.lower()
        .str.replace(r'[- ]+', '_', regex=True)
        .str.replace(r'[^\w]', '', regex=True)
    )

    print(f"Standardized {len(df.columns)} column names")
    return df


def remove_unused_columns(df: pd.DataFrame,
                          columns_to_remove: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove columns not needed for analysis.

    Args:
        df: Input DataFrame
        columns_to_remove: List of column names to remove

    Returns:
        DataFrame with specified columns removed
    """
    df = df.copy()

    if columns_to_remove is None:
        # Default columns to remove (adjust based on actual dataset)
        columns_to_remove = []

    # Find which columns exist
    existing_cols = [c for c in columns_to_remove if c in df.columns]

    if existing_cols:
        df = df.drop(columns=existing_cols)
        print(f"Removed {len(existing_cols)} columns: {existing_cols}")

    return df


def handle_missing_values(df: pd.DataFrame,
                          strategy: str = "drop",
                          threshold: float = 0.5) -> pd.DataFrame:
    """
    Handle missing values in the dataset.

    Args:
        df: Input DataFrame
        strategy: "drop" to remove rows, "impute" to fill values
        threshold: For "drop", remove columns with > threshold missing

    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()
    original_rows = len(df)

    # First, drop columns with too many missing values
    missing_pct = df.isnull().sum() / len(df)
    cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"Dropped {len(cols_to_drop)} columns with >{threshold*100:.0f}% missing: {cols_to_drop}")

    if strategy == "drop":
        # Drop rows with any remaining missing values in key columns
        df = df.dropna()
        print(f"Dropped {original_rows - len(df):,} rows with missing values")

    elif strategy == "impute":
        # Impute numeric columns with median, categorical with mode
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype in ['float64', 'int64']:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown")
        print("Imputed missing values (numeric: median, categorical: mode)")

    return df


def convert_categorical_variables(df: pd.DataFrame,
                                  one_hot_columns: Optional[List[str]] = None,
                                  label_encode_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, dict]:
    """
    Convert categorical variables for analysis.

    Args:
        df: Input DataFrame
        one_hot_columns: Columns to one-hot encode
        label_encode_columns: Columns to label encode

    Returns:
        Tuple of (transformed DataFrame, encoding mappings)
    """
    df = df.copy()
    encodings = {}

    # One-hot encoding
    if one_hot_columns:
        existing_ohe = [c for c in one_hot_columns if c in df.columns]
        if existing_ohe:
            df = pd.get_dummies(df, columns=existing_ohe, prefix=existing_ohe)
            print(f"One-hot encoded: {existing_ohe}")

    # Label encoding (convert to numeric codes)
    if label_encode_columns:
        for col in label_encode_columns:
            if col in df.columns:
                df[col], mapping = pd.factorize(df[col])
                encodings[col] = dict(enumerate(mapping))
                print(f"Label encoded {col}: {len(mapping)} unique values")

    return df, encodings


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features for analysis.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with additional derived features
    """
    df = df.copy()

    # Log transformations for numeric columns (if they exist)
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if df[col].min() > 0:  # Only log-transform positive values
            df[f"log_{col}"] = np.log(df[col])

    # Add any additional derived features based on domain knowledge
    # Example: income deciles, task complexity scores, etc.

    print(f"Created derived features. Total columns: {len(df.columns)}")
    return df


def create_income_deciles(df: pd.DataFrame,
                          income_col: str = "income") -> pd.DataFrame:
    """
    Create income decile categories.

    Args:
        df: Input DataFrame
        income_col: Name of the income column

    Returns:
        DataFrame with income_decile column added
    """
    df = df.copy()

    if income_col in df.columns:
        df["income_decile"] = pd.qcut(
            df[income_col],
            q=10,
            labels=range(1, 11),
            duplicates='drop'
        )
        print(f"Created income deciles from '{income_col}'")
    else:
        print(f"Warning: '{income_col}' not found in dataset")

    return df


def validate_cleaned_data(df: pd.DataFrame) -> dict:
    """
    Validate the cleaned dataset.

    Args:
        df: Cleaned DataFrame

    Returns:
        Dictionary with validation results
    """
    validation = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "has_missing": df.isnull().any().any(),
        "missing_counts": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "memory_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
    }

    print("\n" + "=" * 60)
    print("CLEANED DATA VALIDATION")
    print("=" * 60)
    print(f"Rows: {validation['n_rows']:,}")
    print(f"Columns: {validation['n_columns']}")
    print(f"Has missing values: {validation['has_missing']}")
    print(f"Memory: {validation['memory_mb']:.2f} MB")
    print("=" * 60)

    return validation


def save_cleaned_data(df: pd.DataFrame,
                      filename: str = "aei_cleaned.parquet") -> Path:
    """Save cleaned data to processed directory."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / filename

    df.to_parquet(output_path, index=False)
    print(f"Saved cleaned data to: {output_path}")

    # Also save a CSV version for R users
    csv_path = PROCESSED_DATA_DIR / filename.replace(".parquet", ".csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV version to: {csv_path}")

    return output_path


def run_cleaning_pipeline(df: pd.DataFrame,
                          columns_to_remove: Optional[List[str]] = None,
                          one_hot_columns: Optional[List[str]] = None,
                          label_encode_columns: Optional[List[str]] = None,
                          missing_strategy: str = "drop") -> pd.DataFrame:
    """
    Run the complete data cleaning pipeline.

    Args:
        df: Raw DataFrame
        columns_to_remove: Columns to drop
        one_hot_columns: Columns to one-hot encode
        label_encode_columns: Columns to label encode
        missing_strategy: How to handle missing values

    Returns:
        Cleaned DataFrame
    """
    print("Starting data cleaning pipeline...")
    print("-" * 40)

    # Step 1: Standardize column names
    df = standardize_column_names(df)

    # Step 2: Remove unused columns
    df = remove_unused_columns(df, columns_to_remove)

    # Step 3: Handle missing values
    df = handle_missing_values(df, strategy=missing_strategy)

    # Step 4: Convert categorical variables
    df, encodings = convert_categorical_variables(
        df,
        one_hot_columns=one_hot_columns,
        label_encode_columns=label_encode_columns
    )

    # Step 5: Create derived features
    df = create_derived_features(df)

    # Validate
    validate_cleaned_data(df)

    return df


def main():
    """Main function to run data cleaning pipeline."""
    print("Starting AEI Data Cleaning Pipeline")
    print("=" * 60)

    # Load raw data
    try:
        df = load_raw_data()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run data_ingestion.py first.")
        return None

    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")

    # Run cleaning pipeline with default settings
    # Adjust these based on actual AEI dataset structure
    df_cleaned = run_cleaning_pipeline(
        df,
        columns_to_remove=None,  # Specify columns to remove
        one_hot_columns=None,    # Will be set based on actual data
        label_encode_columns=None,  # Will be set based on actual data
        missing_strategy="drop"
    )

    # Save cleaned data
    save_cleaned_data(df_cleaned)

    print("\nData cleaning complete!")
    return df_cleaned


if __name__ == "__main__":
    main()
