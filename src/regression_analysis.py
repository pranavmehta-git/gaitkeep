"""
Regression & Correlation Analysis Module

Quantifies relationships between AI adoption and inequality proxies.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"


def setup_output_directories():
    """Create output directories if they don't exist."""
    for directory in [RESULTS_DIR, TABLES_DIR]:
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
# DATA PREPARATION FOR REGRESSION
# ============================================================================

def prepare_regression_data(df: pd.DataFrame,
                            target: str,
                            features: List[str],
                            drop_na: bool = True,
                            standardize: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for regression analysis.

    Args:
        df: Input DataFrame
        target: Target variable name
        features: List of feature names
        drop_na: Whether to drop rows with missing values
        standardize: Whether to standardize numeric features

    Returns:
        Tuple of (X features DataFrame, y target Series)
    """
    # Select columns
    all_cols = [target] + features
    data = df[all_cols].copy()

    # Drop missing values
    if drop_na:
        data = data.dropna()

    # Separate target and features
    y = data[target]
    X = data[features]

    # Handle categorical variables
    for col in X.columns:
        if X[col].dtype == 'object':
            # One-hot encode
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
            X = pd.concat([X.drop(columns=[col]), dummies], axis=1)

    # Standardize if requested
    if standardize:
        scaler = StandardScaler()
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    return X, y


def create_log_transform(df: pd.DataFrame,
                         columns: List[str],
                         add_constant: float = 1.0) -> pd.DataFrame:
    """
    Create log-transformed versions of columns.

    Args:
        df: Input DataFrame
        columns: Columns to transform
        add_constant: Constant to add before log (to handle zeros)

    Returns:
        DataFrame with log-transformed columns added
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[f"log_{col}"] = np.log(df[col] + add_constant)
    return df


# ============================================================================
# OLS REGRESSION
# ============================================================================

def run_ols_regression(X: pd.DataFrame,
                       y: pd.Series,
                       add_constant: bool = True) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Run OLS regression.

    Args:
        X: Features DataFrame
        y: Target Series
        add_constant: Whether to add intercept

    Returns:
        Fitted OLS model
    """
    if add_constant:
        X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    return model


def run_robust_regression(X: pd.DataFrame,
                          y: pd.Series,
                          add_constant: bool = True,
                          cov_type: str = "HC3") -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Run OLS with robust standard errors.

    Args:
        X: Features DataFrame
        y: Target Series
        add_constant: Whether to add intercept
        cov_type: Type of robust covariance (HC0, HC1, HC2, HC3)

    Returns:
        Fitted OLS model with robust standard errors
    """
    if add_constant:
        X = sm.add_constant(X)

    model = sm.OLS(y, X).fit(cov_type=cov_type)
    return model


def format_regression_results(model: sm.regression.linear_model.RegressionResultsWrapper) -> Dict:
    """
    Format regression results into a dictionary.

    Args:
        model: Fitted statsmodels regression model

    Returns:
        Dictionary with formatted results
    """
    results = {
        "n_obs": int(model.nobs),
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "f_statistic": model.fvalue,
        "f_pvalue": model.f_pvalue,
        "aic": model.aic,
        "bic": model.bic,
        "coefficients": {},
    }

    # Extract coefficient details
    for var in model.params.index:
        results["coefficients"][var] = {
            "coef": model.params[var],
            "std_err": model.bse[var],
            "t_stat": model.tvalues[var],
            "p_value": model.pvalues[var],
            "ci_lower": model.conf_int().loc[var, 0],
            "ci_upper": model.conf_int().loc[var, 1],
        }

    return results


def print_regression_summary(model: sm.regression.linear_model.RegressionResultsWrapper,
                             title: str = "Regression Results"):
    """Print formatted regression summary."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(model.summary())
    print("=" * 80)


def save_regression_results(model: sm.regression.linear_model.RegressionResultsWrapper,
                            filename: str,
                            output_dir: Path = None):
    """Save regression results to files."""
    if output_dir is None:
        output_dir = TABLES_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary as text
    with open(output_dir / f"{filename}_summary.txt", 'w') as f:
        f.write(model.summary().as_text())

    # Save coefficients as CSV
    coef_df = pd.DataFrame({
        'coefficient': model.params,
        'std_error': model.bse,
        't_statistic': model.tvalues,
        'p_value': model.pvalues,
        'ci_lower': model.conf_int()[0],
        'ci_upper': model.conf_int()[1],
    })
    coef_df.to_csv(output_dir / f"{filename}_coefficients.csv")

    print(f"Saved regression results to: {output_dir / filename}_*")


# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

def compute_correlation_matrix(df: pd.DataFrame,
                               columns: Optional[List[str]] = None,
                               method: str = "pearson") -> pd.DataFrame:
    """
    Compute correlation matrix.

    Args:
        df: Input DataFrame
        columns: Columns to include (default: all numeric)
        method: Correlation method ('pearson', 'spearman', 'kendall')

    Returns:
        Correlation matrix as DataFrame
    """
    if columns is None:
        numeric_df = df.select_dtypes(include=[np.number])
    else:
        numeric_df = df[columns].select_dtypes(include=[np.number])

    return numeric_df.corr(method=method)


def compute_partial_correlations(df: pd.DataFrame,
                                 var1: str,
                                 var2: str,
                                 controls: List[str]) -> Tuple[float, float]:
    """
    Compute partial correlation between two variables controlling for others.

    Args:
        df: Input DataFrame
        var1: First variable
        var2: Second variable
        controls: Variables to control for

    Returns:
        Tuple of (partial correlation, p-value)
    """
    # Prepare data
    all_vars = [var1, var2] + controls
    data = df[all_vars].dropna()

    # Regress both variables on controls
    X_controls = sm.add_constant(data[controls])

    resid1 = sm.OLS(data[var1], X_controls).fit().resid
    resid2 = sm.OLS(data[var2], X_controls).fit().resid

    # Correlate residuals
    corr, pval = stats.pearsonr(resid1, resid2)

    return corr, pval


# ============================================================================
# EXAMPLE REGRESSION SPECIFICATIONS
# ============================================================================

def run_usage_regression(df: pd.DataFrame,
                         usage_col: str = "usage_intensity",
                         income_col: str = "income",
                         education_col: str = "education_level",
                         sector_col: str = "sector") -> Optional[sm.regression.linear_model.RegressionResultsWrapper]:
    """
    Run AI usage ~ income + education + sector regression.

    Example: AI usage ~ income + education + industry + geography

    Args:
        df: Input DataFrame
        usage_col: Target variable (AI usage)
        income_col: Income variable
        education_col: Education level variable
        sector_col: Industry sector variable

    Returns:
        Fitted regression model or None if columns missing
    """
    # Check which columns exist
    available_features = []
    for col in [income_col, education_col, sector_col]:
        if col in df.columns:
            available_features.append(col)
        elif f"log_{col}" in df.columns:
            available_features.append(f"log_{col}")

    if usage_col not in df.columns:
        print(f"Warning: Target column '{usage_col}' not found")
        return None

    if not available_features:
        print("Warning: No feature columns found")
        return None

    print(f"\nRunning regression: {usage_col} ~ {' + '.join(available_features)}")

    # Prepare data
    X, y = prepare_regression_data(df, usage_col, available_features)

    # Run regression
    model = run_robust_regression(X, y)
    print_regression_summary(model, f"AI Usage Regression ({usage_col})")

    return model


def run_success_regression(df: pd.DataFrame,
                           success_col: str = "task_success",
                           occupation_col: str = "occupation_type",
                           platform_col: str = "platform",
                           education_col: str = "education_level") -> Optional[sm.regression.linear_model.RegressionResultsWrapper]:
    """
    Run AI task success ~ occupation + platform + education regression.

    Args:
        df: Input DataFrame
        success_col: Target variable (task success)
        occupation_col: Occupation type variable
        platform_col: Platform variable
        education_col: Education level variable

    Returns:
        Fitted regression model or None if columns missing
    """
    available_features = []
    for col in [occupation_col, platform_col, education_col]:
        if col in df.columns:
            available_features.append(col)

    if success_col not in df.columns:
        print(f"Warning: Target column '{success_col}' not found")
        return None

    if not available_features:
        print("Warning: No feature columns found")
        return None

    print(f"\nRunning regression: {success_col} ~ {' + '.join(available_features)}")

    # Prepare data
    X, y = prepare_regression_data(df, success_col, available_features)

    # Run regression
    model = run_robust_regression(X, y)
    print_regression_summary(model, f"Task Success Regression ({success_col})")

    return model


def run_automation_regression(df: pd.DataFrame,
                              automation_col: str = "automation_likelihood",
                              wage_decile_col: str = "wage_decile",
                              task_type_col: str = "task_type",
                              country_income_col: str = "country_income_group") -> Optional[sm.regression.linear_model.RegressionResultsWrapper]:
    """
    Run automation likelihood ~ wage decile + task type + country income regression.

    Args:
        df: Input DataFrame
        automation_col: Target variable
        wage_decile_col: Wage decile variable
        task_type_col: Task type variable
        country_income_col: Country income group variable

    Returns:
        Fitted regression model or None if columns missing
    """
    available_features = []
    for col in [wage_decile_col, task_type_col, country_income_col]:
        if col in df.columns:
            available_features.append(col)

    if automation_col not in df.columns:
        print(f"Warning: Target column '{automation_col}' not found")
        return None

    if not available_features:
        print("Warning: No feature columns found")
        return None

    print(f"\nRunning regression: {automation_col} ~ {' + '.join(available_features)}")

    # Prepare data
    X, y = prepare_regression_data(df, automation_col, available_features)

    # Run regression
    model = run_robust_regression(X, y)
    print_regression_summary(model, f"Automation Likelihood Regression ({automation_col})")

    return model


# ============================================================================
# FLEXIBLE REGRESSION RUNNER
# ============================================================================

def run_custom_regression(df: pd.DataFrame,
                          target: str,
                          features: List[str],
                          robust: bool = True,
                          save_results: bool = True,
                          output_name: str = "custom_regression") -> Optional[sm.regression.linear_model.RegressionResultsWrapper]:
    """
    Run a custom regression with specified target and features.

    Args:
        df: Input DataFrame
        target: Target variable name
        features: List of feature names
        robust: Whether to use robust standard errors
        save_results: Whether to save results to files
        output_name: Name for output files

    Returns:
        Fitted regression model
    """
    setup_output_directories()

    # Check columns exist
    available_features = [f for f in features if f in df.columns]

    if target not in df.columns:
        print(f"Error: Target column '{target}' not found")
        print(f"Available columns: {list(df.columns)}")
        return None

    if not available_features:
        print(f"Error: No feature columns found from {features}")
        print(f"Available columns: {list(df.columns)}")
        return None

    print(f"\nRunning regression: {target} ~ {' + '.join(available_features)}")

    # Prepare data
    X, y = prepare_regression_data(df, target, available_features)
    print(f"Using {len(X)} observations")

    # Run regression
    if robust:
        model = run_robust_regression(X, y)
    else:
        model = run_ols_regression(X, y)

    # Print summary
    print_regression_summary(model, f"Regression: {target}")

    # Save results
    if save_results:
        save_regression_results(model, output_name)

    return model


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run_regression_analysis(df: pd.DataFrame) -> Dict:
    """
    Run complete regression analysis suite.

    Args:
        df: Cleaned DataFrame

    Returns:
        Dictionary with all regression results
    """
    setup_output_directories()
    results = {}

    print("Running Regression Analysis...")
    print("=" * 80)

    # Get column info
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Available numeric columns: {numeric_cols}")

    # 1. Correlation matrix
    print("\nComputing correlation matrix...")
    corr_matrix = compute_correlation_matrix(df)
    results["correlation_matrix"] = corr_matrix
    corr_matrix.to_csv(TABLES_DIR / "correlation_matrix.csv")
    print("Saved correlation matrix")

    # 2. Try running example regressions with actual columns
    # These will adapt based on available columns
    if len(numeric_cols) >= 2:
        target = numeric_cols[0]
        features = numeric_cols[1:min(4, len(numeric_cols))]

        print(f"\nRunning sample regression with available columns...")
        model = run_custom_regression(
            df, target, features,
            output_name="sample_regression"
        )
        if model:
            results["sample_regression"] = format_regression_results(model)

    # 3. Run predefined regressions (will handle missing columns gracefully)
    usage_model = run_usage_regression(df)
    if usage_model:
        results["usage_regression"] = format_regression_results(usage_model)
        save_regression_results(usage_model, "usage_regression")

    success_model = run_success_regression(df)
    if success_model:
        results["success_regression"] = format_regression_results(success_model)
        save_regression_results(success_model, "success_regression")

    automation_model = run_automation_regression(df)
    if automation_model:
        results["automation_regression"] = format_regression_results(automation_model)
        save_regression_results(automation_model, "automation_regression")

    print("\n" + "=" * 80)
    print("Regression Analysis Complete!")
    print(f"Results saved to: {TABLES_DIR}")

    return results


def main():
    """Main function to run regression analysis."""
    print("Starting AEI Regression Analysis")
    print("=" * 80)

    # Load cleaned data
    try:
        df = load_cleaned_data()
        print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

    # Run analysis
    results = run_regression_analysis(df)

    return results


if __name__ == "__main__":
    main()
