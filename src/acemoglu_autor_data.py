"""
Acemoglu-Autor Data Pipeline

Loads and prepares AEI v4 data for the Acemoglu-Autor toy models.
When real AEI v4 data is unavailable, generates realistic synthetic data
based on the schema described in the AEI v4 release (release_2026_01_15).

Key variables from AEI v4:
- Collaboration modes: directive, task_iteration, learning, feedback_loop
- Economic primitives: human_only_time, human_ai_time, speedup_factor,
  human_education, ai_education, ai_autonomy, task_success, use_case
- Structural: O*NET task codes, SOC codes, wages
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "acemoglu_autor"
DASHBOARD_DIR = PROJECT_ROOT / "results" / "dashboard"

# SOC major group labels
SOC_MAJOR_GROUPS = {
    "11": "Management",
    "13": "Business and Financial Operations",
    "15": "Computer and Mathematical",
    "17": "Architecture and Engineering",
    "19": "Life, Physical, and Social Science",
    "21": "Community and Social Service",
    "23": "Legal",
    "25": "Educational Instruction and Library",
    "27": "Arts, Design, Entertainment, Sports, and Media",
    "29": "Healthcare Practitioners and Technical",
    "31": "Healthcare Support",
    "33": "Protective Service",
    "35": "Food Preparation and Serving Related",
    "37": "Building and Grounds Cleaning and Maintenance",
    "39": "Personal Care and Service",
    "41": "Sales and Related",
    "43": "Office and Administrative Support",
    "45": "Farming, Fishing, and Forestry",
    "47": "Construction and Extraction",
    "49": "Installation, Maintenance, and Repair",
    "51": "Production",
    "53": "Transportation and Material Moving",
}

# Wage profiles by SOC major group (approximate median annual wages, USD)
SOC_WAGE_PROFILES = {
    "11": 110000, "13": 76000, "15": 98000, "17": 83000, "19": 72000,
    "21": 50000, "23": 88000, "25": 55000, "27": 62000, "29": 78000,
    "31": 35000, "33": 48000, "35": 30000, "37": 32000, "39": 33000,
    "41": 42000, "43": 40000, "45": 35000, "47": 50000, "49": 48000,
    "51": 38000, "53": 38000,
}

# Education levels by SOC group (years of schooling typically required)
SOC_EDUCATION_PROFILES = {
    "11": 16, "13": 16, "15": 16, "17": 16, "19": 18,
    "21": 16, "23": 19, "25": 18, "27": 14, "29": 18,
    "31": 12, "33": 13, "35": 10, "37": 10, "39": 12,
    "41": 12, "43": 12, "45": 11, "47": 12, "49": 13,
    "51": 12, "53": 11,
}

# AI usage intensity by SOC group (relative, 0-1 scale)
SOC_AI_USAGE = {
    "11": 0.6, "13": 0.7, "15": 0.95, "17": 0.7, "19": 0.65,
    "21": 0.35, "23": 0.75, "25": 0.55, "27": 0.6, "29": 0.4,
    "31": 0.2, "33": 0.15, "35": 0.1, "37": 0.05, "39": 0.15,
    "41": 0.4, "43": 0.6, "45": 0.1, "47": 0.08, "49": 0.2,
    "51": 0.15, "53": 0.1,
}


def load_existing_occupations() -> pd.DataFrame:
    """Load occupation data from the existing dashboard dataset."""
    dashboard_path = DASHBOARD_DIR / "dashboard_data.json"
    if not dashboard_path.exists():
        return pd.DataFrame()

    with open(dashboard_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    # Get unique occupations
    occ = df[["onet_soc_code", "title"]].drop_duplicates().reset_index(drop=True)
    occ["soc_major"] = occ["onet_soc_code"].str[:2]

    # Get task info
    tasks = df[["onet_soc_code", "title", "task_id", "task", "task_type"]].drop_duplicates()
    return tasks


def generate_synthetic_aei_v4(
    n_tasks: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic AEI v4-shaped data for toy model demonstration.

    Creates task-level records with collaboration modes, economic primitives,
    and occupation mappings that match the real AEI v4 schema. The
    distributions are calibrated to reflect known patterns from the AEI v4
    report (e.g., augmentation edging over automation on Claude.ai at 52% vs
    45%, median ~12x speedup for college-level tasks).

    Args:
        n_tasks: Number of task-level records to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with synthetic AEI v4 data.
    """
    rng = np.random.default_rng(seed)

    # Load real occupation/task structure
    existing = load_existing_occupations()
    if existing.empty:
        raise RuntimeError("No existing occupation data found in dashboard_data.json")

    # Sample tasks from existing occupation data
    if len(existing) >= n_tasks:
        sampled = existing.sample(n=n_tasks, random_state=seed).reset_index(drop=True)
    else:
        # Repeat and sample
        repeats = (n_tasks // len(existing)) + 1
        sampled = pd.concat([existing] * repeats, ignore_index=True).sample(
            n=n_tasks, random_state=seed
        ).reset_index(drop=True)

    sampled["soc_major"] = sampled["onet_soc_code"].str[:2]

    # Map wages and education
    sampled["median_annual_wage"] = sampled["soc_major"].map(SOC_WAGE_PROFILES)
    base_education = sampled["soc_major"].map(SOC_EDUCATION_PROFILES).values.astype(float)
    ai_usage = sampled["soc_major"].map(SOC_AI_USAGE).values.astype(float)

    # --- Collaboration modes ---
    # AEI v4 reports: Claude.ai is ~52% augmentation, ~45% automation, rest learning
    # Vary by occupation: white-collar has more directive, blue-collar more learning

    # Base directive share: higher for high-AI-usage, high-education occupations
    directive_base = 0.3 + 0.3 * ai_usage + 0.1 * (base_education / 20)
    directive_share = np.clip(
        directive_base + rng.normal(0, 0.08, n_tasks), 0.02, 0.85
    )

    # Task iteration: collaborative refinement, higher for moderate-AI-usage
    iteration_base = 0.25 + 0.15 * (1 - np.abs(ai_usage - 0.5))
    task_iteration_share = np.clip(
        iteration_base + rng.normal(0, 0.08, n_tasks), 0.02, 0.7
    )

    # Learning: higher for lower-education, lower-AI-usage occupations
    learning_base = 0.15 + 0.2 * (1 - ai_usage) + 0.1 * (1 - base_education / 20)
    learning_share = np.clip(
        learning_base + rng.normal(0, 0.06, n_tasks), 0.01, 0.5
    )

    # Feedback/validation: remainder
    feedback_share = np.clip(
        1.0 - directive_share - task_iteration_share - learning_share,
        0.01, 0.4,
    )

    # Renormalize to sum to 1
    total = directive_share + task_iteration_share + learning_share + feedback_share
    directive_share /= total
    task_iteration_share /= total
    learning_share /= total
    feedback_share /= total

    sampled["directive_share"] = np.round(directive_share, 4)
    sampled["task_iteration_share"] = np.round(task_iteration_share, 4)
    sampled["learning_share"] = np.round(learning_share, 4)
    sampled["feedback_share"] = np.round(feedback_share, 4)

    # --- Economic primitives ---

    # Human education: years of schooling to understand the prompt
    # Varies within occupation by task complexity
    sampled["human_education"] = np.clip(
        base_education + rng.normal(0, 2.0, n_tasks), 8, 22
    ).round(1)

    # AI education: years of schooling to understand the response
    # Generally >= human_education (AI produces expert-level output)
    sampled["ai_education"] = np.clip(
        sampled["human_education"].values + rng.uniform(0, 4, n_tasks),
        10, 22,
    ).round(1)

    # AI autonomy: how much the user delegates (0-1)
    # Correlated with directive share
    sampled["ai_autonomy"] = np.clip(
        0.3 * directive_share + 0.5 * ai_usage + rng.normal(0, 0.1, n_tasks),
        0.05, 0.95,
    ).round(4)

    # Task success: whether Claude completes the task (0-1)
    # Higher for simpler tasks and high-AI-usage domains
    success_base = 0.6 + 0.2 * ai_usage - 0.01 * (sampled["human_education"].values - 12)
    sampled["task_success"] = np.clip(
        success_base + rng.normal(0, 0.1, n_tasks), 0.2, 0.98
    ).round(4)

    # Human-only time (minutes): how long the task would take without AI
    # Scales with education level and task complexity
    human_only_base = 30 + 20 * (sampled["human_education"].values / 16)
    sampled["human_only_time"] = np.clip(
        human_only_base * rng.lognormal(0, 0.5, n_tasks), 5, 480
    ).round(1)

    # Human+AI time (minutes): ~15 min median per AEI report
    speedup_factor = np.clip(
        8 + 8 * ai_usage + rng.normal(0, 3, n_tasks), 1.5, 50
    )
    sampled["human_ai_time"] = np.clip(
        sampled["human_only_time"].values / speedup_factor, 1, 120
    ).round(1)
    sampled["speedup_factor"] = (
        sampled["human_only_time"] / sampled["human_ai_time"]
    ).round(2)

    # Use case distribution
    use_case_probs = rng.dirichlet([0.55, 0.2, 0.25], size=n_tasks)
    use_cases = ["work", "education", "personal"]
    sampled["use_case"] = [
        use_cases[np.argmax(p)] for p in use_case_probs
    ]

    # Usage weight: proxy for how often this task appears in Claude usage
    sampled["usage_weight"] = np.clip(
        ai_usage * rng.exponential(1, n_tasks), 0.01, 10
    ).round(4)
    # Normalize within occupation
    for occ_code in sampled["onet_soc_code"].unique():
        mask = sampled["onet_soc_code"] == occ_code
        occ_weights = sampled.loc[mask, "usage_weight"]
        if occ_weights.sum() > 0:
            sampled.loc[mask, "usage_weight"] = (
                occ_weights / occ_weights.sum()
            ).round(4)

    # Platform source
    sampled["platform"] = rng.choice(
        ["claude_ai", "api"], size=n_tasks, p=[0.5, 0.5]
    )

    # --- Derived: automation vs augmentation classification ---
    # Per AEI v4: automation = directive + high autonomy; augmentation = rest
    sampled["automation_share"] = (
        sampled["directive_share"] * sampled["ai_autonomy"]
    ).round(4)
    sampled["augmentation_share"] = (1 - sampled["automation_share"]).round(4)

    return sampled


def load_aei_v4_data(
    release: str = "release_2026_01_15",
    use_synthetic: bool = True,
    synthetic_n_tasks: int = 5000,
    synthetic_seed: int = 42,
) -> pd.DataFrame:
    """
    Load AEI v4 data, falling back to synthetic data if unavailable.

    Args:
        release: AEI release version.
        use_synthetic: Whether to generate synthetic data as fallback.
        synthetic_n_tasks: Number of synthetic task records.
        synthetic_seed: Random seed for synthetic generation.

    Returns:
        DataFrame with task-level AEI v4 data.
    """
    # Try to load real data first
    real_data_dir = RAW_DATA_DIR / release / "data"
    if real_data_dir.exists():
        csv_files = list(real_data_dir.glob("aei_raw_claude_ai_*.csv"))
        if csv_files:
            dfs = [pd.read_csv(f) for f in csv_files]
            df = pd.concat(dfs, ignore_index=True)
            print(f"Loaded real AEI v4 data: {len(df):,} rows from {len(csv_files)} files")
            return df

    if use_synthetic:
        print("Real AEI v4 data not available. Generating synthetic data...")
        df = generate_synthetic_aei_v4(
            n_tasks=synthetic_n_tasks,
            seed=synthetic_seed,
        )
        print(f"Generated synthetic AEI v4 data: {len(df):,} rows")
        return df

    raise FileNotFoundError(
        f"No AEI v4 data found at {real_data_dir} and synthetic generation disabled"
    )


def load_wage_data() -> pd.DataFrame:
    """
    Load wage data by SOC code.

    Returns DataFrame with soc_major, soc_major_label, and median_annual_wage.
    """
    records = []
    for code, wage in SOC_WAGE_PROFILES.items():
        records.append({
            "soc_major": code,
            "soc_major_label": SOC_MAJOR_GROUPS.get(code, "Unknown"),
            "median_annual_wage": wage,
            "education_years": SOC_EDUCATION_PROFILES.get(code, 12),
        })
    return pd.DataFrame(records)


def prepare_analysis_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare the analysis-ready dataset with all required fields.

    Ensures consistent column naming, fills missing values, and adds
    derived fields needed by the three toy models.

    Args:
        df: Raw AEI v4 data (real or synthetic).

    Returns:
        Cleaned and enriched DataFrame.
    """
    out = df.copy()

    # Ensure soc_major exists
    if "soc_major" not in out.columns:
        out["soc_major"] = out["onet_soc_code"].str[:2]

    # Add SOC labels
    out["soc_major_label"] = out["soc_major"].map(SOC_MAJOR_GROUPS)

    # Add wage data if not present
    if "median_annual_wage" not in out.columns:
        out["median_annual_wage"] = out["soc_major"].map(SOC_WAGE_PROFILES)

    # Ensure education fields
    if "human_education" not in out.columns:
        out["human_education"] = out["soc_major"].map(SOC_EDUCATION_PROFILES).astype(float)

    # Fill any remaining NaNs
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].fillna(out[numeric_cols].median())

    return out


def save_analysis_data(df: pd.DataFrame, filename: str = "aei_v4_analysis.csv") -> Path:
    """Save the analysis-ready dataset."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / filename
    df.to_csv(output_path, index=False)
    print(f"Saved analysis data: {output_path} ({len(df):,} rows)")
    return output_path


def get_data_summary(df: pd.DataFrame) -> Dict:
    """Generate a summary of the analysis dataset."""
    summary = {
        "n_records": len(df),
        "n_occupations": df["onet_soc_code"].nunique(),
        "n_soc_major_groups": df["soc_major"].nunique(),
        "collaboration_modes": {
            "directive_mean": float(df["directive_share"].mean()),
            "task_iteration_mean": float(df["task_iteration_share"].mean()),
            "learning_mean": float(df["learning_share"].mean()),
            "feedback_mean": float(df["feedback_share"].mean()),
        },
        "economic_primitives": {
            "median_human_education": float(df["human_education"].median()),
            "median_ai_autonomy": float(df["ai_autonomy"].median()),
            "median_task_success": float(df["task_success"].median()),
            "median_speedup_factor": float(df["speedup_factor"].median()),
            "median_human_ai_time": float(df["human_ai_time"].median()),
        },
        "platform_split": df["platform"].value_counts().to_dict()
        if "platform" in df.columns
        else {},
        "use_case_split": df["use_case"].value_counts().to_dict()
        if "use_case" in df.columns
        else {},
    }
    return summary
