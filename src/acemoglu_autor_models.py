"""
Acemoglu-Autor Toy Models

Three indices that marry the Acemoglu-Autor (2026) five-category taxonomy
with the Anthropic Economic Index (AEI v4) revealed-preference data.

Model A: Expertise Commodification Index (ECI)
    Tests which occupations are experiencing "commodification of expertise"
    — where AI renders scarce know-how cheap.

Model B: Pro-Worker AI Score (PWAS)
    Tests which occupations are experiencing the "pro-worker" pattern
    — where AI extends human capabilities rather than substituting for them.

Model C: Junior Pipeline Fragility Index
    Tests the "training wheel" hypothesis — if AI eats codified entry-level
    tasks, where do future seniors come from?
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Model A: Expertise Commodification Index (ECI)
# ---------------------------------------------------------------------------

def compute_eci_task(df: pd.DataFrame) -> pd.Series:
    """
    Compute task-level Expertise Commodification Index.

    ECI_t = directive_share_t * task_success_t * ai_autonomy_t
            * (human_education_t / max(human_education))

    A task scores high when:
    (a) it's done in directive/automated mode
    (b) AI succeeds at it
    (c) AI operates with high autonomy
    (d) the task previously required significant human expertise

    Args:
        df: Task-level DataFrame with required columns.

    Returns:
        Series of task-level ECI scores.
    """
    max_education = df["human_education"].max()
    if max_education == 0:
        max_education = 1.0

    eci = (
        df["directive_share"]
        * df["task_success"]
        * df["ai_autonomy"]
        * (df["human_education"] / max_education)
    )
    return eci


def compute_eci_occupation(
    df: pd.DataFrame,
    weight_col: str = "usage_weight",
) -> pd.DataFrame:
    """
    Compute occupation-level ECI by aggregating task-level scores.

    ECI_o = sum_t(w_t * ECI_t) where w_t is the task's share of usage.

    Args:
        df: Task-level DataFrame with ECI scores.
        weight_col: Column to use as task weights.

    Returns:
        DataFrame indexed by occupation with ECI and supporting stats.
    """
    df = df.copy()
    df["eci_task"] = compute_eci_task(df)
    df["eci_weighted"] = df["eci_task"] * df[weight_col]

    occ = df.groupby("onet_soc_code").agg(
        ECI=("eci_weighted", "sum"),
        eci_mean=("eci_task", "mean"),
        eci_std=("eci_task", "std"),
        eci_max=("eci_task", "max"),
        n_tasks=("eci_task", "count"),
        mean_directive_share=("directive_share", "mean"),
        mean_ai_autonomy=("ai_autonomy", "mean"),
        mean_task_success=("task_success", "mean"),
        mean_human_education=("human_education", "mean"),
        median_annual_wage=("median_annual_wage", "first"),
        title=("title", "first"),
        soc_major=("soc_major", "first"),
    ).reset_index()

    return occ


def compute_eci_soc_major(occ_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate ECI to SOC major group level.

    Args:
        occ_df: Occupation-level ECI DataFrame.

    Returns:
        DataFrame indexed by SOC major group.
    """
    from .acemoglu_autor_data import SOC_MAJOR_GROUPS

    soc = occ_df.groupby("soc_major").agg(
        ECI_mean=("ECI", "mean"),
        ECI_median=("ECI", "median"),
        ECI_std=("ECI", "std"),
        n_occupations=("ECI", "count"),
        mean_wage=("median_annual_wage", "mean"),
        mean_education=("mean_human_education", "mean"),
    ).reset_index()

    soc["soc_major_label"] = soc["soc_major"].map(SOC_MAJOR_GROUPS)
    return soc.sort_values("ECI_mean", ascending=False)


# ---------------------------------------------------------------------------
# Model B: Pro-Worker AI Score (PWAS)
# ---------------------------------------------------------------------------

def compute_pwas_task(
    df: pd.DataFrame,
    alpha: float = 1 / 3,
    beta: float = 1 / 3,
    gamma: float = 1 / 3,
) -> pd.Series:
    """
    Compute task-level Pro-Worker AI Score.

    PWAS_t = alpha * learning_share
             + beta * task_iteration_share
             + gamma * (1 - ai_autonomy) * task_success

    The three terms capture:
    1. Learning: humans acquire new skills (new task-creating)
    2. Task iteration: humans and AI refine work together (augmentation)
    3. Constrained success: AI succeeds but with low autonomy
       (human remains essential)

    Args:
        df: Task-level DataFrame.
        alpha: Weight for learning share term.
        beta: Weight for task iteration term.
        gamma: Weight for constrained success term.

    Returns:
        Series of task-level PWAS scores.
    """
    pwas = (
        alpha * df["learning_share"]
        + beta * df["task_iteration_share"]
        + gamma * (1 - df["ai_autonomy"]) * df["task_success"]
    )
    return pwas


def compute_pwas_occupation(
    df: pd.DataFrame,
    alpha: float = 1 / 3,
    beta: float = 1 / 3,
    gamma: float = 1 / 3,
    weight_col: str = "usage_weight",
) -> pd.DataFrame:
    """
    Compute occupation-level PWAS.

    Args:
        df: Task-level DataFrame.
        alpha, beta, gamma: PWAS term weights.
        weight_col: Column for task weights.

    Returns:
        DataFrame indexed by occupation with PWAS and supporting stats.
    """
    df = df.copy()
    df["pwas_task"] = compute_pwas_task(df, alpha, beta, gamma)
    df["pwas_weighted"] = df["pwas_task"] * df[weight_col]

    occ = df.groupby("onet_soc_code").agg(
        PWAS=("pwas_weighted", "sum"),
        pwas_mean=("pwas_task", "mean"),
        pwas_std=("pwas_task", "std"),
        n_tasks=("pwas_task", "count"),
        mean_learning_share=("learning_share", "mean"),
        mean_task_iteration_share=("task_iteration_share", "mean"),
        mean_ai_autonomy=("ai_autonomy", "mean"),
        mean_task_success=("task_success", "mean"),
        median_annual_wage=("median_annual_wage", "first"),
        title=("title", "first"),
        soc_major=("soc_major", "first"),
    ).reset_index()

    return occ


def compute_pwas_soc_major(occ_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate PWAS to SOC major group level."""
    from .acemoglu_autor_data import SOC_MAJOR_GROUPS

    soc = occ_df.groupby("soc_major").agg(
        PWAS_mean=("PWAS", "mean"),
        PWAS_median=("PWAS", "median"),
        PWAS_std=("PWAS", "std"),
        n_occupations=("PWAS", "count"),
        mean_wage=("median_annual_wage", "mean"),
    ).reset_index()

    soc["soc_major_label"] = soc["soc_major"].map(SOC_MAJOR_GROUPS)
    return soc.sort_values("PWAS_mean", ascending=False)


# ---------------------------------------------------------------------------
# Model C: Junior Pipeline Fragility Index
# ---------------------------------------------------------------------------

def compute_pipeline_fragility(
    df: pd.DataFrame,
    complexity_threshold: str = "median",
    min_directive_floor: float = 0.01,
) -> pd.DataFrame:
    """
    Compute Pipeline Fragility Index per occupation.

    Fragility_o = automation_share(low-complexity tasks in o)
                / max(automation_share(high-complexity tasks in o), floor)

    Ratio > 1 means AI disproportionately eats "training wheel" tasks.
    Higher ratio = more fragile junior pipeline.

    Low-complexity: human_education < median for that occupation
    High-complexity: human_education >= median for that occupation

    Args:
        df: Task-level DataFrame.
        complexity_threshold: How to split tasks ("median" uses per-occupation median).
        min_directive_floor: Floor for denominator to avoid division by zero.

    Returns:
        DataFrame with per-occupation fragility scores and enrichments.
    """
    df = df.copy()

    # Compute per-occupation education median for complexity split
    occ_median_ed = df.groupby("onet_soc_code")["human_education"].transform("median")
    df["is_junior_task"] = df["human_education"] < occ_median_ed

    # Automation share proxy: directive_share * ai_autonomy
    df["auto_proxy"] = df["directive_share"] * df["ai_autonomy"]

    results = []
    for occ_code, group in df.groupby("onet_soc_code"):
        junior = group[group["is_junior_task"]]
        senior = group[~group["is_junior_task"]]

        junior_auto = junior["auto_proxy"].mean() if len(junior) > 0 else 0
        senior_auto = max(senior["auto_proxy"].mean() if len(senior) > 0 else 0,
                          min_directive_floor)

        fragility = junior_auto / senior_auto

        # Learning share enrichment: is AI replacing the apprenticeship function?
        junior_learning = junior["learning_share"].mean() if len(junior) > 0 else 0
        senior_learning = senior["learning_share"].mean() if len(senior) > 0 else 0

        results.append({
            "onet_soc_code": occ_code,
            "pipeline_fragility": round(fragility, 4),
            "junior_automation": round(junior_auto, 4),
            "senior_automation": round(senior_auto, 4),
            "junior_learning_share": round(junior_learning, 4),
            "senior_learning_share": round(senior_learning, 4),
            "n_junior_tasks": len(junior),
            "n_senior_tasks": len(senior),
            "n_total_tasks": len(group),
            "title": group["title"].iloc[0],
            "soc_major": group["soc_major"].iloc[0],
            "median_annual_wage": group["median_annual_wage"].iloc[0],
            "mean_human_education": group["human_education"].mean(),
        })

    frag_df = pd.DataFrame(results)

    # Danger zone classification
    # High fragility + low learning = danger (training wheels gone, nothing replaces them)
    # High fragility + high learning = AI may be replacing the apprenticeship function
    median_fragility = frag_df["pipeline_fragility"].median()
    median_learning = frag_df["junior_learning_share"].median()

    frag_df["zone"] = "neutral"
    frag_df.loc[
        (frag_df["pipeline_fragility"] > median_fragility)
        & (frag_df["junior_learning_share"] < median_learning),
        "zone",
    ] = "danger"
    frag_df.loc[
        (frag_df["pipeline_fragility"] > median_fragility)
        & (frag_df["junior_learning_share"] >= median_learning),
        "zone",
    ] = "ai_apprenticeship"
    frag_df.loc[
        (frag_df["pipeline_fragility"] <= median_fragility)
        & (frag_df["junior_learning_share"] >= median_learning),
        "zone",
    ] = "pro_worker"

    return frag_df


def compute_fragility_soc_major(frag_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pipeline fragility to SOC major group level."""
    from .acemoglu_autor_data import SOC_MAJOR_GROUPS

    soc = frag_df.groupby("soc_major").agg(
        fragility_mean=("pipeline_fragility", "mean"),
        fragility_median=("pipeline_fragility", "median"),
        fragility_std=("pipeline_fragility", "std"),
        mean_junior_learning=("junior_learning_share", "mean"),
        danger_count=("zone", lambda x: (x == "danger").sum()),
        pro_worker_count=("zone", lambda x: (x == "pro_worker").sum()),
        n_occupations=("pipeline_fragility", "count"),
        mean_wage=("median_annual_wage", "mean"),
    ).reset_index()

    soc["soc_major_label"] = soc["soc_major"].map(SOC_MAJOR_GROUPS)
    soc["danger_pct"] = (soc["danger_count"] / soc["n_occupations"] * 100).round(1)
    return soc.sort_values("fragility_mean", ascending=False)


# ---------------------------------------------------------------------------
# Net Exposure and Combined Analysis
# ---------------------------------------------------------------------------

def compute_net_exposure(
    eci_occ: pd.DataFrame,
    pwas_occ: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute net exposure score per occupation: ECI - PWAS.

    Positive = net automation pressure (substitution risk).
    Negative = net pro-worker benefit (augmentation opportunity).

    Args:
        eci_occ: Occupation-level ECI DataFrame.
        pwas_occ: Occupation-level PWAS DataFrame.

    Returns:
        Merged DataFrame with net exposure and classification.
    """
    eci_sub = eci_occ[["onet_soc_code", "ECI", "title", "soc_major",
                        "median_annual_wage", "mean_human_education",
                        "n_tasks"]].copy()
    pwas_sub = pwas_occ[["onet_soc_code", "PWAS"]].copy()

    merged = eci_sub.merge(pwas_sub, on="onet_soc_code", how="inner")
    merged["net_exposure"] = merged["ECI"] - merged["PWAS"]

    # Classification
    merged["exposure_class"] = pd.cut(
        merged["net_exposure"],
        bins=[-np.inf, -0.05, 0.05, np.inf],
        labels=["pro_worker", "neutral", "automation_risk"],
    )

    return merged.sort_values("net_exposure", ascending=False)


def compute_combined_results(
    df: pd.DataFrame,
    alpha: float = 1 / 3,
    beta: float = 1 / 3,
    gamma: float = 1 / 3,
) -> Dict[str, pd.DataFrame]:
    """
    Run all three models and the net exposure analysis.

    Args:
        df: Task-level AEI v4 DataFrame.
        alpha, beta, gamma: PWAS weights.

    Returns:
        Dictionary with all result DataFrames.
    """
    # Model A: ECI
    eci_occ = compute_eci_occupation(df)
    eci_soc = compute_eci_soc_major(eci_occ)

    # Model B: PWAS
    pwas_occ = compute_pwas_occupation(df, alpha, beta, gamma)
    pwas_soc = compute_pwas_soc_major(pwas_occ)

    # Model C: Pipeline Fragility
    frag_occ = compute_pipeline_fragility(df)
    frag_soc = compute_fragility_soc_major(frag_occ)

    # Net Exposure
    net_exp = compute_net_exposure(eci_occ, pwas_occ)

    # Add fragility to net exposure
    frag_sub = frag_occ[["onet_soc_code", "pipeline_fragility", "zone",
                          "junior_learning_share"]].copy()
    net_exp = net_exp.merge(frag_sub, on="onet_soc_code", how="left")

    return {
        "eci_occupation": eci_occ,
        "eci_soc_major": eci_soc,
        "pwas_occupation": pwas_occ,
        "pwas_soc_major": pwas_soc,
        "fragility_occupation": frag_occ,
        "fragility_soc_major": frag_soc,
        "net_exposure": net_exp,
    }


def print_model_summary(results: Dict[str, pd.DataFrame]) -> None:
    """Print a summary of all model results."""
    print("\n" + "=" * 70)
    print("ACEMOGLU-AUTOR TOY MODEL RESULTS SUMMARY")
    print("=" * 70)

    # ECI Summary
    eci = results["eci_occupation"]
    print("\n--- Model A: Expertise Commodification Index (ECI) ---")
    print(f"  Occupations analyzed: {len(eci)}")
    print(f"  Mean ECI: {eci['ECI'].mean():.4f}")
    print(f"  Median ECI: {eci['ECI'].median():.4f}")
    print(f"  Top 5 commodified occupations:")
    for _, row in eci.nlargest(5, "ECI").iterrows():
        print(f"    {row['title']}: ECI={row['ECI']:.4f}")

    # PWAS Summary
    pwas = results["pwas_occupation"]
    print("\n--- Model B: Pro-Worker AI Score (PWAS) ---")
    print(f"  Occupations analyzed: {len(pwas)}")
    print(f"  Mean PWAS: {pwas['PWAS'].mean():.4f}")
    print(f"  Median PWAS: {pwas['PWAS'].median():.4f}")
    print(f"  Top 5 pro-worker occupations:")
    for _, row in pwas.nlargest(5, "PWAS").iterrows():
        print(f"    {row['title']}: PWAS={row['PWAS']:.4f}")

    # Fragility Summary
    frag = results["fragility_occupation"]
    print("\n--- Model C: Junior Pipeline Fragility ---")
    print(f"  Occupations analyzed: {len(frag)}")
    print(f"  Mean fragility: {frag['pipeline_fragility'].mean():.4f}")
    print(f"  Median fragility: {frag['pipeline_fragility'].median():.4f}")
    zone_counts = frag["zone"].value_counts()
    for zone, count in zone_counts.items():
        print(f"  Zone '{zone}': {count} occupations ({count/len(frag)*100:.1f}%)")
    print(f"  Top 5 most fragile pipelines:")
    for _, row in frag.nlargest(5, "pipeline_fragility").iterrows():
        print(f"    {row['title']}: fragility={row['pipeline_fragility']:.2f} ({row['zone']})")

    # Net Exposure Summary
    net = results["net_exposure"]
    print("\n--- Net Exposure (ECI - PWAS) ---")
    class_counts = net["exposure_class"].value_counts()
    for cls, count in class_counts.items():
        print(f"  {cls}: {count} occupations ({count/len(net)*100:.1f}%)")
    print(f"  Most automation-exposed:")
    for _, row in net.nlargest(3, "net_exposure").iterrows():
        print(f"    {row['title']}: net={row['net_exposure']:.4f}")
    print(f"  Most pro-worker:")
    for _, row in net.nsmallest(3, "net_exposure").iterrows():
        print(f"    {row['title']}: net={row['net_exposure']:.4f}")

    print("\n" + "=" * 70)
