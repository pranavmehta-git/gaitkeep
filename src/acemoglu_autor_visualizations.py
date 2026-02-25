"""
Acemoglu-Autor Visualizations

Generates all plots for the toy model analysis:
1. ECI vs. median wage scatter
2. PWAS vs. SOC major group bar chart
3. Net exposure (ECI - PWAS) by SOC major group
4. Pipeline fragility by occupation, colored by zone
5. Speedup factor vs. ECI scatter
6. Collaboration mode breakdown by SOC group
7. ECI-PWAS quadrant plot
8. Fragility danger zone heatmap
"""

from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

RESULTS_DIR = Path(__file__).parent.parent / "results" / "acemoglu_autor"
FIGURES_DIR = RESULTS_DIR / "figures"

# Color scheme
COLORS = {
    "eci": "#e74c3c",
    "pwas": "#2ecc71",
    "fragility": "#f39c12",
    "danger": "#c0392b",
    "pro_worker": "#27ae60",
    "neutral": "#95a5a6",
    "ai_apprenticeship": "#3498db",
    "automation_risk": "#e74c3c",
}

ZONE_COLORS = {
    "danger": COLORS["danger"],
    "pro_worker": COLORS["pro_worker"],
    "neutral": COLORS["neutral"],
    "ai_apprenticeship": COLORS["ai_apprenticeship"],
}


def setup_style():
    """Configure matplotlib style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.figsize": (10, 8),
    })


def save_figure(fig: plt.Figure, name: str, fmt: str = "png") -> Path:
    """Save a figure to the results directory."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.{fmt}"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Plot 1: ECI vs. Median Wage
# ---------------------------------------------------------------------------

def plot_eci_vs_wage(eci_occ: pd.DataFrame) -> plt.Figure:
    """
    Scatter plot of ECI vs. median annual wage by occupation.

    Expects negative correlation: high-ECI occupations should have
    lower wage growth pressure (expertise being commodified).
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color by SOC major group
    soc_groups = sorted(eci_occ["soc_major"].unique())
    cmap = plt.cm.get_cmap("tab20", len(soc_groups))
    color_map = {g: cmap(i) for i, g in enumerate(soc_groups)}

    for soc, group in eci_occ.groupby("soc_major"):
        ax.scatter(
            group["ECI"],
            group["median_annual_wage"] / 1000,
            c=[color_map[soc]],
            alpha=0.6,
            s=group["n_tasks"] * 8,
            label=soc,
            edgecolors="white",
            linewidth=0.5,
        )

    # Trend line
    z = np.polyfit(eci_occ["ECI"], eci_occ["median_annual_wage"] / 1000, 1)
    p = np.poly1d(z)
    x_line = np.linspace(eci_occ["ECI"].min(), eci_occ["ECI"].max(), 100)
    ax.plot(x_line, p(x_line), "--", color="gray", alpha=0.8, linewidth=2)

    corr = eci_occ["ECI"].corr(eci_occ["median_annual_wage"])
    ax.text(
        0.02, 0.98, f"r = {corr:.3f}",
        transform=ax.transAxes, fontsize=12, va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax.set_xlabel("Expertise Commodification Index (ECI)")
    ax.set_ylabel("Median Annual Wage ($K)")
    ax.set_title("Model A: Expertise Commodification vs. Wage Level\n"
                  "(Size = number of tasks)")
    ax.legend(
        bbox_to_anchor=(1.05, 1), loc="upper left",
        title="SOC Major", fontsize=8, ncol=1,
    )

    return fig


# ---------------------------------------------------------------------------
# Plot 2: PWAS by SOC Major Group
# ---------------------------------------------------------------------------

def plot_pwas_by_soc(pwas_soc: pd.DataFrame) -> plt.Figure:
    """
    Horizontal bar chart of PWAS by SOC major group.

    Acemoglu-Autor predict higher pro-worker potential in non-degreed jobs.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(14, 9))

    df = pwas_soc.sort_values("PWAS_mean")
    bars = ax.barh(
        df["soc_major_label"],
        df["PWAS_mean"],
        xerr=df["PWAS_std"],
        color=COLORS["pwas"],
        alpha=0.8,
        edgecolor="white",
        capsize=3,
    )

    ax.set_xlabel("Pro-Worker AI Score (PWAS)")
    ax.set_title("Model B: Pro-Worker AI Score by SOC Major Group\n"
                  "(Error bars = std across occupations within group)")

    # Add count annotations
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(
            row["PWAS_mean"] + row["PWAS_std"] + 0.005,
            i, f"n={int(row['n_occupations'])}",
            va="center", fontsize=9, color="gray",
        )

    ax.axvline(df["PWAS_mean"].median(), color="gray", linestyle="--",
               alpha=0.6, label="Median")
    ax.legend()

    return fig


# ---------------------------------------------------------------------------
# Plot 3: Net Exposure (ECI - PWAS) by SOC Major Group
# ---------------------------------------------------------------------------

def plot_net_exposure_by_soc(net_exp: pd.DataFrame) -> plt.Figure:
    """
    Horizontal bar chart of net exposure by SOC major group.

    Positive = automation risk. Negative = pro-worker benefit.
    """
    setup_style()
    from .acemoglu_autor_data import SOC_MAJOR_GROUPS

    soc_agg = net_exp.groupby("soc_major").agg(
        net_exposure_mean=("net_exposure", "mean"),
        net_exposure_std=("net_exposure", "std"),
        n=("net_exposure", "count"),
    ).reset_index()
    soc_agg["soc_major_label"] = soc_agg["soc_major"].map(SOC_MAJOR_GROUPS)
    soc_agg = soc_agg.sort_values("net_exposure_mean")

    fig, ax = plt.subplots(figsize=(14, 9))

    colors = [
        COLORS["automation_risk"] if x > 0 else COLORS["pro_worker"]
        for x in soc_agg["net_exposure_mean"]
    ]

    ax.barh(
        soc_agg["soc_major_label"],
        soc_agg["net_exposure_mean"],
        xerr=soc_agg["net_exposure_std"],
        color=colors,
        alpha=0.8,
        edgecolor="white",
        capsize=3,
    )

    ax.axvline(0, color="black", linewidth=1.5)
    ax.set_xlabel("Net Exposure (ECI - PWAS)")
    ax.set_title("Net AI Exposure by SOC Major Group\n"
                  "Positive = Automation Risk | Negative = Pro-Worker Benefit")

    # Legend
    patches = [
        mpatches.Patch(color=COLORS["automation_risk"], label="Automation Risk (net > 0)"),
        mpatches.Patch(color=COLORS["pro_worker"], label="Pro-Worker Benefit (net < 0)"),
    ]
    ax.legend(handles=patches, loc="lower right")

    return fig


# ---------------------------------------------------------------------------
# Plot 4: Pipeline Fragility by Occupation
# ---------------------------------------------------------------------------

def plot_fragility_scatter(frag_occ: pd.DataFrame) -> plt.Figure:
    """
    Scatter of pipeline fragility vs. junior learning share, colored by zone.

    High fragility + low learning = danger zone.
    High fragility + high learning = AI may replace apprenticeship.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 8))

    for zone, color in ZONE_COLORS.items():
        mask = frag_occ["zone"] == zone
        subset = frag_occ[mask]
        ax.scatter(
            subset["junior_learning_share"],
            subset["pipeline_fragility"],
            c=color,
            alpha=0.6,
            s=50,
            label=zone.replace("_", " ").title(),
            edgecolors="white",
            linewidth=0.5,
        )

    # Quadrant lines
    med_frag = frag_occ["pipeline_fragility"].median()
    med_learn = frag_occ["junior_learning_share"].median()
    ax.axhline(med_frag, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(med_learn, color="gray", linestyle="--", alpha=0.5)

    # Quadrant labels (x=learning share, y=fragility)
    ax.text(0.02, 0.98, "High Fragility\nLow Learning\n(DANGER ZONE)",
            transform=ax.transAxes, fontsize=9, va="top", ha="left",
            color=COLORS["danger"], fontweight="bold")
    ax.text(0.98, 0.98, "High Fragility\nHigh Learning\n(AI Apprenticeship)",
            transform=ax.transAxes, fontsize=9, va="top", ha="right",
            color=COLORS["ai_apprenticeship"], fontweight="bold")
    ax.text(0.02, 0.02, "Low Fragility\nLow Learning\n(Neutral)",
            transform=ax.transAxes, fontsize=9, va="bottom", ha="left",
            color=COLORS["neutral"], fontweight="bold")
    ax.text(0.98, 0.02, "Low Fragility\nHigh Learning\n(Pro-Worker)",
            transform=ax.transAxes, fontsize=9, va="bottom", ha="right",
            color=COLORS["pro_worker"], fontweight="bold")

    ax.set_xlabel("Junior Task Learning Share")
    ax.set_ylabel("Pipeline Fragility Index")
    ax.set_title("Model C: Junior Pipeline Fragility vs. Learning Share\n"
                  "(Each point = one occupation)")
    ax.legend(loc="center right")

    return fig


# ---------------------------------------------------------------------------
# Plot 5: Fragility by SOC Major Group (bar chart)
# ---------------------------------------------------------------------------

def plot_fragility_by_soc(frag_soc: pd.DataFrame) -> plt.Figure:
    """Bar chart of pipeline fragility by SOC major group with danger %."""
    setup_style()
    fig, ax = plt.subplots(figsize=(14, 9))

    df = frag_soc.sort_values("fragility_mean")

    colors = [
        COLORS["danger"] if dp > 40 else
        COLORS["fragility"] if dp > 20 else
        COLORS["pro_worker"]
        for dp in df["danger_pct"]
    ]

    ax.barh(
        df["soc_major_label"],
        df["fragility_mean"],
        xerr=df["fragility_std"],
        color=colors,
        alpha=0.8,
        edgecolor="white",
        capsize=3,
    )

    # Annotate with danger %
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(
            row["fragility_mean"] + row["fragility_std"] + 0.02,
            i, f"{row['danger_pct']:.0f}% danger",
            va="center", fontsize=9, color="gray",
        )

    ax.axvline(1.0, color="red", linestyle="--", alpha=0.6,
               label="Fragility = 1.0 (equal)")
    ax.set_xlabel("Pipeline Fragility Index")
    ax.set_title("Model C: Junior Pipeline Fragility by SOC Major Group\n"
                  "(Ratio > 1 = AI disproportionately eats junior tasks)")
    ax.legend()

    return fig


# ---------------------------------------------------------------------------
# Plot 6: Collaboration Mode Breakdown by SOC Group
# ---------------------------------------------------------------------------

def plot_collaboration_modes(df: pd.DataFrame) -> plt.Figure:
    """Stacked bar chart of collaboration mode shares by SOC group."""
    setup_style()
    from .acemoglu_autor_data import SOC_MAJOR_GROUPS

    modes = df.groupby("soc_major")[
        ["directive_share", "task_iteration_share", "learning_share", "feedback_share"]
    ].mean()
    modes.index = modes.index.map(lambda x: SOC_MAJOR_GROUPS.get(x, x))
    modes = modes.sort_values("directive_share")

    fig, ax = plt.subplots(figsize=(14, 9))

    mode_colors = {
        "directive_share": "#e74c3c",
        "task_iteration_share": "#3498db",
        "learning_share": "#2ecc71",
        "feedback_share": "#95a5a6",
    }
    mode_labels = {
        "directive_share": "Directive (Automation)",
        "task_iteration_share": "Task Iteration (Augmentation)",
        "learning_share": "Learning (Expertise-Leveling)",
        "feedback_share": "Feedback/Validation (Monitoring)",
    }

    bottom = np.zeros(len(modes))
    for col in modes.columns:
        ax.barh(
            modes.index,
            modes[col],
            left=bottom,
            color=mode_colors[col],
            label=mode_labels[col],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += modes[col].values

    ax.set_xlabel("Share of AI Interactions")
    ax.set_title("Collaboration Mode Breakdown by SOC Major Group\n"
                  "(Acemoglu-Autor Category Mapping)")
    ax.legend(bbox_to_anchor=(0.5, -0.12), loc="upper center", ncol=2)
    ax.set_xlim(0, 1)

    return fig


# ---------------------------------------------------------------------------
# Plot 7: ECI-PWAS Quadrant Plot
# ---------------------------------------------------------------------------

def plot_eci_pwas_quadrant(net_exp: pd.DataFrame) -> plt.Figure:
    """
    Scatter of ECI vs. PWAS per occupation with quadrant classification.

    Top-left: High PWAS, Low ECI = Pro-Worker sweet spot
    Bottom-right: Low PWAS, High ECI = Automation danger
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 10))

    class_colors = {
        "pro_worker": COLORS["pro_worker"],
        "neutral": COLORS["neutral"],
        "automation_risk": COLORS["automation_risk"],
    }

    for cls, color in class_colors.items():
        mask = net_exp["exposure_class"] == cls
        subset = net_exp[mask]
        ax.scatter(
            subset["ECI"],
            subset["PWAS"],
            c=color,
            alpha=0.5,
            s=subset["median_annual_wage"] / 2000,
            label=cls.replace("_", " ").title(),
            edgecolors="white",
            linewidth=0.5,
        )

    # Diagonal line (ECI = PWAS)
    lim_max = max(net_exp["ECI"].max(), net_exp["PWAS"].max()) * 1.1
    ax.plot([0, lim_max], [0, lim_max], "--", color="gray", alpha=0.5, label="ECI = PWAS")

    # Annotate outliers
    top_auto = net_exp.nlargest(3, "net_exposure")
    top_pro = net_exp.nsmallest(3, "net_exposure")
    for _, row in pd.concat([top_auto, top_pro]).iterrows():
        short_title = row["title"][:30] + "..." if len(row["title"]) > 30 else row["title"]
        ax.annotate(
            short_title,
            (row["ECI"], row["PWAS"]),
            fontsize=7,
            alpha=0.8,
            textcoords="offset points",
            xytext=(5, 5),
        )

    ax.set_xlabel("Expertise Commodification Index (ECI)")
    ax.set_ylabel("Pro-Worker AI Score (PWAS)")
    ax.set_title("ECI vs. PWAS Quadrant Analysis\n"
                  "(Size = median wage | Below diagonal = automation risk)")
    ax.legend(loc="upper left")

    return fig


# ---------------------------------------------------------------------------
# Plot 8: Speedup Factor vs. ECI
# ---------------------------------------------------------------------------

def plot_speedup_vs_eci(df: pd.DataFrame) -> plt.Figure:
    """
    Scatter of speedup factor vs. task-level ECI.

    Tests: are the most "sped up" tasks the most commodified?
    """
    setup_style()
    from .acemoglu_autor_models import compute_eci_task

    df = df.copy()
    df["eci_task"] = compute_eci_task(df)

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(
        df["eci_task"],
        df["speedup_factor"],
        c=df["human_education"],
        cmap="YlOrRd",
        alpha=0.4,
        s=20,
        edgecolors="none",
    )

    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("Human Education (years)")

    # Trend line
    mask = np.isfinite(df["eci_task"]) & np.isfinite(df["speedup_factor"])
    z = np.polyfit(df.loc[mask, "eci_task"], df.loc[mask, "speedup_factor"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df["eci_task"].min(), df["eci_task"].max(), 100)
    ax.plot(x_line, p(x_line), "--", color="gray", linewidth=2)

    corr = df["eci_task"].corr(df["speedup_factor"])
    ax.text(
        0.02, 0.98, f"r = {corr:.3f}",
        transform=ax.transAxes, fontsize=12, va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax.set_xlabel("Task-Level ECI")
    ax.set_ylabel("Speedup Factor (human_only_time / human_ai_time)")
    ax.set_title("Speedup Factor vs. Expertise Commodification\n"
                  "(Color = education level required)")

    return fig


# ---------------------------------------------------------------------------
# Master Function
# ---------------------------------------------------------------------------

def generate_all_visualizations(
    task_df: pd.DataFrame,
    results: Dict[str, pd.DataFrame],
) -> Dict[str, Path]:
    """
    Generate all eight visualization plots.

    Args:
        task_df: Task-level AEI v4 DataFrame.
        results: Dictionary of model results from compute_combined_results().

    Returns:
        Dictionary mapping plot names to saved file paths.
    """
    setup_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    paths = {}

    print("\nGenerating visualizations...")

    print("  [1/8] ECI vs. Wage scatter...")
    fig = plot_eci_vs_wage(results["eci_occupation"])
    paths["eci_vs_wage"] = save_figure(fig, "01_eci_vs_wage")

    print("  [2/8] PWAS by SOC major group...")
    fig = plot_pwas_by_soc(results["pwas_soc_major"])
    paths["pwas_by_soc"] = save_figure(fig, "02_pwas_by_soc")

    print("  [3/8] Net exposure by SOC major group...")
    fig = plot_net_exposure_by_soc(results["net_exposure"])
    paths["net_exposure_by_soc"] = save_figure(fig, "03_net_exposure_by_soc")

    print("  [4/8] Pipeline fragility scatter...")
    fig = plot_fragility_scatter(results["fragility_occupation"])
    paths["fragility_scatter"] = save_figure(fig, "04_fragility_scatter")

    print("  [5/8] Fragility by SOC major group...")
    fig = plot_fragility_by_soc(results["fragility_soc_major"])
    paths["fragility_by_soc"] = save_figure(fig, "05_fragility_by_soc")

    print("  [6/8] Collaboration mode breakdown...")
    fig = plot_collaboration_modes(task_df)
    paths["collaboration_modes"] = save_figure(fig, "06_collaboration_modes")

    print("  [7/8] ECI-PWAS quadrant plot...")
    fig = plot_eci_pwas_quadrant(results["net_exposure"])
    paths["eci_pwas_quadrant"] = save_figure(fig, "07_eci_pwas_quadrant")

    print("  [8/8] Speedup vs. ECI...")
    fig = plot_speedup_vs_eci(task_df)
    paths["speedup_vs_eci"] = save_figure(fig, "08_speedup_vs_eci")

    print(f"\nAll {len(paths)} visualizations saved to {FIGURES_DIR}")
    return paths
