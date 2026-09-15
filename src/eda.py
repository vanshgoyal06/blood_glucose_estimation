"""
Exploratory data analysis helpers.
Mirrors notebook cells 8-10.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import FIGURES_DIR


def summarize_distribution(df: pd.DataFrame, label: str, age_col="Age", target_col="BGL_Status"):
    print(f"\n================ {label} ================")
    print(f"Total Count: {len(df)}")

    if age_col in df.columns and target_col in df.columns:
        summary = (
            pd.crosstab(df[age_col], df[target_col], margins=True)
            .rename_axis(index="Age", columns="Status")
        )
        print(summary)
    elif target_col in df.columns:
        print(df[target_col].value_counts())
    else:
        print("Age / Target status not available for breakdown.")


def plot_histograms(df: pd.DataFrame, save_as="histograms.png"):
    df.hist(figsize=(16, 12), bins=30, edgecolor="black", grid=True)
    plt.suptitle("Histograms of Original Dataset Features", fontsize=16)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / save_as, dpi=150)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, save_as="correlation_heatmap.png"):
    corr_matrix = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, cmap="YlGnBu", annot=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / save_as, dpi=150)
    plt.close()
    return corr_matrix
