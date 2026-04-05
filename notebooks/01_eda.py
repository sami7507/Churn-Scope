"""
01_eda.py  —  Exploratory Data Analysis
========================================
Run as a script or paste cells into a Jupyter notebook.
Produces all EDA plots and prints a data quality report.

Sections:
  1. Dataset overview & data types
  2. Target distribution & class imbalance
  3. Numeric feature distributions
  4. Categorical feature vs churn rates
  5. Correlation heatmap
  6. Key business insights summary
"""

# ── Imports ────────────────────────────────────────────────────────────────
import os
import sys
sys.path.insert(0, os.path.abspath(".."))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from src import load_config
from src.data_loader import DataLoader

# ── Style ──────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PALETTE = {"No Churn": "#4C9BE8", "Churn": "#E85D5D"}
os.makedirs("plots", exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("CUSTOMER CHURN — EXPLORATORY DATA ANALYSIS")
print("=" * 60)

cfg = load_config("../config/config.yaml")
df = DataLoader.generate_synthetic_data(n_samples=7043, random_state=42)

# Map target back to labels for plots
df["Churn_Label"] = df["Churn"].map({1: "Churn", 0: "No Churn"})

print(f"\nDataset shape : {df.shape}")
print(f"Churn rate    : {df['Churn'].mean():.1%}  ({df['Churn'].sum():,} churners)")
print(f"\nColumn dtypes:\n{df.dtypes.value_counts()}")

# ══════════════════════════════════════════════════════════════════════════
# 2. DATA QUALITY REPORT
# ══════════════════════════════════════════════════════════════════════════
print("\n── Missing Values ──────────────────────────────────────")
null_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
cols_with_nulls = null_pct[null_pct > 0]
if cols_with_nulls.empty:
    print("  No missing values detected ✓")
else:
    print(cols_with_nulls.to_string())

print("\n── Numeric Feature Summary ─────────────────────────────")
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
print(df[num_cols].describe().round(2).to_string())

# ══════════════════════════════════════════════════════════════════════════
# 3. TARGET DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Target Distribution — Customer Churn", fontsize=14, fontweight="bold")

# Pie chart
churn_counts = df["Churn_Label"].value_counts()
axes[0].pie(
    churn_counts,
    labels=churn_counts.index,
    autopct="%1.1f%%",
    colors=[PALETTE["Churn"], PALETTE["No Churn"]],
    startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 2},
)
axes[0].set_title("Overall Churn Split")

# Bar chart with counts
sns.countplot(data=df, x="Churn_Label", palette=PALETTE, ax=axes[1], order=["No Churn", "Churn"])
for p in axes[1].patches:
    axes[1].annotate(
        f"{int(p.get_height()):,}\n({p.get_height()/len(df):.1%})",
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center", va="bottom", fontsize=11,
    )
axes[1].set_title("Customer Count by Churn Status")
axes[1].set_xlabel("")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig("plots/01_target_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved: plots/01_target_distribution.png")

# ══════════════════════════════════════════════════════════════════════════
# 4. NUMERIC FEATURE DISTRIBUTIONS vs CHURN
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Numeric Feature Distributions by Churn Status", fontsize=14, fontweight="bold")

num_features = ["tenure", "MonthlyCharges", "TotalCharges"]

for col_idx, col in enumerate(num_features):
    # KDE plot
    ax_kde = axes[0, col_idx]
    for label, grp in df.groupby("Churn_Label"):
        grp[col].dropna().plot.kde(ax=ax_kde, label=label, color=PALETTE[label], linewidth=2)
    ax_kde.set_title(f"{col} — Density")
    ax_kde.set_xlabel(col)
    ax_kde.legend()

    # Box plot
    ax_box = axes[1, col_idx]
    sns.boxplot(
        data=df, x="Churn_Label", y=col,
        palette=PALETTE, ax=ax_box, order=["No Churn", "Churn"],
        linewidth=1.2,
    )
    ax_box.set_title(f"{col} — Box Plot")
    ax_box.set_xlabel("")

plt.tight_layout()
plt.savefig("plots/02_numeric_distributions.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved: plots/02_numeric_distributions.png")

# Print group means
print("\n── Numeric Feature Means by Churn Status ───────────────")
print(df.groupby("Churn_Label")[num_features].mean().round(2).to_string())

# ══════════════════════════════════════════════════════════════════════════
# 5. CATEGORICAL FEATURE CHURN RATES
# ══════════════════════════════════════════════════════════════════════════
cat_features = [
    "Contract", "InternetService", "PaymentMethod",
    "TechSupport", "OnlineSecurity", "PaperlessBilling",
]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Churn Rate by Categorical Feature", fontsize=14, fontweight="bold")
axes = axes.flatten()

for idx, col in enumerate(cat_features):
    churn_rate = df.groupby(col)["Churn"].mean().sort_values(ascending=False)
    bars = axes[idx].bar(
        range(len(churn_rate)),
        churn_rate.values,
        color=[
            "#E85D5D" if v > df["Churn"].mean() else "#4C9BE8"
            for v in churn_rate.values
        ],
        edgecolor="white",
    )
    axes[idx].axhline(df["Churn"].mean(), color="#888", linestyle="--", linewidth=1.2, label="Overall avg")
    axes[idx].set_xticks(range(len(churn_rate)))
    axes[idx].set_xticklabels(churn_rate.index, rotation=25, ha="right", fontsize=9)
    axes[idx].set_title(col)
    axes[idx].set_ylabel("Churn Rate")
    axes[idx].set_ylim(0, min(1.0, churn_rate.max() * 1.3))
    axes[idx].legend(fontsize=8)

    # Annotate bars
    for bar, val in zip(bars, churn_rate.values):
        axes[idx].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.1%}", ha="center", va="bottom", fontsize=8,
        )

plt.tight_layout()
plt.savefig("plots/03_categorical_churn_rates.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved: plots/03_categorical_churn_rates.png")

# ══════════════════════════════════════════════════════════════════════════
# 6. TENURE vs CHURN RATE (BINNED)
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5))

df["tenure_bin"] = pd.cut(df["tenure"], bins=[0, 6, 12, 24, 36, 48, 72],
                           labels=["0-6m", "7-12m", "13-24m", "25-36m", "37-48m", "49-72m"])
tenure_churn = df.groupby("tenure_bin", observed=True)["Churn"].agg(["mean", "count"])

color_bars = ["#E85D5D" if v > 0.25 else "#F4A37B" if v > 0.15 else "#4C9BE8"
              for v in tenure_churn["mean"]]
bars = ax.bar(range(len(tenure_churn)), tenure_churn["mean"], color=color_bars, edgecolor="white")
ax.axhline(df["Churn"].mean(), color="#555", linestyle="--", label=f"Overall avg ({df['Churn'].mean():.1%})")

for bar, (idx, row) in zip(bars, tenure_churn.iterrows()):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{row['mean']:.1%}\n(n={row['count']:,})", ha="center", va="bottom", fontsize=9)

ax.set_xticks(range(len(tenure_churn)))
ax.set_xticklabels(tenure_churn.index)
ax.set_xlabel("Tenure Band")
ax.set_ylabel("Churn Rate")
ax.set_title("Churn Rate by Customer Tenure Band\n(Red = High risk, Blue = Low risk)")
ax.legend()
plt.tight_layout()
plt.savefig("plots/04_tenure_churn_rate.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved: plots/04_tenure_churn_rate.png")

# ══════════════════════════════════════════════════════════════════════════
# 7. MONTHLY CHARGES vs TENURE SCATTER (CHURN vs NON-CHURN)
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))

for label, grp in df.groupby("Churn_Label"):
    ax.scatter(
        grp["tenure"], grp["MonthlyCharges"],
        label=label, alpha=0.3, s=15,
        color=PALETTE[label],
    )

ax.set_xlabel("Tenure (months)")
ax.set_ylabel("Monthly Charges ($)")
ax.set_title("Monthly Charges vs Tenure — Churn Pattern\n(Top-left = High risk zone)")
ax.legend()

# Annotate high-risk zone
ax.axvspan(0, 12, alpha=0.05, color="red", label="High-risk zone")
ax.text(3, 115, "High-risk\nzone", color="#E85D5D", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("plots/05_scatter_charges_tenure.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved: plots/05_scatter_charges_tenure.png")

# ══════════════════════════════════════════════════════════════════════════
# 8. CORRELATION HEATMAP (NUMERIC FEATURES)
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 6))

corr_cols = ["Churn", "tenure", "MonthlyCharges", "TotalCharges"]
corr = df[corr_cols].corr()

mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f",
    cmap="RdBu_r", center=0, vmin=-1, vmax=1,
    square=True, linewidths=0.5, ax=ax,
    annot_kws={"size": 11},
)
ax.set_title("Correlation Matrix — Numeric Features", fontweight="bold")
plt.tight_layout()
plt.savefig("plots/06_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved: plots/06_correlation_heatmap.png")

# ══════════════════════════════════════════════════════════════════════════
# 9. KEY BUSINESS INSIGHTS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("KEY BUSINESS INSIGHTS FROM EDA")
print("=" * 60)

mtm_churn = df[df["Contract"] == "Month-to-month"]["Churn"].mean()
two_yr_churn = df[df["Contract"] == "Two year"]["Churn"].mean()
fiber_churn = df[df["InternetService"] == "Fiber optic"]["Churn"].mean()
new_cust_churn = df[df["tenure"] <= 6]["Churn"].mean()
loyal_churn = df[df["tenure"] >= 24]["Churn"].mean()
elec_churn = df[df["PaymentMethod"] == "Electronic check"]["Churn"].mean()

print(f"\n1. Contract type is the strongest predictor:")
print(f"   Month-to-month churn rate : {mtm_churn:.1%}")
print(f"   Two-year contract churn   : {two_yr_churn:.1%}")
print(f"   Ratio                     : {mtm_churn/two_yr_churn:.1f}x higher risk")

print(f"\n2. Tenure is inversely correlated with churn:")
print(f"   New customers (≤6 months) churn at : {new_cust_churn:.1%}")
print(f"   Loyal customers (≥24 months) churn : {loyal_churn:.1%}")

print(f"\n3. Fiber optic customers churn more (premium pricing sensitivity):")
print(f"   Fiber optic churn rate : {fiber_churn:.1%}")

print(f"\n4. Electronic check payment is a high-risk signal:")
print(f"   Electronic check churn rate : {elec_churn:.1%}")

print(f"\n5. Class imbalance: {df['Churn'].mean():.1%} positive class")
print(f"   → Use stratified splits, class_weight='balanced', PR-AUC over ROC-AUC")

print("\n" + "=" * 60)
print(f"All EDA plots saved to: plots/")
print("=" * 60)
