"""
02_feature_engineering.py  —  Feature Engineering Deep Dive
=============================================================
Demonstrates every engineered feature with:
  - Business rationale
  - Distribution before/after
  - Correlation with churn target
  - Feature importance preview

Run from notebooks/ directory or adjust sys.path as needed.
"""

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
from src.feature_engineering import FeatureEngineer

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)
os.makedirs("plots", exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════════
cfg = load_config("../config/config.yaml")
df_raw = DataLoader.generate_synthetic_data(n_samples=7043, random_state=42)

fe = FeatureEngineer(cfg)
df_eng = fe.transform(df_raw)

print("=" * 60)
print("FEATURE ENGINEERING ANALYSIS")
print("=" * 60)
print(f"Raw features     : {df_raw.shape[1]}")
print(f"Engineered total : {df_eng.shape[1]}")
print(f"New features     : {df_eng.shape[1] - df_raw.shape[1]}")

# ══════════════════════════════════════════════════════════════════════════
# 1. TENURE FEATURES
# ══════════════════════════════════════════════════════════════════════════
print("\n── 1. Tenure Features ──────────────────────────────────")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Engineered Tenure Features vs Churn", fontweight="bold")

# Tenure band churn rate
band_labels = {0: "New\n(0-6m)", 1: "Dev\n(7-12m)", 2: "Est\n(13-24m)",
               3: "Loyal\n(25-48m)", 4: "Champ\n(49m+)"}
band_churn = df_eng.groupby("tenure_band")["Churn"].mean()
colors = ["#E85D5D" if v > 0.25 else "#F4A37B" if v > 0.15 else "#4C9BE8" for v in band_churn]
axes[0].bar([band_labels[b] for b in band_churn.index], band_churn.values, color=colors)
axes[0].axhline(df_eng["Churn"].mean(), color="#555", linestyle="--", alpha=0.7)
axes[0].set_title("Churn Rate by Tenure Band")
axes[0].set_ylabel("Churn Rate")
for i, v in enumerate(band_churn.values):
    axes[0].text(i, v + 0.005, f"{v:.1%}", ha="center", fontsize=9)

# Log tenure distribution
for label, grp in df_eng.groupby("Churn"):
    grp["log_tenure"].plot.kde(ax=axes[1], label=f"Churn={label}", linewidth=2)
axes[1].set_title("Log Tenure Distribution by Churn")
axes[1].set_xlabel("log(tenure + 1)")
axes[1].legend()

# Is new customer
new_cust_churn = df_eng.groupby("is_new_customer")["Churn"].mean()
axes[2].bar(["Established\n(>6m)", "New\n(≤6m)"], new_cust_churn.values,
             color=["#4C9BE8", "#E85D5D"])
axes[2].set_title("Churn Rate: New vs Established")
axes[2].set_ylabel("Churn Rate")
for i, v in enumerate(new_cust_churn.values):
    axes[2].text(i, v + 0.005, f"{v:.1%}", ha="center", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig("plots/fe_01_tenure_features.png", dpi=150, bbox_inches="tight")
plt.show()

# ══════════════════════════════════════════════════════════════════════════
# 2. CHARGE RATIO FEATURES
# ══════════════════════════════════════════════════════════════════════════
print("── 2. Charge Ratio Features ─────────────────────────────")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Engineered Charge Features vs Churn", fontweight="bold")

charge_features = ["monthly_charge_per_tenure", "log_monthly_charges", "is_high_value"]
titles = ["Monthly Charge per Tenure Month", "Log(Monthly Charges)", "High Value Customer"]

for idx, (feat, title) in enumerate(zip(charge_features, titles)):
    if feat == "is_high_value":
        grp_churn = df_eng.groupby(feat)["Churn"].mean()
        axes[idx].bar(["Standard", "High Value"], grp_churn.values, color=["#4C9BE8", "#E85D5D"])
        axes[idx].set_ylabel("Churn Rate")
        for i, v in enumerate(grp_churn.values):
            axes[idx].text(i, v + 0.005, f"{v:.1%}", ha="center", fontsize=11)
    else:
        for label, grp in df_eng.groupby("Churn"):
            color = "#E85D5D" if label == 1 else "#4C9BE8"
            grp[feat].dropna().clip(
                grp[feat].quantile(0.01), grp[feat].quantile(0.99)
            ).plot.kde(ax=axes[idx], label=f"Churn={label}", color=color, linewidth=2)
        axes[idx].legend()
        axes[idx].set_xlabel(feat)
    axes[idx].set_title(title)

plt.tight_layout()
plt.savefig("plots/fe_02_charge_features.png", dpi=150, bbox_inches="tight")
plt.show()

# ══════════════════════════════════════════════════════════════════════════
# 3. SERVICE COUNT FEATURES
# ══════════════════════════════════════════════════════════════════════════
print("── 3. Service Count Features ─────────────────────────────")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Service Count Features vs Churn", fontweight="bold")

# Total services vs churn rate
svc_churn = df_eng.groupby("total_services")["Churn"].mean()
colors = ["#E85D5D" if v > df_eng["Churn"].mean() else "#4C9BE8" for v in svc_churn.values]
axes[0].bar(svc_churn.index, svc_churn.values, color=colors)
axes[0].axhline(df_eng["Churn"].mean(), color="#555", linestyle="--", label="Overall avg")
axes[0].set_xlabel("Number of Services Subscribed")
axes[0].set_ylabel("Churn Rate")
axes[0].set_title("Churn Rate by Total Services\n(More services = lower churn = stickier)")
axes[0].legend()

# Stickiness score distribution
for label, grp in df_eng.groupby("Churn"):
    color = "#E85D5D" if label == 1 else "#4C9BE8"
    grp["stickiness_score"].dropna().plot.kde(
        ax=axes[1], label=f"Churn={label}", color=color, linewidth=2
    )
axes[1].set_title("Stickiness Score (services × log_tenure)\nChurn vs Non-Churn")
axes[1].set_xlabel("Stickiness Score")
axes[1].legend()

plt.tight_layout()
plt.savefig("plots/fe_03_service_features.png", dpi=150, bbox_inches="tight")
plt.show()

# ══════════════════════════════════════════════════════════════════════════
# 4. CONTRACT RISK FLAGS
# ══════════════════════════════════════════════════════════════════════════
print("── 4. Contract Risk Features ─────────────────────────────")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Contract & Payment Risk Features", fontweight="bold")

risk_flags = ["is_month_to_month", "is_electronic_check",
              "is_auto_payment", "highest_churn_risk_profile"]
risk_labels = ["Month-to-Month", "Electronic Check", "Auto Payment", "Highest Risk Profile"]
churn_rates = [df_eng[f]["Churn" if f == "Churn" else f].pipe(lambda s: df_eng.groupby(f)["Churn"].mean().get(1, 0))
               for f in risk_flags]

# Compute properly
risk_data = {}
for flag, lbl in zip(risk_flags, risk_labels):
    if flag in df_eng.columns:
        risk_data[lbl] = df_eng.groupby(flag)["Churn"].mean().get(1, 0)

overall_rate = df_eng["Churn"].mean()
axes[0].barh(
    list(risk_data.keys()),
    list(risk_data.values()),
    color=["#E85D5D" if v > overall_rate else "#4C9BE8" for v in risk_data.values()],
)
axes[0].axvline(overall_rate, color="#555", linestyle="--", label=f"Overall {overall_rate:.1%}")
axes[0].set_xlabel("Churn Rate when Flag = 1")
axes[0].set_title("Churn Rate for High-Risk Profile Flags")
axes[0].legend()
for i, (lbl, val) in enumerate(risk_data.items()):
    axes[0].text(val + 0.005, i, f"{val:.1%}", va="center", fontsize=10)

# 2×2 grid: interaction effect of month-to-month × electronic_check
if "is_month_to_month" in df_eng.columns and "is_electronic_check" in df_eng.columns:
    pivot = df_eng.groupby(["is_month_to_month", "is_electronic_check"])["Churn"].mean().unstack()
    pivot.index = ["Annual/2yr", "Month-to-Month"]
    pivot.columns = ["Other Pay", "Elec. Check"]
    sns.heatmap(pivot, annot=True, fmt=".1%", cmap="Reds", ax=axes[1],
                linewidths=0.5, annot_kws={"size": 13})
    axes[1].set_title("Interaction: Contract × Payment Method\n(Churn Rate %)")
    axes[1].set_xlabel("Payment Method")
    axes[1].set_ylabel("Contract Type")

plt.tight_layout()
plt.savefig("plots/fe_04_risk_flags.png", dpi=150, bbox_inches="tight")
plt.show()

# ══════════════════════════════════════════════════════════════════════════
# 5. FEATURE CORRELATION WITH TARGET
# ══════════════════════════════════════════════════════════════════════════
print("── 5. Top Engineered Features by Target Correlation ──────")

numeric_eng_cols = df_eng.select_dtypes(include=[np.number]).columns.tolist()
target_corr = df_eng[numeric_eng_cols].corr()["Churn"].drop("Churn").abs().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 8))
top_n = 20
top_corr = target_corr.head(top_n)
colors = ["#E85D5D" if f in fe.get_new_feature_names() else "#4C9BE8" for f in top_corr.index]
ax.barh(range(len(top_corr)), top_corr.values[::-1], color=colors[::-1])
ax.set_yticks(range(len(top_corr)))
ax.set_yticklabels(top_corr.index[::-1])
ax.set_xlabel("|Pearson Correlation with Churn|")
ax.set_title(f"Top {top_n} Features by Absolute Correlation with Target\n(Red = Engineered, Blue = Raw)")

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor="#E85D5D", label="Engineered feature"),
                   Patch(facecolor="#4C9BE8", label="Raw feature")]
ax.legend(handles=legend_elements, loc="lower right")

plt.tight_layout()
plt.savefig("plots/fe_05_target_correlation.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n── Top 15 features by correlation with Churn ───────────")
print(target_corr.head(15).to_string())

print("\n" + "=" * 60)
print("Feature engineering analysis complete.")
print(f"Engineered {len(fe.get_new_feature_names())} new features.")
print("Plots saved to: plots/")
print("=" * 60)
