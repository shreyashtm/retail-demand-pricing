# %% [markdown]
# # Retail Demand Forecasting & Dynamic Pricing Engine
# ### Exploratory Analysis & Results Notebook
#
# **Dataset**: Corporación Favorita Grocery Sales (Kaggle)
#
# **Goal**: Forecast item-level demand and identify which promotions
# drive profitable revenue through price elasticity analysis.
#
# ---

# %% [markdown]
# ## Setup

# %%
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path

# Use a clean style
plt.rcParams.update({
    "figure.facecolor" : "#0f1117",
    "axes.facecolor"   : "#1a1d27",
    "axes.edgecolor"   : "#3a3d4d",
    "axes.labelcolor"  : "#e0e0e0",
    "text.color"       : "#e0e0e0",
    "xtick.color"      : "#9a9db0",
    "ytick.color"      : "#9a9db0",
    "grid.color"       : "#2a2d3d",
    "grid.alpha"       : 0.5,
    "font.family"      : "monospace",
})

PROCESSED = Path("data/processed")

# %% [markdown]
# ## 1. Data Overview

# %%
df = pd.read_parquet(PROCESSED / "favorita_enriched.parquet")
df["date"] = pd.to_datetime(df["date"])
print(f"Shape: {df.shape}")
df.head()

# %%
print("Date range :", df["date"].min().date(), "→", df["date"].max().date())
print("Stores     :", df["store_nbr"].nunique())
print("Items      :", df["item_nbr"].nunique())
print("Families   :", df["family"].nunique() if "family" in df.columns else "N/A")
print("Promo rate :", f"{df['onpromotion'].mean():.1%}")

# %% [markdown]
# ## 2. Demand Distribution

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Raw distribution
axes[0].hist(df["unit_sales"].clip(0, 100), bins=60, color="#4fd1c5", alpha=0.85, edgecolor="none")
axes[0].set_title("Unit Sales Distribution (clipped at 100)")
axes[0].set_xlabel("Unit Sales")
axes[0].set_ylabel("Count")

# Log distribution
log_sales = np.log1p(df["unit_sales"].clip(lower=0))
axes[1].hist(log_sales, bins=60, color="#b794f4", alpha=0.85, edgecolor="none")
axes[1].set_title("Log(1 + Unit Sales) Distribution")
axes[1].set_xlabel("log(1 + Unit Sales)")

plt.tight_layout()
plt.savefig("data/processed/demand_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Sales Over Time

# %%
daily_sales = df.groupby("date")["unit_sales"].sum().reset_index()
daily_promo = df.groupby("date")["onpromotion"].mean().reset_index()

fig, ax1 = plt.subplots(figsize=(16, 5))

ax1.fill_between(daily_sales["date"], daily_sales["unit_sales"],
                 color="#4fd1c5", alpha=0.3)
ax1.plot(daily_sales["date"], daily_sales["unit_sales"],
         color="#4fd1c5", linewidth=0.8, label="Total Daily Sales")
ax1.set_ylabel("Total Unit Sales")
ax1.set_xlabel("")

ax2 = ax1.twinx()
ax2.plot(daily_promo["date"], daily_promo["onpromotion"],
         color="#f6ad55", linewidth=1.2, alpha=0.7, label="Promo Rate")
ax2.set_ylabel("Promo Rate")
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

ax1.set_title("Daily Sales Volume vs. Promotion Rate Over Time")
plt.tight_layout()
plt.savefig("data/processed/sales_over_time.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Sales by Day of Week

# %%
dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
df["day_of_week"] = df["date"].dt.dayofweek
dow_sales = df.groupby("day_of_week")["unit_sales"].mean().reset_index()

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(dow_sales["day_of_week"], dow_sales["unit_sales"],
              color=["#4fd1c5" if d < 5 else "#f6ad55" for d in dow_sales["day_of_week"]],
              alpha=0.9, edgecolor="none", width=0.7)
ax.set_xticks(range(7))
ax.set_xticklabels(dow_labels)
ax.set_title("Average Unit Sales by Day of Week")
ax.set_ylabel("Avg Unit Sales")
plt.tight_layout()
plt.savefig("data/processed/sales_by_dow.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Promotion Effect on Demand

# %%
promo_effect = (
    df.groupby("onpromotion")["unit_sales"]
    .agg(["mean", "median", "count"])
    .reset_index()
)
promo_effect.columns = ["on_promo", "mean_sales", "median_sales", "count"]
promo_effect["on_promo"] = promo_effect["on_promo"].map({0: "Not Promoted", 1: "On Promotion"})
print(promo_effect.to_string(index=False))

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

colors = ["#9a9db0", "#4fd1c5"]
for i, metric in enumerate(["mean_sales", "median_sales"]):
    axes[i].bar(promo_effect["on_promo"], promo_effect[metric],
                color=colors, alpha=0.9, edgecolor="none", width=0.5)
    axes[i].set_title(f"{'Mean' if i == 0 else 'Median'} Sales: Promo vs No Promo")
    axes[i].set_ylabel("Unit Sales")

plt.tight_layout()
plt.savefig("data/processed/promo_effect.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Model Comparison

# %%
results_path = PROCESSED / "model_results.parquet"
if results_path.exists():
    results = pd.read_parquet(results_path)
    print(results.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(results["model"], results["MAE"],
                   color=["#9a9db0", "#4fd1c5", "#b794f4"],
                   alpha=0.9, edgecolor="none")
    ax.set_xlabel("Mean Absolute Error (lower = better)")
    ax.set_title("Model Comparison on Validation Set")
    ax.invert_yaxis()

    for bar, val in zip(bars, results["MAE"]):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig("data/processed/model_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
else:
    print("Run main.py first to generate model results.")

# %% [markdown]
# ## 7. Feature Importances

# %%
fi_path = PROCESSED / "feature_importances.parquet"
if fi_path.exists():
    fi = pd.read_parquet(fi_path).head(15)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(fi["feature"][::-1], fi["importance"][::-1],
            color="#4fd1c5", alpha=0.85, edgecolor="none")
    ax.set_xlabel("Importance")
    ax.set_title("Top 15 Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.savefig("data/processed/feature_importances.png", dpi=150, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## 8. Elasticity Distribution

# %%
el_path = PROCESSED / "elasticity.parquet"
if el_path.exists():
    el = pd.read_parquet(el_path)
    print(f"Segments: {len(el)}")
    print(el["elasticity"].describe())

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(el["elasticity"], bins=40, color="#b794f4", alpha=0.85, edgecolor="none")
    ax.axvline(0, color="#f6ad55", linestyle="--", linewidth=1.5, label="Zero effect")
    ax.axvline(el["elasticity"].mean(), color="#4fd1c5", linestyle="--",
               linewidth=1.5, label=f"Mean = {el['elasticity'].mean():.2f}")
    ax.set_title("Distribution of Promotion Elasticity Across Segments")
    ax.set_xlabel("Elasticity Coefficient")
    ax.legend()
    plt.tight_layout()
    plt.savefig("data/processed/elasticity_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## 9. Pricing Recommendations

# %%
rec_path = PROCESSED / "pricing_recommendations.parquet"
if rec_path.exists():
    recs = pd.read_parquet(rec_path)
    promote = recs[recs["promote"]]

    print(f"Segments recommended for promotion: {len(promote)} / {len(recs)}")
    print(f"Average revenue uplift: +{promote['revenue_delta_pct'].mean():.1f}%")
    print("\nTop 10 by Revenue Upside:")
    print(promote[["family", "cluster", "elasticity",
                   "recommended_discount_pct", "revenue_delta_pct"]].head(10).to_string(index=False))

    # Visualise discount distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    promote["recommended_discount_pct"].value_counts().sort_index().plot.bar(
        ax=ax, color="#4fd1c5", alpha=0.85, edgecolor="none"
    )
    ax.set_xlabel("Recommended Discount (%)")
    ax.set_ylabel("Number of Segments")
    ax.set_title("Distribution of Recommended Discount Rates")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("data/processed/discount_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()
