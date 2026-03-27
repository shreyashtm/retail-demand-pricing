# %% [markdown]
# # Retail Demand Forecasting & Dynamic Pricing Engine
# ### Exploratory Analysis & Results
# 
# **Dataset**: Corporación Favorita Grocery Sales (Kaggle)  
# **Goal**: Forecast item-level demand and identify which promotions drive profitable revenue.
# 
# ---
# 

# %% [markdown]
# ## 1. Setup & Data Load

# %%
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path

PROCESSED = Path('data/processed')

plt.rcParams.update({
    'figure.facecolor' : '#0f1117',
    'axes.facecolor'   : '#1a1d27',
    'axes.edgecolor'   : '#3a3d4d',
    'axes.labelcolor'  : '#e0e0e0',
    'text.color'       : '#e0e0e0',
    'xtick.color'      : '#9a9db0',
    'ytick.color'      : '#9a9db0',
    'grid.color'       : '#2a2d3d',
    'grid.alpha'       : 0.5,
    'font.family'      : 'monospace',
    'axes.titlesize'   : 13,
})
ACCENT = ['#4fd1c5','#b794f4','#f6ad55','#fc8181','#68d391']
print('Setup complete')

# %%
df = pd.read_parquet(PROCESSED/ 'favorita_enriched.parquet')
df['date'] = pd.to_datetime(df['date'])

print(f'Shape          : {df.shape}')
print(f'Date range     : {df["date"].min().date()} -> {df["date"].max().date()}')
print(f'Unique stores  : {df["store_nbr"].nunique()}')
print(f'Unique items   : {df["item_nbr"].nunique()}')
print(f'Promo rate     : {df["onpromotion"].mean():.1%}')
print(f'Zero-sale rows : {(df["unit_sales"]==0).mean():.1%}')
df.head()

# %% [markdown]
# ## 2. Demand Distribution

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].hist(df['unit_sales'].clip(0, 100), bins=60, color=ACCENT[0], alpha=0.85, edgecolor='none')
axes[0].set_title('Unit Sales Distribution (clipped at 100)')
axes[0].set_xlabel('Unit Sales'); axes[0].set_ylabel('Count')

log_sales = np.log1p(df['unit_sales'].clip(lower=0))
axes[1].hist(log_sales, bins=60, color=ACCENT[1], alpha=0.85, edgecolor='none')
axes[1].set_title('log(1 + Unit Sales) — near Normal shape')
axes[1].set_xlabel('log(1 + unit_sales)')

plt.tight_layout()
plt.savefig(PROCESSED / 'demand_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'Skewness: {df["unit_sales"].skew():.2f}')

# %% [markdown]
# ## 3. Sales Over Time

# %%
daily_sales = df.groupby('date')['unit_sales'].sum().reset_index()
daily_promo = df.groupby('date')['onpromotion'].mean().reset_index()

fig, ax1 = plt.subplots(figsize=(16, 5))
ax1.fill_between(daily_sales['date'], daily_sales['unit_sales'], color=ACCENT[0], alpha=0.25)
ax1.plot(daily_sales['date'], daily_sales['unit_sales'], color=ACCENT[0], linewidth=0.9, label='Total Daily Sales')
ax1.set_ylabel('Total Unit Sales')

ax2 = ax1.twinx()
ax2.plot(daily_promo['date'], daily_promo['onpromotion'], color=ACCENT[2], linewidth=1.2, alpha=0.8, label='Promo Rate')
ax2.set_ylabel('Promo Rate')
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

lines1, l1 = ax1.get_legend_handles_labels()
lines2, l2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, l1+l2, loc='upper left')
ax1.set_title('Daily Sales Volume vs. Promotion Rate (2013-2017)')
plt.tight_layout()
plt.savefig(PROCESSED / 'sales_over_time.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 4. Seasonality

# %%
df['day_of_week'] = df['date'].dt.dayofweek
df['month']       = df['date'].dt.month

dow_labels   = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

dow_sales   = df.groupby('day_of_week')['unit_sales'].mean()
month_sales = df.groupby('month')['unit_sales'].mean()

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].bar(range(7), dow_sales.values,
            color=[ACCENT[0] if d<5 else ACCENT[2] for d in range(7)],
            alpha=0.9, edgecolor='none', width=0.7)
axes[0].set_xticks(range(7)); axes[0].set_xticklabels(dow_labels)
axes[0].set_title('Avg Sales by Day of Week')
axes[0].set_ylabel('Avg Unit Sales')

axes[1].bar(range(1,13), month_sales.values, color=ACCENT[1], alpha=0.85, edgecolor='none', width=0.8)
axes[1].set_xticks(range(1,13)); axes[1].set_xticklabels(month_labels, rotation=45)
axes[1].set_title('Avg Sales by Month')
axes[1].set_ylabel('Avg Unit Sales')

plt.tight_layout()
plt.savefig(PROCESSED / 'seasonality.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 5. Promotion Effect on Demand

# %%
promo_stats = (df.groupby('onpromotion')['unit_sales']
               .agg(['mean','median','count']).reset_index())
promo_stats['onpromotion'] = promo_stats['onpromotion'].map({0:'Not Promoted',1:'On Promotion'})
print(promo_stats.to_string(index=False))

lift = (promo_stats.loc[1,'mean'] / promo_stats.loc[0,'mean'] - 1)*100 # type: ignore
print(f'Mean demand lift from promotion: +{lift:.1f}%')

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
colors = ['#9a9db0', ACCENT[0]]

for i, metric in enumerate(['mean','median']):
    axes[i].bar(promo_stats['onpromotion'], promo_stats[metric],
                color=colors, alpha=0.9, edgecolor='none', width=0.5)
    axes[i].set_title(f'{metric.capitalize()} Sales: Promo vs No Promo')
    axes[i].set_ylabel('Unit Sales')

plt.tight_layout()
plt.savefig(PROCESSED / 'promo_effect.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 6. Holiday Impact

# %%
if 'is_national_holiday' in df.columns:
    h = df.groupby('is_national_holiday')['unit_sales'].mean()
    h.index = h.index.map({0:'Regular Day',1:'National Holiday'})
    print(h.to_string())

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(h.index, h.values, color=['#9a9db0', ACCENT[3]], alpha=0.9, edgecolor='none', width=0.5) # type: ignore
    ax.set_title('Avg Unit Sales: Regular Day vs National Holiday')
    ax.set_ylabel('Avg Unit Sales')
    plt.tight_layout()
    plt.savefig(PROCESSED / 'holiday_effect.png', dpi=150, bbox_inches='tight')
    plt.show()
else:
    print('Run main.py first.')

# %% [markdown]
# ## 7. Model Comparison

# %%
rp = PROCESSED / 'model_results.parquet'
if rp.exists():
    results = pd.read_parquet(rp)
    print(results.to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(results['model'], results['MAE'],
                   color=ACCENT[:len(results)], alpha=0.9, edgecolor='none')
    ax.set_xlabel('Mean Absolute Error (lower = better)')
    ax.set_title('Model Comparison - Validation Set (last 8 weeks)')
    ax.invert_yaxis()
    for bar, val in zip(bars, results['MAE']):
        ax.text(bar.get_width()+0.05, bar.get_y()+bar.get_height()/2,
                f'{val:.2f}', va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(PROCESSED / 'model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
else:
    print('Run python main.py first.')

# %% [markdown]
# ## 8. Feature Importances

# %%
fp = PROCESSED / 'feature_importances.parquet'
if fp.exists():
    fi = pd.read_parquet(fp).head(15)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(fi['feature'][::-1], fi['importance'][::-1], color=ACCENT[0], alpha=0.85, edgecolor='none')
    ax.set_xlabel('Importance Score')
    ax.set_title('Top 15 Feature Importances (Random Forest)')
    plt.tight_layout()
    plt.savefig(PROCESSED / 'feature_importances.png', dpi=150, bbox_inches='tight')
    plt.show()
else:
    print('Run python main.py first.')

# %% [markdown]
# ## 9. Elasticity Distribution

# %%
ep = PROCESSED / 'elasticity.parquet'
if ep.exists():
    el = pd.read_parquet(ep)
    print(f'Segments: {len(el)}')
    print(el['elasticity'].describe().to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].hist(el['elasticity'], bins=40, color=ACCENT[1], alpha=0.85, edgecolor='none')
    axes[0].axvline(0, color=ACCENT[2], linestyle='--', linewidth=1.5, label='Zero')
    axes[0].axvline(el['elasticity'].mean(), color=ACCENT[0], linestyle='--',
                    linewidth=1.5, label=f"Mean={el['elasticity'].mean():.2f}")
    axes[0].set_title('Promotion Elasticity Distribution')
    axes[0].set_xlabel('Elasticity Coefficient'); axes[0].legend()

    fam_el = el.groupby('family')['elasticity'].mean().sort_values(ascending=False)
    axes[1].barh(fam_el.index[::-1], fam_el.values[::-1], color=ACCENT[0], alpha=0.85, edgecolor='none')
    axes[1].axvline(0, color=ACCENT[2], linestyle='--', linewidth=1)
    axes[1].set_title('Avg Elasticity by Product Family')
    axes[1].set_xlabel('Mean Elasticity')

    plt.tight_layout()
    plt.savefig(PROCESSED / 'elasticity_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
else:
    print('Run python main.py first.')

# %% [markdown]
# ## 10. Pricing Recommendations

# %%
rcp = PROCESSED / 'pricing_recommendations.parquet'
if rcp.exists():
    recs    = pd.read_parquet(rcp)
    promote = recs[recs['promote']]

    print(f'Segments evaluated          : {len(recs)}')
    print(f'Recommended for promotion   : {len(promote)} ({len(promote)/len(recs)*100:.1f}%)')
    print(f'Avg projected revenue uplift: +{promote["revenue_delta_pct"].mean():.1f}%')
    print(f'Avg recommended discount    : {promote["recommended_discount_pct"].mean():.1f}%')

    print('\nTop 15 by Revenue Upside:')
    print(promote[['family','cluster','elasticity','recommended_discount_pct','revenue_delta_pct']]
          .head(15).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    promote['recommended_discount_pct'].value_counts().sort_index().plot.bar(
        ax=axes[0], color=ACCENT[0], alpha=0.85, edgecolor='none')
    axes[0].set_xlabel('Recommended Discount (%)')
    axes[0].set_ylabel('Number of Segments')
    axes[0].set_title('Distribution of Recommended Discount Rates')
    axes[0].tick_params(axis='x', rotation=0)

    axes[1].scatter(promote['elasticity'], promote['revenue_delta_pct'],
                    c=ACCENT[0], alpha=0.7, edgecolors='none', s=60)
    axes[1].axhline(0, color=ACCENT[2], linestyle='--', linewidth=1)
    axes[1].set_xlabel('Elasticity')
    axes[1].set_ylabel('Projected Revenue Delta (%)')
    axes[1].set_title('Elasticity vs Revenue Uplift')

    plt.tight_layout()
    plt.savefig(PROCESSED / 'pricing_recommendations.png', dpi=150, bbox_inches='tight')
    plt.show()
else:
    print('Run python main.py first.')


