# Data

## Source

**Corporación Favorita Grocery Sales Forecasting**
https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data

## Required files — place in `data/raw/`

| File                  | Required    | Rows   | Description                         |
|-----------------------|-------------|--------|-------------------------------------|
| `train.csv`           | ✅          | ~125M  | Sales transactions (2013–2017)      |
| `stores.csv`          | ✅          | 54     | Store city / state / type / cluster |
| `items.csv`           | ✅          | ~4,100 | Item family / class / perishable    |
| `holidays_events.csv` | ✅          | ~350   | Ecuador holiday calendar            |
| `transactions.csv`    | Recommended | ~83K   | Daily store transaction counts      |
| `oil.csv`             | Optional    | ~1,200 | WTI crude oil price                 |
|-----------------------|-------------|----------------------------------------------|

## How to download

1. Go to: https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data
2. Accept competition rules
3. Download and extract all CSV files
4. Place them in `data/raw/`

## Working with a smaller sample

`train.csv` is ~5 GB and will OOM-kill the process on most laptops if loaded
in full. The pipeline handles this automatically — it reads the file in chunks
and keeps a random sample without ever loading the full file into memory.

To control the sample size, edit line 25 of `src/load_data.py`:
```python
SAMPLE_ROWS = 2_000_000   # default — works on 4GB RAM
SAMPLE_ROWS = 5_000_000   # needs ~8GB RAM
SAMPLE_ROWS = None        # full dataset — needs 16GB+ RAM
```

Do not manually overwrite or truncate `train.csv` — keep the raw file intact
so you can re-run with a different sample size without re-downloading.

## Processed outputs

`data/processed/` is populated automatically by `python main.py`:

| File                              | Description                                 |
|-----------------------------------|---------------------------------------------|
| `favorita_enriched.parquet`       | Merged dataset with all features            |
| `model_results.parquet`           | MAE / RMSE for all models                   |
| `feature_importances.parquet`     | Random Forest feature importances           |
| `elasticity.parquet`              | Segment-level elasticity estimates          |
| `pricing_recommendations.parquet` | Discount recommendations per segment        |
|-----------------------------------|---------------------------------------------|