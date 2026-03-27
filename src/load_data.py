"""
load_data.py
------------
Loads and merges all Favorita dataset components.

Memory strategy
---------------
The full train.csv is ~125M rows / ~5GB and will OOM-kill most laptops
during the merge step. By default we load a random sample of N rows,
which is sufficient for all modelling and analysis in this project.

To change the sample size set the SAMPLE_ROWS constant below, or pass
sample_rows=None to load_and_merge() to load everything (needs ~16GB RAM).

Recommended sample sizes by available RAM:
  4 GB  RAM  →  1_000_000  rows
  8 GB  RAM  →  3_000_000  rows
  16 GB RAM  →  8_000_000  rows  (or None for full dataset)
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# Default sample size — change this if you have more RAM
SAMPLE_ROWS = None


# ---------------------------------------------------------------------------
# Individual loaders
# ---------------------------------------------------------------------------

def load_sales(path: Path, sample_rows: int = SAMPLE_ROWS) -> pd.DataFrame:  # type: ignore
    """
    Load train.csv with optional row sampling.

    Sampling is done by reading the file in chunks and keeping a random
    subset — this avoids loading the full 5GB file into RAM at once.
    """
    print("  Loading train.csv ...")

    if sample_rows is not None:
        print(f"  Sampling {sample_rows:,} rows (change SAMPLE_ROWS in load_data.py to adjust)")

        # Count total rows efficiently without loading data
        # then sample row indices to read
        chunks   = []
        chunksize = 200_000
        kept      = 0
        total_read = 0

        for chunk in pd.read_csv(
            path,
            parse_dates=["date"],
            low_memory=False,
            chunksize=chunksize,
        ):
            total_read += len(chunk)

            # Keep each row with probability = sample_rows / estimated_total
            # We estimate ~125M rows; adjust keep_prob as we go
            keep_prob = min(1.0, (sample_rows - kept) / max(1, 125_000_000 - total_read + len(chunk)))
            mask      = np.random.random(len(chunk)) < keep_prob
            chunks.append(chunk[mask])
            kept += mask.sum()

            if kept >= sample_rows:
                break

            if total_read % 2_000_000 == 0:
                print(f"    Read {total_read:,} rows, kept {kept:,} ...")

        df = pd.concat(chunks, ignore_index=True)
        # Trim to exactly sample_rows if we overshot
        if len(df) > sample_rows:
            df = df.sample(sample_rows, random_state=42).reset_index(drop=True)

    else:
        print("  Loading full dataset (this may use 10GB+ RAM) ...")
        df = pd.read_csv(path, parse_dates=["date"], low_memory=False)

    print(f"    Columns found : {list(df.columns)}")
    print(f"    Rows loaded   : {len(df):,}")

    # Handle both 'sales' and 'unit_sales' column names
    if "unit_sales" not in df.columns and "sales" in df.columns:
        df = df.rename(columns={"sales": "unit_sales"})
        print("    Renamed column 'sales' -> 'unit_sales'")

    # Validate required columns
    required = {"date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"\n\nMissing columns in train.csv: {missing}\n"
            f"Columns found: {list(df.columns)}\n\n"
            f"Make sure you placed train.csv (not test.csv) in data/raw/.\n"
            f"Download from: https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data\n"
        )

    df["unit_sales"]  = df["unit_sales"].clip(lower=0).astype("float32")
    df["onpromotion"] = df["onpromotion"].fillna(0).astype("int8")
    df["store_nbr"]   = df["store_nbr"].astype("int8")
    df["item_nbr"]    = df["item_nbr"].astype("int32")

    return df


def load_stores(path: Path) -> pd.DataFrame:
    """Load stores.csv — store city / state / type / cluster."""
    return pd.read_csv(path, dtype={"store_nbr": "int8", "cluster": "int8"})


def load_items(path: Path) -> pd.DataFrame:
    """Load items.csv — item family / class / perishable."""
    return pd.read_csv(path, dtype={"item_nbr": "int32", "perishable": "int8"})


def load_transactions(path: Path) -> pd.DataFrame:
    """Load transactions.csv — daily store transaction counts."""
    return pd.read_csv(
        path,
        parse_dates=["date"],
        dtype={"store_nbr": "int8", "transactions": "int32"},
    )


def load_holidays(path: Path) -> pd.DataFrame:
    """Load holidays_events.csv."""
    return pd.read_csv(path, parse_dates=["date"])


def load_oil(path: Path) -> pd.DataFrame:
    """Load oil.csv and interpolate missing days."""
    oil = pd.read_csv(path, parse_dates=["date"])
    oil = oil.rename(columns={"dcoilwtico": "oil_price"})
    oil["oil_price"] = (
        oil["oil_price"]
        .interpolate(method="linear")
        .ffill()
        .bfill()
        .astype("float32")
    )
    return oil


# ---------------------------------------------------------------------------
# Holiday flag builder
# ---------------------------------------------------------------------------

def build_holiday_flags(holidays: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the holidays table to one binary-flag row per date.

    Flags:
      is_national_holiday  — national event, not transferred
      is_local_holiday     — local/regional event, not transferred
      is_transferred       — holiday officially moved to another date
      is_bridge            — bridge day extending a long weekend
    """
    h = holidays.copy()

    nat = (
        h[(h["locale"] == "National") & (~h["transferred"])]
        [["date"]].drop_duplicates().assign(is_national_holiday=1)
    )
    local = (
        h[(h["locale"].isin(["Local", "Regional"])) & (~h["transferred"])]
        [["date"]].drop_duplicates().assign(is_local_holiday=1)
    )
    transferred = (
        h[h["transferred"]][["date"]].drop_duplicates().assign(is_transferred=1)
    )
    bridge = (
        h[h["type"] == "Bridge"][["date"]].drop_duplicates().assign(is_bridge=1)
    )

    flags = (
        nat
        .merge(local,       on="date", how="outer")
        .merge(transferred, on="date", how="outer")
        .merge(bridge,      on="date", how="outer")
        .fillna(0)
    )
    int_cols = [c for c in flags.columns if c != "date"]
    flags[int_cols] = flags[int_cols].astype("int8")
    return flags


# ---------------------------------------------------------------------------
# Master merge  (lean — drops columns not needed downstream)
# ---------------------------------------------------------------------------

def merge_all(sales, stores, items, holidays,
              transactions=None, oil=None) -> pd.DataFrame:
    """Join all sources onto the sales spine and drop unused columns."""
    print("  Merging datasets ...")

    df = sales.merge(stores, on="store_nbr", how="left")
    df = df.merge(items,  on="item_nbr",  how="left")

    # Drop high-cardinality / unused columns to save RAM
    drop_cols = [c for c in ["id", "class"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    flags = build_holiday_flags(holidays)
    df    = df.merge(flags, on="date", how="left")
    for col in ["is_national_holiday", "is_local_holiday",
                "is_transferred", "is_bridge"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0).astype("int8")

    if transactions is not None:
        df = df.merge(transactions, on=["date", "store_nbr"], how="left")
        df["transactions"] = df["transactions"].fillna(0).astype("int32")

    if oil is not None:
        df = df.merge(oil, on="date", how="left")
        df["oil_price"] = df["oil_price"].ffill().bfill().astype("float32")

    # Force lean dtypes on string columns
    for col in ["family", "city", "state", "type"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    print(f"    Merged shape : {df.shape}")
    mem_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"    Memory usage : {mem_mb:.0f} MB")
    return df


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_and_merge(sample_rows: int = SAMPLE_ROWS) -> pd.DataFrame: # type: ignore
    """
    Full pipeline: load → merge → save parquet.

    Args:
        sample_rows : Number of rows to sample from train.csv.
                      Set to None to load the full dataset (needs ~16GB RAM).

    Returns:
        Merged DataFrame saved to data/processed/favorita_enriched.parquet
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    #print(f"[load_data] Loading raw files (sample_rows={sample_rows:,} — edit SAMPLE_ROWS in load_data.py to change) ...")

    sales    = load_sales(RAW_DIR / "train.csv", sample_rows=sample_rows)
    stores   = load_stores(RAW_DIR / "stores.csv")
    items    = load_items(RAW_DIR / "items.csv")
    holidays = load_holidays(RAW_DIR / "holidays_events.csv")

    transactions = None
    tx_path      = RAW_DIR / "transactions.csv"
    if tx_path.exists():
        print("  Loading transactions.csv ...")
        transactions = load_transactions(tx_path)

    oil      = None
    oil_path = RAW_DIR / "oil.csv"
    if oil_path.exists():
        print("  Loading oil.csv ...")
        oil = load_oil(oil_path)

    df       = merge_all(sales, stores, items, holidays, transactions, oil)
    out_path = PROCESSED_DIR / "favorita_enriched.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  Saved → {out_path}\n")
    return df


if __name__ == "__main__":
    load_and_merge()
