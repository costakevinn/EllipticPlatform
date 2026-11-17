"""Build the Bronze layer from the Raw layer.

Reads the three Elliptic CSV files, gives each column a name and writes
one Parquet file per table. The values are not changed.

Run:  docker compose run --rm dev uv run python src/build_bronze.py
"""

from pathlib import Path

import pandas as pd

# --- Settings ---

RAW_DIR = Path("data/raw/elliptic/elliptic_bitcoin_dataset")
BRONZE_DIR = Path("data/bronze/elliptic")

# --- Column names ---

CLASSES_COLUMNS = ["tx_id", "class"]
EDGES_COLUMNS = ["tx_from", "tx_to"]

# The features file has no header, so its columns are named by position.
# The split into 93 local and 72 aggregated features comes from the dataset
# documentation; the data itself does not show it.
FEATURES_COLUMNS = (
    ["tx_id", "time_step"]
    + [f"local_{i}" for i in range(1, 94)]
    + [f"agg_{i}" for i in range(1, 73)]
)

# --- Conversion ---


def convert(table, csv_name, columns, has_header):
    """Read one raw CSV, name its columns and write it as Parquet."""
    header = 0 if has_header else None
    df = pd.read_csv(RAW_DIR / csv_name, header=header, names=columns)
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    path = BRONZE_DIR / f"{table}.parquet"
    df.to_parquet(path, index=False)
    print(f"{table}: {len(df):,} rows, {len(df.columns)} columns -> {path}")


# --- Check ---

# Shapes found in the EDA (rows, columns).
EXPECTED_SHAPES = {
    "classes": (203769, 2),
    "edges": (234355, 2),
    "features": (203769, 167),
}


def check():
    """Read the Parquet files back and compare their shapes with the EDA."""
    for table, expected in EXPECTED_SHAPES.items():
        df = pd.read_parquet(BRONZE_DIR / f"{table}.parquet")
        if df.shape != expected:
            raise SystemExit(f"{table}: expected {expected}, got {df.shape}")
    print("Bronze check passed: all shapes match the EDA.")


# --- Entry point ---


def main():
    convert("classes", "elliptic_txs_classes.csv", CLASSES_COLUMNS, True)
    convert("edges", "elliptic_txs_edgelist.csv", EDGES_COLUMNS, True)
    convert("features", "elliptic_txs_features.csv", FEATURES_COLUMNS, False)
    check()


if __name__ == "__main__":
    main()
