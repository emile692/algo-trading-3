# data/make_splits.py
import json
import os

import pandas as pd
from pathlib import Path

from tools.logger import setup_logger

logger = setup_logger()

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH  = PROJECT_ROOT / "config" / "config_test.json"
INTERIM_DIR  = PROJECT_ROOT / "data" / "interim"

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_splits(symbol: str, timeframe: str):
    cfg = load_config()
    train_ratio = cfg["split"]["train_ratio"]
    val_ratio = cfg["split"]["val_ratio"]

    df = pd.read_parquet(INTERIM_DIR / f"{symbol}_{timeframe}.parquet")
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)

    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": {"start": df["time"].iloc[0].isoformat(), "end": df["time"].iloc[n_train - 1].isoformat()},
        "val":   {"start": df["time"].iloc[n_train].isoformat(), "end": df["time"].iloc[n_train + n_val - 1].isoformat()},
        "test":  {"start": df["time"].iloc[n_train + n_val].isoformat(), "end": df["time"].iloc[-1].isoformat()}
    }

    logger.info(f"Calcul des splits (ratios train={train_ratio}, val={val_ratio})")
    logger.info(f"→ TRAIN: {splits['train']['start']} → {splits['train']['end']}")
    logger.info(f"→ VAL:   {splits['val']['start']} → {splits['val']['end']}")
    logger.info(f"→ TEST:  {splits['test']['start']} → {splits['test']['end']}")
    return splits

if __name__ == "__main__":
    compute_splits("EURUSD", "H1")
