import csv
import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional

from tools.logger import setup_logger

logger = setup_logger()

DATA_DIR = Path("../data")
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
CONFIG_PATH = Path(f"../config/data_sources.yaml")

CANONICAL_COLS = ["time", "open", "high", "low", "close", "volume"]

def _auto_sep(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except Exception:
        return ","  # fallback

def _read_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _pick_first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in df.columns:
            return c.lower()
    return None

def _build_time(df: pd.DataFrame, time_cfg: dict, tz: str) -> pd.Series:
    """
    Construit une colonne temporelle unifiée à partir des champs spécifiés.
    """
    if "from_unix" in time_cfg:
        unit = time_cfg.get("unit", "s")
        col = time_cfg["from_unix"]
        s = pd.to_datetime(df[col], unit=unit, utc=True)
    elif "from_column" in time_cfg:
        col = time_cfg["from_column"]
        s = pd.to_datetime(df[col], utc=True, errors="coerce")
    elif {"date_str", "hour_str"} <= set(time_cfg.keys()):
        ds = df[time_cfg["date_str"]]
        hs = df[time_cfg["hour_str"]].astype(str).str.zfill(2)
        s = pd.to_datetime(ds + " " + hs + ":00", utc=True, errors="coerce")
    else:
        raise ValueError("Config temps non reconnue (from_unix / from_column / date_str+hour_str).")

    # Gestion de la timezone
    if tz.upper() != "UTC":
        try:
            s = s.dt.tz_convert(tz)
        except TypeError:
            s = s.dt.tz_localize("UTC").dt.tz_convert(tz)
    # On retire le tz pour garder une datetime naive locale
    s = s.dt.tz_localize(None)
    return s


def make_interim(symbol: str, timeframe: str, source: str):
    """
    Normalise un CSV brut (spécifique à une source) vers le schéma canonique,
    puis sauvegarde en Parquet dans data/interim/.
    """
    cfg = _read_config()
    defaults = cfg.get("defaults", {})
    src = cfg["sources"][source]

    pattern = src.get("filename_pattern", "{symbol}_{timeframe}.csv")
    raw_path = RAW_DIR / pattern.format(symbol=symbol, timeframe=timeframe)

    # Pré-détection séparateur si demandé
    sep = src.get("sep", defaults.get("sep", "auto"))
    encoding = src.get("encoding", defaults.get("encoding", "utf-8"))
    decimal = src.get("decimal", defaults.get("decimal", "."))
    tz = src.get("timezone", defaults.get("timezone", "UTC"))

    if sep == "auto":
        with open(raw_path, "r", encoding=encoding, errors="ignore") as f:
            sample = f.read(2048)
        sep = _auto_sep(sample)

    logger.info(f"Reading {raw_path} with sep='{sep}', encoding='{encoding}'")

    df = pd.read_csv(raw_path, sep=sep, encoding=encoding, decimal=decimal)
    # uniformiser les noms pour la suite
    df.columns = [c.strip() for c in df.columns]

    # rename → canonique
    rename_map: Dict[str, str] = src.get("rename", {})
    # support des clés insensibles à la casse
    lower_map = {k.lower(): v for k, v in rename_map.items()}
    df.columns = [lower_map.get(c.lower(), c) for c in df.columns]

    # construire la colonne time
    time_cfg = src.get("time", {})
    df["time"] = _build_time(df, time_cfg, tz)

    # choisir la colonne volume selon préférence
    vol_pref = src.get("volume_preference", defaults.get("volume_preference", ["volume"]))
    vol_col = _pick_first_present(df, vol_pref)
    if vol_col and vol_col != "volume":
        df.rename(columns={vol_col: "volume"}, inplace=True)

    # garder uniquement les colonnes essentielles
    missing = [c for c in ["open","high","low","close","volume"] if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes après mapping: {missing}")

    out = df[["time","open","high","low","close","volume"]].copy()

    # nettoyage
    out = out.dropna(subset=["time"]).sort_values("time").drop_duplicates("time")
    if src.get("drop_missing", True):
        out = out.dropna(subset=["open","high","low","close"])

    # validations simples
    if not out["time"].is_monotonic_increasing:
        logger.warning("time non strictement croissant — tri appliqué.")
    if out.duplicated("time").any():
        raise ValueError("Duplicates sur time après nettoyage.")

    # I/O
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    out_path = INTERIM_DIR / f"{symbol}_{timeframe}.parquet"
    out.to_parquet(out_path, index=False)
    logger.info(f"Saved cleaned data to {out_path}")

    return out

if __name__ == "__main__":

    make_interim(symbol="EURUSD", timeframe="H1", source="generic")
