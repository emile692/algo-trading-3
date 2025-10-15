# data/make_features.py
import json
from pathlib import Path
import pandas as pd
import numpy as np

from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator, KAMAIndicator
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator, CCIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice

from tools.logger import setup_logger
from exogenous_model_v0.utils.fracdiff import FracDifferentiator
from data.make_splits import compute_splits

logger = setup_logger()

# --- Paths ---
DATA_DIR = Path("../data")
INTERIM_DIR = DATA_DIR / "interim"
FEATURES_DIR = DATA_DIR / "features"
EXTERNAL_DIR = DATA_DIR / "external"
CONFIG_PATH = Path("../config/config_test.json")


# === Helper functions ===

def _load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_cols(df):
    needed = ["open", "high", "low", "close", "volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le dataset : {missing}")


def _set_time_index(df):
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").drop_duplicates("time").set_index("time")
    return df


def _slice(df, start_iso, end_iso):
    mask = (df.index >= pd.Timestamp(start_iso)) & (df.index <= pd.Timestamp(end_iso))
    return df.loc[mask].copy()


def _fit_fracdiff_on_train(df_train, config):
    """Fit d uniquement sur TRAIN."""
    if not config["features"]["fracdiff"]["enable"]:
        logger.info("[FFD] désactivée dans la config.")
        return None

    series = np.log(df_train["close"]) - np.log(df_train["close"]).mean()
    frac = FracDifferentiator(thresh=config["features"]["fracdiff"]["thresh"])
    d_opt, pval = frac.fit(series.ffill().bfill())

    config["features"]["fracdiff"]["d_optimal"] = d_opt
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    logger.info(f"[FFD] d(fit on train)={d_opt:.4f}, ADF p-val={pval:.4f}")
    return FracDifferentiator(d=d_opt)


def _apply_fracdiff(df_slice, frac):
    if frac is None:
        df_slice["frac_diff"] = np.nan
        return df_slice
    s = np.log(df_slice["close"]) - np.log(df_slice["close"]).mean()
    try:
        df_slice["frac_diff"] = frac.transform(s.ffill().bfill())
    except ValueError as e:
        logger.warning(f"FracDiff échouée sur split : {e}")
        df_slice["frac_diff"] = np.nan
    return df_slice


def _add_technical_indicators(df, cfg):
    f = cfg["features"]["tech"]

    # SMA & EMA
    for w in f.get("sma", []):
        df[f"sma_{w}"] = SMAIndicator(df["close"], window=int(w)).sma_indicator()
    for w in f.get("ema", []):
        df[f"ema_{w}"] = EMAIndicator(df["close"], window=int(w)).ema_indicator()

    # RSI
    rsi_w = int(f.get("rsi", 14))
    rsi = RSIIndicator(df["close"], window=rsi_w).rsi()
    df["rsi"] = rsi
    df["rsi_dist_oversold"] = df["rsi"] - 30
    df["rsi_dist_overbought"] = 70 - df["rsi"]
    df["rsi_signal"] = ((df["rsi"] > 70) | (df["rsi"] < 30)).astype(int)

    # Bollinger Bands
    bb = BollingerBands(df["close"], window=f["bb"]["window"], window_dev=f["bb"]["dev"])
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]
    df["bb_dist_upper"] = df["bb_upper"] - df["close"]
    df["bb_dist_lower"] = df["close"] - df["bb_lower"]

    # MACD
    macd = MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_diff"] = macd.macd_diff()

    # Stochastic Oscillator
    stoch = StochasticOscillator(df["high"], df["low"], df["close"])
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # OBV
    df["obv"] = OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()

    # Volatility / Trend
    df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"], window=f["atr"]).average_true_range()
    df["adx"] = ADXIndicator(df["high"], df["low"], df["close"], window=f["adx"]).adx()
    df["cci"] = CCIIndicator(df["high"], df["low"], df["close"], window=f["cci"]).cci()
    df["williams_r"] = WilliamsRIndicator(df["high"], df["low"], df["close"], lbp=f["stoch"]).williams_r()
    df["roc"] = ROCIndicator(df["close"], window=f["roc"]).roc()
    k = f["kama"]
    df["kama"] = KAMAIndicator(df["close"], window=k["window"], pow1=k["pow1"], pow2=k["pow2"]).kama()
    df["vwap"] = VolumeWeightedAveragePrice(df["high"], df["low"], df["close"], df["volume"], window=f["vwap"]).volume_weighted_average_price()

    # Relations SMA
    if "sma_50" in df and "sma_200" in df:
        df["above_sma_50"] = (df["close"] > df["sma_50"]).astype(int)
        df["above_sma_200"] = (df["close"] > df["sma_200"]).astype(int)
        df["sma_50_vs_200"] = (df["sma_50"] > df["sma_200"]).astype(int)

    return df


def _add_vix(df, cfg):
    vcfg = cfg["exogenous"]["vix"]
    if not vcfg.get("enable", False):
        return df
    cache_path = Path(vcfg["cache_path"])
    if not cache_path.exists():
        logger.warning(f"⚠️ VIX cache introuvable: {cache_path}")
        return df

    vix = pd.read_parquet(cache_path)
    vix.index = pd.to_datetime(vix.index)
    vix_hourly = vix.resample(vcfg.get("resample", "h")).ffill()
    df = df.merge(vix_hourly, left_index=True, right_index=True, how="left")
    return df


def _compute_split(df_full, start, end, frac, cfg, split_name):
    df_s = _slice(df_full, start, end)
    df_s = _apply_fracdiff(df_s, frac)
    df_s = _add_technical_indicators(df_s, cfg)
    df_s = _add_vix(df_s, cfg)
    df_s["split"] = split_name
    return df_s.dropna()


# === MAIN FUNCTION ===

def make_features(symbol: str, timeframe: str):
    cfg = _load_config()
    splits = compute_splits(symbol, timeframe)  # ← basé sur ratios

    df = pd.read_parquet(INTERIM_DIR / f"{symbol}_{timeframe}.parquet")
    _ensure_cols(df)
    df = _set_time_index(df)

    # Fit fracdiff uniquement sur TRAIN
    df_train = _slice(df, splits["train"]["start"], splits["train"]["end"])
    frac = _fit_fracdiff_on_train(df_train, cfg)

    # Compute par split (no leak)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    parts = []
    saved_paths = {}

    for name in ("train", "val", "test"):
        s = splits[name]
        part = _compute_split(df, s["start"], s["end"], frac, cfg, name)
        out_path = FEATURES_DIR / f"{symbol}_{timeframe}_{name}.parquet"
        part.to_parquet(out_path)
        saved_paths[name] = out_path
        parts.append(part)
        logger.info(f"{name.upper()} sauvegardé: {out_path} | shape={part.shape}")

    # Optionnel : un fichier global fusionné
    out = pd.concat(parts, axis=0).sort_index()
    merged_path = FEATURES_DIR / f"{symbol}_{timeframe}_full.parquet"
    out.to_parquet(merged_path)
    logger.info(f" Fichier combiné sauvegardé: {merged_path} | shape={out.shape}")

    return saved_paths


if __name__ == "__main__":
    make_features("EURUSD", "H1")
