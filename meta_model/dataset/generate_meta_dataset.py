# meta_model/dataset/generate_meta_dataset.py
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import joblib

from tools.logger import setup_logger
from exogenous_model.model.core import LSTMClassifier
from exogenous_model.training.datasets import SequenceDataset

logger = setup_logger()

# === Paths ===
PROJECT_ROOT  = Path(__file__).resolve().parents[2]
INTERIM_DIR   = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CKPT_DIR      = PROJECT_ROOT / "exogenous_model" / "model" / "checkpoints"
OUT_DIR       = PROJECT_ROOT / "meta_model" / "dataset"


# === Load helpers ===
def load_processed_and_raw(seed: int, split: str, symbol: str, timeframe: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Charge deux versions du dataset :
    - RAW : prix réels depuis data/interim
    - PROCESSED : features normalisées depuis data/processed
    """
    p_proc = PROCESSED_DIR / f"seed_{seed}" / f"{symbol}_{timeframe}_{split}.parquet"
    p_raw  = INTERIM_DIR / f"{symbol}_{timeframe}.parquet"

    if not p_proc.exists():
        raise FileNotFoundError(f"Processed introuvable: {p_proc}")
    if not p_raw.exists():
        raise FileNotFoundError(f"Raw introuvable: {p_raw}")

    df_proc = pd.read_parquet(p_proc)
    df_raw  = pd.read_parquet(p_raw)

    # Assure alignement temporel
    if "time" not in df_raw.columns:
        df_raw = df_raw.reset_index().rename(columns={"index": "time"})
    if "time" not in df_proc.columns:
        df_proc = df_proc.reset_index().rename(columns={"index": "time"})

    # Filtre même période
    df_raw = df_raw[df_raw["time"].isin(df_proc["time"])]
    df_raw = df_raw.reset_index(drop=True)
    df_proc = df_proc[df_proc["label"].notna()].reset_index(drop=True)

    return df_proc, df_raw



def load_meta_scaler_model_paths(symbol: str, timeframe: str, seed: int):
    meta_path   = CKPT_DIR / f"meta_{symbol}_{timeframe}_seed{seed}.json"
    scaler_path = CKPT_DIR / f"scaler_{symbol}_{timeframe}_seed{seed}.pkl"
    model_path  = CKPT_DIR / f"lstm_{symbol}_{timeframe}_seed{seed}.pt"
    for p in (meta_path, scaler_path, model_path):
        if not p.exists():
            raise FileNotFoundError(f"Manquant: {p}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    scaler = joblib.load(scaler_path)
    return meta, scaler, model_path


# === Utils ===
def entropy_from_probs(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def rolling_shannon_entropy(x: pd.Series, window=100, bins=30, min_periods_ratio=0.3, standardize=True):
    """
    Shannon entropy sur fenêtre glissante, calculée sur une PMF (pas une densité).
    - Standardise la fenêtre avant histogramme pour éviter les valeurs "bloquées" proches de 0.
    - Retourne une entropie en nats ∈ [0, ln(bins)].
    """
    x = x.astype(float)
    min_periods = max(1, int(window * float(min_periods_ratio)))
    eps = 1e-12

    def _ent(arr):
        arr = np.asarray(arr)
        arr = arr[np.isfinite(arr)]
        n = arr.size
        if n == 0:
            return np.nan
        if standardize:
            mu = arr.mean()
            sd = arr.std()
            if not np.isfinite(sd) or sd < 1e-12:
                # fenêtre quasi constante → faible entropie
                return 0.0
            arr = (arr - mu) / (sd + eps)
        # PMF (pas density), puis normalisation à 1
        counts, _ = np.histogram(arr, bins=bins, density=False)
        total = counts.sum()
        if total == 0:
            return np.nan
        p = counts.astype(float) / total
        p = p[p > 0]
        H = -np.sum(p * np.log(p + eps))
        return float(np.clip(H, 0.0, np.log(bins)))

    return x.shift(1).rolling(window=window, min_periods=min_periods).apply(_ent, raw=False)


def compute_vrapd(df_raw: pd.DataFrame, y_pred: np.ndarray) -> np.ndarray:
    """Volatility Regime Adjusted Prediction Divergence (robuste si 'vix' absent)."""
    df = df_raw.copy()

    if "vix" not in df.columns:
        df["vix"] = 0.0  # fallback silencieux

    # log returns
    df["log_price"] = np.log(df["close"].astype(float).clip(lower=1e-12))
    df["log_return"] = df["log_price"].diff()

    # Indicateurs "naïfs"
    df["naive_persistence"] = df["log_return"].shift(1)
    df["naive_trend"] = np.sign(df["close"] - df["close"].rolling(50).mean())

    w = 20
    df["bb_ma"] = df["close"].rolling(w).mean()
    df["bb_std"] = df["close"].rolling(w).std()
    df["bb_upper"] = df["bb_ma"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_ma"] - 2 * df["bb_std"]
    df["naive_mean_reversion"] = np.where(
        df["close"] > df["bb_upper"], -1,
        np.where(df["close"] < df["bb_lower"], 1, 0)
    )

    df["consensus_naif"] = (
        0.5 * df["naive_persistence"].fillna(0)
        + 0.3 * df["naive_trend"].fillna(0)
        + 0.2 * df["naive_mean_reversion"].fillna(0)
    )

    # Volatilité / VIX
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    df["volatility_regime"] = np.tanh(
        0.7 * (df["atr"] / df["atr"].rolling(100).mean())
        + 0.3 * (df["vix"].fillna(0) / 30.0)
    ) + 0.5

    df["volume_z"] = (
        (df["volume"] - df["volume"].rolling(50).mean())
        / df["volume"].rolling(50).std()
    )
    df["liquidity_factor"] = np.exp(-df["volume_z"].abs())

    y_score = np.where(y_pred == 1, 1.0, np.where(y_pred == 2, -1.0, 0.0))
    N = len(y_pred)
    denom = (
        df["volatility_regime"].values[-N:]
        * df["liquidity_factor"].values[-N:]
        + 1e-8
    )
    vrapd = np.abs(y_score - df["consensus_naif"].values[-N:]) / denom
    return vrapd



# === Main generator ===
def generate_meta_dataset(symbol="EURUSD", timeframe="H1", seed=42, batch_size=512):
    logger.info(f"[META] Génération dataset pour {symbol}_{timeframe} | seed={seed}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load processed TEST (RAW)
    df_test, df_test_raw = load_processed_and_raw(seed, "test", symbol, timeframe)  # sera scalé pour le LSTM

    # 2) Charger scaler, modèle et meta.json
    meta, scaler, model_path = load_meta_scaler_model_paths(symbol, timeframe, seed)
    feature_cols = meta["feature_cols"]
    seq_len = int(meta["sequence_length"])

    # 3) Scaling (transform only sur la copie)
    df_test[feature_cols] = df_test[feature_cols].astype(float)
    df_test.loc[:, feature_cols] = scaler.transform(df_test[feature_cols])

    # 4) Dataset séquentiel
    ds_test = SequenceDataset(df_test, seq_len, feature_cols)
    if len(ds_test) == 0:
        raise RuntimeError("Aucun échantillon séquentiel sur TEST.")
    loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    # 5) Charger le modèle LSTM
    model = LSTMClassifier(input_dim=len(feature_cols)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 6) Inférence LSTM
    y_true, y_pred, y_proba = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            y_true.append(yb.numpy())
            y_pred.append(preds)
            y_proba.append(probs)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_proba = np.concatenate(y_proba)

    # Alignement temporel
    start = seq_len - 1
    end = start + len(y_true)
    times = df_test_raw["time"].iloc[start:end].to_numpy()

    # 7) Features méta
    p0, p1, p2 = y_proba[:, 0], y_proba[:, 1], y_proba[:, 2]
    sortp = np.sort(y_proba, axis=1)
    pmax, p2nd = sortp[:, -1], sortp[:, -2]
    margin = pmax - p2nd
    entropy_pred = entropy_from_probs(y_proba)
    entropy_roll = pd.Series(entropy_pred).shift(1).rolling(window=50, min_periods=10).mean().values

    # ENTROPIE PRIX sur données RAW
    df_test_raw["log_return"] = np.log(df_test_raw["close"].astype(float).clip(lower=1e-12)).diff()
    entropy_full = rolling_shannon_entropy(
        df_test_raw["log_return"], window=100, bins=30, min_periods_ratio=0.3, standardize=True
    )
    entropy_price = entropy_full.iloc[start:end].to_numpy()
    logger.info(f"[DEBUG] entropy_price stats: min={np.nanmin(entropy_price):.3f} "
                f"median={np.nanmedian(entropy_price):.3f} max={np.nanmax(entropy_price):.3f}")

    # VRAPD sur RAW
    df_align_raw = df_test_raw.iloc[start:end].copy()
    vrapd = compute_vrapd(df_align_raw, y_pred)

    # 8) Dataset méta
    df_meta = pd.DataFrame({
        "time": times,
        "y_true": y_true,
        "y_pred": y_pred,
        "meta_label": (y_true == y_pred).astype(int),
        "p0": p0, "p1": p1, "p2": p2,
        "pmax": pmax, "margin": margin,
        "entropy_pred": entropy_pred,
        "entropy_pred_roll": entropy_roll,
        "entropy_price": entropy_price,
        "vrapd": vrapd
    })

    # 9) Nettoyage & sauvegarde
    df_meta = df_meta.replace([np.inf, -np.inf], np.nan)
    logger.info(f"[META] Avant dropna: shape={df_meta.shape}, NaN={df_meta.isna().sum().sum()}")
    df_meta = df_meta.dropna(thresh=5)
    logger.info(f"[META] Après dropna partiel: shape={df_meta.shape}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f"meta_{symbol}_{timeframe}_seed_{seed}.csv"
    df_meta.to_csv(out_csv, index=False)
    logger.info(f"[META] Dataset sauvegardé → {out_csv} | shape={df_meta.shape}")

    return df_meta


if __name__ == "__main__":
    generate_meta_dataset(symbol="EURUSD", timeframe="H1", seed=42)
