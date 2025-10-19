# exogenous_model/eval/evaluate_lstm.py
import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, matthews_corrcoef,
    precision_recall_fscore_support, confusion_matrix, average_precision_score,
    top_k_accuracy_score
)

from tools.logger import setup_logger
from exogenous_model.model.core import LSTMClassifier
from exogenous_model.training.datasets import SequenceDataset  # même Dataset que le training

logger = setup_logger()

# --- Paths / Const ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CKPT_DIR      = PROJECT_ROOT / "exogenous_model" / "model" / "checkpoints"
RESULTS_DIR   = PROJECT_ROOT / "exogenous_model" / "results"


# --- IO helpers ---
def load_processed(seed: int, split: str, symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Charge le parquet du split demandé (ici 'test'), tel qu'écrit par data/make_processed.py.
    """
    p = PROCESSED_DIR / f"seed_{seed}" / f"{symbol}_{timeframe}_{split}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Split {split} introuvable: {p}")
    df = pd.read_parquet(p)
    # garantir une colonne 'time' exploitable pour l'export CSV d'analyse
    if "time" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "time"})
        else:
            df = df.reset_index(drop=False)
            if "index" in df.columns:
                df = df.rename(columns={"index": "time"})
    return df


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


# --- Metrics ---
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray):
    acc  = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    mcc  = matthews_corrcoef(y_true, y_pred)

    prec_c, rec_c, f1_c, sup_c = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    prec_macro = float(np.mean(prec_c))
    rec_macro  = float(np.mean(rec_c))
    f1_macro   = float(np.mean(f1_c))

    ap_per_class = []
    for k in range(y_proba.shape[1]):
        mask = (y_true == k).astype(int)
        ap = average_precision_score(mask, y_proba[:, k]) if mask.sum() > 0 else np.nan
        ap_per_class.append(ap)
    pr_auc_macro = float(np.nanmean(ap_per_class))

    try:
        top2 = float(top_k_accuracy_score(y_true, y_proba, k=2, labels=np.arange(y_proba.shape[1])))
    except Exception:
        top2 = None

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "mcc": float(mcc),
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "pr_auc_macro": pr_auc_macro,
        "top2_accuracy": top2,
        "per_class": [
            {
                "class": int(k),
                "precision": float(prec_c[k]),
                "recall": float(rec_c[k]),
                "f1": float(f1_c[k]),
                "support": int(sup_c[k]),
                "pr_auc": None if np.isnan(ap_per_class[k]) else float(ap_per_class[k]),
            }
            for k in range(len(prec_c))
        ],
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
    }


# --- Main eval ---
def evaluate_lstm(symbol="EURUSD", timeframe="H1", seed=42, batch_size=64):
    """
    Évalue le checkpoint LSTM *uniquement sur le split TEST*.
    - charge data/processed/.../{symbol}_{tf}_test.parquet
    - applique le scaler du train
    - recrée les séquences (SequenceDataset)
    - calcule métriques + sauvegarde artefacts
    """
    logger.info(f"=== Evaluate LSTM | {symbol}_{timeframe} | seed={seed} ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) DATA: TEST uniquement
    df_test = load_processed(seed, "test", symbol, timeframe)
    df_test = df_test[df_test["label"].notna()].reset_index(drop=True)

    # 2) META + SCALER + CHECKPOINT
    meta, scaler, model_path = load_meta_scaler_model_paths(symbol, timeframe, seed)
    feature_cols = meta["feature_cols"]
    seq_len      = int(meta["sequence_length"])

    # 3) SCALING (transform only, fit a été fait sur train)
    df_test[feature_cols] = df_test[feature_cols].astype(float)
    df_test.loc[:, feature_cols] = scaler.transform(df_test[feature_cols])

    # 4) Dataset/Loader (séquençage lazy, aligné avec train)
    ds_test = SequenceDataset(df_test, seq_len, feature_cols)  # importé de training/
    if len(ds_test) == 0:
        raise RuntimeError("Aucun échantillon séquentiel sur le TEST.")
    loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    # 5) Modèle
    model = LSTMClassifier(input_dim=len(feature_cols)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info(f"Checkpoint chargé: {Path(model_path).name} | device={device}")

    # 6) Inference
    y_true, y_pred, y_proba = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            y_true.append(yb.numpy())
            y_pred.append(preds.cpu().numpy())
            y_proba.append(probs.cpu().numpy())

    y_true  = np.concatenate(y_true)
    y_pred  = np.concatenate(y_pred)
    y_proba = np.concatenate(y_proba)

    # 7) Métriques + logs de distribution
    metrics = compute_metrics(y_true, y_pred, y_proba)
    unique, counts = np.unique(y_true, return_counts=True)
    dist = {int(k): int(v) for k, v in zip(unique, counts)}
    logger.info(f"Test class distribution: {dist}")
    logger.info(
        "Acc {:.4f} | BAcc {:.4f} | MCC {:.4f} | F1(macro) {:.4f} | PR-AUC(macro) {:.4f}{}".format(
            metrics["accuracy"], metrics["balanced_accuracy"], metrics["mcc"],
            metrics["f1_macro"], metrics["pr_auc_macro"],
            "" if metrics["top2_accuracy"] is None else f" | Top2 {metrics['top2_accuracy']:.4f}"
        )
    )

    # 8) Sauvegardes (metrics + prédictions + CSV d’analyse horodaté)
    out_dir = RESULTS_DIR / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # prédictions brutes
    np.save(out_dir / f"y_true_seed_{seed}.npy",  y_true)
    np.save(out_dir / f"y_pred_seed_{seed}.npy",  y_pred)
    np.save(out_dir / f"y_proba_seed_{seed}.npy", y_proba)

    # métriques détaillées
    with open(out_dir / "metrics_detailed.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # CSV d’analyse par timestamp (alignement “seq_len-1: ”)
    try:
        times = df_test["time"].iloc[seq_len - 1 : seq_len - 1 + len(y_true)].to_numpy()
    except Exception:
        times = np.arange(len(y_true))
    proba_max = y_proba.max(axis=1)
    an_df = pd.DataFrame({"time": times, "y_true": y_true, "y_pred": y_pred, "proba_max": proba_max})
    an_df.to_csv(out_dir / "test_preds.csv", index=False)

    logger.info(f"Résultats sauvegardés: {out_dir}")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Eval LSTM sur le split TEST")
    ap.add_argument("--symbol", type=str, default="EURUSD")
    ap.add_argument("--timeframe", type=str, default="H1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    evaluate_lstm(symbol=args.symbol, timeframe=args.timeframe, seed=args.seed, batch_size=args.batch_size)
