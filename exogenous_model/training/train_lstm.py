# exogenous_model/training/train_lstm.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import torch
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight

from tools.logger import setup_logger
from exogenous_model.model.core import LSTMClassifier
from exogenous_model.training.datasets import SequenceDataset
from exogenous_model.training.losses import FocalLoss
from exogenous_model.training.utils import seed_all, ensure_dir, save_json

logger = setup_logger()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH  = PROJECT_ROOT / "config" / "config_test.json"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def load_processed(seed: int, split: str, symbol: str, timeframe: str):
    p = DATA_PROCESSED_DIR / f"seed_{seed}" / f"{symbol}_{timeframe}_{split}.parquet"
    df = pd.read_parquet(p)
    if "time" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={"index": "time"})
    return df

def train_lstm(symbol="EURUSD", timeframe="H1", seed=42):
    cfg = load_config()
    seq_len = int(cfg["model"]["sequence_length"])
    batch_size = int(cfg["model"]["batch_size"])
    epochs = int(cfg["model"]["epochs"])
    lr = float(cfg["model"]["learning_rate"])
    patience = int(cfg["model"]["patience"])

    logger.info(f"=== LSTM training | {symbol}_{timeframe} | seed={seed} ===")
    seed_all(seed)

    # 1) Charger les splits déjà normalisés
    train_df = load_processed(seed, "train", symbol, timeframe)
    val_df   = load_processed(seed, "val",   symbol, timeframe)
    test_df  = load_processed(seed, "test",  symbol, timeframe)
    logger.info(f"Loaded processed: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    # 2) Charger le scaler et les métadonnées (créés par make_processed)
    ckpt_dir = PROJECT_ROOT / "exogenous_model" / "model" / "checkpoints"
    meta_path = ckpt_dir / f"meta_{symbol}_{timeframe}_seed{seed}.json"
    scaler_path = ckpt_dir / f"scaler_{symbol}_{timeframe}_seed{seed}.pkl"

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]

    # 3) Nettoyage des NaN label
    train_df = train_df[train_df["label"].notna()]
    val_df   = val_df[val_df["label"].notna()]
    test_df  = test_df[test_df["label"].notna()]

    # 4) Datasets & loaders (séquençage lazy)
    ds_train = SequenceDataset(train_df, seq_len, feature_cols)
    ds_val   = SequenceDataset(val_df,   seq_len, feature_cols)
    ds_test  = SequenceDataset(test_df,  seq_len, feature_cols)

    if len(ds_train) == 0:
        raise RuntimeError("Aucun échantillon séquentiel sur le train — augmente sequence_length ou vérifie les labels.")

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False, drop_last=False)

    # 5) Modèle / loss / optim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(input_dim=len(feature_cols)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Poids de classes sur train
    y_train_all = train_df["label"].astype(int).to_numpy()
    cls = np.unique(y_train_all)
    weights = compute_class_weight(class_weight="balanced", classes=cls, y=y_train_all)
    criterion = FocalLoss(alpha=torch.tensor(weights, dtype=torch.float32, device=device))

    # 6) Entraînement + early stopping
    best_val = float("inf")
    patience_counter = 0

    ensure_dir(ckpt_dir)
    best_model_path = ckpt_dir / f"lstm_{symbol}_{timeframe}_seed{seed}.pt"

    logger.info(f"Start training for {epochs} epochs (batch={batch_size}, seq={seq_len}) on {device}")

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= max(1, len(train_loader))

        # --- Validation ---
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                va_loss += criterion(model(xb), yb).item()
        va_loss /= max(1, len(val_loader))

        if epoch % 2 == 0:
            logger.info(f"[{epoch+1}/{epochs}] train_loss={tr_loss:.4f} val_loss={va_loss:.4f} best={best_val:.4f}")

        if va_loss < best_val:
            best_val = va_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # 7) Sauvegarde finale (mise à jour du meta)
    meta["best_val_loss"] = float(best_val)
    save_json(meta, meta_path)
    logger.info(f"Saved model={best_model_path.name} | updated meta.json")

    # 8) Quick test loss
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    te_loss = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            te_loss += criterion(model(xb), yb).item()
    te_loss /= max(1, len(test_loader))
    logger.info(f"TEST loss = {te_loss:.4f}")

    return str(best_model_path), str(scaler_path), float(best_val), float(te_loss)


if __name__ == "__main__":
    train_lstm(symbol="EURUSD", timeframe="H1", seed=42)
