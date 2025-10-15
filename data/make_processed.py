# data/make_processed.py

import json
import numpy as np
import pandas as pd
from pathlib import Path
from tools.logger import setup_logger

logger = setup_logger()

# === Paths ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_DIR = DATA_DIR / "features"
CONFIG_PATH = PROJECT_ROOT / "config" / "config_test.json"
OUTPUT_DIR = PROJECT_ROOT / "exogenous_model_v0" / "dataset" / "splits"

# === Load config ===
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)


# === Utils ===
def remove_highly_correlated_features(df, threshold=0.95, protected_cols=None):
    """
    Supprime les features fortement corrélées entre elles, sauf les colonnes protégées.
    Ignore automatiquement les colonnes non numériques comme 'split', 'label', etc.
    """
    if protected_cols is None:
        protected_cols = {"open", "high", "low", "close", "frac_diff"}

    # Sélection uniquement des colonnes numériques
    numeric_df = df.select_dtypes(include=[np.number])

    # Corrélation absolue
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    # Ne pas supprimer les colonnes protégées
    to_drop = [col for col in to_drop if col not in protected_cols]

    logger.info(f"Suppression {len(to_drop)} colonnes corrélées > {threshold}")
    if to_drop:
        logger.debug(f"Colonnes supprimées : {to_drop}")

    # Supprime les colonnes corrélées du DataFrame original
    df = df.drop(columns=to_drop, errors="ignore")
    return df, to_drop



def generate_label_with_triple_barrier_on_frac_diff_cumsum(
    df: pd.DataFrame,
    tp_pips: float,
    sl_pips: float,
    pips_value: float,
    window: int,
) -> list[int]:
    """
    Labels triple barrière (0 neutre, 1 long, 2 short).
    """
    labels = []
    frac = df["frac_diff"].values
    close = df["close"].values

    for i in range(len(df) - window):
        entry_price = close[i]
        future_returns = frac[i + 1 : i + 1 + window]
        cum_returns = np.cumsum(future_returns)

        tp_return = np.log(1 + (tp_pips * pips_value) / entry_price)
        sl_return = -np.log(1 + (sl_pips * pips_value) / entry_price)

        hit_tp = next((j for j, r in enumerate(cum_returns) if r >= tp_return), None)
        hit_sl = next((j for j, r in enumerate(cum_returns) if r <= sl_return), None)

        if hit_tp is not None and (hit_sl is None or hit_tp < hit_sl):
            label = 1  # Long
        elif hit_sl is not None and (hit_tp is None or hit_sl < hit_tp):
            label = 2  # Short
        else:
            label = 0  # Neutre

        labels.append(label)

    labels = [np.nan] * window + labels
    return labels


def select_best_prediction_window(df, tp_pips, sl_pips, pips_size, candidate_windows, seed):
    """
    Choisit la fenêtre avec entropie max sur le TRAIN.
    """
    results = {}
    for w in candidate_windows:
        labels = generate_label_with_triple_barrier_on_frac_diff_cumsum(
            df, tp_pips, sl_pips, pips_size, w
        )
        s = pd.Series(labels).dropna()
        if len(s) == 0:
            entropy = -np.inf
        else:
            probs = s.value_counts(normalize=True)
            ps = probs.values
            entropy = -np.sum(ps * np.log(ps + 1e-12))
        results[w] = entropy

    best_window = max(results, key=results.get)
    logger.info(f"Fenêtre optimale (seed={seed}): {best_window} (entropie={results[best_window]:.4f})")

    CONFIG["label"]["window"] = int(best_window)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=4)

    return best_window


# === Processing pipeline ===
def process_split(df, split_name, seed, protected_cols, threshold):
    """
    Gère la suppression des features corrélées et la cohérence des colonnes.
    """
    checkpoints_dir = PROJECT_ROOT / "exogenous_model_v0" / "model" / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    features_path = checkpoints_dir / f"features_used_seed_{seed}.txt"
    dropped_path = checkpoints_dir / f"dropped_cols_seed_{seed}.txt"

    if split_name == "train":
        df, dropped = remove_highly_correlated_features(df, threshold, protected_cols)

        with open(features_path, "w") as f:
            for c in df.columns:
                f.write(c + "\n")
        with open(dropped_path, "w") as f:
            for c in dropped:
                f.write(c + "\n")

    else:
        with open(dropped_path, "r") as f:
            dropped = [line.strip() for line in f.readlines()]
        df = df.drop(columns=dropped, errors="ignore")

        with open(features_path, "r") as f:
            keep_cols = [line.strip() for line in f.readlines()]
        df = df.reindex(columns=keep_cols, fill_value=np.nan)

    return df


def add_label(df, prediction_window, split_name):
    """
    Applique le triple-barrier labelling.
    """
    tp_pips = CONFIG["label"]["take_profit_pips"]
    sl_pips = CONFIG["label"]["stop_loss_pips"]
    pips_size = CONFIG["label"]["pips_size"]

    df["label"] = generate_label_with_triple_barrier_on_frac_diff_cumsum(
        df, tp_pips, sl_pips, pips_size, prediction_window
    )
    dist = df["label"].value_counts(normalize=True).mul(100).round(2)
    logger.info(f"Distribution des labels ({split_name}):\n{dist.to_string()}")
    return df


def save_processed(df, split_name, seed, debug=False):
    """
    Sauvegarde les features + label au format parquet (et CSV optionnel) dans data/processed.
    """
    processed_dir = DATA_DIR / "processed" / f"seed_{seed}"
    processed_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = processed_dir / f"{split_name}.parquet"
    df.to_parquet(parquet_path, index=True)
    logger.info(f"{split_name.upper()} sauvegardé (parquet): {parquet_path}")

    if debug:
        csv_path = processed_dir / f"{split_name}.csv"
        df.to_csv(csv_path, index=True)
        logger.info(f"Export CSV (debug): {csv_path}")

    return parquet_path



def make_processed(symbol="EURUSD", timeframe="H1", seed=42):
    """
    Pipeline complet : charge les splits -> nettoie -> labellise -> sauvegarde
    """
    logger.info(f"=== [make_processed] seed={seed} | {symbol}_{timeframe} ===")

    protected_cols = set(CONFIG["features"]["protected_cols"])
    threshold = CONFIG["features"]["remove_corr_threshold"]

    # 1️⃣ Charger les splits
    paths = {
        "train": FEATURES_DIR / f"{symbol}_{timeframe}_train.parquet",
        "val": FEATURES_DIR / f"{symbol}_{timeframe}_val.parquet",
        "test": FEATURES_DIR / f"{symbol}_{timeframe}_test.parquet",
    }

    dfs = {k: pd.read_parquet(v) for k, v in paths.items()}
    logger.info(f"Splits chargés : {[f'{k}={len(v)}' for k, v in dfs.items()]}")

    # 2️⃣ Process features (remove corr, align columns)
    train = process_split(dfs["train"], "train", seed, protected_cols, threshold)
    val = process_split(dfs["val"], "val", seed, protected_cols, threshold)
    test = process_split(dfs["test"], "test", seed, protected_cols, threshold)

    # 3️⃣ Sélection de la meilleure fenêtre (entropie max sur train)
    candidate_windows = [4, 8, 12, 24, 48]
    best_window = select_best_prediction_window(
        train,
        CONFIG["label"]["take_profit_pips"],
        CONFIG["label"]["stop_loss_pips"],
        CONFIG["label"]["pips_size"],
        candidate_windows,
        seed,
    )

    # 4️⃣ Ajouter labels triple barrière
    train = add_label(train, best_window, "train")
    val = add_label(val, best_window, "val")
    test = add_label(test, best_window, "test")

    # 5️⃣ Sauvegarder les splits finaux
    save_processed(train, "train", seed)
    save_processed(val, "val", seed)
    save_processed(test, "test", seed)

    logger.info("make_processed terminé avec succès.")
    return True


if __name__ == "__main__":
    make_processed("EURUSD", "H1", seed=42)
