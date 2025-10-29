# data/make_processed.py
import json
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
import joblib

from tools.logger import setup_logger
from pandas.api.types import is_numeric_dtype

logger = setup_logger()

# === Paths ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_DIR = DATA_DIR / "features"
CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"

# === Load config ===
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)


# ===============================
# Utils génériques
# ===============================
def _ensure_numeric_corr(df: pd.DataFrame, protected_cols):
    """
    Retourne le sous-ensemble numérique de df pour calculer la matrice de corrélation,
    en excluant les colonnes non-numériques et en s'assurant que protected_cols restent éligibles au 'keep'.
    """
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    missing_protected = [c for c in protected_cols if c not in numeric_df.columns]
    if missing_protected:
        logger.debug(f"[Corr] Colonnes protégées absentes du bloc numérique: {missing_protected}")
    return numeric_df


def remove_highly_correlated_features(df: pd.DataFrame, threshold=0.95, protected_cols=None):
    """
    Supprime les features fortement corrélées (corr abs > threshold), sauf les colonnes protégées.
    Ignore automatiquement les colonnes non numériques.
    """
    if protected_cols is None:
        protected_cols = {"open", "high", "low", "close", "frac_diff"}

    numeric_df = _ensure_numeric_corr(df, protected_cols)
    if numeric_df.shape[1] == 0:
        logger.info("[Corr] Aucune colonne numérique, on ne supprime rien.")
        return df, []

    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    to_drop = [col for col in to_drop if col not in protected_cols]

    logger.info(f"Suppression {len(to_drop)} colonnes corrélées > {threshold}")
    if to_drop:
        logger.debug(f"Colonnes supprimées : {to_drop}")

    df = df.drop(columns=to_drop, errors="ignore")
    return df, to_drop


# ===============================
# Structural Breaks Features
# ===============================
def cusum_levels(logp: pd.Series, k_sigma=5.0, h=0.01, win=250) -> pd.DataFrame:
    """
    CUSUM “levels” sur log-prix (proxy rupture). Marche bien en H1.
    - k_sigma : seuil en écart-types normalisés
    - h       : drift/min cutoff
    - win     : fenêtre pour estimer la std des increments (rolling)
    """
    logp = logp.astype(float)
    sigma = logp.diff().rolling(win, min_periods=win // 2).std()

    s_pos = pd.Series(0.0, index=logp.index)
    s_neg = pd.Series(0.0, index=logp.index)
    breaks = pd.Series(0, index=logp.index, dtype=int)

    for i in range(1, len(logp)):
        denom = sigma.iloc[i] if pd.notna(sigma.iloc[i]) else np.nan
        if not np.isfinite(denom) or denom == 0:
            s_pos.iloc[i] = s_pos.iloc[i - 1]
            s_neg.iloc[i] = s_neg.iloc[i - 1]
            continue

        z = (logp.iloc[i] - logp.iloc[i - 1]) / (denom + 1e-12)
        s_pos.iloc[i] = max(0.0, s_pos.iloc[i - 1] + z - h)
        s_neg.iloc[i] = min(0.0, s_neg.iloc[i - 1] + z + h)

        if s_pos.iloc[i] > k_sigma:
            breaks.iloc[i] = 1
            s_pos.iloc[i] = 0.0
        if s_neg.iloc[i] < -k_sigma:
            breaks.iloc[i] = 1
            s_neg.iloc[i] = 0.0

    return pd.DataFrame(
        {"cusum_pos": s_pos, "cusum_neg": s_neg, "cusum_break_flag": breaks},
        index=logp.index
    )


def sadf_lite(logp: pd.Series, min_win=250, starts_per_year=10, freq_per_day=24) -> pd.Series:
    """
    Approximation rapide de SADF (Supremum ADF).
    """
    logp = logp.astype(float)
    n = len(logp)
    out = np.full(n, np.nan, dtype=float)
    year_points = 365 * freq_per_day
    step = max(10, year_points // max(1, int(starts_per_year)))

    for t in range(min_win, n):
        best = -np.inf
        start_min = max(0, t - 10 * step)
        for s in range(start_min, t - min_win + 1, step):
            try:
                stat = adfuller(logp.iloc[s:t].values, maxlag=1, regression='c', autolag=None)[0]
                score = -stat
                if score > best:
                    best = score
            except Exception:
                pass
        out[t] = best
    return pd.Series(out, index=logp.index, name="sadf_score")


def add_structural_break_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Ajoute (optionnellement) les features de breaks structurels à df (par split, no-leak).
    """
    sb_cfg = cfg["features"].get("structural_breaks", {})
    if not sb_cfg.get("enable", False):
        return df

    if "close" not in df.columns:
        logger.warning("[SB] 'close' manquant, impossible de calculer les features de rupture.")
        return df

    logp = np.log(df["close"].astype(float).clip(lower=1e-12))

    # CUSUM
    if sb_cfg.get("cusum", {}).get("enable", True):
        k_sigma = float(sb_cfg["cusum"].get("k_sigma", 5.0))
        h = float(sb_cfg["cusum"].get("h", 0.01))
        win = int(sb_cfg.get("sadf", {}).get("min_window", 250))
        cus = cusum_levels(logp, k_sigma=k_sigma, h=h, win=win)
        for c in cus.columns:
            df[f"{c}"] = cus[c]

    # SADF-lite
    if sb_cfg.get("sadf", {}).get("enable", True):
        min_win = int(sb_cfg["sadf"].get("min_window", 250))
        starts_per_year = int(sb_cfg["sadf"].get("starts_per_year", 10))
        sadf = sadf_lite(logp, min_win=min_win, starts_per_year=starts_per_year, freq_per_day=24)
        df["sadf_score"] = sadf

    return df


# ===============================
# Labelling
# ===============================
def generate_label_with_triple_barrier_on_frac_diff_cumsum(
    df: pd.DataFrame,
    tp_pips: float,
    sl_pips: float,
    pips_value: float,
    window: int,
) -> list[int]:
    """
    Labels triple barrière (0 neutre, 1 long, 2 short), en cumulant les returns frac_diff.
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


# ===============================
# Pipeline split-safe
# ===============================
def _load_splits(symbol: str, timeframe: str):
    paths = {
        "train": FEATURES_DIR / f"{symbol}_{timeframe}_train.parquet",
        "val":   FEATURES_DIR / f"{symbol}_{timeframe}_val.parquet",
        "test":  FEATURES_DIR / f"{symbol}_{timeframe}_test.parquet",
    }
    dfs = {k: pd.read_parquet(v) for k, v in paths.items()}
    logger.info(f"Splits chargés : {[f'{k}={len(v)}' for k, v in dfs.items()]}")
    return dfs


def process_split(df: pd.DataFrame, split_name: str, seed: int, protected_cols, threshold):
    """
    Ajoute features de rupture (si activées), puis gère la suppression corrélée (train-only),
    et l'alignement de colonnes pour val/test.
    """
    # 1) Structural breaks par split (no-leak)
    df = add_structural_break_features(df, CONFIG)

    # 2) Corrélation (train only) + alignement
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


def add_label(df: pd.DataFrame, prediction_window: int, split_name: str):
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


def _save_split_parquet(df: pd.DataFrame, split_name: str, seed: int, symbol: str, timeframe: str, debug=False):
    processed_dir = DATA_DIR / "processed" / f"seed_{seed}"
    processed_dir.mkdir(parents=True, exist_ok=True)

    out = processed_dir / f"{symbol}_{timeframe}_{split_name}.parquet"
    df.to_parquet(out, index=True)
    logger.info(f"{split_name.upper()} sauvegardé (parquet): {out}")

    if debug:
        csv_path = processed_dir / f"{symbol}_{timeframe}_{split_name}.csv"
        df.to_csv(csv_path, index=True)
        logger.info(f"Export CSV (debug): {csv_path}")

    return out


# ===============================
# Scaling utils (robustes)
# ===============================
IGNORE_COLS = {
    "label", "split", "dataset", "set", "partition",
    "symbol", "time", "timestamp", "date", "datetime", "index"
}

def get_numeric_feature_cols(df: pd.DataFrame, ignore: set) -> list:
    """
    Retourne les colonnes numériques (dtype numérique) en excluant explicitement 'ignore'.
    """
    cols = [c for c in df.columns if c not in ignore and is_numeric_dtype(df[c])]
    return cols

def sanitize_for_scaling(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Remplace inf/-inf par NaN, puis ffill/bfill pour éviter des NaN avant StandardScaler.
    """
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)
    # ffill/bfill pour séries temporelles ; si tout NaN, restera NaN → on gérera via remplissage zéro
    df[cols] = df[cols].ffill().bfill()
    # S'il reste des NaN (ex: début de série entière NaN), on met 0
    df[cols] = df[cols].fillna(0.0)
    return df


def make_processed(symbol="EURUSD", timeframe="H1", seed=42):
    """
    Pipeline complet : charge features par split -> ajoute SB features -> corr-prune (train) -> label -> scale -> save.
    """
    logger.info(f"=== [make_processed] seed={seed} | {symbol}_{timeframe} ===")

    protected_cols = set(CONFIG["features"]["protected_cols"])
    threshold = CONFIG["features"]["remove_corr_threshold"]

    # 1) Charger splits
    dfs = _load_splits(symbol, timeframe)

    # 2) Process features par split (no-leak)
    train = process_split(dfs["train"].copy(), "train", seed, protected_cols, threshold)
    val   = process_split(dfs["val"].copy(),   "val",   seed, protected_cols, threshold)
    test  = process_split(dfs["test"].copy(),  "test",  seed, protected_cols, threshold)

    # (Debug) Log dtypes utiles
    logger.debug("[Dtypes][TRAIN]\n" + train.dtypes.sort_index().to_string())
    obj_cols = [c for c in train.columns if train[c].dtype == "object"]
    if obj_cols:
        logger.warning(f"[TRAIN] Colonnes dtype=object détectées: {obj_cols}")

    # 3) Choix de la fenêtre (entropie) sur train
    candidate_windows =  set(CONFIG["label"]["candidate_windows"])
    best_window = select_best_prediction_window(
        train,
        CONFIG["label"]["take_profit_pips"],
        CONFIG["label"]["stop_loss_pips"],
        CONFIG["label"]["pips_size"],
        candidate_windows,
        seed,
    )

    # 4) Labelling
    train = add_label(train, best_window, "train")
    val   = add_label(val,   best_window, "val")
    test  = add_label(test,  best_window, "test")

    # 4.5) Sélection robuste des features numériques (train-only) + scaling split-safe
    feature_cols = get_numeric_feature_cols(train, IGNORE_COLS)
    if not feature_cols:
        raise ValueError("[Scaling] Aucune feature numérique éligible après filtrage. Vérifier le pipeline upstream.")

    # Sanitize (NaN/inf) avant fit/transform
    train = sanitize_for_scaling(train, feature_cols)
    val   = sanitize_for_scaling(val,   feature_cols)
    test  = sanitize_for_scaling(test,  feature_cols)

    train[feature_cols] = train[feature_cols].astype(float)
    val[feature_cols] = val[feature_cols].astype(float)
    test[feature_cols] = test[feature_cols].astype(float)

    scaler = StandardScaler()
    scaler.fit(train[feature_cols])

    train.loc[:, feature_cols] = scaler.transform(train[feature_cols])
    val.loc[:, feature_cols] = scaler.transform(val[feature_cols])
    test.loc[:, feature_cols] = scaler.transform(test[feature_cols])

    logger.info(f"[make_processed] Nb features retenues pour scaling: {len(feature_cols)}")
    logger.debug(f"[make_processed] Features: {feature_cols}")

    # Sauvegarde du scaler (dans checkpoints/)
    # NOTE : chemin harmonisé avec 'exogenous_model' (ton code original avait v0 et non-v0).
    ckpt_dir = PROJECT_ROOT / "exogenous_model" / "model" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, ckpt_dir / f"scaler_{symbol}_{timeframe}_seed{seed}.pkl")

    # Sauvegarde des métadonnées
    meta = {
        "feature_cols": feature_cols,
        "sequence_length": CONFIG["model"]["sequence_length"]
    }
    with open(ckpt_dir / f"meta_{symbol}_{timeframe}_seed{seed}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # 5) Sauvegarde
    _save_split_parquet(train, "train", seed, symbol, timeframe)
    _save_split_parquet(val,   "val",   seed, symbol, timeframe)
    _save_split_parquet(test,  "test",  seed, symbol, timeframe)

    logger.info("✅ make_processed terminé avec succès.")
    return True


if __name__ == "__main__":
    make_processed("EURUSD", "H1", seed=42)
