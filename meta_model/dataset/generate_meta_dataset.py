"""generate_meta_datasets.py"""
import json
import os
import torch
import numpy as np
import pandas as pd
from exogenous_model_v0.model.core import LSTMClassifier
from exogenous_model_v0.prediction.prediction import predict_exo_model
from exogenous_model_v0.dataset.generate_dataset import logger


# === ENTROPIE (De Prado inspired) === #
def rolling_shannon_entropy(x: pd.Series, window=200, bins=30):
    """
    Entropie de Shannon sur fenêtre glissante, basée uniquement sur le passé (causale).
    Plus l'entropie est basse, plus le marché est "compressé".
    """
    x = x.astype(float)

    def _entropy(arr):
        hist, _ = np.histogram(arr, bins=bins, density=True)
        p = hist[hist > 0]
        return -np.sum(p * np.log(p))

    ent = (
        x.shift(1)  # exclure la valeur courante
         .rolling(window=window, min_periods=window)
         .apply(_entropy, raw=False)
    )
    return ent


# === ENTROPIE DES SOFTMAX DU LSTM === #
def entropy_from_softmax_probs(probs: np.ndarray) -> np.ndarray:
    """
    Calcule l'entropie de Shannon pour chaque ligne de probas (N, C).
    H = -sum_i p_i log(p_i)
    """
    p = np.clip(probs, 1e-12, 1.0)   # stabilité num.
    H = -np.sum(p * np.log(p), axis=1)
    return H


def predict_lstm_softmax(model: torch.nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:
    """
    Renvoie un array (N, C) des probabilités softmax du LSTM sur X.
    """
    model.eval()
    probs_list = []
    with torch.no_grad():
        n = X.shape[0]
        for i in range(0, n, batch_size):
            xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32, device=device)
            logits = model(xb)                         # (B, C)
            pb = torch.softmax(logits, dim=1)         # (B, C)
            probs_list.append(pb.detach().cpu().numpy())
    return np.concatenate(probs_list, axis=0)


# === FEATURE VRAPD === #
def compute_vrapd_features(df_raw: pd.DataFrame, y_pred: np.ndarray) -> np.ndarray:
    """
    Calcule la feature VRAPD (Volatility-Regime Adjusted Prediction Divergence).
    """
    df = df_raw.copy()
    df['log_price'] = np.log(df['close'])
    df['log_return'] = df['log_price'].diff()

    # Consensus naïf
    df['naive_persistence'] = df['log_return'].shift(1)
    df['naive_trend'] = np.sign(df['close'] - df['close'].rolling(50).mean())

    # Mean reversion via bandes de Bollinger
    window_bb = 20
    df['bb_ma'] = df['close'].rolling(window_bb).mean()
    df['bb_std'] = df['close'].rolling(window_bb).std()
    df['bb_upper'] = df['bb_ma'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_ma'] - 2 * df['bb_std']

    df['naive_mean_reversion'] = np.where(
        df['close'] > df['bb_upper'], -1,
        np.where(df['close'] < df['bb_lower'], 1, 0)
    )

    # Consensus combiné
    df['consensus_naif'] = (
        0.5 * df['naive_persistence'].fillna(0) +
        0.3 * df['naive_trend'].fillna(0) +
        0.2 * df['naive_mean_reversion'].fillna(0)
    )

    # Régime de volatilité (ATR + VIX)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close  = np.abs(df['low']  - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr']  = true_range.rolling(14).mean()

    df['volatility_regime'] = np.tanh(
        0.7 * (df['atr'] / df['atr'].rolling(100).mean()) +
        0.3 * (df['vix'] / 30)
    ) + 0.5

    # Facteur de liquidité
    df['volume_z'] = (df['volume'] - df['volume'].rolling(50).mean()) / df['volume'].rolling(50).std()
    df['liquidity_factor'] = np.exp(-np.abs(df['volume_z']))

    # VRAPD
    y_score = 2 * y_pred - 1  # 0 -> -1, 1 -> 1 (si 3 classes, adapter au mapping désiré)
    vrapd = np.abs(y_score - df['consensus_naif'].values[-len(y_pred):]) / (
        df['volatility_regime'].values[-len(y_pred):] *
        df['liquidity_factor'].values[-len(y_pred):] + 1e-8
    )

    return vrapd


# === MAIN FUNCTION === #
def generate_meta_dataset(seed, logger):
    """
    Génére le dataset pour le méta-modèle (XGBoost) :
    - Prédictions du LSTM (classes)
    - Probas softmax du LSTM + entropy_pred (+ version lissée causale)
    - entropy_price (marché)
    - VRAPD
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # Chargement config
    config_path = os.path.join(project_root, 'config', 'config.json')
    with open(config_path) as f:
        config = json.load(f)

    SEQUENCE_LENGTH = config['model']['sequence_length']

    # Dossiers
    base_dir   = os.path.join(project_root, 'exogenous_model_v0', 'dataset', 'splits')
    model_dir  = os.path.join(project_root, 'exogenous_model_v0', 'model', 'checkpoints')
    output_dir = os.path.join(project_root, 'meta_model', 'dataset', 'features_and_target')

    base_path  = os.path.join(base_dir,  f'seed_{seed}')
    model_path = os.path.join(model_dir, f'model_seed_{seed}.pt')

    # Données test
    X_test = np.load(os.path.join(base_path, 'X_test.npy'))
    y_test = np.load(os.path.join(base_path, 'y_test.npy'))

    raw_test_path = os.path.join(base_path, 'df_test_processed.csv')
    if not os.path.exists(raw_test_path):
        raise FileNotFoundError(f"{raw_test_path} n'existe pas")

    df_raw_test = pd.read_csv(raw_test_path)
    logger.info(f"Données de test brutes chargées : {df_raw_test.shape[0]} points")

    # Alignement temporel (mêmes N que X_test/y_test)
    df_raw_aligned = df_raw_test.iloc[-len(y_test):].copy()
    df_raw_aligned = df_raw_aligned[['time','open','high','low','close','volume','vix']]
    logger.info(f"Données de test alignées : {df_raw_aligned.shape[0]} points")

    # Modèle LSTM
    model = LSTMClassifier(input_dim=X_test.shape[2]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Prédictions classes
    logger.info("Prédiction du modèle principal (classes)...")
    test_preds = predict_exo_model(model, X_test, device)  # array (N,)

    # Probas softmax + entropie des prédictions
    logger.info("Prédiction du modèle principal (probas softmax) & entropy_pred...")
    probs = predict_lstm_softmax(model, X_test, device)    # (N, C)
    entropy_pred = entropy_from_softmax_probs(probs)       # (N,)
    # Lissage causal optionnel (utile pour le méta) : moyenne glissante sur passé
    entropy_pred_roll = pd.Series(entropy_pred).shift(1).rolling(window=50, min_periods=10).mean().values

    # === Features additionnelles === #
    logger.info("Calcul de la feature VRAPD...")
    vrapd_feature = compute_vrapd_features(df_raw_aligned, test_preds)

    logger.info("Calcul de la feature d'entropie de marché (entropy_price)...")
    df_raw_aligned['log_return']   = np.log(df_raw_aligned['close']).diff()
    df_raw_aligned['entropy_price'] = rolling_shannon_entropy(df_raw_aligned['log_return'], window=200, bins=30)

    # === Construction du dataset méta === #
    df_meta = df_raw_aligned.copy()
    df_meta['y_pred']  = test_preds
    df_meta['y_true']  = y_test
    df_meta['meta_label'] = (df_meta['y_true'] == df_meta['y_pred']).astype(int)

    # Ajouts
    df_meta['vrapd']            = vrapd_feature
    df_meta['entropy_pred']     = entropy_pred
    df_meta['entropy_pred_roll'] = entropy_pred_roll  # version lissée, causale

    # Nettoyage (fenêtres)
    df_meta.dropna(inplace=True)

    # Sauvegarde
    output_file_path = os.path.join(output_dir, f'meta_dataset_seed_{seed}.csv')
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    df_meta.to_csv(output_file_path, index=False)
    logger.info(f"Dataset méta sauvegardé : {output_file_path}")


if __name__ == "__main__":
    generate_meta_dataset(42, logger)
