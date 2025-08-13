"""generate_meta_datasets.py"""
import json
import os
import torch
import numpy as np
import pandas as pd
from exogenous_model.model.core import LSTMClassifier
from exogenous_model.prediction.prediction import predict_exo_model
from exogenous_model.dataset.generate_dataset import logger


def temporal_feature_engineering(X):
    features = {
        'mean': X.mean(axis=1),
        'std': X.std(axis=1),
        'min': X.min(axis=1),
        'max': X.max(axis=1),
        'last': X[:, -1, :],
        'first': X[:, 0, :],
        'diff': X[:, -1, :] - X[:, 0, :]
    }
    features_concat = np.concatenate(list(features.values()), axis=1)
    return features_concat, list(features.keys())


def compute_vrapd_features(df_raw: pd.DataFrame, y_pred: np.ndarray) -> np.ndarray:
    """
    Calcule la feature VRAPD (Volatility-Regime Adjusted Prediction Divergence)
    pour chaque point de prédiction.

    Args:
        df_raw: DataFrame contenant les données brutes du marché
        y_pred: Prédictions du modèle LSTM

    Returns:
        Array numpy de la feature VRAPD
    """
    # 1. Calcul des indicateurs de base
    df = df_raw.copy()
    df['log_price'] = np.log(df['close'])
    df['log_return'] = df['log_price'].diff()

    # 2. Consensus naïf
    df['naive_persistence'] = df['log_return'].shift(1)
    df['naive_trend'] = np.sign(df['close'] - df['close'].rolling(50).mean())

    # Mean Reversion basée sur les bandes de Bollinger
    window_bb = 20
    df['bb_ma'] = df['close'].rolling(window_bb).mean()
    df['bb_std'] = df['close'].rolling(window_bb).std()
    df['bb_upper'] = df['bb_ma'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_ma'] - 2 * df['bb_std']

    df['naive_mean_reversion'] = np.where(
        df['close'] > df['bb_upper'],
        -1,  # Sur-achat -> prédit baisse
        np.where(
            df['close'] < df['bb_lower'],
            1,  # Sur-vente -> prédit hausse
            0  # Neutre
        )
    )

    # 3. Consensus combiné
    df['consensus_naif'] = (
            0.5 * df['naive_persistence'].fillna(0) +
            0.3 * df['naive_trend'].fillna(0) +
            0.2 * df['naive_mean_reversion'].fillna(0)
    )

    # 4. Régime de volatilité
    # Calcul ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(14).mean()

    # Régime de volatilité combiné (ATR + VIX)
    df['volatility_regime'] = np.tanh(
        0.7 * (df['atr'] / df['atr'].rolling(100).mean()) +
        0.3 * (df['vix'] / 30)  # Normalisation empirique du VIX
    ) + 0.5  # Décalage pour valeurs positives

    # 5. Facteur de liquidité
    df['volume_z'] = (df['volume'] - df['volume'].rolling(50).mean()) / df['volume'].rolling(50).std()
    df['liquidity_factor'] = np.exp(-np.abs(df['volume_z']))

    # 6. Calcul VRAPD
    # Transformation des prédictions en score continu [-1, 1]
    y_score = 2 * y_pred - 1  # 0 → -1, 1 → 1

    # Calcul de la divergence ajustée
    vrapd = np.abs(y_score - df['consensus_naif'].values[-len(y_pred):]) / (
            df['volatility_regime'].values[-len(y_pred):] *
            df['liquidity_factor'].values[-len(y_pred):] + 1e-8
    )

    return vrapd


def generate_meta_dataset(seed, logger):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # === CONFIGURATION === #
    config_path = os.path.join(project_root, 'config', 'config.json')
    with open(config_path) as f:
        config = json.load(f)

    SEQUENCE_LENGTH = config['model']['sequence_length']

    base_dir = os.path.join(project_root, 'exogenous_model', 'dataset', 'splits')
    model_dir = os.path.join(project_root, 'exogenous_model', 'model', 'checkpoints')
    output_dir = os.path.join(project_root, 'meta_model', 'dataset', 'features_and_target')

    base_path = os.path.join(base_dir, f'seed_{seed}')
    model_path = os.path.join(model_dir, f'model_seed_{seed}.pt')

    # Charger les données test
    X_test = np.load(os.path.join(base_path, 'X_test.npy'))
    y_test = np.load(os.path.join(base_path, 'y_test.npy'))
    time_test = np.load(os.path.join(base_path, 'time_test.npy'))

    # Charger les données brutes correspondantes
    raw_test_path = os.path.join(base_path, 'df_test_processed.csv')
    if not os.path.exists(raw_test_path):
        logger.error("Fichier de données brutes test_raw.csv manquant")
        raise FileNotFoundError(f"{raw_test_path} n'existe pas")

    df_raw_test = pd.read_csv(raw_test_path)
    logger.info(f"Données de test brutes chargées : {df_raw_test.shape[0]} points")

    df_raw_aligned = df_raw_test.iloc[SEQUENCE_LENGTH:].copy()
    df_raw_aligned = df_raw_aligned[['time','open','high','low','close', 'volume','vix']]
    logger.info(f"Données de test brutes alignées et nettoyées : {df_raw_test.shape[0]} points")

    model = LSTMClassifier(input_dim=X_test.shape[2]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    logger.info("Prédiction du modèle principal...")
    test_preds = predict_exo_model(model, X_test, device)

    logger.info("Calcul de la feature VRAPD...")
    vrapd_feature = compute_vrapd_features(df_raw_aligned, test_preds)

    df_meta = df_raw_aligned.copy()
    df_meta['y_pred'] = test_preds
    df_meta['y_true'] = y_test
    df_meta['meta_label'] = (df_meta['y_true'] == df_meta['y_pred']).astype(int)

    # Ajout de la feature VRAPD
    df_meta['vrapd'] = vrapd_feature
    df_meta.dropna(inplace=True)

    output_file_path = os.path.join(output_dir, f'meta_dataset_seed_{seed}.csv')
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    try:
        df_meta.to_csv(output_file_path, index=False)
        logger.info(f"Dataset sauvegardé sous {output_file_path}")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde : {e}")
        raise


if __name__ == "__main__":

    generate_meta_dataset(42, logger)