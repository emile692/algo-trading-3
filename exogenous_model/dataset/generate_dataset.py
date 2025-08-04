import os
import pandas as pd
import numpy as np
import kagglehub
import json
from typing import List

from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator, KAMAIndicator
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator, CCIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice

from config.logger.logger import setup_logger
from exogenous_model.dataset.external_source.evz_loader import download_vix

from exogenous_model.utils.fracdiff import FracDifferentiator

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
logger = setup_logger()

config_path = os.path.join(project_root, 'config', 'config.json')

with open(config_path, "r") as f:
    config = json.load(f)


def enrich_with_vix(eurusd_df):
    vix_df = download_vix(start=eurusd_df.index.min().date().isoformat(),
                          end=eurusd_df.index.max().date().isoformat())
    vix_hourly = vix_df.resample('h').ffill()
    eurusd_df = eurusd_df.merge(vix_hourly, how='left', left_index=True, right_index=True)
    eurusd_df = eurusd_df.rename(columns={'VIX': 'vix'})
    return eurusd_df


def set_time_as_index(df):
    df['time'] = pd.to_datetime(df['time'])
    return df.set_index('time')


def remove_highly_correlated_features(df, threshold=0.95):
    protected_cols = {'open', 'high', 'low', 'close'}  # Colonnes à ne jamais supprimer

    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # Ne pas supprimer les colonnes protégées même si elles sont corrélées
    to_drop = [col for col in to_drop if col not in protected_cols]

    logger.info(f"Colonnes supprimées à cause d'une corrélation > {threshold} (hors OHLC) : {to_drop}")
    return df.drop(columns=to_drop), to_drop


def generate_label_with_triple_barrier_on_cumsum(
    df: pd.DataFrame,
    tp_pips: float,
    sl_pips: float,
    window: int
) -> List[int]:
    labels = []

    tp_threshold = tp_pips * 0.0001
    sl_threshold = sl_pips * 0.0001

    log_returns = df['log_return'].values

    for i in range(len(df) - window):
        # chemin local de prix centré à 0
        future_returns = log_returns[i + 1 : i + 1 + window]
        local_path = np.cumsum(future_returns)

        # barrières long
        upper_barrier_long = tp_threshold
        lower_barrier_long = -sl_threshold

        # barrières short
        upper_barrier_short = -tp_threshold
        lower_barrier_short = sl_threshold

        # logiques de touch
        long_tp_hit = next((j for j, p in enumerate(local_path) if p >= upper_barrier_long), None)
        long_sl_hit = next((j for j, p in enumerate(local_path) if p <= lower_barrier_long), None)

        short_tp_hit = next((j for j, p in enumerate(local_path) if p <= upper_barrier_short), None)
        short_sl_hit = next((j for j, p in enumerate(local_path) if p >= lower_barrier_short), None)

        if long_tp_hit is not None and (long_sl_hit is None or long_tp_hit < long_sl_hit):
            label = 1  # long
        elif short_tp_hit is not None and (short_sl_hit is None or short_tp_hit < short_sl_hit):
            label = 2  # short
        else:
            label = 0  # neutre

        labels.append(label)

    # Padding pour aligner la taille
    labels += [0] * window
    return labels


def process_data(df: pd.DataFrame, seed: int, inference: bool = False) -> pd.DataFrame:

    df['log_price'] = np.log(df['close'])
    df['log_return'] = df['log_price'].diff()

    if inference:
        # Utilise le d déjà défini
        d_optimal = config['model']['d_optimal']
        logger.info(f"[Inférence] Utilisation du d enregistré : {d_optimal}")
        frac = FracDifferentiator(d=d_optimal)
    else:
        if 'd_optimal' in config['model']:
            d_optimal = config['model']['d_optimal']
            logger.info(f"d déjà défini dans le config : {d_optimal}")
            frac = FracDifferentiator(d=d_optimal)
        else:
            logger.info("Recherche automatique du d optimal avec différentiation fractionnaire...")
            d_list = [0.2, 0.3, 0.4, 0.5, 0.6]
            frac = FracDifferentiator(d_values=d_list)
            d_optimal, pval = frac.fit(df['log_price'])
            logger.info(f"d sélectionné : {d_optimal} (p-value = {pval:.4f})")
            config['model']['d_optimal'] = d_optimal
            with open("config/config.json", "w") as f:
                json.dump(config, f, indent=4)

    try:
        df['frac_diff'] = frac.transform(df['log_price'])
    except ValueError as e:
        logger.warning(f"FracDiff échouée : {e}")
        df['frac_diff'] = np.nan

    df['frac_diff_cumsum'] = df['frac_diff'].cumsum()

    logger.info("Calcul des indicateurs techniques...")
    df['sma_50'] = SMAIndicator(df['close'], window=50).sma_indicator()
    df['sma_200'] = SMAIndicator(df['close'], window=200).sma_indicator()
    df['ema_20'] = EMAIndicator(df['close'], window=20).ema_indicator()
    df['ema_100'] = EMAIndicator(df['close'], window=100).ema_indicator()

    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    df['rsi_dist_oversold'] = df['rsi'] - 30
    df['rsi_dist_overbought'] = 70 - df['rsi']
    df['rsi_signal'] = ((df['rsi'] > 70) | (df['rsi'] < 30)).astype(int)

    bb = BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_dist_upper'] = df['bb_upper'] - df['close']
    df['bb_dist_lower'] = df['close'] - df['bb_lower']
    df['bb_width'] = df['bb_upper'] - df['bb_lower']

    df['above_sma_50'] = (df['close'] > df['sma_50']).astype(int)
    df['above_sma_200'] = (df['close'] > df['sma_200']).astype(int)
    df['sma_50_vs_200'] = (df['sma_50'] > df['sma_200']).astype(int)

    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_diff'] = macd.macd_diff()

    stoch = StochasticOscillator(df['high'], df['low'], df['close'])
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()

    obv = OnBalanceVolumeIndicator(df['close'], df['volume'])
    df['obv'] = obv.on_balance_volume()

    df['price_diff_1'] = df['close'].diff()
    df['price_diff_2'] = df['price_diff_1'].diff()

    returns = df['close'].pct_change()
    df['autocorr_return_1'] = returns.rolling(10).apply(lambda x: x.autocorr(lag=1), raw=False)
    df['autocorr_return_5'] = returns.rolling(20).apply(lambda x: x.autocorr(lag=5), raw=False)
    df['autocorr_price_5'] = df['close'].rolling(20).apply(lambda x: x.autocorr(lag=5), raw=False)

    df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    df['adx'] = ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
    df['cci'] = CCIIndicator(df['high'], df['low'], df['close'], window=20).cci()
    df['williams_r'] = WilliamsRIndicator(df['high'], df['low'], df['close'], lbp=14).williams_r()
    df['roc'] = ROCIndicator(df['close'], window=12).roc()
    df['kama'] = KAMAIndicator(df['close'], window=10, pow1=2, pow2=30).kama()
    df['vwap'] = VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume'], window=14).volume_weighted_average_price()

    df = enrich_with_vix(df)

    logger.info("Nettoyage des données...")
    df.dropna(inplace=True)

    if inference:
        logger.info("Mode inférence : application des colonnes de l'entraînement")
        features_path = os.path.join(project_root,'exogenous_model', 'model', 'checkpoints', f'features_used_seed_{seed}.txt')

        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Fichier des features manquant : {features_path}")

        with open(features_path, 'r') as f:
            expected_columns = [line.strip() for line in f.readlines()]

        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Colonnes manquantes dans le DataFrame d'inférence : {missing_cols}")

        df = df.reindex(columns=expected_columns)
    else:
        df, _ = remove_highly_correlated_features(df, threshold=0.95)
        features_path = os.path.join(project_root,'exogenous_model', 'model', 'checkpoints', f'features_used_seed_{seed}.txt')
        os.makedirs(os.path.dirname(features_path), exist_ok=True)
        with open(features_path, 'w') as f:
            for col in df.columns:
                f.write(f"{col}\n")
        logger.info("Features du modèle conservée")

    logger.info("Data et features processées")
    return df


def process_target(df: pd.DataFrame) -> pd.DataFrame:

    TAKE_PROFIT_PIPS = config['dataset']['take_profit_pips']
    STOP_LOSS_PIPS = config['dataset']['stop_loss_pips']
    PREDICTION_WINDOW = config['dataset']['window']

    logger.info("Génération des labels triple barrière...")
    for w in [4, 8, 12, 24, 48]:
        labels = generate_label_with_triple_barrier_on_cumsum(df, TAKE_PROFIT_PIPS, STOP_LOSS_PIPS, w)
        counter = pd.Series(labels).value_counts(normalize=True)
        logger.debug(f"Window: {w}h - Distribution des labels: {counter.to_dict()}")

    logger.info(f"PREDICTION_WINDOW sélectionnée : {PREDICTION_WINDOW}")
    df['label'] = generate_label_with_triple_barrier_on_cumsum(df, TAKE_PROFIT_PIPS, STOP_LOSS_PIPS, PREDICTION_WINDOW)

    features = [col for col in df.columns if col not in ['label', 'time', 'log_price', 'frac_diff_cumsum']]
    df_final = df[features + ['label']]

    return df_final

def save_processed_dataframe(df: pd.DataFrame, split_name: str, seed: int):
    """
    Sauvegarde les données brutes (non séquencées) pour le méta-modèle au format CSV.

    Args:
        df (pd.DataFrame): Données brutes avec features + colonne 'label'
        split_name (str): 'train', 'val' ou 'test'
        seed (int): Seed pour la structure des dossiers
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    split_dir = os.path.join(project_root, 'exogenous_model', 'dataset', 'splits', f'seed_{seed}')
    os.makedirs(split_dir, exist_ok=True)

    # On sauvegarde uniquement les colonnes utiles (features + label)
    raw_path = os.path.join(split_dir, f'df_{split_name}_processed.csv')
    df.to_csv(raw_path, index=False)

    return raw_path


def process_split_compute_target_and_save(df: pd.DataFrame, seed: int):

    df_train, df_val, df_test = purge_train_test_split(
        df,
        train_ratio=0.7,
        val_ratio=0.15,
        max_dependency=0
    )

    train_processed = process_data(df_train, seed, False)
    val_processed = process_data(df_val, seed, False)
    test_processed = process_data(df_test, seed, False)

    train_processed_with_target = process_target(train_processed)
    val_processed_with_target = process_target(val_processed)
    test_processed_with_target = process_target(test_processed)

    # === 3. Sauvegarde brute pour méta-modèle (non séquencée) === #
    save_processed_dataframe(train_processed, 'train', seed)
    save_processed_dataframe(val_processed, 'val', seed)
    test_raw_path = save_processed_dataframe(test_processed, 'test', seed)
    logger.info(f"Données brutes test sauvegardées sous: {test_raw_path}")

    return train_processed_with_target, val_processed_with_target, test_processed_with_target

def purge_train_test_split(df, train_ratio=0.7, val_ratio=0.15, max_dependency=0):
    """
    Split le dataset en entraînement/validation/test en purgeant les données pour éviter le leakage.

    Args:
        df (pd.DataFrame): Données à splitter.
        train_ratio (float): Proportion pour l'entraînement (par défaut 70%).
        val_ratio (float): Proportion pour la validation (par défaut 15%).
        max_dependency (int): Longueur maximale de dépendance temporelle (ex: horizon de prédiction).
                             Si >0, applique un embargo pour éviter le look-ahead.

    Returns:
        df_train, df_val, df_test (pd.DataFrame): Ensembles purgés.
    """
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    # === Split initial === #
    train_end = n_train
    val_end = train_end + n_val

    # === Purge si dépendance temporelle === #
    if max_dependency > 0:
        # On retire `max_dependency` points avant/après chaque split
        train_end_purged = train_end - max_dependency
        val_end_purged = val_end - max_dependency

        df_train = df.iloc[:train_end_purged].reset_index(drop=True)
        df_val = df.iloc[train_end:val_end_purged].reset_index(drop=True)
        df_test = df.iloc[val_end:].reset_index(drop=True)
    else:
        # Split classique (pas de purge)
        df_train = df.iloc[:train_end].reset_index(drop=True)
        df_val = df.iloc[train_end:val_end].reset_index(drop=True)
        df_test = df.iloc[val_end:].reset_index(drop=True)

    return df_train, df_val, df_test


def generate_exogenous_dataset(seed):

    logger.info("Chargement des données...")
    path = kagglehub.dataset_download("orkunaktas/eurusd-1h-2020-2024-september-forex")
    csv_path = os.path.join(path, 'EURUSD_1H_2020-2024.csv')

    df = pd.read_csv(csv_path)
    df.dropna(inplace=True)
    df['time'] = pd.to_datetime(df['time'])
    df.sort_values('time', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.drop(columns=['real_volume','spread'], inplace=True)
    df = df.rename(columns={'tick_volume':'volume'})
    df = set_time_as_index(df)

    train_processed, val_processed, test_processed = process_split_compute_target_and_save(df, seed)

    logger.info(f"Taille train: {len(train_processed)}, val: {len(val_processed)}, test: {len(test_processed)}")


if __name__ == "__main__":
    generate_exogenous_dataset(42)
