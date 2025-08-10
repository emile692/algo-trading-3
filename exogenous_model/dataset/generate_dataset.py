import os
import pandas as pd
import numpy as np
import kagglehub
import json
from statsmodels.tsa.stattools import adfuller

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
    protected_cols = {'open', 'high', 'low', 'close', 'frac_diff'}  # Colonnes à ne jamais supprimer

    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # Ne pas supprimer les colonnes protégées même si elles sont corrélées
    to_drop = [col for col in to_drop if col not in protected_cols]

    logger.info(f"Colonnes supprimées à cause d'une corrélation > {threshold} (hors OHLC) : {to_drop}")
    return df.drop(columns=to_drop), to_drop


def generate_label_with_triple_barrier_on_frac_diff_cumsum(
    df: pd.DataFrame,
    tp_pips: float,
    sl_pips: float,
    pips_value: float,
    window: int
) -> list[int]:
    """
    Labels triple barrière (0 neutre, 1 long, 2 short) inspirés de López de Prado.

    - df['close'] : prix de clôture
    - df['frac_diff'] : rendements fractionnés (log-returns fractionnés)
    - tp_pips, sl_pips : seuils en pips (1 pip = 0.0001 pour EURUSD)
    """
    labels = []

    for i in range(len(df) - window):
        # Point d'entrée
        entry_price = df['close'].iloc[i]

        # Chemin des rendements cumulés (en log-returns)
        future_returns = df['frac_diff'].iloc[i+1 : i+1+window]
        cum_returns = np.cumsum(future_returns)

        # Conversion TP/SL en termes de log-return
        tp_return = np.log(1 + (tp_pips * pips_value) / entry_price)
        sl_return = -np.log(1 + (sl_pips * pips_value) / entry_price)

        # Trouver le premier moment où on touche une barrière
        hit_tp = next((j for j, r in enumerate(cum_returns) if r >= tp_return), None)
        hit_sl = next((j for j, r in enumerate(cum_returns) if r <= sl_return), None)

        if hit_tp is not None and (hit_sl is None or hit_tp < hit_sl):
            label = 1  # long
        elif hit_sl is not None and (hit_tp is None or hit_sl < hit_tp):
            label = 2  # short
        else:
            label = 0  # neutre

        labels.append(label)

    # Padding de tête pour aligner les indices
    labels = [np.nan] * window + labels
    return labels


def select_best_prediction_window(df, tp_pips, sl_pips, pips_size, candidate_windows, seed):
    """
    Évalue chaque fenêtre candidate sur le jeu fourni (train) en calculant
    l'entropie de la distribution des labels. Retourne la fenêtre qui maximise l'entropie.
    Sauvegarde aussi la fenêtre choisie dans les checkpoints et dans config.json.
    """
    results = {}
    for w in candidate_windows:
        labels = generate_label_with_triple_barrier_on_frac_diff_cumsum(df, tp_pips, sl_pips, pips_size, w)
        s = pd.Series(labels).dropna()
        if len(s) == 0:
            entropy = -np.inf
            dist = {}
        else:
            probs = s.value_counts(normalize=True)
            ps = probs.values
            entropy = -np.sum(ps * np.log(ps + 1e-12))  # entropie (base e)
            dist = probs.to_dict()
        results[w] = {'entropy': entropy, 'dist': dist, 'n_samples': int(len(s))}

    # choisir la fenêtre avec entropie maximale
    best_window = max(results.items(), key=lambda x: x[1]['entropy'])[0]

    # sauvegarder choix (fichier seed-specific)
    features_dir = os.path.join(project_root, 'exogenous_model', 'model', 'checkpoints')
    os.makedirs(features_dir, exist_ok=True)
    pred_window_path = os.path.join(features_dir, f'prediction_window_seed_{seed}.txt')
    with open(pred_window_path, 'w') as f:
        f.write(str(best_window))

    # mettre à jour config et sauvegarder proprement (utilise config_path défini en haut du fichier)
    config.setdefault('dataset', {})['window'] = int(best_window)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    logger.info(f"Fenêtre sélectionnée (seed={seed}) : {best_window}. Sauvegardée dans {pred_window_path} et config.")
    logger.debug(f"Détails sélection fenêtre: {results[best_window]}")

    return best_window, results


def process_data(df_entry: pd.DataFrame, seed: int, inference: bool = False,
                 split_type: str = "train") -> pd.DataFrame:
    df = df_entry.copy()

    # 1. Log-price
    df['log_price'] = np.log(df['close'])

    # 2. (Optionnel) centrer la série
    series_to_diff = df['log_price'] - df['log_price'].mean()

    # === Fractional Differencing ===
    if inference:
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
            frac = FracDifferentiator(thresh=1e-5)
            d_optimal, pval = frac.fit(series_to_diff)
            logger.info(f"d sélectionné : {d_optimal} (p-value = {pval:.4f})")
            config['model']['d_optimal'] = d_optimal

            with open(config_path, "w") as f:
                json.dump(config, f, indent=4)

    try:
        # Nettoyage et application FFD
        clean_series = series_to_diff.ffill().bfill()
        df['frac_diff'] = frac.transform(clean_series)

        # Vérification de la stationnarité
        adf_result = adfuller(df['frac_diff'].dropna(), maxlag=1, regression='c', autolag=None)
        logger.info(f"ADF après FFD: Stat={adf_result[0]:.4f}, p-value={adf_result[1]:.4f}")

        # Statistiques descriptives
        logger.info(f"Stats frac_diff: Moy={df['frac_diff'].mean():.6f}, Méd={df['frac_diff'].median():.6f}, "
                    f"%Pos={100 * (df['frac_diff'] > 0).mean():.2f}%")

    except ValueError as e:
        logger.warning(f"FracDiff échouée : {e}")
        df['frac_diff'] = np.nan

    df['frac_diff'].plot(title='EURUSD frac_diff')

    # === Technical Indicators ===
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

    # === Gestion des features conservées ===
    features_dir = os.path.join(project_root, 'exogenous_model', 'model', 'checkpoints')
    features_path = os.path.join(features_dir, f'features_used_seed_{seed}.txt')
    dropped_path = os.path.join(features_dir, f'dropped_cols_seed_{seed}.txt')

    if inference:
        logger.info("Mode inférence : application des colonnes de l'entraînement")

        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Fichier des features manquant : {features_path}")

        with open(features_path, 'r') as f:
            expected_columns = [line.strip() for line in f.readlines()]

        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Colonnes manquantes dans le DataFrame d'inférence : {missing_cols}")

        df = df.reindex(columns=expected_columns)

    elif split_type == "train":
        df, dropped_cols = remove_highly_correlated_features(df, threshold=0.95)

        os.makedirs(features_dir, exist_ok=True)

        with open(features_path, 'w') as f:
            for col in df.columns:
                f.write(f"{col}\n")
        with open(dropped_path, 'w') as f:
            for col in dropped_cols:
                f.write(f"{col}\n")

        logger.info("Features du modèle conservées (train split)")

    else:  # val ou test
        if not os.path.exists(dropped_path):
            raise FileNotFoundError(f"Fichier des colonnes supprimées manquant : {dropped_path}")

        with open(dropped_path, 'r') as f:
            dropped_cols = [line.strip() for line in f.readlines()]

        df = df.drop(columns=dropped_cols, errors='ignore')
        logger.info(f"Colonnes supprimées pour split {split_type} : {dropped_cols}")

    logger.info("Data et features processées")
    return df



def process_target(df: pd.DataFrame, prediction_window: int, split_name : str) -> pd.DataFrame:
    """
    Génère la colonne 'label' pour df en utilisant prediction_window choisi sur le train.
    """
    TAKE_PROFIT_PIPS = config['dataset']['take_profit_pips']
    STOP_LOSS_PIPS = config['dataset']['stop_loss_pips']
    PIPS_SIZE = config['dataset']['pips_size']

    logger.info(f"Utilisation de PREDICTION_WINDOW={prediction_window} pour les labels du jeu {split_name}.")
    df['label'] = generate_label_with_triple_barrier_on_frac_diff_cumsum(
        df,
        TAKE_PROFIT_PIPS,
        STOP_LOSS_PIPS,
        PIPS_SIZE,
        prediction_window
    )

    label_distribution = df['label'].value_counts(normalize=True).mul(100).round(2)
    logger.info(f"Distribution des labels ({split_name}):\n{label_distribution.to_string()}")
    logger.info(f"Nombre total d'échantillons: {len(df)}")

    features = [col for col in df.columns if col not in ['label', 'time', 'log_price']]
    return df[features + ['label']]



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
    """
    Split -> process_data -> sélectionner la PREDICTION_WINDOW sur le train -> générer labels
    et sauvegarder les fichiers bruts (non séquencés).
    """
    df_train, df_val, df_test = purge_train_test_split(
        df,
        train_ratio=0.7,
        val_ratio=0.15,
        max_dependency=0
    )

    # process features (columns) pour chaque split
    train_processed = process_data(df_train, seed, False, 'train')
    val_processed = process_data(df_val, seed, False, 'val')
    test_processed = process_data(df_test, seed, False, 'test')

    # --- Sélection de la fenêtre sur le train uniquement ---
    candidate_windows = [4, 8, 12, 24, 48]
    best_window, window_results = select_best_prediction_window(train_processed,
                                                                config['dataset']['take_profit_pips'],
                                                                config['dataset']['stop_loss_pips'],
                                                                config['dataset']['pips_size'],
                                                                candidate_windows,
                                                                seed)

    # Appliquer la même fenêtre à train/val/test
    train_processed_with_target = process_target(train_processed, prediction_window=best_window, split_name='train')
    val_processed_with_target = process_target(val_processed, prediction_window=best_window, split_name='val')
    test_processed_with_target = process_target(test_processed, prediction_window=best_window, split_name='test')

    # Sauvegarde brute pour méta-modèle (non séquencée)
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
        df_train, df_val, df_test (pd.DataFrame): Ensembles purgés, index conservé.
    """
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_end = n_train
    val_end = train_end + n_val

    if max_dependency > 0:
        train_end_purged = train_end - max_dependency
        val_start_purged = train_end + max_dependency
        val_end_purged = val_end - max_dependency
        test_start = val_end + max_dependency

        df_train = df.iloc[:train_end_purged]
        df_val = df.iloc[val_start_purged:val_end_purged]
        df_test = df.iloc[test_start:]
    else:
        df_train = df.iloc[:train_end]
        df_val = df.iloc[train_end:val_end]
        df_test = df.iloc[val_end:]

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
    df['close'].plot(title='EURUSD close')

    train_processed, val_processed, test_processed = process_split_compute_target_and_save(df, seed)

    logger.info(f"Taille train: {len(train_processed)}, val: {len(val_processed)}, test: {len(test_processed)}")


if __name__ == "__main__":
    generate_exogenous_dataset(42)
