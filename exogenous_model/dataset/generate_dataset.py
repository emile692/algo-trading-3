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
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    logger.info(f"Colonnes supprimées à cause d'une corrélation > {threshold} : {to_drop}")
    return df.drop(columns=to_drop), to_drop


def frac_diff(series, d, thresh=1e-5):
    w = [1.]
    for k in range(1, len(series)):
        w_ = -w[-1] * (d - k + 1) / k
        if abs(w_) < thresh:
            break
        w.append(w_)
    w = np.array(w[::-1])
    width = len(w)
    diff_series = []
    for i in range(width, len(series)):
        window = series.iloc[i - width:i]
        if window.isnull().any():
            diff_series.append(np.nan)
        else:
            diff_series.append(np.dot(w, window))
    return pd.Series([np.nan] * width + diff_series, index=series.index)


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


def process_data(df : pd.DataFrame) -> pd.DataFrame:

    df['log_price'] = np.log(df['close'])
    df['log_return'] = df['log_price'].diff()

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

        # Sauvegarde du config avec le d trouvé
        with open("config/config.json", "w") as f:
            json.dump(config, f, indent=4)

    # Application de la transformation
    try:
        df['frac_diff'] = frac.transform(df['log_price'])
    except ValueError as e:
        logger.warning(f"FracDiff échouée : {e}")
        df['frac_diff'] = np.nan

    df['log_price_cumsum'] = df['frac_diff'].cumsum()

    logger.info("Calcul des indicateurs techniques...")
    # Tendances
    df['sma_50'] = SMAIndicator(df['close'], window=50).sma_indicator()
    df['sma_200'] = SMAIndicator(df['close'], window=200).sma_indicator()
    df['ema_20'] = EMAIndicator(df['close'], window=20).ema_indicator()
    df['ema_100'] = EMAIndicator(df['close'], window=100).ema_indicator()

    # RSI
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    df['rsi_dist_oversold'] = df['rsi'] - 30
    df['rsi_dist_overbought'] = 70 - df['rsi']
    df['rsi_signal'] = ((df['rsi'] > 70) | (df['rsi'] < 30)).astype(int)

    # Bollinger
    bb = BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_dist_upper'] = df['bb_upper'] - df['close']
    df['bb_dist_lower'] = df['close'] - df['bb_lower']
    df['bb_width'] = df['bb_upper'] - df['bb_lower']

    # Moyennes
    df['above_sma_50'] = (df['close'] > df['sma_50']).astype(int)
    df['above_sma_200'] = (df['close'] > df['sma_200']).astype(int)
    df['sma_50_vs_200'] = (df['sma_50'] > df['sma_200']).astype(int)

    # MACD
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    # df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()

    # Stochastique
    stoch = StochasticOscillator(df['high'], df['low'], df['close'])
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()

    # OBV
    obv = OnBalanceVolumeIndicator(df['close'], df['tick_volume'])
    df['obv'] = obv.on_balance_volume()

    # Features dérivées du prix
    df['price_diff_1'] = df['close'].diff()
    df['price_diff_2'] = df['price_diff_1'].diff()

    # Autocorrélation
    returns = df['close'].pct_change()
    df['autocorr_return_1'] = returns.rolling(10).apply(lambda x: x.autocorr(lag=1), raw=False)
    df['autocorr_return_5'] = returns.rolling(20).apply(lambda x: x.autocorr(lag=5), raw=False)
    df['autocorr_price_5'] = df['close'].rolling(20).apply(lambda x: x.autocorr(lag=5), raw=False)

    # Indicateurs Quantreo
    df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    df['adx'] = ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
    df['cci'] = CCIIndicator(df['high'], df['low'], df['close'], window=20).cci()
    df['williams_r'] = WilliamsRIndicator(df['high'], df['low'], df['close'], lbp=14).williams_r()
    df['roc'] = ROCIndicator(df['close'], window=12).roc()
    df['kama'] = KAMAIndicator(df['close'], window=10, pow1=2, pow2=30).kama()
    df['vwap'] = VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['tick_volume'],
                                            window=14).volume_weighted_average_price()
    df = enrich_with_vix(df)

    logger.info("Nettoyage des données...")
    df.dropna(inplace=True)
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

    features = [col for col in df.columns if col not in ['label', 'time', 'log_price', 'log_price_cumsum']]
    df_final, _ = remove_highly_correlated_features(df[features + ['label']], threshold=0.95, logger=logger)

    csv_output_path = os.path.join(project_root, config['dataset']['output_dataset_path'])
    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
    df_final.to_csv(csv_output_path, index=False)
    logger.info(f"Dataset sauvegardé sous {csv_output_path}")

    return df_final

def generate_exogenous_dataset(logger):

    SEQUENCE_LENGTH = config['model']['sequence_length']

    logger.info("Chargement des données...")
    path = kagglehub.dataset_download("orkunaktas/eurusd-1h-2020-2024-september-forex")
    csv_path = os.path.join(path, 'EURUSD_1H_2020-2024.csv')

    df = pd.read_csv(csv_path)
    df.dropna(inplace=True)
    df['time'] = pd.to_datetime(df['time'])
    df.sort_values('time', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.drop(columns='real_volume', inplace=True)
    df = set_time_as_index(df)

    df_final = process_data(df)
    df_final_with_target = process_target(df_final)

    logger.info("Construction des séquences...")
    sequence_data = []
    sequence_labels = []
    feature_columns = df_final_with_target.columns.drop('label')

    for i in range(SEQUENCE_LENGTH, len(df_final_with_target)):
        seq = df_final_with_target.iloc[i - SEQUENCE_LENGTH:i][feature_columns]
        label = df_final_with_target.iloc[i]['label']
        sequence_data.append(seq.values)
        sequence_labels.append(label)

    X = np.array(sequence_data)
    y = np.array(sequence_labels)

    npz_output_path = os.path.join(project_root, config['dataset']['output_Xy_npz_path'])
    np.savez_compressed(npz_output_path, X=X, y=y, columns=feature_columns.to_list())
    logger.info(f"Dataset séquentiel sauvegardé sous {npz_output_path} avec noms de colonnes")

