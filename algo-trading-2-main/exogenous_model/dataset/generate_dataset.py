"""generate_dataset.py"""
import pandas as pd
import numpy as np
import kagglehub
import json

from ta.momentum import (
    RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator, KAMAIndicator
)
from ta.trend import (
    SMAIndicator, EMAIndicator, MACD, ADXIndicator, CCIIndicator
)
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import (
    OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
)

from exogenous_model.dataset.external_source.evz_loader import download_vix

def enrich_with_vix(eurusd_df):

    vix_df = download_vix(start=eurusd_df.index.min().date().isoformat(),
                          end=eurusd_df.index.max().date().isoformat())

    vix_hourly = vix_df.resample('h').ffill()

    eurusd_df = eurusd_df.merge(vix_hourly, how='left', left_index=True, right_index=True)

    eurusd_df = eurusd_df.rename(columns={'VIX' : 'vix'})

    return eurusd_df

def set_time_as_index(df):
    df['time'] = pd.to_datetime(df['time'])
    return df.set_index('time')


def clean_dataset_robust(df: pd.DataFrame) -> pd.DataFrame:
    df_cleaned = df.copy()

    # Retirer les lignes avec des valeurs extrêmes ou infinies
    df_cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Filtrer les colonnes numériques pour le nettoyage
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        q1 = df_cleaned[col].quantile(0.01)
        q99 = df_cleaned[col].quantile(0.99)
        df_cleaned[col] = df_cleaned[col].clip(lower=q1, upper=q99)

    df_cleaned.dropna(inplace=True)
    df_cleaned.reset_index(drop=True, inplace=True)
    return df_cleaned


def generate_label_with_dd(df, tp_pips, window, max_dd_pips):
    labels = []
    tp_threshold = tp_pips * 0.0001
    dd_threshold = max_dd_pips * 0.0001

    close_prices = df['close'].values

    for i in range(len(df) - window):
        current_price = close_prices[i]
        future_prices = close_prices[i + 1:i + 1 + window]

        # BUY scenario : on cherche à atteindre TP sans drawdown > DD
        peak = current_price
        drawdown = 0
        for price in future_prices:
            peak = max(peak, price)
            drawdown = max(drawdown, peak - price)
            if (peak - current_price >= tp_threshold) and (drawdown <= dd_threshold):
                labels.append(1)
                break
        else:
            # SELL scenario
            trough = current_price
            drawup = 0
            for price in future_prices:
                trough = min(trough, price)
                drawup = max(drawup, price - trough)
                if (current_price - trough >= tp_threshold) and (drawup <= dd_threshold):
                    labels.append(2)
                    break
            else:
                labels.append(0)  # HOLD

    labels += [0] * window
    return labels

def generate_exogenous_dataset(seed):

    seed.set(seed)

    # === CONFIGURATION === #
    with open('/config.json') as f:
        config = json.load(f)

    TAKE_PROFIT_PIPS = config['dataset']["take_profit_pips"]
    MAX_DD_PIPS = config['dataset']["max_dd_pips"]
    PREDICTION_WINDOW = config['dataset']["window"]
    SEQUENCE_LENGTH = config['dataset']["sequence_length"]

    # === TÉLÉCHARGEMENT DES DONNÉES === #
    print("Téléchargement des données...")
    path = kagglehub.dataset_download("orkunaktas/eurusd-1h-2020-2024-september-forex")
    csv_path = path + '/EURUSD_1H_2020-2024.csv'

    df = pd.read_csv(csv_path)
    df.dropna(inplace=True)
    df['time'] = pd.to_datetime(df['time'])
    df.sort_values('time', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df.drop(columns='real_volume')

    # === INDICATEURS TECHNIQUES === #
    print("Calcul des indicateurs techniques...")
    df = set_time_as_index(df)

    # returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # Tendances
    df['sma_50'] = SMAIndicator(df['close'], window=50).sma_indicator()
    df['sma_200'] = SMAIndicator(df['close'], window=200).sma_indicator()
    df['ema_20'] = EMAIndicator(df['close'], window=20).ema_indicator()
    df['ema_100'] = EMAIndicator(df['close'], window=100).ema_indicator()

    # RSI & distances
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    df['rsi_dist_oversold'] = df['rsi'] - 30
    df['rsi_dist_overbought'] = 70 - df['rsi']
    df['rsi_signal'] = ((df['rsi'] > 70) | (df['rsi'] < 30)).astype(int)

    # Bollinger Bands
    bb = BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_dist_upper'] = df['bb_upper'] - df['close']
    df['bb_dist_lower'] = df['close'] - df['bb_lower']
    df['bb_width'] = df['bb_upper'] - df['bb_lower']

    # Position relative aux moyennes
    df['above_sma_50'] = (df['close'] > df['sma_50']).astype(int)
    df['above_sma_200'] = (df['close'] > df['sma_200']).astype(int)
    df['sma_50_vs_200'] = (df['sma_50'] > df['sma_200']).astype(int)

    # MACD
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()

    # Stochastique
    stoch = StochasticOscillator(df['high'], df['low'], df['close'])
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()

    # OBV
    obv = OnBalanceVolumeIndicator(df['close'], df['tick_volume'])
    df['obv'] = obv.on_balance_volume()

    # === DÉRIVÉES DU PRIX === #
    df['price_diff_1'] = df['close'].diff()
    df['price_diff_2'] = df['price_diff_1'].diff()

    # === AUTOCORRÉLATION === #
    returns = df['close'].pct_change()
    df['autocorr_return_1'] = returns.rolling(window=10).apply(lambda x: x.autocorr(lag=1), raw=False)
    df['autocorr_return_5'] = returns.rolling(window=20).apply(lambda x: x.autocorr(lag=5), raw=False)
    df['autocorr_price_5'] = df['close'].rolling(window=20).apply(lambda x: x.autocorr(lag=5), raw=False)

    # === FEATURES QUANTREO === #
    df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    df['adx'] = ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
    df['cci'] = CCIIndicator(df['high'], df['low'], df['close'], window=20).cci()
    df['williams_r'] = WilliamsRIndicator(df['high'], df['low'], df['close'], lbp=14).williams_r()
    df['roc'] = ROCIndicator(df['close'], window=12).roc()
    df['kama'] = KAMAIndicator(df['close'], window=10, pow1=2, pow2=30).kama()
    df['vwap'] = VolumeWeightedAveragePrice(
        high=df['high'], low=df['low'], close=df['close'], volume=df['tick_volume'], window=14
    ).volume_weighted_average_price()
    df = enrich_with_vix(df)

    # === NETTOYAGE ROBUSTE === #
    # df = clean_dataset_robust(df)
    print(f"Lignes avant nettoyage : {len(df)}")
    df = df.dropna()
    print(f"Lignes après nettoyage : {len(df)}")

    # === LABELING === #
    print("Création des labels...")
    df['label'] = generate_label_with_dd(df, TAKE_PROFIT_PIPS, PREDICTION_WINDOW, MAX_DD_PIPS)

    # === SÉLECTION FINALE === #
    print("Sélection des colonnes...")

    features = [
        'log_return',
        'rsi', 'rsi_dist_oversold', 'rsi_dist_overbought', 'rsi_signal',
        'sma_50', 'sma_200', 'ema_20', 'ema_100',
        'above_sma_50', 'above_sma_200', 'sma_50_vs_200',
        'bb_upper', 'bb_lower', 'bb_dist_upper', 'bb_dist_lower', 'bb_width',
        'macd', 'macd_signal', 'macd_diff',
        'stoch_k', 'stoch_d',
        'obv', 'atr', 'adx', 'cci', 'williams_r', 'vix',
        'roc', 'kama', 'vwap',
        'price_diff_1', 'price_diff_2',
        'autocorr_return_1', 'autocorr_return_5', 'autocorr_price_5'
    ]

    df_final = df[features + ['label']].copy()

    # === EXPORT CSV === #
    df_final.to_csv(config['dataset']["output_dataset_path"], index=False)
    print(f"✅ Dataset sauvegardé sous {config['dataset']['output_dataset_path']}")

    # === CONSTRUCTION DES SÉQUENCES === #
    sequence_data = []
    sequence_labels = []

    for i in range(SEQUENCE_LENGTH, len(df_final)):
        seq = df_final.iloc[i - SEQUENCE_LENGTH:i]
        label = df_final.iloc[i]['label']
        sequence_data.append(seq.drop(columns=['label']).values)
        sequence_labels.append(label)

    X = np.array(sequence_data)
    y = np.array(sequence_labels)

    np.save(config['dataset']["output_X_path"], X)
    np.save(config['dataset']["output_y_path"], y)

    print(
        f"✅ Dataset séquentiel sauvegardé sous {config['dataset']['output_X_path']} et {config['dataset']['output_y_path']}")