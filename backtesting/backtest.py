import json
import joblib
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from tools.logger import setup_logger

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
logger = setup_logger()

config_path = os.path.join(project_root, 'config', 'config.json')
with open(config_path, "r") as f:
    CONFIG = json.load(f)


def run_backtest_triple_barrier(
    seed: int,
    y_proba_meta_path: str,
    y_pred_exo_path: str,
    time_pred_meta_path: str,
    meta_model_path: str,
    df_test_raw_exo_path: str,
    capital: float = 10000.0,
    transaction_fee: float = 0.001,
    output_dir: str = "../backtesting/results",
    verbose: bool = True
):
    """
    Backtest triple-barrier aligné sur la config de labeling.
    """

    os.makedirs(output_dir, exist_ok=True)

    # === Lire les paramètres depuis la config ===
    cfg_label = CONFIG["label"]
    TAKE_PROFIT_PIPS = cfg_label["take_profit_pips"]
    STOP_LOSS_PIPS = cfg_label["stop_loss_pips"]
    PIP_SIZE = cfg_label["pips_size"]
    PREDICTION_WINDOW = cfg_label["window"]

    logger.info(f"Paramètres issus du config.json → TP={TAKE_PROFIT_PIPS} | SL={STOP_LOSS_PIPS} | window={PREDICTION_WINDOW}")

    # === Chargement des données ===
    y_pred = np.load(y_pred_exo_path).astype(int)
    y_proba_meta = np.load(y_proba_meta_path)
    time_test = np.load(time_pred_meta_path, allow_pickle=True)

    # Lecture du dataframe test
    if df_test_raw_exo_path.endswith(".parquet"):
        df_test_raw = pd.read_parquet(df_test_raw_exo_path)
    else:
        df_test_raw = pd.read_csv(df_test_raw_exo_path)

    if 'time' in df_test_raw.columns:
        df_test_raw['time'] = pd.to_datetime(df_test_raw['time'])
        df_test_raw.set_index('time', inplace=True)
    else:
        df_test_raw.index = pd.to_datetime(df_test_raw.index)

    close_prices = df_test_raw.loc[time_test, 'close']
    high_prices = df_test_raw.loc[time_test, 'high']
    low_prices = df_test_raw.loc[time_test, 'low']

    model_data = joblib.load(meta_model_path)
    threshold = float(model_data["threshold"])
    logger.info(f"Seuil méta-modèle chargé : {threshold:.4f}")

    # Harmonisation des tailles
    n = min(len(y_pred), len(y_proba_meta))
    y_pred, y_proba_meta = y_pred[:n], y_proba_meta[:n]

    trades = []
    returns_time = np.zeros(n - 1, dtype=float)
    in_cooldown_until = -1
    i = 0

    # === Boucle principale ===
    while i < n - PREDICTION_WINDOW - 1:
        if i >= n or i + PREDICTION_WINDOW >= n:
            break
        if i < in_cooldown_until:
            i += 1
            continue

        pred = int(y_pred[i])
        meta_p = float(y_proba_meta[i])

        # Entrée autorisée uniquement si le méta-veto est OK
        entry_direction = None
        if pred == 1 and meta_p < threshold:
            entry_direction = 'long'
        elif pred == 2 and meta_p < threshold:
            entry_direction = 'short'
        else:
            i += 1
            continue

        entry_idx = i
        entry_price = float(close_prices.iloc[entry_idx])

        tp_price = entry_price + TAKE_PROFIT_PIPS * PIP_SIZE if entry_direction == 'long' else entry_price - TAKE_PROFIT_PIPS * PIP_SIZE
        sl_price = entry_price - STOP_LOSS_PIPS * PIP_SIZE if entry_direction == 'long' else entry_price + STOP_LOSS_PIPS * PIP_SIZE

        exit_idx, exit_price, exit_type = None, None, 'time'

        for h in range(1, PREDICTION_WINDOW + 1):
            bar_high = float(high_prices.iloc[entry_idx + h])
            bar_low = float(low_prices.iloc[entry_idx + h])

            if entry_direction == 'long':
                if bar_low <= sl_price and bar_high >= tp_price:
                    exit_idx, exit_price, exit_type = entry_idx + h, sl_price, 'sl'
                    break
                elif bar_high >= tp_price:
                    exit_idx, exit_price, exit_type = entry_idx + h, tp_price, 'tp'
                    break
                elif bar_low <= sl_price:
                    exit_idx, exit_price, exit_type = entry_idx + h, sl_price, 'sl'
                    break
            else:
                if bar_high >= sl_price and bar_low <= tp_price:
                    exit_idx, exit_price, exit_type = entry_idx + h, sl_price, 'sl'
                    break
                elif bar_low <= tp_price:
                    exit_idx, exit_price, exit_type = entry_idx + h, tp_price, 'tp'
                    break
                elif bar_high >= sl_price:
                    exit_idx, exit_price, exit_type = entry_idx + h, sl_price, 'sl'
                    break

        if exit_idx is None or exit_idx >= len(close_prices):
            exit_idx = min(entry_idx + PREDICTION_WINDOW, len(close_prices) - 1)
            exit_price = float(close_prices.iloc[exit_idx])
            exit_type = 'time'

        gross = (exit_price - entry_price) / entry_price if entry_direction == 'long' else (entry_price - exit_price) / entry_price
        net = gross - 2.0 * transaction_fee

        trades.append({
            "entry_idx": entry_idx,
            "exit_idx": exit_idx,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "direction": entry_direction,
            "exit_type": exit_type,
            "gross_return": gross,
            "net_return": net,
            "meta_prob_at_entry": meta_p,
            "pred_at_entry": pred
        })

        if entry_idx < len(returns_time):
            returns_time[entry_idx] = net

        in_cooldown_until = entry_idx + PREDICTION_WINDOW
        i = in_cooldown_until

    # === Analyse des résultats ===
    cumulative_returns = (1 + returns_time).cumprod()
    final_capital = capital * cumulative_returns[-1]
    total_return_pct = (final_capital - capital) / capital * 100

    n_trades = len(trades)
    wins = [t for t in trades if t['net_return'] > 0]
    winrate = (len(wins) / n_trades * 100) if n_trades > 0 else 0.0
    avg_net = np.mean([t['net_return'] for t in trades]) if n_trades > 0 else 0.0
    tp_count = sum(1 for t in trades if t['exit_type'] == 'tp')
    sl_count = sum(1 for t in trades if t['exit_type'] == 'sl')

    # Résumé
    if verbose:
        print("\n=== Résultat Triple-Barrier Backtest ===")
        print(f"TP={TAKE_PROFIT_PIPS} | SL={STOP_LOSS_PIPS} | Window={PREDICTION_WINDOW}")
        print(f"N trades={n_trades} | Winrate={winrate:.2f}% | Avg net={avg_net:.5f}")
        print(f"PNL={total_return_pct:.2f}% | Final capital={final_capital:.2f}€")


    return {
        "seed": seed,
        "n_trades": n_trades,
        "pnl_pct": total_return_pct,
        "final_capital": final_capital,
        "winrate_trades_pct": winrate,
        "avg_net_return": avg_net,
        "tp_count": tp_count,
        "sl_count": sl_count
    }


if __name__ == "__main__":
    seed = 42
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    res = run_backtest_triple_barrier(
        seed=seed,
        y_proba_meta_path=os.path.join(project_root, "meta_model", "results", f"seed_{seed}", "y_prob_class0.npy"),
        meta_model_path=os.path.join(project_root, "meta_model", "results", f"seed_{seed}", f"xgboost_meta_model_seed_{seed}.joblib"),
        y_pred_exo_path=os.path.join(project_root, "exogenous_model", "results", f"seed_{seed}", f"y_pred_seed_{seed}.npy"),
        time_pred_meta_path=os.path.join(project_root, "exogenous_model", "results", f"seed_{seed}", "time.npy"),
        df_test_raw_exo_path=os.path.join(project_root, "data", "processed", f"seed_{seed}", f"EURUSD_H1_test.parquet"),
        capital=10000,
        transaction_fee=0.001,
        output_dir=os.path.join(project_root, "backtesting", "results")
    )


