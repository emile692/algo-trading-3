import json
import joblib
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from config.logger.logger import setup_logger

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
logger = setup_logger()

config_path = os.path.join(project_root, 'config', 'config.json')

with open(config_path, "r") as f:
    config = json.load(f)



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
    verbose: bool = True,
    stop_loss_pips: Optional[float] = None,
    pip_size: Optional[float] = None,
    prediction_window: Optional[int] = None,
    take_profit_pips: Optional[float] = None
):
    """
    Backtest triple-barrier :
      - utilise y_pred (1=long, 2=short, 0=wait) produit par le LSTM
      - veto du méta-modèle : si proba(classe0) >= threshold => on n'ouvre pas la position (HOLD)
      - si position ouverte : on surveille les closes des heures suivantes pendant `prediction_window`
        pour TP (take_profit_pips) ou SL (stop_loss_pips). Si aucune barrière touchée -> sortie à la verticale.
      - après ouverture d'un trade, on ignore les prédictions suivantes jusqu'à la fin de la prédiction window.
    Retourne dictionnaire métriques + DataFrame des trades.
    """

    os.makedirs(output_dir, exist_ok=True)

    # lecture config fallback
    cfg_ds = config.get('dataset', {}) if 'config' in globals() else {}
    TAKE_PROFIT_PIPS = take_profit_pips if take_profit_pips is not None else cfg_ds.get('take_profit_pips')
    PREDICTION_WINDOW = prediction_window if prediction_window is not None else cfg_ds.get('window')
    STOP_LOSS_PIPS = stop_loss_pips if stop_loss_pips is not None else cfg_ds.get('stop_loss_pips', TAKE_PROFIT_PIPS)
    PIP_SIZE = pip_size if pip_size is not None else cfg_ds.get('pips_size')

    if TAKE_PROFIT_PIPS is None or PREDICTION_WINDOW is None:
        raise ValueError("TAKE_PROFIT_PIPS et PREDICTION_WINDOW doivent être définis (passés en arg ou dans config).")

    # charger données
    y_pred = np.load(y_pred_exo_path).astype(int)    # 0/1/2
    y_proba_meta = np.load(y_proba_meta_path)       # prob classe 0 (erreur)
    df_test_raw = pd.read_csv(df_test_raw_exo_path)
    time_test = np.load(time_pred_meta_path, allow_pickle=True)
    df_test_raw.set_index('time', inplace=True)
    close_prices = df_test_raw.loc[time_test,'close']
    open_prices = df_test_raw.loc[time_test,'open']
    high_prices = df_test_raw.loc[time_test,'high']
    low_prices = df_test_raw.loc[time_test,'low']

    model_data = joblib.load(meta_model_path)
    threshold = model_data.get('threshold', 0.5)

    # stockage
    n = len(y_pred)
    trades = []  # dicts: entry_idx, exit_idx, entry_price, exit_price, direction, return, exit_type
    returns_time = np.zeros(n-1, dtype=float)  # per-step returns (0 when no trade open at that step)
    in_cooldown_until = -1
    i = 0

    while i < n - PREDICTION_WINDOW:
        # si on est en cooldown, skip
        if i < in_cooldown_until:
            i += 1
            continue

        pred = int(y_pred[i])
        meta_p = float(y_proba_meta[i])

        # decide entry
        entry_direction = None
        if pred == 1 and meta_p < threshold:
            entry_direction = 'long'
        elif pred == 2 and meta_p < threshold:
            entry_direction = 'short'
        else:
            i += 1
            continue

        entry_idx = i
        entry_price = float(close_prices[entry_idx])

        # définir barrières en prix
        tp_price = entry_price + TAKE_PROFIT_PIPS * PIP_SIZE if entry_direction == 'long' else entry_price - TAKE_PROFIT_PIPS * PIP_SIZE
        sl_price = entry_price - STOP_LOSS_PIPS * PIP_SIZE if entry_direction == 'long' else entry_price + STOP_LOSS_PIPS * PIP_SIZE

        exit_idx = None
        exit_price = None
        exit_type = 'time'  # default

        # parcourir la fenêtre pour détecter TP/SL sur les closes (approx.)
        # parcourir la fenêtre pour détecter TP/SL avec OHLC
        for h in range(1, PREDICTION_WINDOW + 1):

            bar_high = float(high_prices[entry_idx + h])
            bar_low = float(low_prices[entry_idx + h])

            if entry_direction == 'long':
                # SL touché si low <= sl_price avant que high >= tp_price
                if bar_low <= sl_price and bar_high >= tp_price:
                    # ordre d'atteinte impossible à déterminer sans intrabar -> on sort par SL par prudence
                    exit_idx = entry_idx + h
                    exit_price = sl_price
                    exit_type = 'sl'
                    break
                elif bar_high >= tp_price:
                    exit_idx = entry_idx + h
                    exit_price = tp_price
                    exit_type = 'tp'
                    break
                elif bar_low <= sl_price:
                    exit_idx = entry_idx + h
                    exit_price = sl_price
                    exit_type = 'sl'
                    break

            else:  # short
                if bar_high >= sl_price and bar_low <= tp_price:
                    exit_idx = entry_idx + h
                    exit_price = sl_price
                    exit_type = 'sl'
                    break
                elif bar_low <= tp_price:
                    exit_idx = entry_idx + h
                    exit_price = tp_price
                    exit_type = 'tp'
                    break
                elif bar_high >= sl_price:
                    exit_idx = entry_idx + h
                    exit_price = sl_price
                    exit_type = 'sl'
                    break

        # si aucune barrière atteinte -> sortie au time barrier
        if exit_idx is None:
            exit_idx = entry_idx + PREDICTION_WINDOW
            exit_price = float(close_prices[exit_idx])
            exit_type = 'time'

        # calcul du return brut et net (entrée->sortie)
        if entry_direction == 'long':
            gross = (exit_price - entry_price) / entry_price
        else:
            gross = (entry_price - exit_price) / entry_price

        net = gross - 2.0 * transaction_fee  # entrée + sortie

        # enregistrer le trade
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

        # enregistrer return dans la time-series (au index d'entrée)
        if entry_idx < len(returns_time):
            returns_time[entry_idx] = net

        # activer cooldown jusqu'à la fin de la préd window (on ignore autres préd)
        in_cooldown_until = entry_idx + PREDICTION_WINDOW
        i = in_cooldown_until  # on saute directement
    # fin boucle

    # calcul equity timeline (par timestamp)
    cumulative_returns = (1 + returns_time).cumprod()
    if len(cumulative_returns) == 0:
        cumulative_returns = np.array([1.0])
    final_capital = capital * cumulative_returns[-1]
    total_return_pct = (final_capital - capital) / capital * 100

    # métriques sur trades
    n_trades = len(trades)
    wins = [t for t in trades if t['net_return'] > 0]
    loss = [t for t in trades if t['net_return'] <= 0]
    winrate = (len(wins) / n_trades * 100) if n_trades > 0 else 0.0
    avg_net = np.mean([t['net_return'] for t in trades]) if n_trades > 0 else 0.0
    median_net = np.median([t['net_return'] for t in trades]) if n_trades > 0 else 0.0
    tp_count = sum(1 for t in trades if t['exit_type'] == 'tp')
    sl_count = sum(1 for t in trades if t['exit_type'] == 'sl')
    time_count = sum(1 for t in trades if t['exit_type'] == 'time')

    # Sharpe sur timeline (approx. per-step)
    excess_returns = returns_time - 0.0 / 252
    sharpe = (np.mean(excess_returns) / (np.std(excess_returns) + 1e-8)) * np.sqrt(252)

    # max drawdown timeline
    peak = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - peak) / peak
    max_dd = drawdowns.min() * 100

    # DataFrame trades
    trades_df = pd.DataFrame(trades)
    trades_csv = os.path.join(output_dir, f"trades_triple_barrier_seed_{seed}.csv")
    trades_df.to_csv(trades_csv, index=False)

    # affichage résumé
    if verbose:
        print("\n=== Résultat Triple-Barrier Backtest ===")
        print(f"Seed: {seed}")
        print(f"Seuil meta-model: {threshold:.4f}")
        print(f"TAKE_PROFIT_PIPS: {TAKE_PROFIT_PIPS}, STOP_LOSS_PIPS: {STOP_LOSS_PIPS}, PREDICTION_WINDOW: {PREDICTION_WINDOW}")
        print(f"N trades: {n_trades}  |  TP: {tp_count}  SL: {sl_count}  TIME: {time_count}")
        print(f"Winrate trades: {winrate:.2f}%  | Avg net: {avg_net:.5f}  | Median net: {median_net:.5f}")
        print(f"PNL total: {total_return_pct:.2f}%  | Final capital: {final_capital:.2f}€")
        print(f"Sharpe (timeline approx): {sharpe:.3f}  | Max drawdown: {max_dd:.2f}%")
        print(f"Trades saved to: {trades_csv}")

    # plot equity with entry/exit markers
    plt.figure(figsize=(12, 5))
    plt.plot(np.arange(len(cumulative_returns)), cumulative_returns * capital, label="Equity (timeline)", lw=1.2)

    # markers
    for t in trades:
        ei = t['entry_idx']
        xi = t['exit_idx']
        ep = t['entry_price']
        xp = t['exit_price']
        if t['direction'] == 'long':
            plt.scatter(ei, cumulative_returns[ei] * capital, marker='^', c='g', s=40, label='Entry long' if 'Entry long' not in plt.gca().get_legend_handles_labels()[1] else "")
        else:
            plt.scatter(ei, cumulative_returns[ei] * capital, marker='v', c='r', s=40, label='Entry short' if 'Entry short' not in plt.gca().get_legend_handles_labels()[1] else "")

        # exit marker colored by type
        color = 'green' if t['exit_type'] == 'tp' else ('red' if t['exit_type'] == 'sl' else 'orange')
        plt.scatter(xi, cumulative_returns[xi] * capital, marker='o', c=color, s=30,
                    label=f"Exit {t['exit_type']}" if f"Exit {t['exit_type']}" not in plt.gca().get_legend_handles_labels()[1] else "")

    plt.title(f"Equity Curve (Triple Barrier) - Seed {seed}")
    plt.xlabel("Time index")
    plt.ylabel("Capital (€)")
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"equity_triple_barrier_seed_{seed}.png"))
    plt.show()

    return {
        "seed": seed,
        "n_trades": n_trades,
        "pnl_pct": total_return_pct,
        "final_capital": final_capital,
        "winrate_trades_pct": winrate,
        "avg_net_return": avg_net,
        "median_net_return": median_net,
        "tp_count": tp_count,
        "sl_count": sl_count,
        "time_count": time_count,
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd,
        "trades_df": trades_df,
        "cumulative_returns": cumulative_returns
    }



if __name__ == "__main__":
    seed = 42
    res = run_backtest_triple_barrier(
        seed=seed,
        y_proba_meta_path=os.path.join(project_root, "meta_model", "results", f"seed_{seed}", "xgboost_meta_model_probs.npy"),
        y_pred_exo_path=os.path.join(project_root, "meta_model", "results", f"seed_{seed}", "exo_model_y_pred.npy"),
        time_pred_meta_path=os.path.join(project_root, "meta_model", "results", f"seed_{seed}", "xgboost_meta_model_time_test.npy"),
        meta_model_path=os.path.join(project_root, "meta_model", "results", f"seed_{seed}", f"xgboost_meta_model_seed_{seed}.joblib"),
        df_test_raw_exo_path=os.path.join(project_root, "exogenous_model", "dataset", "splits", f"seed_{seed}", "df_test_processed.csv"),
        capital=10000,
        transaction_fee=0.001,
        output_dir=os.path.join(project_root, "backtesting", "results")
    )
