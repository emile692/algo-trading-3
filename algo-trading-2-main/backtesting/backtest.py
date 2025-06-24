import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def run_backtest(
    y_pred_path: str,
    close_prices_path: str = "../exogenous_model/dataset/close_prices.npy",
    capital: float = 10000.0,
    transaction_fee: float = 0.001,  # 0.1% par trade (entrée + sortie)
    risk_free_rate: float = 0.0,     # Pour le Sharpe ratio
    output_dir: str = "../backtesting/results",
    save_equity: bool = True,
) -> dict:
    """
    Exécute un backtest et retourne les métriques de performance clés.

    Args:
        y_pred_path (str): Chemin du fichier .npy contenant les prédictions.
        close_prices_path (str): Chemin du fichier .npy contenant les prix de clôture.
        capital (float): Capital initial.
        transaction_fee (float): Frais de transaction proportionnels (par opération).
        risk_free_rate (float): Taux sans risque (pour Sharpe ratio).
        output_dir (str): Dossier pour sauvegarder la courbe de capital.
        save_equity (bool): Sauvegarder ou non la courbe de capital.

    Returns:
        dict: Résultats du backtest (pnl, winrate, sharpe, max_drawdown, etc.).
    """

    y_pred = np.load(y_pred_path)
    close_prices = np.load(close_prices_path)

    # Troncature au même nombre de points
    min_len = min(len(y_pred), len(close_prices))
    y_pred = y_pred[:min_len]
    close_prices = close_prices[:min_len]

    returns = []

    for i in range(min_len - 1):
        signal = y_pred[i]
        entry_price = close_prices[i]
        exit_price = close_prices[i + 1]

        if signal == 1:  # BUY
            gross_return = (exit_price - entry_price) / entry_price
            net_return = gross_return - 2 * transaction_fee  # Entrée + sortie
        elif signal == 0:  # SELL
            gross_return = (entry_price - exit_price) / entry_price
            net_return = gross_return - 2 * transaction_fee
        else:
            net_return = 0  # HOLD = aucune position

        returns.append(net_return)

    returns = np.array(returns)
    cumulative_returns = (1 + returns).cumprod()
    final_capital = capital * cumulative_returns[-1]
    total_return_pct = (final_capital - capital) / capital * 100
    winrate_pct = np.mean(returns > 0) * 100

    # === METRIQUES AVANCEES === #

    # Sharpe ratio
    excess_returns = returns - risk_free_rate / 252  # daily rate
    sharpe_ratio = np.mean(excess_returns) / (np.std(excess_returns) + 1e-8) * np.sqrt(252)

    # Max drawdown
    peak = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - peak) / peak
    max_drawdown_pct = drawdowns.min() * 100

    # Sauvegarde éventuelle de la courbe
    if save_equity:
        os.makedirs(output_dir, exist_ok=True)
        seed = extract_seed_from_path(y_pred_path)
        equity_path = os.path.join(output_dir, f"equity_curve_seed_{seed}.csv")
        pd.Series(cumulative_returns * capital).to_csv(equity_path)
        print(f"💾 Courbe de capital sauvegardée dans : {equity_path}")

    return {
        "seed": int(seed) if seed.isdigit() else seed,
        "pnl": round(total_return_pct, 2),
        "winrate": round(winrate_pct, 2),
        "sharpe_ratio": round(sharpe_ratio, 3),
        "max_drawdown": round(max_drawdown_pct, 2),
        "final_capital": round(final_capital, 2),
        "n_trades": len(returns),
    }


def extract_seed_from_path(path: str) -> str:
    """
    Extrait le numéro de seed depuis le nom de fichier, ex: y_pred_seed_42.npy → 42
    """
    base = os.path.basename(path)
    if "seed_" in base:
        return base.split("seed_")[-1].split(".")[0]
    return "unknown"



print()
