# run_inference.py
import numpy as np
import pandas as pd
import torch

from exogenous_model_v0.dataset.generate_dataset import process_data
from inference_utils import load_models
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 1. Try to detect tester path
tester_path = os.path.expandvars(r"%APPDATA%\MetaQuotes\Tester")
live_path = os.path.expandvars(r"%APPDATA%\MetaQuotes\Terminal")

# 2. Choix du bon répertoire
if os.path.exists(tester_path):
    # Regarde dans les sous-dossiers de tester pour trouver ton agent actif
    for agent in os.listdir(tester_path):
        candidate = os.path.join(tester_path, agent, "Agent-127.0.0.1-3000", "MQL5", "Files")
        if os.path.exists(candidate):
            base_path = candidate
            break
else:
    # Mode normal/live
    base_path = os.path.join(live_path, "<TON_ID_TERMINAL>", "MQL5", "Files")

# Accès aux fichiers
input_path = os.path.join(base_path, "input.json")
signal_path = os.path.join(base_path, "signal.txt")

def main():

    best_seed = 42

    df = pd.read_json(input_path)
    df = df.iloc[::-1].reset_index(drop=True)
    df.set_index('time', drop= True, inplace=True)
    df.index = pd.to_datetime(df.index)

    lstm, scaler, xgb = load_models(best_seed)

    X = process_data(df, best_seed, True)

    X_scaled = scaler.transform(X)

    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled).float().unsqueeze(0)  # shape: (1, seq_len, n_features)
        lstm_out = lstm(X_tensor).numpy().flatten()

    meta_input = np.concatenate([X_scaled[-1], lstm_out])
    meta_input = meta_input.reshape(1, -1)
    signal = xgb.predict(meta_input)[0]

    with open(signal_path, 'w') as f:
        f.write(signal)

if __name__ == "__main__":
    main()
