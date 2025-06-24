# run_inference.py
import numpy as np
import pandas as pd
import torch
from inference_utils import load_models
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
input_path = os.path.join(ROOT, 'mt5_interface', 'input_data.json')
signal_path = os.path.join(ROOT, 'mt5_interface', 'signal.txt')

def preprocess(df):
    # Exemple simple, à adapter avec tes vrais features
    df['return'] = df['close'].pct_change().fillna(0)
    return df[['open', 'high', 'low', 'close', 'volume', 'return']].values[-60:]


def main():
    df = pd.read_json(input_path)
    lstm, scaler, xgb = load_models()

    X = preprocess(df)
    X_scaled = scaler.transform(X)

    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled).float().unsqueeze(0)  # shape: (1, seq_len, n_features)
        lstm_out = lstm(X_tensor).numpy().flatten()

    meta_input = np.concatenate([X_scaled[-1], lstm_out])
    signal = xgb.predict([meta_input])[0]

    with open(signal_path, 'w') as f:
        f.write(signal)

if __name__ == "__main__":
    main()
