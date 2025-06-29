# inference_utils.py
import os
import torch
import joblib
import pickle

from exogenous_model.model.core import LSTMClassifier  # Assure-toi que ce chemin est correct

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def load_models():

    best_seed = 42
    # === Paths === #
    lstm_weights_path = os.path.join(ROOT, 'exogenous_model', 'model', 'checkpoints', f'model_seed_{best_seed}.pt')
    scaler_path = os.path.join(ROOT, 'exogenous_model', 'model', 'checkpoints', f'scaler_seed_{best_seed}.pkl')
    xgb_path = os.path.join(ROOT, 'meta_model', 'results', f'seed_{best_seed}', f'xgboost_meta_model_seed_{best_seed}.joblib')

    # === Load scaler === #
    scaler = joblib.load(scaler_path)

    # === Recreate model and load weights === #
    input_dim = scaler.mean_.shape[0]  # input_dim = number of features
    lstm = LSTMClassifier(input_dim=input_dim)
    state_dict = torch.load(lstm_weights_path, map_location=torch.device('cpu'))
    lstm.load_state_dict(state_dict)
    lstm.eval()

    # === Load XGBoost meta-model === #
    xgb = joblib.load(xgb_path)

    return lstm, scaler, xgb
