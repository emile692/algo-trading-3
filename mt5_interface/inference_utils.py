# inference_utils.py
import torch
import joblib
import pickle
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def load_models():
    lstm_model_path = os.path.join(ROOT, 'exogenous_model', 'model', 'checkpoints', 'model_seed_42.pt')
    scaler_path = os.path.join(ROOT, 'exogenous_model', 'model', 'checkpoints', 'scaler_seed_42.pkl')
    xgb_path = os.path.join(ROOT, 'meta_model', 'results', 'xgboost_meta_model_seed_42.joblib')

    lstm = torch.load(lstm_model_path, map_location=torch.device('cpu'))
    lstm.eval()

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    xgb = joblib.load(xgb_path)
    return lstm, scaler, xgb
