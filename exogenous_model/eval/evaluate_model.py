# exogenous_model/eval/evaluate_model.py
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from exogenous_model.model.core import LSTMClassifier

BATCH_SIZE = 64

class ForexLSTMDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def evaluate_model(model_path: str, logger):

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    seed = int(model_path.split('_')[-1].split('.')[0])

    split_prefix = os.path.join(project_root, 'exogenous_model', 'dataset','splits', f'seed_{seed}')

    # Chargement des données
    X_test_path = os.path.join(split_prefix, 'X_test.npy')
    Y_test_path = os.path.join(split_prefix, 'Y_test.npy')

    X_test = np.load(X_test_path)
    y_test = np.load(Y_test_path)

    test_loader = DataLoader(ForexLSTMDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    # Modèle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(input_dim=X_test.shape[2]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Prédictions
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = torch.argmax(model(X_batch), dim=1)
            y_true.extend(y_batch.numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    y_pred_path = os.path.join(split_prefix, f'y_pred_seed_{seed}.npy')
    y_true_path = os.path.join(split_prefix, f'y_true_seed_{seed}.npy')

    np.save(y_pred_path, y_pred)
    np.save(y_true_path, y_true)

    logger.info(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    return {
        'accuracy': acc,
        'precision_macro': precision,
        'recall_macro': recall,
        'f1_macro': f1
    }
