# exogenous_model_v0/eval/evaluate_model.py
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, matthews_corrcoef,
    precision_recall_fscore_support, confusion_matrix, average_precision_score
)

from tools.logger import setup_logger
from exogenous_model_v0.model.core import LSTMClassifier

BATCH_SIZE = 64


class ForexLSTMDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def evaluate_model(model_path: str, logger=None):
    """Évalue un modèle LSTM sauvegardé sur le jeu de test et calcule des métriques détaillées."""

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    seed = int(model_path.split('_')[-1].split('.')[0])
    split_prefix = os.path.join(project_root, 'exogenous_model_v0', 'dataset', 'splits', f'seed_{seed}')

    # === Chargement des données ===
    X_test = np.load(os.path.join(split_prefix, 'X_test.npy'))
    y_test = np.load(os.path.join(split_prefix, 'Y_test.npy'))
    test_loader = DataLoader(ForexLSTMDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    # === Chargement du modèle ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(input_dim=X_test.shape[2]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # === Prédictions ===
    y_true, y_pred, y_proba = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(y_batch.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_proba.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_proba = np.array(y_proba)

    # === Calcul des métriques ===
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    prec_c, rec_c, f1_c, support_c = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    prec_macro = np.mean(prec_c)
    rec_macro = np.mean(rec_c)
    f1_macro = np.mean(f1_c)

    # PR-AUC par classe
    ap_per_class = []
    for k in range(y_proba.shape[1]):
        mask = (y_true == k).astype(int)
        if mask.sum() > 0:
            ap = average_precision_score(mask, y_proba[:, k])
        else:
            ap = np.nan
        ap_per_class.append(ap)
    ap_macro = np.nanmean(ap_per_class)

    # Matrices de confusion
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)

    # === Logs ===
    if logger is not None:
        logger.info(f"Accuracy: {acc:.4f} | Balanced Acc: {bacc:.4f} | MCC: {mcc:.4f}")
        logger.info(f"Macro -> Precision: {prec_macro:.4f} | Recall: {rec_macro:.4f} | F1: {f1_macro:.4f} | PR-AUC: {ap_macro:.4f}")
        logger.info("Matrice de confusion (valeurs absolues):")
        logger.info(f"\n{cm}")
        logger.info("Matrice de confusion normalisée:")
        logger.info(f"\n{np.round(cm_norm, 3)}")
    else:
        print(f"\n=== Résultats seed {seed} ===")
        print(f"Accuracy: {acc:.4f} | Balanced Acc: {bacc:.4f} | MCC: {mcc:.4f}")
        print(f"Macro -> Precision: {prec_macro:.4f} | Recall: {rec_macro:.4f} | F1: {f1_macro:.4f} | PR-AUC: {ap_macro:.4f}")
        print("\nMatrice de confusion :\n", cm)
        print("\nMatrice normalisée :\n", np.round(cm_norm, 3))

    # === Sauvegarde des résultats ===
    metrics = {
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "mcc": mcc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "pr_auc_macro": ap_macro,
        "per_class": [
            {
                "class": int(k),
                "precision": float(prec_c[k]),
                "recall": float(rec_c[k]),
                "f1": float(f1_c[k]),
                "support": int(support_c[k]),
                "pr_auc": None if np.isnan(ap_per_class[k]) else float(ap_per_class[k]),
            }
            for k in range(len(prec_c))
        ],
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
    }

    results_dir = os.path.join(project_root, 'exogenous_model_v0', 'results', f'seed_{seed}')
    os.makedirs(results_dir, exist_ok=True)

    np.save(os.path.join(results_dir, f'y_pred_seed_{seed}.npy'), y_pred)
    np.save(os.path.join(results_dir, f'y_true_seed_{seed}.npy'), y_true)
    np.save(os.path.join(results_dir, f'y_proba_seed_{seed}.npy'), y_proba)

    with open(os.path.join(results_dir, 'metrics_detailed.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics


if __name__ == "__main__":

    logger = setup_logger()

    model_path = r"..//model//checkpoints//model_seed_42.pt"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")

    logger.info(f"Évaluation directe du modèle : {model_path}")
    metrics = evaluate_model(model_path, logger)
    logger.info("Évaluation terminée")
