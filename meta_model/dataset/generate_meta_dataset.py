"""generate_meta_datasets.py"""
import os

import torch
import numpy as np
import pandas as pd

from exogenous_model.model.core import LSTMClassifier
from exogenous_model.prediction.prediction import predict_exo_model

def temporal_feature_engineering(X):
    features = {
        'mean': X.mean(axis=1),
        'std': X.std(axis=1),
        'min': X.min(axis=1),
        'max': X.max(axis=1),
        'last': X[:, -1, :],
        'first': X[:, 0, :],
        'diff': X[:, -1, :] - X[:, 0, :]
    }
    features_concat = np.concatenate(list(features.values()), axis=1)
    return features_concat, list(features.keys())

def generate_meta_dataset(seed, logger):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    base_dir = os.path.join(project_root, 'exogenous_model', 'dataset', 'splits')
    model_dir = os.path.join(project_root, 'exogenous_model', 'model', 'checkpoints')
    output_dir = os.path.join(project_root, 'meta_model', 'dataset', 'features_and_target')

    base_path = os.path.join(base_dir, f'seed_{seed}')
    model_path = os.path.join(model_dir, f'model_seed_{seed}.pt')

    # Charger les données test
    X_test = np.load(os.path.join(base_path, 'X_test.npy'))
    y_test = np.load(os.path.join(base_path, 'y_test.npy'))

    model = LSTMClassifier(input_dim=X_test.shape[2]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    logger.info("Prédiction du modèle principal ...")
    test_preds = predict_exo_model(model, X_test, device)

    logger.info("Extraction de features temporelles ...")
    X_meta_feats, stat_keys = temporal_feature_engineering(X_test)

    feature_names = []
    for stat in stat_keys:
        feature_names.extend([f'{stat}_f{i}' for i in range(X_test.shape[2])])

    df_meta = pd.DataFrame(X_meta_feats, columns=feature_names)
    df_meta['y_pred'] = test_preds
    df_meta['y_true'] = y_test
    df_meta['meta_label'] = (df_meta['y_true'] == df_meta['y_pred']).astype(int)

    output_file_path = os.path.join(output_dir, f'meta_dataset_seed_{seed}.csv')
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    try:
        df_meta.to_csv(output_file_path, index=False)
        logger.info(f"Dataset sauvegardé sous {output_file_path}")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde : {e}")
        raise


if __name__ == "__main__":

    from exogenous_model.dataset.generate_dataset import logger
    generate_meta_dataset(42, logger)