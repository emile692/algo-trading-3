"""generate_meta_datasets.py"""
import os

import torch
import numpy as np
import pandas as pd
from exogenous_model.model.core import LSTMClassifier
from exogenous_model.prediction.prediction import predict_exo_model


def process_seed(seed, device, base_dir, model_dir, output_path, logger):

    base_path = os.path.join(base_dir ,f'seed_{seed}')
    model_path =  os.path.join(model_dir , f'model_seed_{seed}.pt')

    y_train = np.load(os.path.join(base_path, 'y_train.npy'))
    y_val = np.load(os.path.join(base_path, 'y_val.npy'))
    y_test = np.load(os.path.join(base_path, 'y_test.npy'))

    X_train = np.load(os.path.join(base_path, 'X_train.npy'))
    X_val =  np.load(os.path.join(base_path, 'X_val.npy'))
    X_test =  np.load(os.path.join(base_path, 'X_test.npy'))

    model = LSTMClassifier(input_dim=X_train.shape[2]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    logger.info("Prédiction du modèle principal ...")
    train_preds = predict_exo_model(model, X_train, device)
    val_preds = predict_exo_model(model, X_val, device)
    test_preds = predict_exo_model(model, X_test, device)

    # Concaténation des données
    X_meta = np.vstack([X_train, X_val, X_test])
    y_meta = np.concatenate([y_train, y_val, y_test])
    preds_meta = np.concatenate([train_preds, val_preds, test_preds])

    # Aplatir la séquence 3D en 2D (samples, features)
    X_meta_flat = X_meta.reshape(X_meta.shape[0], -1)

    # Créer DataFrame
    df_meta = pd.DataFrame(X_meta_flat)

    # Ajouter la prédiction primaire et la cible méta
    df_meta['y_pred'] = preds_meta
    df_meta['y_true'] = y_meta
    df_meta['meta_label'] = (df_meta['y_true'] == df_meta['y_pred']).astype(int)

    # Sauvegarder
    output_file_path = os.path.join(output_path , f'meta_dataset_seed_{seed}.csv')
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    logger.info("Méta labels créés")

    try:
        df_meta.to_csv(output_file_path, index=False)
        logger.info(f"Dataset sauvegardé sous {output_file_path}")
    except Exception as e:
        logger.error(f"Failed to save the dataset: {e}")
        raise


def generate_meta_dataset(seed, logger) :

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    base_dir = os.path.join(project_root,'exogenous_model','dataset','splits')
    model_dir = os.path.join(project_root,'exogenous_model','model','checkpoints')
    output_dir = os.path.join(project_root,'meta_model','dataset','features_and_target')

    process_seed(seed, device, base_dir, model_dir, output_dir, logger)