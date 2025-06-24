"""generate_meta_datasets.py"""
import json

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from exogenous_model.model.core import LSTMClassifier
from exogenous_model.prediction.prediction import predict_exo_model


def process_seed(seed, device, base_dir, model_dir, output_dir):

    print(f"\n[•] Traitement du seed {seed}...")

    base_path = base_dir / f'seed_{seed}'
    model_path = model_dir / f'model_seed_{seed}.pt'
    output_path = output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    y_train = np.load(base_path / 'y_train.npy')
    y_val = np.load(base_path / 'y_val.npy')
    y_test = np.load(base_path / 'y_test.npy')

    X_train = np.load(base_path / 'X_train.npy')
    X_val = np.load(base_path / 'X_val.npy')
    X_test = np.load(base_path / 'X_test.npy')

    model = LSTMClassifier(input_dim=X_train.shape[2]).to(device)
    print(f"[INFO] Seed {seed} - X_train shape: {X_train.shape}")
    model.load_state_dict(torch.load(model_path, map_location=device))

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
    output_file = output_path / f'meta_dataset_seed_{seed}.csv'
    df_meta.to_csv(output_file, index=False)
    print(f"[✓] Fichier sauvegardé : {output_file}")


def generate_meta_dataset(seed) :

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dir = Path(r'/exogenous_model/dataset/splits')
    model_dir = Path(r'/exogenous_model/model/checkpoints')
    output_dir = Path(r'/meta_model/dataset/features_and_target')

    process_seed(seed, device, base_dir, model_dir, output_dir)