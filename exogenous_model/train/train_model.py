"""exogenous_model/train_and_test/train_model.py"""
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from exogenous_model.model.core import LSTMClassifier
from exogenous_model.dataset.generate_dataset import logger

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
config_path = os.path.join(project_root, 'config', 'config.json')

# === CONFIGURATION === #
with open(config_path) as f:
    config = json.load(f)

BATCH_SIZE = config['model']["batch_size"]
EPOCHS = config['model']["epochs"]
LR = config['model']["learning_rate"]
PATIENCE = config['model']["patience"]
SEQUENCE_LENGTH = config['model']['sequence_length']


class ForexLSTMDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha if alpha is not None else torch.tensor([1.0])
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def save_raw_dataframe_for_meta_model(df: pd.DataFrame, split_name: str, seed: int):
    """
    Sauvegarde les données brutes (non séquencées) pour le méta-modèle au format CSV.

    Args:
        df (pd.DataFrame): Données brutes avec features + colonne 'label'
        split_name (str): 'train', 'val' ou 'test'
        seed (int): Seed pour la structure des dossiers
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    split_dir = os.path.join(project_root, 'exogenous_model', 'dataset', 'splits', f'seed_{seed}')
    os.makedirs(split_dir, exist_ok=True)

    # On sauvegarde uniquement les colonnes utiles (features + label)
    raw_path = os.path.join(split_dir, f'{split_name}_raw.csv')
    df.to_csv(raw_path, index=False)

    return raw_path



def create_sequences(df: pd.DataFrame, sequence_length: int, label_col: str = 'label'):
    """
    Transforme un DataFrame en données séquentielles (X, y) pour l'entraînement d'un modèle.

    Args:
        df (pd.DataFrame): DataFrame contenant les features + une colonne de label.
        sequence_length (int): Longueur des séquences à générer.
        label_col (str): Nom de la colonne cible.

    Returns:
        X (np.ndarray): Données séquentielles de forme (n_samples, sequence_length, n_features).
        y (np.ndarray): Labels associés de forme (n_samples,).
        feature_columns (list): Liste des colonnes utilisées comme features.
    """
    sequence_data = []
    sequence_labels = []

    feature_columns = df.columns.drop(label_col)

    for i in range(sequence_length, len(df)):
        seq = df.iloc[i - sequence_length:i][feature_columns]
        label = df.iloc[i][label_col]
        sequence_data.append(seq.values)
        sequence_labels.append(label)

    X = np.array(sequence_data)
    y = np.array(sequence_labels)

    return X, y, feature_columns.to_list()


def train_and_save_model(seed: int, logger):
    torch.manual_seed(seed)
    np.random.seed(seed)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # === 1. Charger le DataFrame final avec les labels === #
    df_final_path = os.path.join(project_root, 'exogenous_model', 'dataset', 'features_and_target', f'seed_{seed}', f'features_and_target.csv')
    df = pd.read_csv(df_final_path)

    # === 2. Split brut (avant scaling / séquençage) === #
    n = len(df)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    df_train = df.iloc[:n_train].reset_index(drop=True)
    df_val = df.iloc[n_train:n_train+n_val].reset_index(drop=True)
    df_test = df.iloc[n_train+n_val:].reset_index(drop=True)

    # === 3. Sauvegarde brute pour méta-modèle (non séquencée) === #
    save_raw_dataframe_for_meta_model(df_train, 'train', seed)
    save_raw_dataframe_for_meta_model(df_val, 'val', seed)
    test_raw_path = save_raw_dataframe_for_meta_model(df_test, 'test', seed)
    logger.info(f"Données brutes test sauvegardées sous: {test_raw_path}")

    # === 4. Scaling (fit uniquement sur train) === #
    feature_cols = df.columns.drop('label')
    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols])

    df_train[feature_cols] = scaler.transform(df_train[feature_cols])
    df_val[feature_cols] = scaler.transform(df_val[feature_cols])
    df_test[feature_cols] = scaler.transform(df_test[feature_cols])

    # === 5. Séquençage === #
    X_train, y_train, _ = create_sequences(df_train, SEQUENCE_LENGTH)
    X_val, y_val, _ = create_sequences(df_val, SEQUENCE_LENGTH)
    X_test, y_test, _ = create_sequences(df_test, SEQUENCE_LENGTH)

    # === 6. Sauvegarde des splits === #
    split_prefix = os.path.join(project_root, 'exogenous_model', 'dataset', 'splits', f'seed_{seed}')
    os.makedirs(split_prefix, exist_ok=True)
    np.save(os.path.join(split_prefix, 'X_train.npy'), X_train)
    np.save(os.path.join(split_prefix, 'y_train.npy'), y_train)
    np.save(os.path.join(split_prefix, 'X_val.npy'), X_val)
    np.save(os.path.join(split_prefix, 'y_val.npy'), y_val)
    np.save(os.path.join(split_prefix, 'X_test.npy'), X_test)
    np.save(os.path.join(split_prefix, 'y_test.npy'), y_test)

    # Sauvegarde de la feature "close" au dernier timestep pour analyse
    close_prices_test = X_test[:, -1, 0]  # Dernier pas de temps, 1ère colonne = 'close'
    np.save(os.path.join(split_prefix, 'close_prices.npy'), close_prices_test)

    # === 7. Préparation des DataLoaders === #
    train_loader = DataLoader(ForexLSTMDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ForexLSTMDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    # === 8. Entraînement du modèle === #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(input_dim=X_train.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    class_weights_np = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(device)
    criterion = FocalLoss(alpha=class_weights)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                val_loss += criterion(model(X_batch), y_batch).item()
        val_loss /= len(val_loader)

        if epoch % 2 == 0 or epoch == EPOCHS - 1:
            logger.debug(
                f"[Epoch {epoch + 1}/{EPOCHS}] Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Best Val Loss: {best_val_loss:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(project_root, 'exogenous_model', 'model', 'checkpoints', f'model_seed_{seed}.pt')
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    # === 9. Sauvegarde du scaler === #
    scaler_path = os.path.join(project_root, 'exogenous_model', 'model', 'checkpoints', f'scaler_seed_{seed}.pkl')
    joblib.dump(scaler, scaler_path)

    return best_model_path, scaler_path



if __name__ == "__main__":

    train_and_save_model(42, logger)