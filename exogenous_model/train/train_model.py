"""exogenous_model/train_and_test/train_model.py"""
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

from torch.utils.data import DataLoader, Dataset
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


def create_sequences(
    df: pd.DataFrame,
    sequence_length: int,
    label_col: str = "label",
    time_col: str = "time",
    pred_window: int = 0,
    start_idx: int = 0,
    end_idx: int | None = None,
    embargo: int = 0,
):
    """
    Transforme un DataFrame en données séquentielles (X, y, times) sans fuite temporelle.

    Args:
        df (pd.DataFrame): DataFrame contenant les features + une colonne label + time.
        sequence_length (int): Longueur de chaque séquence.
        label_col (str): Nom de la colonne cible.
        time_col (str): Nom de la colonne contenant les timestamps.
        pred_window (int): Horizon de prédiction (nombre de pas dans le futur pour le label).
        start_idx (int): Index de départ dans df (inclus).
        end_idx (int | None): Index de fin (exclu). Si None, prend len(df).
        embargo (int): Nombre d'échantillons à ignorer à la frontière (séparation train/val/test).

    Returns:
        X (np.ndarray): (n_samples, sequence_length, n_features)
        y (np.ndarray): (n_samples,)
        times (np.ndarray): timestamps associés à chaque y
        feature_columns (list): liste des features utilisées
    """
    if end_idx is None:
        end_idx = len(df)
    assert end_idx > start_idx, "end_idx doit être > start_idx"

    feature_columns = df.columns.drop([label_col, time_col])

    sequence_data = []
    sequence_labels = []
    sequence_times = []

    # t = dernier index du bloc de contexte
    t_min = start_idx + sequence_length - 1
    # t_max = on s'arrête avant de dépasser end_idx - pred_window
    t_max = end_idx - 1 - pred_window - embargo

    for t in range(t_min, t_max + 1):
        seq = df.iloc[t - sequence_length + 1 : t + 1][feature_columns]
        label = df.iloc[t + pred_window][label_col]
        time_val = df.iloc[t + pred_window][time_col]

        sequence_data.append(seq.values)
        sequence_labels.append(label)
        sequence_times.append(time_val)

    X = np.array(sequence_data, dtype=np.float32)
    y = np.array(sequence_labels, dtype=np.int64)
    times = np.array(sequence_times)

    return X, y, times, feature_columns.to_list()



def train_and_save_model(seed: int, logger):
    logger.info(f"=== Début de l'entraînement avec seed {seed} ===")

    torch.manual_seed(seed)
    np.random.seed(seed)
    logger.debug("Seeds PyTorch et NumPy initialisés")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    logger.debug(f"Racine du projet: {project_root}")

    # === 1. Charger les DataFrame finaux avec les labels === #
    logger.info("Chargement des données d'entraînement, validation et test...")
    df_train_path = os.path.join(project_root, 'exogenous_model', 'dataset', 'splits', f'seed_{seed}',
                                 f'df_train_processed.csv')
    df_val_path = os.path.join(project_root, 'exogenous_model', 'dataset', 'splits', f'seed_{seed}',
                               f'df_val_processed.csv')
    df_test_path = os.path.join(project_root, 'exogenous_model', 'dataset', 'splits', f'seed_{seed}',
                                f'df_test_processed.csv')

    train_processed = pd.read_csv(df_train_path)
    val_processed = pd.read_csv(df_val_path)
    test_processed = pd.read_csv(df_test_path)
    logger.info(
        f"Données chargées - Train: {len(train_processed)} | Val: {len(val_processed)} | Test: {len(test_processed)}")

    # === 4. Scaling (fit uniquement sur train) === #
    feature_cols = train_processed.columns.drop(['label', 'time'])
    logger.debug(f"Colonnes features: {feature_cols.tolist()}")

    logger.info("Normalisation des données avec StandardScaler...")
    scaler = StandardScaler()
    scaler.fit(train_processed[feature_cols])

    train_processed[feature_cols] = scaler.transform(train_processed[feature_cols])
    val_processed[feature_cols] = scaler.transform(val_processed[feature_cols])
    test_processed[feature_cols] = scaler.transform(test_processed[feature_cols])
    logger.info("Normalisation terminée")

    # === 5. Séquençage === #
    logger.info(f"Création des séquences (longueur: {SEQUENCE_LENGTH})...")

    with open(config_path) as f: #rechargement config
        config = json.load(f)

    PREDICTION_WINDOW = config['label']['window']
    EMBARGO = SEQUENCE_LENGTH + PREDICTION_WINDOW  # petit tampon de sécurité

    X_train, y_train, time_train_seq, _ = create_sequences(
        train_processed,
        SEQUENCE_LENGTH,
        pred_window=PREDICTION_WINDOW,
        start_idx=0,
        end_idx=len(train_processed),
        embargo=0,  # inutile ici car c’est un split indépendant
    )

    X_val, y_val, time_val_seq, _ = create_sequences(
        val_processed,
        SEQUENCE_LENGTH,
        pred_window=PREDICTION_WINDOW,
        start_idx=0,
        end_idx=len(val_processed),
        embargo=EMBARGO,  # optionnel mais safe
    )

    X_test, y_test, time_test_seq, _ = create_sequences(
        test_processed,
        SEQUENCE_LENGTH,
        pred_window=PREDICTION_WINDOW,
        start_idx=0,
        end_idx=len(test_processed),
        embargo=EMBARGO,
    )

    logger.debug(f"Dernier timestamp train: {time_train_seq[-1]}")
    logger.debug(f"Premier timestamp val: {time_val_seq[0]}")
    logger.debug(f"Premier timestamp test: {time_test_seq[0]}")

    logger.info(f"Séquences créées - X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}")

    # === 6. Sauvegarde des splits === #
    split_prefix = os.path.join(project_root, 'exogenous_model', 'dataset', 'splits', f'seed_{seed}')
    os.makedirs(split_prefix, exist_ok=True)
    logger.info(f"Sauvegarde des séquences dans {split_prefix}...")

    np.save(os.path.join(split_prefix, 'X_train.npy'), X_train)
    np.save(os.path.join(split_prefix, 'y_train.npy'), y_train)
    np.save(os.path.join(split_prefix, 'time_train.npy'), time_train_seq)

    np.save(os.path.join(split_prefix, 'X_val.npy'), X_val)
    np.save(os.path.join(split_prefix, 'y_val.npy'), y_val)
    np.save(os.path.join(split_prefix, 'time_val.npy'), time_val_seq)

    np.save(os.path.join(split_prefix, 'X_test.npy'), X_test)
    np.save(os.path.join(split_prefix, 'y_test.npy'), y_test)
    np.save(os.path.join(split_prefix, 'time_test.npy'), time_test_seq)
    logger.info("Sauvegarde des séquences terminée")

    # === 7. Préparation des DataLoaders === #
    logger.info(f"Préparation des DataLoaders (batch_size={BATCH_SIZE})...")
    train_loader = DataLoader(ForexLSTMDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ForexLSTMDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    logger.info(f"DataLoaders créés - Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # === 8. Entraînement du modèle === #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Utilisation du device: {device}")

    model = LSTMClassifier(input_dim=X_train.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    logger.info(f"Modèle initialisé - Architecture: {model}")

    class_weights_np = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(device)
    criterion = FocalLoss(alpha=class_weights)
    logger.info(f"Poids des classes calculés: {class_weights_np}")

    best_val_loss = float('inf')
    patience_counter = 0
    logger.info(f"Début de l'entraînement pour {EPOCHS} epochs (patience={PATIENCE})...")

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
            best_model_path = os.path.join(project_root, 'exogenous_model', 'model', 'checkpoints',
                                           f'model_seed_{seed}.pt')
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            logger.debug(f"Nouveau meilleur modèle sauvegardé à {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"Early stopping à l'epoch {epoch + 1}")
                break

    logger.info(f"Entraînement terminé - Meilleure val_loss: {best_val_loss:.4f}")

    # === 9. Sauvegarde du scaler === #
    scaler_path = os.path.join(project_root, 'exogenous_model', 'model', 'checkpoints', f'scaler_seed_{seed}.pkl')
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler sauvegardé à {scaler_path}")

    logger.info(f"=== Fin de l'entraînement avec seed {seed} ===")
    return best_model_path, scaler_path



if __name__ == "__main__":

    train_and_save_model(42, logger)