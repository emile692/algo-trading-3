"""exogenous_model/train_and_test/train_model.py"""
import json
import os
import numpy as np
import torch
import torch.nn as nn
import joblib

from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from exogenous_model.model.core import LSTMClassifier

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
config_path = os.path.join(project_root, 'config', 'config.json')

# === CONFIGURATION === #
with open(config_path) as f:
    config = json.load(f)

BATCH_SIZE = config['model']["batch_size"]
EPOCHS = config['model']["epochs"]
LR = config['model']["learning_rate"]
PATIENCE = config['model']["patience"]


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

def train_and_save_model(seed: int, logger):

    torch.manual_seed(seed)
    np.random.seed(seed)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # === Load raw data === #
    data = np.load(os.path.join(project_root, config['dataset']["output_Xy_npz_path"]), allow_pickle=True)
    X = data["X"]  # shape: (N, T, F)
    y = data["y"]
    columns = data["columns"]
    logger.info(f"Features list : {columns}")

    N, T, F = X.shape

    # === Split raw data first === #
    indices = np.arange(N)
    np.random.shuffle(indices)

    n_train = int(N * 0.7)
    n_val = int(N * 0.15)
    n_test = N - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    X_train_raw, y_train = X[train_idx], y[train_idx]
    X_val_raw, y_val = X[val_idx], y[val_idx]
    X_test_raw, y_test = X[test_idx], y[test_idx]

    # === Scale using only train set === #
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw.reshape(-1, F)).reshape(-1, T, F)
    X_val = scaler.transform(X_val_raw.reshape(-1, F)).reshape(-1, T, F)
    X_test = scaler.transform(X_test_raw.reshape(-1, F)).reshape(-1, T, F)

    # === Save the splits === #
    split_prefix = os.path.join(project_root, 'exogenous_model', 'dataset', 'splits', f'seed_{seed}')
    os.makedirs(split_prefix, exist_ok=True)
    np.save(os.path.join(split_prefix, 'X_train.npy'), X_train)
    np.save(os.path.join(split_prefix, 'y_train.npy'), y_train)
    np.save(os.path.join(split_prefix, 'X_val.npy'), X_val)
    np.save(os.path.join(split_prefix, 'y_val.npy'), y_val)
    np.save(os.path.join(split_prefix, 'X_test.npy'), X_test)
    np.save(os.path.join(split_prefix, 'y_test.npy'), y_test)

    # === Save raw close prices for test set === #
    close_prices_test = X_test_raw[:, -1, 0]  # dernière timestep, première feature (close)
    np.save(os.path.join(split_prefix, 'close_prices.npy'), close_prices_test)

    # === DataLoaders === #
    train_set = ForexLSTMDataset(X_train, y_train)
    val_set = ForexLSTMDataset(X_val, y_val)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    # === Model init === #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(input_dim=F).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    class_weights_np = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
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

        # === Validation === #
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                val_loss += criterion(model(X_batch), y_batch).item()
        val_loss /= len(val_loader)

        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            logger.debug(
                f"[Epoch {epoch + 1}/{EPOCHS}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Best Val Loss: {best_val_loss:.4f}"
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

    # === Save scaler === #
    scaler_path = os.path.join(project_root, 'exogenous_model', 'model', 'checkpoints', f'scaler_seed_{seed}.pkl')
    joblib.dump(scaler, scaler_path)

    return best_model_path, scaler_path

