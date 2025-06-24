# exogenous_model/train_and_test/train_model.py
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

# === CONFIGURATION === #
with open('C:\\Donnees\\ECL\\11-S9\\Deep Learning\\algo-trading-2\\config.json') as f:
    config = json.load(f)

BATCH_SIZE = config['model']["batch_size"]
EPOCHS = config['model']["batch_size"]
LR = config['model']["batch_size"]
PATIENCE = config['model']["batch_size"]

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

def train_and_save_model(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # === Load and scale data === #
    X = np.load('dataset/X_lstm.npy')
    y = np.load('dataset/y_lstm.npy')
    N, T, F = X.shape

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.reshape(-1, F)).reshape(N, T, F)

    # === Split === #
    dataset = ForexLSTMDataset(X_scaled, y)
    n_total = len(dataset)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.15)
    n_test = n_total - n_train - n_val

    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(seed))

    X_train = X_scaled[train_set.indices]
    y_train = y[train_set.indices]
    X_val = X_scaled[val_set.indices]
    y_val = y[val_set.indices]
    X_test = X_scaled[test_set.indices]
    y_test = y[test_set.indices]

    # === Save split === #
    split_prefix = f'dataset/splits/seed_{seed}'
    os.makedirs(split_prefix, exist_ok=True)
    np.save(f'{split_prefix}/X_train.npy', X_train)
    np.save(f'{split_prefix}/y_train.npy', y_train)
    np.save(f'{split_prefix}/X_val.npy', X_val)
    np.save(f'{split_prefix}/y_val.npy', y_val)
    np.save(f'{split_prefix}/X_test.npy', X_test)
    np.save(f'{split_prefix}/y_test.npy', y_test)

    # === Save raw close prices for test set (non-scaled) === #
    close_prices_test = X[test_set.indices, -1, 0]  # dernière timestep, première feature (close)
    np.save(f'{split_prefix}/close_prices.npy', close_prices_test)

    # === DataLoader === #
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

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
        train_loss = 0
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
        val_loss = 0
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                val_loss += criterion(model(X_val_batch), y_val_batch).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = f'model/checkpoints/model_seed_{seed}.pt'
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"⏹️ Early stopping at epoch {epoch+1}")
                break

    scaler_path = f'model/checkpoints/scaler_seed_{seed}.pkl'
    joblib.dump(scaler, scaler_path)

    return best_model_path, scaler_path
