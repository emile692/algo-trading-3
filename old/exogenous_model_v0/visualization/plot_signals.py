import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import joblib
import plotly.graph_objects as go
from old.exogenous_model_v0.model.core import LSTMClassifier

import plotly.io as pio
pio.renderers.default = "browser"

# === CONFIG === #
MODEL_PATH = '../model/checkpoints/lstm_model.pt'
SCALER_PATH = '../model/checkpoints/scaler.pkl'
X_PATH = '../dataset/features_and_target/X_lstm.npy'
Y_PATH = '../dataset/features_and_target/y_lstm.npy'
PRICE_PATH = '../dataset/primary_source/eurusd_1h.csv'  # pour récupérer les prix d'origine
BATCH_SIZE = 64

# === Dataset custom === #
class ForexLSTMDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# === Chargement des données === #
print("Chargement des données...")
X = np.load(X_PATH)
y = np.load(Y_PATH)
prices = np.genfromtxt(PRICE_PATH, delimiter=',', skip_header=1, usecols=4)  # Close price (col 4)

# Mise à l'échelle
scaler = joblib.load(SCALER_PATH)
N, T, F = X.shape
X_flat = X.reshape(-1, F)
X_scaled = scaler.transform(X_flat).reshape(N, T, F)

# Découpe train_and_test/test
test_size = int(0.2 * len(X_scaled))
X_test = X_scaled[-test_size:]
y_test = y[-test_size:]
prices_test = prices[-test_size:]  # les prix associés

# Dataset & DataLoader
test_dataset = ForexLSTMDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# === Chargement du modèle === #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier(input_dim=F).to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# === Prédictions === #
y_pred = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        y_pred.extend(preds)

y_pred = np.array(y_pred)

# === Visualisation avec plotly === #
subset_len = 300  # portion affichée
start = 0
end = start + subset_len

plot_prices = prices_test[start:end]
plot_preds = y_pred[start:end]

# Création du graphe
fig = go.Figure()

# Prix
fig.add_trace(go.Scatter(
    x=list(range(len(plot_prices))),
    y=plot_prices,
    mode='lines',
    name='Prix (close)',
    line=dict(color='black')
))

# Ajout des signaux
for i, signal in enumerate(plot_preds):
    if signal == 1:  # Buy
        fig.add_trace(go.Scatter(
            x=[i], y=[plot_prices[i]],
            mode='markers',
            marker=dict(color='green', symbol='triangle-up', size=12),
            name='Buy',
            showlegend=False
        ))
    elif signal == 2:  # Sell
        fig.add_trace(go.Scatter(
            x=[i], y=[plot_prices[i]],
            mode='markers',
            marker=dict(color='red', symbol='triangle-down', size=12),
            name='Sell',
            showlegend=False
        ))

fig.update_layout(
    title='Prix et signaux du modèle (portion test)',
    xaxis_title='Index',
    yaxis_title='Prix (Close)',
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    height=500
)

fig.show()
