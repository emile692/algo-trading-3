import torch.nn as nn

# === MODÈLE LSTM === #
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 3)  # 3 classes: BUY, SELL, HOLD

    def forward(self, x):
        out, (h_n, _) = self.lstm(x)  # h_n: (num_layers, batch, hidden_dim)
        return self.fc(h_n[-1])  # dernière couche du dernier pas de temps