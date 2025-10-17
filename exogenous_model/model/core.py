# exogenous_model/model/core.py
from __future__ import annotations
import math
import torch
import torch.nn as nn


__all__ = ["LSTMClassifier"]


def _init_lstm_forget_bias(lstm: nn.LSTM, bias_value: float = 1.0) -> None:
    # Ouvre les biais par défaut (meilleure mémoire à court terme au début)
    for layer in range(lstm.num_layers):
        for direction in range(2 if lstm.bidirectional else 1):
            b = getattr(lstm, f"bias_ih_l{layer}" + (f"_{direction}" if lstm.bidirectional else ""))
            # biais = [i, f, g, o] gates dans cet ordre (PyTorch)
            hidden_size = lstm.hidden_size
            start, end = hidden_size, 2 * hidden_size  # gate "forget"
            b.data[start:end].fill_(bias_value)


class LSTMClassifier(nn.Module):
    """
    Classif séquentielle simple:
      - LSTM (batch_first=True)
      - On récupère la représentation du dernier pas (concat directions si bidirectionnel)
      - Tête MLP légère -> logits
    """
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        num_classes: int = 3,
        fc_hidden: int | None = None,
        layernorm: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        proj_in = hidden_size * self.num_directions
        head = []
        if layernorm:
            head.append(nn.LayerNorm(proj_in))
        if fc_hidden is not None and fc_hidden > 0:
            head += [nn.Linear(proj_in, fc_hidden), nn.ReLU(), nn.Dropout(dropout)]
            proj_in = fc_hidden
        head.append(nn.Linear(proj_in, num_classes))
        self.head = nn.Sequential(*head)

        self.reset_parameters()

    def reset_parameters(self):
        # init orthogonale pour LSTM, xavier pour les linear
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
        _init_lstm_forget_bias(self.lstm, 1.0)

        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_dim)
        return: logits (batch, num_classes)
        """
        # out: (batch, seq_len, hidden*dirs)
        out, _ = self.lstm(x)
        last = out[:, -1, :]  # dernier pas temporel
        logits = self.head(last)
        return logits

    @staticmethod
    def from_config(input_dim: int, cfg: dict) -> "LSTMClassifier":
        mcfg = cfg.get("model", {})
        # Valeurs par défaut si absentes
        return LSTMClassifier(
            input_dim=input_dim,
            hidden_size=int(mcfg.get("hidden_size", 128)),
            num_layers=int(mcfg.get("num_layers", 2)),
            dropout=float(mcfg.get("dropout", 0.2)),
            bidirectional=bool(mcfg.get("bidirectional", False)),
            num_classes=int(mcfg.get("num_classes", 3)),
            fc_hidden=mcfg.get("fc_hidden", None),
            layernorm=bool(mcfg.get("layernorm", False)),
        )

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
