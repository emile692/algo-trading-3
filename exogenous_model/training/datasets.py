import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    """
    Dataset séquentiel *lazy* (pas de .npy intermédiaires).
    Assumptions:
      - df contient 'label'
      - les features = df.columns \ {'label','time','split'} (time optionnelle si index)
      - on ne veut AUCUN NaN dans label -> filtré avant
    """
    def __init__(self, df: pd.DataFrame, sequence_length: int, feature_cols=None, time_col="time"):
        df = df.copy()
        # garder time si présent
        if time_col in df.columns:
            self.times = df[time_col].to_numpy()
        elif isinstance(df.index, pd.DatetimeIndex):
            self.times = df.index.to_numpy()
        else:
            self.times = np.arange(len(df))

        # drop NaN label (ex. padding du triple-barrier)
        df = df[df["label"].notna()].reset_index(drop=True)

        # features auto si non fournies
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c not in ("label", "time", "split")]
        self.feature_cols = feature_cols

        self.X = df[feature_cols].to_numpy(dtype=np.float32)
        self.y = df["label"].astype(np.int64).to_numpy()
        self.sequence_length = int(sequence_length)

        # nombre d'échantillons séquentiels disponibles
        self.n = max(0, len(self.y) - self.sequence_length + 1)

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int):
        # fenêtre [idx : idx+seq_len]
        x_seq = self.X[idx: idx + self.sequence_length]
        y_t   = self.y[idx + self.sequence_length - 1]
        return torch.from_numpy(x_seq), torch.tensor(y_t, dtype=torch.long)
