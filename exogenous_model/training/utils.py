import os, json, random
import numpy as np
import torch
from pathlib import Path

def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def save_json(obj, path: str | Path):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
