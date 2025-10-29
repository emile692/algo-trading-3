import json
import os

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


def predict_exo_model(model, X, device):

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    config_path = os.path.join(project_root, 'config', 'config_old.json')

    with open(config_path) as f:
        config = json.load(f)

    BATCH_SIZE = config['model']["batch_size"]

    model.eval()
    dataset = TensorDataset(torch.tensor(X).float())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    preds = []
    with torch.no_grad():
        for batch in loader:
            x_batch = batch[0].to(device)
            output = model(x_batch)
            pred_class = torch.argmax(output, dim=1).cpu().numpy()
            preds.append(pred_class)
    return np.concatenate(preds)