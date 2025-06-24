import json
import random
import numpy as np
import torch
import pandas as pd
import logging
import os
from datetime import datetime

from exogenous_model.dataset.generate_dataset import generate_exogenous_dataset
from exogenous_model.train.train_model import train_and_save_model
from exogenous_model.eval.evaluate_model import evaluate_model
from meta_model.dataset.generate_meta_dataset import generate_meta_dataset
from meta_model.train_and_test.train_test_meta_xgboost import train_and_test_meta_xgboost
from strategy.utils import analyse_capture_ratio


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_logger():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"run_multi_seed_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def run_pipeline_for_seed(seed: int, logger) -> dict:
    logger.info(f"🔁 Lancement du run avec seed {seed}")
    set_seed(seed)

    try:
        logger.info("📦 Génération du dataset exogène...")
        generate_exogenous_dataset(seed=seed)

        logger.info("🏋️ Entraînement du modèle principal...")
        model_path, scaler_path = train_and_save_model(seed=seed)

        logger.info("🧪 Évaluation du modèle principal...")
        metrics = evaluate_model(model_path, scaler_path)
        metrics['seed'] = seed

        logger.info("🔧 Génération du dataset méta...")
        generate_meta_dataset(seed=seed)

        logger.info("🎯 Entraînement et évaluation du méta-modèle...")
        train_and_test_meta_xgboost(seed=seed)

        logger.info("📈 Analyse du capture ratio...")
        analyse_capture_ratio(seed=seed)

        logger.info(f"✅ Run terminé pour seed {seed}")
        return metrics

    except Exception as e:
        logger.exception(f"❌ Erreur durant le run pour seed {seed} : {e}")
        return {"seed": seed, "error": str(e)}


def run_multi_seed():
    logger = setup_logger()

    with open("config.json") as f:
        config = json.load(f)

    seeds = config['general']['seeds']
    results = []

    logger.info(f"🚀 Démarrage de l'exécution multi-seeds : {seeds}")

    for seed in seeds:
        metrics = run_pipeline_for_seed(seed, logger)
        results.append(metrics)

    df = pd.DataFrame(results)

    logger.info("\n📊 Moyenne des scores sur toutes les seeds :")
    logger.info(df.mean(numeric_only=True).round(4))

    logger.info("\n📉 Écart-type des scores :")
    logger.info(df.std(numeric_only=True).round(4))

    df.to_csv("results/results_multi_seed.csv", index=False)
    logger.info("✅ Résultats enregistrés dans results/results_multi_seed.csv")


if __name__ == "__main__":
    run_multi_seed()
