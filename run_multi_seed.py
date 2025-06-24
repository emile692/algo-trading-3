import json
import random
import numpy as np
import torch
import pandas as pd
import logging
import os
import colorlog
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
    # Créer un répertoire pour les logs s'il n'existe pas
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)

    # Ajouter un horodatage au nom du fichier log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"run_multi_seed_{timestamp}.log")

    logger = logging.getLogger("multi_seed_logger")
    logger.setLevel(logging.DEBUG)  # Capturer les messages au niveau DEBUG

    # Text formatter (pour le fichier)
    file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)  # Capturer les messages DEBUG dans le fichier

    # Colored formatter (pour la console)
    color_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'yellow',
            'WARNING': 'green',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }
    )
    console_handler = colorlog.StreamHandler()
    console_handler.setFormatter(color_formatter)
    console_handler.setLevel(logging.DEBUG)  # Capturer les messages DEBUG dans la console

    # Éviter les handlers en doublon
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger




def run_multi_seed():
    # Logger
    logger = setup_logger()

    # Configuration
    with open('config.json') as f:
        config = json.load(f)

    logger.info(f"Génération de l'exogenous dataset")
    generate_exogenous_dataset(logger)

    seeds = config['general']['seeds']

    logger.info(f"Démarrage de l'exécution multi-seeds : {seeds}\n")

    results = []

    for seed in seeds:
        logger.info(f"--- Début du run avec seed {seed} ---")

        try:
            set_seed(seed)

            # Entraînement
            logger.info("Entraînement du modèle principal")
            model_path, scaler_path = train_and_save_model(seed=seed, logger=logger)

            # Évaluation
            logger.info("Évaluation du modèle principal")
            metrics = evaluate_model(model_path, logger=logger)
            metrics['seed'] = seed
            results.append(metrics)

            # Dataset méta
            logger.info("Génération du dataset méta")
            generate_meta_dataset(seed=seed, logger=logger)

            # Entraînement du modèle méta
            logger.info("Entraînement et évaluation du modèle XGBoost méta")
            meta_metrics = train_and_test_meta_xgboost(seed=seed, logger=logger)

            # Combine les résultats LSTM + XGBoost
            metrics.update(meta_metrics)

            # Analyse stratégie
            logger.info("Analyse du capture ratio")
            analyse_capture_ratio(seed=seed)

            logger.info(f"--- Fin du run avec seed {seed} ---\n")

        except Exception as e:
            logger.exception(f"Erreur lors du traitement du seed {seed}: {e}")

    # Résultats finaux
    df = pd.DataFrame(results)
    logger.info("Moyenne des scores sur tous les seeds :")
    logger.info(df.mean(numeric_only=True).round(4))

    logger.info("Écart-type des scores :")
    logger.info(df.std(numeric_only=True).round(4))

    output_path = "meta_model/results/results_multi_seed.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Résultats enregistrés dans {output_path}")


if __name__ == "__main__":
    run_multi_seed()
