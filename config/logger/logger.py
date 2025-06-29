import logging
import os
import colorlog
from datetime import datetime


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