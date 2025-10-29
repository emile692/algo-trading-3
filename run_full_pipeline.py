import subprocess
from tools.logger import setup_logger

logger = setup_logger()
seed = 42

steps = [
    ("Préparation des données", f"python -m data.main_data"),
    ("Entraînement LSTM", f"python -m exogenous_model.training.train_lstm"),
    ("Évaluation LSTM", f"python -m exogenous_model.eval.evaluate_lstm"),
    ("Entraînement méta XGBoost", f"python -m meta_model.training.train_meta_xgboost"),
    ("Backtest final", f"python -m backtesting.backtest"),
]

for name, cmd in steps:
    logger.info(f"=== Étape : {name} ===")
    subprocess.run(cmd, shell=True, check=True)
