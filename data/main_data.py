# main_data.py
from data.make_dataset import make_interim
from data.make_splits import compute_splits
from data.make_features import make_features
from data.make_processed import make_processed
from tools.logger import setup_logger

logger = setup_logger()

if __name__ == "__main__":

    symbol, timeframe, seed, source = "EURUSD", "H1", 42, "generic"

    logger.info(f"Démarrage du pipeline data {symbol}_{timeframe} avec data model {source} (seed={seed})")

    make_interim(symbol, timeframe, source)
    splits = compute_splits(symbol, timeframe)
    make_features(symbol, timeframe, splits)
    make_processed(symbol, timeframe, seed)

    logger.info("Pipeline complet terminé avec succès")
