# tools/fetch_vix_cache.py
import pandas_datareader.data as web
import pandas as pd
from pathlib import Path
from tools.logger import setup_logger

logger = setup_logger()

# === Configuration === #
CACHE = Path("../data/external/vix.parquet")
CACHE.parent.mkdir(parents=True, exist_ok=True)

def fetch_vix(start="2010-01-01", end="2030-01-01"):
    """Télécharge le VIX depuis FRED (VIXCLS) entre deux dates."""
    vix = web.DataReader("VIXCLS", "fred", start=start, end=end)
    vix = vix.rename(columns={"VIXCLS": "vix"})
    vix.index = pd.to_datetime(vix.index)
    return vix.sort_index()

def main():
    if CACHE.exists():
        # Charger le cache existant
        vix = pd.read_parquet(CACHE)
        last_date = vix.index.max()
        logger.info(f"Cache existant trouvé jusqu'à {last_date.date()}")

        # Télécharger la suite
        new = fetch_vix(start=last_date + pd.Timedelta(days=1))
        if not new.empty:
            vix = pd.concat([vix, new])
            vix = vix[~vix.index.duplicated(keep="last")]
            vix.to_parquet(CACHE)
            logger.info(f"Cache VIX mis à jour jusqu'à {vix.index.max().date()}")
        else:
            logger.info("Aucune donnée VIX supplémentaire disponible.")
    else:
        # Premier run : créer le cache complet
        logger.info("Aucun cache VIX trouvé. Téléchargement initial en cours...")
        vix = fetch_vix()
        vix.to_parquet(CACHE)
        logger.info(f"Cache VIX créé jusqu'à {vix.index.max().date()}")

if __name__ == "__main__":
    main()
