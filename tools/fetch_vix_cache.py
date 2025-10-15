# tools/fetch_vix_cache.py
import pandas_datareader.data as web
import pandas as pd
from pathlib import Path

from tools.logger import setup_logger

logger = setup_logger()

CACHE = Path("../data/external/vix.parquet")
CACHE.parent.mkdir(parents=True, exist_ok=True)

vix = pd.read_parquet(CACHE)
last_date = vix.index.max()

new = web.DataReader("VIXCLS", "fred", start=last_date, end="2030-01-01")
new = new.rename(columns={"VIXCLS": "vix"})
new.index = pd.to_datetime(new.index)
new = new.sort_index()

vix = pd.concat([vix, new[~new.index.isin(vix.index)]])
vix.to_parquet(CACHE)
logger.info(f"Cache VIX mis à jour jusqu'à {vix.index.max().date()}")