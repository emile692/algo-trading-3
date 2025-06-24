import yfinance as yf
import pandas as pd

def download_evz(start='2010-01-01', end='2025-12-31'):
    evz = yf.download('^EVZ', start=start, end=end, interval='1d')
    evz = evz[['Close']].rename(columns={'Close': 'EVZ'})
    evz.index = pd.to_datetime(evz.index)
    return evz


import pandas_datareader.data as web

def download_vix(start='2010-01-01', end='2025-12-31'):
    vix = web.DataReader('VIXCLS', 'fred', start=start, end=end)
    vix = vix.rename(columns={'VIXCLS': 'VIX'})
    return vix