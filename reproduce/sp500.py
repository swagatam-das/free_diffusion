"""Real-data spectrum used in the paper: S&P 500 daily closing prices, 2013-02-08 .. 2018-02-07
(Plotly's public copy of the Kaggle 'S&P 500 stock data' set).  The first N=100 tickers
(alphabetically, fixed a priori) among the 470 with complete coverage; standardised log-returns."""
import os, urllib.request, numpy as np, pandas as pd

URL = "https://raw.githubusercontent.com/plotly/datasets/master/all_stocks_5yr.csv"


def load_spectrum(cache_dir="../data", csv=None, N=100):
    if csv is None:
        os.makedirs(cache_dir, exist_ok=True)
        csv = os.path.join(cache_dir, "all_stocks_5yr.csv")
        if not os.path.exists(csv):
            urllib.request.urlretrieve(URL, csv)
    df = pd.read_csv(csv)
    piv = df.pivot(index="date", columns="Name", values="close").sort_index()
    full = piv.dropna(axis=1, how="any")                       # complete coverage only, no imputation
    logret = np.log(full / full.shift(1)).dropna()
    assets = sorted(logret.columns)[:N]
    R = logret[assets].values
    T = R.shape[0]
    R = (R - R.mean(axis=0)) / R.std(axis=0)
    eigs = np.linalg.eigvalsh(np.corrcoef(R, rowvar=False))
    return eigs, N, T, full.shape[1]
