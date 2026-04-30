# fetch_prices.py
import os
from pathlib import Path

import pandas as pd
import yfinance as yf

# ---- edit these if you want ----
WATCHLIST = ["AAPL","MSFT","NVDA","AMZN","META","TSLA","AMD","GOOGL","NFLX","AVGO","SPY"]
START_DATE = "2018-01-01"
OUT_DIR = Path("data/raw")
OUT_PATH = OUT_DIR / "prices.parquet"
# --------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[fetch] downloading {len(WATCHLIST)} tickers from {START_DATE} ...")
    data = yf.download(WATCHLIST, start=START_DATE, auto_adjust=True, progress=True)

    # yfinance returns a column MultiIndex like ('Open','AAPL'), ('Close','MSFT'), etc.
    # Stack to long format indexed by (Date, Ticker).
    panel = (
        data
        .stack(level=1)
        .rename_axis(["Date", "Ticker"])
        .reset_index()
        [["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]]
        .set_index(["Date", "Ticker"])
        .sort_index()
    )

    # quick sanity prints
    print(f"[fetch] shape: {panel.shape}")
    print(panel.head(5))

    panel.to_parquet(OUT_PATH)
    print(f"[fetch] saved → {OUT_PATH.resolve()}")

if __name__ == "__main__":
    main()
