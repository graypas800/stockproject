# fetch_prices.py
import os
from pathlib import Path

import pandas as pd
import yfinance as yf

# ----------- Universe -----------
# The small set was used for the draft. The expanded set is the new default —
# ~55 large- and mid-cap US names across sectors, plus SPY as benchmark.
# Set UNIVERSE = "small" to fall back to the original 11-ticker watchlist.
UNIVERSE = "large"   # "small" or "large"

SMALL_WATCHLIST = [
    "AAPL","MSFT","NVDA","AMZN","META","TSLA","AMD","GOOGL","NFLX","AVGO","SPY"
]

LARGE_WATCHLIST = [
    # tech / semis / software
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AMD","NFLX","AVGO",
    "ORCL","CRM","ADBE","INTC","CSCO","QCOM","IBM","TXN","AMAT","MU",
    "ADI","INTU","NOW","PANW",
    # financials
    "JPM","BAC","WFC","GS","MS","BLK","V","MA","AXP","C",
    # consumer / retail / travel
    "WMT","COST","HD","NKE","MCD","SBUX","TGT","LULU","BKNG","DIS",
    # healthcare
    "UNH","JNJ","LLY","PFE","ABBV","MRK","TMO","ABT",
    # industrials / energy / staples
    "CAT","BA","XOM","CVX","HON","RTX","DE","KO","PEP","T",
    # benchmark
    "SPY",
]

WATCHLIST = LARGE_WATCHLIST if UNIVERSE == "large" else SMALL_WATCHLIST

START_DATE = "2018-01-01"
OUT_DIR = Path("data/raw")
OUT_PATH = OUT_DIR / "prices.parquet"
# --------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[fetch] universe='{UNIVERSE}' ({len(WATCHLIST)} tickers)")
    print(f"[fetch] downloading from {START_DATE} ...")
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

    # drop rows with NaN Close — happens for tickers that IPO'd partway through
    # the window. Keeps the panel clean for downstream scripts.
    before = len(panel)
    panel = panel.dropna(subset=["Close"])
    after = len(panel)
    if before != after:
        print(f"[fetch] dropped {before-after} rows with missing Close (likely pre-IPO)")

    # quick sanity prints
    n_tickers = panel.index.get_level_values("Ticker").nunique()
    print(f"[fetch] shape: {panel.shape}, tickers: {n_tickers}")
    print(panel.head(5))

    panel.to_parquet(OUT_PATH)
    print(f"[fetch] saved → {OUT_PATH.resolve()}")

if __name__ == "__main__":
    main()
