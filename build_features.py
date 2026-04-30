# build_features.py
from pathlib import Path
import numpy as np
import pandas as pd

RAW_PRICES_PARQUET = Path("data/raw/prices.parquet")
OUT_DIR            = Path("data/processed")
OUT_FEATS_PARQUET  = OUT_DIR / "features.parquet"

def load_prices() -> pd.DataFrame:
    """Load the (Date, Ticker)-indexed OHLCV panel produced by fetch_prices.py"""
    if RAW_PRICES_PARQUET.exists():
        df = pd.read_parquet(RAW_PRICES_PARQUET)
    elif RAW_PRICES_CSV.exists():
        df = pd.read_csv(RAW_PRICES_CSV, parse_dates=["Date"])
        df = df.set_index(["Date","Ticker"]).sort_index()
    else:
        raise FileNotFoundError(
            f"Missing {RAW_PRICES_PARQUET} (or {RAW_PRICES_CSV}). "
            "Run fetch_prices.py first."
        )
    # sanity checks
    expected = {"Open","High","Low","Close","Volume"}
    have = set(df.columns)
    if not expected.issubset(have):
        raise ValueError(f"Expected columns {expected}, found {have}")
    if not isinstance(df.index, pd.MultiIndex) or df.index.names != ["Date","Ticker"]:
        df = df.reset_index().set_index(["Date","Ticker"]).sort_index()
    return df

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    """Simple RSI-like oscillator in [0,1]."""
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.rolling(n, min_periods=n).mean()
    roll_down = down.rolling(n, min_periods=n).mean()
    rs = roll_up / (roll_down + 1e-9)
    return 1.0 - (1.0 / (1.0 + rs))  # 0..1

def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Input: panel with MultiIndex (Date, Ticker) and columns [Open,High,Low,Close,Volume]
    Output: features DataFrame aligned to the same index.
    """
    g = panel.groupby(level=1)  # group by Ticker
    X = pd.DataFrame(index=panel.index)

    # --- Returns
    X["ret_1"]  = g["Close"].transform(lambda s: s.pct_change(1))
    X["ret_5"]  = g["Close"].transform(lambda s: s.pct_change(5))
    X["ret_20"] = g["Close"].transform(lambda s: s.pct_change(20))

    # --- SMAs via transform (avoid duplicate 'Ticker' index issue)
    sma10 = g["Close"].transform(lambda s: s.rolling(10,  min_periods=10).mean())
    sma20 = g["Close"].transform(lambda s: s.rolling(20,  min_periods=20).mean())
    sma50 = g["Close"].transform(lambda s: s.rolling(50,  min_periods=50).mean())

    close = panel["Close"]
    X["dist_sma10"] = (close - sma10) / (sma10 + 1e-9)
    X["dist_sma20"] = (close - sma20) / (sma20 + 1e-9)
    X["dist_sma50"] = (close - sma50) / (sma50 + 1e-9)

    # --- 52-week percentile position
    roll_min = g["Close"].transform(lambda s: s.rolling(252, min_periods=50).min())
    roll_max = g["Close"].transform(lambda s: s.rolling(252, min_periods=50).max())
    X["pct_52w"] = (close - roll_min) / (roll_max - roll_min + 1e-9)

    # --- Volume z-score (20d) via transform
    vol_mean20 = g["Volume"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    vol_std20  = g["Volume"].transform(lambda s: s.rolling(20, min_periods=20).std())
    X["vol_z20"] = (panel["Volume"] - vol_mean20) / (vol_std20 + 1e-9)

    # --- RSI(14) scaled 0..1 (apply is OK; keeps index)
# --- RSI(14) scaled 0..1
    def rsi(series: pd.Series, n: int = 14) -> pd.Series:
        delta = series.diff()
        up = delta.clip(lower=0.0)
        down = (-delta).clip(lower=0.0)
        roll_up = up.rolling(n, min_periods=n).mean()
        roll_down = down.rolling(n, min_periods=n).mean()
        rs = roll_up / (roll_down + 1e-9)
        return 1.0 - (1.0 / (1.0 + rs))  # 0..1

    # use transform to preserve (Date, Ticker) index
    X["rsi14"] = g["Close"].transform(lambda s: rsi(s, 14))

    # Clean up
    X = X.replace([np.inf, -np.inf], np.nan).dropna()

    return X

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[features] loading prices...")
    panel = load_prices()
    print(f"[features] prices shape: {panel.shape}, index names: {panel.index.names}")

    print("[features] building feature set...")
    X = build_features(panel)
    print(f"[features] features shape: {X.shape}")
    print(X.head(5))

    # Save both parquet and csv for convenience
    X.to_parquet(OUT_FEATS_PARQUET)
    print(f"[features] saved → {OUT_FEATS_PARQUET.resolve()}")

if __name__ == "__main__":
    main()
