# build_features.py
from pathlib import Path
import numpy as np
import pandas as pd

RAW_PRICES_PARQUET = Path("data/raw/prices.parquet")
RAW_PRICES_CSV     = Path("data/raw/prices.csv")        # fallback
OUT_DIR            = Path("data/processed")
OUT_FEATS_PARQUET  = OUT_DIR / "features.parquet"
OUT_FEATS_CSV      = OUT_DIR / "features.csv"

# Feature columns produced by build_features (kept in one place so other
# scripts — tests, EDA, training — can import this list).
FEATURE_COLS = [
    # original features
    "ret_1", "ret_5", "ret_20",
    "dist_sma10", "dist_sma20", "dist_sma50",
    "pct_52w", "vol_z20", "rsi14",
    # new — added for the mean-reversion finding from EDA
    "dist_sma200",        # distance from the 200-day SMA (long trend context)
    "down_streak",        # number of consecutive negative-return days
    "vol_regime",         # 20-day realized vol z-scored over 252 days
    "mom_rank_xs",        # cross-sectional rank of ret_20 within each date (0..1)
]


def load_prices() -> pd.DataFrame:
    """Load the (Date, Ticker)-indexed OHLCV panel produced by fetch_prices.py."""
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
    up   = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up   = up.rolling(n, min_periods=n).mean()
    roll_down = down.rolling(n, min_periods=n).mean()
    rs = roll_up / (roll_down + 1e-9)
    return 1.0 - (1.0 / (1.0 + rs))   # 0..1


def _down_streak_per_ticker(ret1: pd.Series) -> pd.Series:
    """For a single ticker, count consecutive down (negative ret_1) days.
    Resets to 0 on any non-negative day. Today counts if today is down.
    """
    neg = (ret1 < 0).astype(int)
    # Each maximal run of identical values gets its own group id.
    grp = (neg != neg.shift()).cumsum()
    # Within each "down" run, cumcount gives 0,1,2,... -> add 1 for "today is the Nth down day".
    # Multiply by neg so non-down days are 0.
    streak = neg.groupby(grp).cumcount().add(1) * neg
    return streak


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Input : OHLCV panel, MultiIndex (Date, Ticker).
    Output: feature DataFrame on the same index, columns == FEATURE_COLS.
    """
    g = panel.groupby(level=1)  # group by Ticker
    X = pd.DataFrame(index=panel.index)

    # -------- Returns --------
    X["ret_1"]  = g["Close"].transform(lambda s: s.pct_change(1))
    X["ret_5"]  = g["Close"].transform(lambda s: s.pct_change(5))
    X["ret_20"] = g["Close"].transform(lambda s: s.pct_change(20))

    # -------- SMA distances --------
    close = panel["Close"]
    sma10  = g["Close"].transform(lambda s: s.rolling(10,  min_periods=10 ).mean())
    sma20  = g["Close"].transform(lambda s: s.rolling(20,  min_periods=20 ).mean())
    sma50  = g["Close"].transform(lambda s: s.rolling(50,  min_periods=50 ).mean())
    # 200-day SMA: use min_periods=100 so it warms up sooner. The first ~100
    # rows per ticker still drop later via dropna(), but later rows fill in.
    sma200 = g["Close"].transform(lambda s: s.rolling(200, min_periods=100).mean())

    X["dist_sma10"]  = (close - sma10)  / (sma10  + 1e-9)
    X["dist_sma20"]  = (close - sma20)  / (sma20  + 1e-9)
    X["dist_sma50"]  = (close - sma50)  / (sma50  + 1e-9)
    X["dist_sma200"] = (close - sma200) / (sma200 + 1e-9)

    # -------- 52-week percentile position --------
    roll_min = g["Close"].transform(lambda s: s.rolling(252, min_periods=50).min())
    roll_max = g["Close"].transform(lambda s: s.rolling(252, min_periods=50).max())
    X["pct_52w"] = (close - roll_min) / (roll_max - roll_min + 1e-9)

    # -------- Volume z-score (20d) --------
    vol_mean20 = g["Volume"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    vol_std20  = g["Volume"].transform(lambda s: s.rolling(20, min_periods=20).std())
    X["vol_z20"] = (panel["Volume"] - vol_mean20) / (vol_std20 + 1e-9)

    # -------- RSI(14) --------
    X["rsi14"] = g["Close"].transform(lambda s: rsi(s, 14))

    # -------- NEW: down-day streak --------
    # Use the ret_1 we already computed; reset per ticker.
    X["down_streak"] = X.groupby(level=1)["ret_1"].transform(_down_streak_per_ticker)

    # -------- NEW: volatility regime --------
    # 20-day realized volatility of daily returns, then z-scored against its own
    # 252-day rolling distribution. Positive = unusually choppy. Negative = unusually calm.
    realized_vol_20 = X.groupby(level=1)["ret_1"].transform(
        lambda s: s.rolling(20, min_periods=20).std()
    )
    vol_mean_252 = realized_vol_20.groupby(level=1).transform(
        lambda s: s.rolling(252, min_periods=60).mean()
    )
    vol_std_252 = realized_vol_20.groupby(level=1).transform(
        lambda s: s.rolling(252, min_periods=60).std()
    )
    X["vol_regime"] = (realized_vol_20 - vol_mean_252) / (vol_std_252 + 1e-9)

    # -------- NEW: cross-sectional momentum rank --------
    # On each date, rank every ticker by ret_20. pct=True gives 0..1.
    # This is the key cross-sectional feature: "where does this stock rank
    # against the rest of the universe TODAY". Index level 0 == Date.
    X["mom_rank_xs"] = X["ret_20"].groupby(level=0).rank(pct=True)

    # -------- Clean up --------
    # Infinities can arise from zero-division corner cases; convert to NaN, drop.
    X = X.replace([np.inf, -np.inf], np.nan).dropna()

    # Ensure column order matches FEATURE_COLS so downstream code is stable.
    X = X[FEATURE_COLS]

    return X


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[features] loading prices...")
    panel = load_prices()
    print(f"[features] prices shape: {panel.shape}, "
          f"index names: {panel.index.names}, "
          f"tickers: {panel.index.get_level_values(1).nunique()}")

    print(f"[features] building feature set with {len(FEATURE_COLS)} columns...")
    X = build_features(panel)
    print(f"[features] features shape: {X.shape}")
    print(X.head(5))

    # Save both parquet and csv for convenience
    X.to_parquet(OUT_FEATS_PARQUET)
    X.reset_index().to_csv(OUT_FEATS_CSV, index=False)
    print(f"[features] saved → {OUT_FEATS_PARQUET.resolve()}")
    print(f"[features] saved → {OUT_FEATS_CSV.resolve()}")

if __name__ == "__main__":
    main()
