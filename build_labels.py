# build_labels.py
from pathlib import Path
import numpy as np
import pandas as pd

RAW_PRICES_PARQUET = Path("data/raw/prices.parquet")
RAW_PRICES_CSV     = Path("data/raw/prices.csv")        # fallback if you saved CSV
OUT_DIR            = Path("data/processed")
OUT_LABELS_PARQUET = OUT_DIR / "labels.parquet"
OUT_LABELS_CSV     = OUT_DIR / "labels.csv"

# --- you can tweak these ---
HORIZON_DAYS = 20      # look-ahead window H
TARGET_RET   = 0.10    # +10% target
DRAWDOWN_CAP = 0.15    # optional: allow at most -15% during window (set to None to ignore)
# ---------------------------

def load_prices() -> pd.DataFrame:
    """Load the (Date, Ticker)-indexed OHLCV panel produced by fetch_prices.py"""
    if RAW_PRICES_PARQUET.exists():
        df = pd.read_parquet(RAW_PRICES_PARQUET)
    elif RAW_PRICES_CSV.exists():
        df = pd.read_csv(RAW_PRICES_CSV, parse_dates=["Date"])
        df = df.set_index(["Date","Ticker"]).sort_index()
    else:
        raise FileNotFoundError(
            f"Missing {RAW_PRICES_PARQUET} (or {RAW_PRICES_CSV}). Run fetch_prices.py first."
        )
    # sanity checks
    expected = {"Open","High","Low","Close","Volume"}
    have = set(df.columns)
    if not expected.issubset(have):
        raise ValueError(f"Expected columns {expected}, found {have}")
    if not isinstance(df.index, pd.MultiIndex) or df.index.names != ["Date","Ticker"]:
        df = df.reset_index().set_index(["Date","Ticker"]).sort_index()
    return df

def make_labels(panel: pd.DataFrame, H: int, T: float, dd_cap: float | None) -> pd.DataFrame:
    """
    Label = 1 if max future close in next H days >= Close_t * (1+T).
    If dd_cap is set, require min future close in next H days >= Close_t * (1 - dd_cap).
    """
    close = panel["Close"]
    g = close.groupby(level=1)  # by Ticker

    # future max/min over next H days, using shift(-1) so tomorrow is day 1
    fwd_max = g.transform(lambda s: s.shift(-1).rolling(H, min_periods=H).max())
    fwd_min = g.transform(lambda s: s.shift(-1).rolling(H, min_periods=H).min())

    ret_max = (fwd_max / close) - 1.0
    label = (ret_max >= T)

    if dd_cap is not None:
        dd = (fwd_min / close) - 1.0
        label = label & (dd >= -abs(dd_cap))

    y = pd.DataFrame({"label": label.astype(int)}, index=panel.index)

    # drop rows where the future window is incomplete (last H days per ticker)
    y = y.dropna()

    return y

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[labels] loading prices...")
    panel = load_prices()
    print(f"[labels] prices shape: {panel.shape}, index names: {panel.index.names}")

    print(f"[labels] building labels with H={HORIZON_DAYS}, T={TARGET_RET}, dd_cap={DRAWDOWN_CAP} ...")
    y = make_labels(panel, HORIZON_DAYS, TARGET_RET, DRAWDOWN_CAP)
    print(f"[labels] labels shape: {y.shape}")
    # small summary
    pos = int(y["label"].sum())
    total = int(len(y))
    pct = 100.0 * pos / max(total, 1)
    print(f"[labels] positives: {pos} / {total} = {pct:.2f}%")

    # Save both parquet and csv for convenience
    y.to_parquet(OUT_LABELS_PARQUET)
    y.reset_index().to_csv(OUT_LABELS_CSV, index=False)
    print(f"[labels] saved → {OUT_LABELS_PARQUET.resolve()}")
    print(f"[labels] saved → {OUT_LABELS_CSV.resolve()}")

if __name__ == "__main__":
    main()
