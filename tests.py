# tests.py
# Validation tests for the stock ML pipeline. Run after build_features.py and
# build_labels.py have produced their outputs.
#
# Usage:
#     python tests.py
#
# Each test prints PASS/FAIL and a short message. Exits non-zero if any test fails.
from pathlib import Path
import sys
import numpy as np
import pandas as pd

PRICES_PATH = Path("data/raw/prices.parquet")
FEATS_PATH  = Path("data/processed/features.parquet")
LABELS_PATH = Path("data/processed/labels.parquet")

EXPECTED_OHLCV = {"Open", "High", "Low", "Close", "Volume"}
EXPECTED_FEATURES = {"ret_1","ret_5","ret_20","dist_sma10","dist_sma20","dist_sma50",
                     "pct_52w","vol_z20","rsi14"}

# Pipeline parameters that must stay aligned with build_labels.py
HORIZON_DAYS = 20
TARGET_RET   = 0.10
DRAWDOWN_CAP = 0.15

# ----- test runner -----
results = []
def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    results.append((name, condition, detail))
    msg = f"[{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)


# ----- loaders that work with parquet or CSV fallback -----
def _load(parquet_path: Path):
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except Exception:
            pass
    csv_path = parquet_path.with_suffix(".csv")
    if csv_path.exists():
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        if {"Date","Ticker"}.issubset(df.columns):
            df = df.set_index(["Date","Ticker"]).sort_index()
        return df
    return None


# ----- tests -----
def test_prices_schema(prices):
    if prices is None:
        check("prices file exists", False, f"missing {PRICES_PATH}")
        return False
    check("prices file exists", True)
    check("prices has OHLCV columns", EXPECTED_OHLCV.issubset(prices.columns),
          f"got {set(prices.columns)}")
    check("prices indexed by (Date, Ticker)",
          isinstance(prices.index, pd.MultiIndex) and prices.index.names == ["Date","Ticker"],
          f"got {prices.index.names}")
    check("prices has no negative close", (prices["Close"] > 0).all(),
          f"min close = {prices['Close'].min()}")
    return True


def test_features_schema(features):
    if features is None:
        check("features file exists", False, f"missing {FEATS_PATH}")
        return False
    check("features file exists", True)
    check("features has all 9 feature columns",
          EXPECTED_FEATURES.issubset(features.columns),
          f"missing: {EXPECTED_FEATURES - set(features.columns)}")
    check("features indexed by (Date, Ticker)",
          isinstance(features.index, pd.MultiIndex) and features.index.names == ["Date","Ticker"],
          f"got {features.index.names}")
    return True


def test_features_no_nan_or_inf(features):
    if features is None: return
    feature_only = features[list(EXPECTED_FEATURES & set(features.columns))]
    n_nan = int(feature_only.isna().sum().sum())
    n_inf = int(np.isinf(feature_only.to_numpy()).sum())
    check("features have no NaN", n_nan == 0, f"{n_nan} NaN values")
    check("features have no inf", n_inf == 0, f"{n_inf} inf values")


def test_features_one_row_per_date_ticker(features):
    if features is None: return
    n_dup = int(features.index.duplicated().sum())
    check("no duplicate (Date, Ticker) in features", n_dup == 0,
          f"{n_dup} duplicates")


def test_pct_52w_in_range(features):
    if features is None or "pct_52w" not in features.columns: return
    p = features["pct_52w"].dropna()
    check("pct_52w stays within [0, 1]", ((p >= -1e-9) & (p <= 1+1e-9)).all(),
          f"min={p.min():.4f}, max={p.max():.4f}")


def test_rsi_in_range(features):
    if features is None or "rsi14" not in features.columns: return
    r = features["rsi14"].dropna()
    check("rsi14 stays within [0, 1]", ((r >= -1e-9) & (r <= 1+1e-9)).all(),
          f"min={r.min():.4f}, max={r.max():.4f}")


def test_labels_schema(labels):
    if labels is None:
        check("labels file exists", False, f"missing {LABELS_PATH}")
        return False
    check("labels file exists", True)
    check("labels indexed by (Date, Ticker)",
          isinstance(labels.index, pd.MultiIndex) and labels.index.names == ["Date","Ticker"],
          f"got {labels.index.names}")
    check("labels are 0 or 1 only",
          set(labels["label"].unique()).issubset({0, 1}),
          f"got {set(labels['label'].unique())}")
    return True


def test_label_positive_rate_sane(labels):
    if labels is None: return
    pr = float(labels["label"].mean())
    # +10%-in-20-days with 15% drawdown cap on tech-heavy universe should land in 5%-50%
    check("label positive rate looks sane (5%–50%)", 0.05 < pr < 0.50,
          f"positive rate = {pr:.3f}")


def test_labels_drop_recent_horizon(labels, prices):
    """Last HORIZON_DAYS rows per ticker should be dropped (unknown future)."""
    if labels is None or prices is None: return
    last_label_date = labels.index.get_level_values(0).max()
    last_price_date = prices.index.get_level_values(0).max()
    gap_business = len(pd.bdate_range(last_label_date, last_price_date)) - 1
    check(f"last labeled date is at least {HORIZON_DAYS} business days before last price date",
          gap_business >= HORIZON_DAYS,
          f"gap = {gap_business} business days")


def test_label_correctness_on_sample(prices, labels):
    """Recompute labels on a few random rows and verify they match build_labels.py."""
    if prices is None or labels is None: return
    rng = np.random.default_rng(42)
    sample = labels.sample(min(50, len(labels)), random_state=rng)
    mismatches = 0
    skipped = 0
    for (date, ticker), row in sample.iterrows():
        try:
            ticker_close = prices.xs(ticker, level=1)["Close"].sort_index()
        except KeyError:
            skipped += 1; continue
        if date not in ticker_close.index:
            skipped += 1; continue
        i = ticker_close.index.get_loc(date)
        # need next H bars
        if i + HORIZON_DAYS >= len(ticker_close):
            skipped += 1; continue
        base = ticker_close.iloc[i]
        window = ticker_close.iloc[i+1:i+1+HORIZON_DAYS]
        ret_max = window.max() / base - 1.0
        ret_min = window.min() / base - 1.0
        expected = int((ret_max >= TARGET_RET) and (ret_min >= -DRAWDOWN_CAP))
        if expected != int(row["label"]):
            mismatches += 1
    n_checked = len(sample) - skipped
    check(f"random-sample label recomputation matches saved labels ({n_checked} rows)",
          mismatches == 0,
          f"{mismatches} mismatches" if mismatches else "")


def test_features_labels_alignable(features, labels):
    if features is None or labels is None: return
    joined = features.join(labels[["label"]], how="inner")
    check("features and labels share at least 80% of feature dates",
          len(joined) >= 0.8 * len(features),
          f"{len(joined)} joined rows vs {len(features)} feature rows")


# ----- main -----
def main():
    print("Loading inputs...")
    prices   = _load(PRICES_PATH)
    features = _load(FEATS_PATH)
    labels   = _load(LABELS_PATH)
    print()
    print("Running tests...")
    print()

    test_prices_schema(prices)
    test_features_schema(features)
    test_features_no_nan_or_inf(features)
    test_features_one_row_per_date_ticker(features)
    test_pct_52w_in_range(features)
    test_rsi_in_range(features)
    test_labels_schema(labels)
    test_label_positive_rate_sane(labels)
    test_labels_drop_recent_horizon(labels, prices)
    test_label_correctness_on_sample(prices, labels)
    test_features_labels_alignable(features, labels)

    n_pass = sum(1 for _, ok, _ in results if ok)
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print()
    print(f"=== {n_pass} passed, {n_fail} failed ===")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
