# evaluate_backtest.py
#
# Backtest using **next-day open** as the entry price.
#
# Why: subtracting flat slippage from same-day close overstates how easy it is
# to capture the move. In real life you see the model's pick at end of day D,
# place the trade, and get filled at the open of day D+1. So the entry price
# is the next-day open, and the realized return is measured against that.
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# Inputs
PRICES_PATH  = Path("data/raw/prices.parquet")
FEATS_PATH   = Path("data/processed/features.parquet")
LABELS_PATH  = Path("data/processed/labels.parquet")
MODEL_PATH   = Path("models/model_lgbm.pkl")

TOP_N        = 5          # evaluate precision@N
H            = 20         # lookahead horizon (must match build_labels)
SLIPPAGE_BPS = 5          # 5 bps round-trip; lower than the 10 bps flat charge
                          # the old script used, because the open-entry already
                          # absorbs the worst of the same-day price-discovery cost.


def load_all():
    prices = pd.read_parquet(PRICES_PATH)
    X = pd.read_parquet(FEATS_PATH)
    y = pd.read_parquet(LABELS_PATH)["label"]
    model = joblib.load(MODEL_PATH)
    # align feature/label panel
    df = X.join(y, how="inner").dropna()
    # split by dates like train_and_score.py
    dates = df.index.get_level_values(0).unique().sort_values()
    cut = int(0.80 * len(dates))
    test_dates = dates[cut:]
    return prices, X, y, model, test_dates


def future_stats_open_entry(prices: pd.DataFrame, H: int) -> pd.DataFrame:
    """Realized H-day return statistics assuming entry at NEXT-DAY open.

    For each (Date=D, Ticker), the returned row describes a trade that:
      * is signaled at close of day D
      * is entered at open of day D+1
      * is exited on close of day D+H

    ret_max_H  : (max close over days D+1..D+H) / open(D+1) - 1
    ret_min_H  : (min close over days D+1..D+H) / open(D+1) - 1
    ret_end_H  : close(D+H) / open(D+1) - 1
    """
    close = prices["Close"]
    open_ = prices["Open"]
    g_close = close.groupby(level=1)
    g_open  = open_.groupby(level=1)

    # Entry = next-day open. shift(-1) on each ticker so the value at D is open(D+1).
    entry = g_open.transform(lambda s: s.shift(-1))

    # Forward close window from D+1 through D+H, computed once and aligned at D.
    fwd_max = g_close.transform(lambda s: s.shift(-1).rolling(H, min_periods=H).max())
    fwd_min = g_close.transform(lambda s: s.shift(-1).rolling(H, min_periods=H).min())
    fwd_end = g_close.transform(lambda s: s.shift(-H))

    ret_max = (fwd_max / entry) - 1.0
    ret_min = (fwd_min / entry) - 1.0
    ret_end = (fwd_end / entry) - 1.0

    return pd.DataFrame({
        "ret_max_H": ret_max,
        "ret_min_H": ret_min,
        "ret_end_H": ret_end,
    })


def predict_proba(model, X_day: pd.DataFrame) -> pd.Series:
    return pd.Series(model.predict_proba(X_day)[:, 1], index=X_day.index)


def evaluate_day(model, X, y, fwd, day, top_n=TOP_N, slippage_bps=SLIPPAGE_BPS):
    # features/labels on this day
    if day not in X.index.get_level_values(0):
        return None
    Xd = X.loc[day].dropna()
    if Xd.empty:
        return None
    proba = predict_proba(model, Xd)
    ranked = proba.sort_values(ascending=False)
    picks = ranked.head(top_n).index

    # label precision@N
    if day in y.index.get_level_values(0):
        yd = y.loc[day].reindex(picks)
        prec = float(yd.mean())
        hits = int(yd.sum())
    else:
        prec, hits = np.nan, 0

    # realized stats using next-day-open entry
    fwd_day = fwd.loc[day].reindex(picks)
    # 5 bps slippage applied to entry (1 way; one-way fill on the open)
    slip = slippage_bps / 10000.0
    ret_max = float((fwd_day["ret_max_H"] - slip).mean())
    ret_end = float((fwd_day["ret_end_H"] - slip).mean())
    dd_min  = float((fwd_day["ret_min_H"]).min())  # worst drawdown among picks

    return {
        "date": day,
        "precision_at_N": prec,
        "hits": hits,
        "N": int(top_n),
        "avg_ret_max_H": ret_max,
        "avg_ret_end_H": ret_end,
        "worst_dd_H": dd_min,
    }


def main():
    print("[eval] loading data/model...")
    prices, X, y, model, test_dates = load_all()
    print(f"[eval] test dates: {test_dates.min().date()} → {test_dates.max().date()} "
          f"(n={len(test_dates)})")

    print("[eval] precomputing forward returns (next-day open entry)...")
    fwd = future_stats_open_entry(prices, H=H)

    rows = []
    for day in test_dates:
        row = evaluate_day(model, X, y, fwd, day)
        if row:
            rows.append(row)

    if not rows:
        print("[eval] no rows evaluated. Check your splits.")
        return

    res = pd.DataFrame(rows).set_index("date").sort_index()

    # summary
    prec_mean = res["precision_at_N"].mean()
    ret_max_mean = res["avg_ret_max_H"].mean()
    ret_end_mean = res["avg_ret_end_H"].mean()
    worst_dd = res["worst_dd_H"].min()
    print("\n=== Backtest summary (test period, next-day open entry) ===")
    print(f"precision@{TOP_N} (mean over days): {prec_mean:.3f}")
    print(f"avg of daily mean ret_max_H (after {SLIPPAGE_BPS}bps slip): {ret_max_mean:.3f}")
    print(f"avg of daily mean ret_end_H (after {SLIPPAGE_BPS}bps slip): {ret_end_mean:.3f}")
    print(f"worst intra-horizon drawdown among picks: {worst_dd:.3f}")

    # save detailed results
    out_dir = Path("signals")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"backtest_test_period_top{TOP_N}.parquet"
    res.to_parquet(out_path)
    print(f"[eval] daily results saved → {out_path.resolve()}")


if __name__ == "__main__":
    main()
