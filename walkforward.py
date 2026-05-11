# walkforward.py
#
# Rolling walk-forward backtest.
#
# Approach: instead of one 80/20 train/test split, we train on a growing
# window of years and test on the *next* year, then slide the window forward.
# Each fold trains a fresh LightGBM model and evaluates it on the held-out
# year using the same next-day-open execution model as evaluate_backtest.py.
#
# This is the honest way to read out-of-sample performance because it never
# allows a fold to peek at future data, and it averages across many market
# regimes instead of betting everything on which year(s) land in one test split.

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb

from evaluate_backtest import future_stats_open_entry, predict_proba

# ---- inputs ----
PRICES_PATH  = Path("data/raw/prices.parquet")
FEATS_PATH   = Path("data/processed/features.parquet")
LABELS_PATH  = Path("data/processed/labels.parquet")

# ---- config ----
TOP_N             = 5
H                 = 20        # must match build_labels.HORIZON_DAYS
SLIPPAGE_BPS      = 5
MIN_TRAIN_YEARS   = 3         # need at least this many years before first test fold
OUT_DIR           = Path("signals")
OUT_DAILY_PATH    = OUT_DIR / f"walkforward_daily_top{TOP_N}.parquet"
OUT_FOLDS_PATH    = OUT_DIR / f"walkforward_folds_top{TOP_N}.parquet"
OUT_MODELS_DIR    = Path("models/walkforward")


def load_all():
    prices = pd.read_parquet(PRICES_PATH)
    X = pd.read_parquet(FEATS_PATH)
    y = pd.read_parquet(LABELS_PATH)["label"]
    df = X.join(y, how="inner").dropna()
    print(f"[wf] joined panel: {df.shape}")
    return prices, df


def fit_one_fold(X_train: pd.DataFrame, y_train: pd.Series) -> lgb.LGBMClassifier:
    """Train a fresh LightGBM with imbalance handling. Same hyperparams as
    train_and_score.py so results are comparable."""
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    scale_pos_weight = float(neg) / float(max(pos, 1))
    print(f"  [fold] train n={len(y_train)}, pos={pos}, neg={neg}, "
          f"scale_pos_weight={scale_pos_weight:.2f}")

    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_fold(model, X_test, y_test, fwd, top_n=TOP_N, slippage_bps=SLIPPAGE_BPS):
    """Daily precision@N and realized returns on the test fold."""
    test_dates = X_test.index.get_level_values(0).unique().sort_values()
    rows = []
    slip = slippage_bps / 10000.0

    for day in test_dates:
        if day not in X_test.index.get_level_values(0):
            continue
        Xd = X_test.loc[day].dropna()
        if Xd.empty:
            continue
        proba = predict_proba(model, Xd)
        picks = proba.sort_values(ascending=False).head(top_n).index

        # precision@N
        if day in y_test.index.get_level_values(0):
            yd = y_test.loc[day].reindex(picks)
            prec = float(yd.mean()) if not yd.isna().all() else np.nan
            hits = int(yd.fillna(0).sum())
        else:
            prec, hits = np.nan, 0

        # realized returns using next-day open entry
        if day in fwd.index.get_level_values(0):
            fwd_day = fwd.loc[day].reindex(picks)
            ret_max = float((fwd_day["ret_max_H"] - slip).mean())
            ret_end = float((fwd_day["ret_end_H"] - slip).mean())
            dd_min  = float((fwd_day["ret_min_H"]).min())
        else:
            ret_max = ret_end = dd_min = np.nan

        rows.append({
            "date": day,
            "precision_at_N": prec,
            "hits": hits,
            "N": top_n,
            "avg_ret_max_H": ret_max,
            "avg_ret_end_H": ret_end,
            "worst_dd_H": dd_min,
        })

    return pd.DataFrame(rows).set_index("date").sort_index()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    prices, df = load_all()
    print("[wf] precomputing forward returns (open-entry)...")
    fwd = future_stats_open_entry(prices, H=H)

    # Build fold boundaries by calendar year
    all_dates = df.index.get_level_values(0)
    years = sorted(all_dates.year.unique())
    print(f"[wf] years in data: {years}")

    if len(years) < MIN_TRAIN_YEARS + 1:
        print(f"[wf] not enough years (need ≥ {MIN_TRAIN_YEARS+1}, got {len(years)})")
        return

    fold_summaries = []
    daily_all = []

    for i, test_year in enumerate(years[MIN_TRAIN_YEARS:]):
        # train on every year strictly before test_year
        train_mask = all_dates.year < test_year
        test_mask  = all_dates.year == test_year

        X_train = df.loc[train_mask].drop(columns=["label"])
        y_train = df.loc[train_mask, "label"]
        X_test  = df.loc[test_mask].drop(columns=["label"])
        y_test  = df.loc[test_mask, "label"]

        if X_test.empty:
            print(f"[wf] fold {i+1}: test year {test_year} empty — skipping")
            continue

        print(f"\n[wf] === Fold {i+1}: train < {test_year}, test = {test_year} ===")
        model = fit_one_fold(X_train, y_train)
        # save fold model so it can be inspected later
        joblib.dump(model, OUT_MODELS_DIR / f"model_lgbm_test{test_year}.pkl")

        daily = evaluate_fold(model, X_test, y_test, fwd)
        daily["test_year"] = test_year
        daily_all.append(daily)

        # fold-level summary
        summary = {
            "test_year": test_year,
            "n_train_rows": int(len(y_train)),
            "n_test_days": int(len(daily)),
            "precision_at_N_mean": float(daily["precision_at_N"].mean(skipna=True)),
            "avg_ret_max_H_mean":  float(daily["avg_ret_max_H"].mean(skipna=True)),
            "avg_ret_end_H_mean":  float(daily["avg_ret_end_H"].mean(skipna=True)),
            "worst_dd_H_min":      float(daily["worst_dd_H"].min(skipna=True)),
        }
        fold_summaries.append(summary)
        print(f"  -> precision@{TOP_N}: {summary['precision_at_N_mean']:.3f}, "
              f"avg ret_end: {summary['avg_ret_end_H_mean']:.3f}, "
              f"worst dd: {summary['worst_dd_H_min']:.3f}")

    if not daily_all:
        print("[wf] no folds produced results.")
        return

    daily_df = pd.concat(daily_all)
    folds_df = pd.DataFrame(fold_summaries)

    # overall
    overall_prec = daily_df["precision_at_N"].mean(skipna=True)
    overall_ret_end = daily_df["avg_ret_end_H"].mean(skipna=True)
    print("\n=== Walk-forward summary ===")
    print(folds_df.to_string(index=False))
    print()
    print(f"overall daily precision@{TOP_N} (across all folds): {overall_prec:.3f}")
    print(f"overall avg ret_end_H (after slippage):              {overall_ret_end:.3f}")

    daily_df.to_parquet(OUT_DAILY_PATH)
    folds_df.to_parquet(OUT_FOLDS_PATH)
    print(f"\n[wf] saved daily results → {OUT_DAILY_PATH.resolve()}")
    print(f"[wf] saved fold summary → {OUT_FOLDS_PATH.resolve()}")


if __name__ == "__main__":
    main()
