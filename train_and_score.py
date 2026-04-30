# train_and_score.py
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import lightgbm as lgb

FEATS_PATH   = Path("data/processed/features.parquet")
LABELS_PATH  = Path("data/processed/labels.parquet")
MODEL_PATH   = Path("models/model_lgbm.pkl")
SIGNALS_DIR  = Path("signals")

TOP_N        = 5   # how many picks you want
MIN_DATES_PCT_TRAIN = 0.80  # first 80% dates for train, last 20% for simple test

def load_xy():
    if not FEATS_PATH.exists():
        raise FileNotFoundError("Missing features. Run build_features.py first.")
    if not LABELS_PATH.exists():
        raise FileNotFoundError("Missing labels. Run build_labels.py first.")

    X = pd.read_parquet(FEATS_PATH)
    y = pd.read_parquet(LABELS_PATH)["label"]

    df = X.join(y, how="inner").dropna()
    print(f"[train] joined shape: {df.shape}")

    y = df["label"]
    X = df.drop(columns=["label"])

    # time split by the Date level (index is MultiIndex: Date, Ticker)
    dates = X.index.get_level_values(0).unique().sort_values()
    cut = int(MIN_DATES_PCT_TRAIN * len(dates))
    train_dates = dates[:cut]
    test_dates  = dates[cut:]

    X_train = X.loc[X.index.get_level_values(0).isin(train_dates)]
    y_train = y.loc[y.index.get_level_values(0).isin(train_dates)]
    X_test  = X.loc[X.index.get_level_values(0).isin(test_dates)]
    y_test  = y.loc[y.index.get_level_values(0).isin(test_dates)]

    print(f"[train] train: {X_train.shape}, test: {X_test.shape}")
    return X_train, y_train, X_test, y_test, X

def train_or_load(X_train, y_train):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # imbalance handling
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = float(neg) / float(max(pos, 1))
    print(f"[train] positives={int(pos)}, negatives={int(neg)}, scale_pos_weight={scale_pos_weight:.2f}")

    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    print(f"[train] model saved → {MODEL_PATH.resolve()}")
    return model

def evaluate(model, X_test, y_test, K=TOP_N, horizon_days=20):
    # simple precision@K on the last test day
    if X_test.empty:
        print("[eval] test set is empty. Skipping.")
        return
    last_day = X_test.index.get_level_values(0).max()
    X_last = X_test.loc[last_day]
    y_last = y_test.loc[last_day]
    proba = model.predict_proba(X_last)[:,1]
    rank = pd.Series(proba, index=X_last.index).sort_values(ascending=False)
    topk = rank.head(K).index
    prec_at_k = y_last.loc[topk].mean()
    print(f"[eval] {str(last_day.date())} precision@{K}: {prec_at_k:.3f} "
          f"({int(y_last.loc[topk].sum())}/{K})")

def reason_codes(X_row: pd.Series) -> str:
    # minimal, human-readable reason codes
    feats = []
    for name in ["rsi14","dist_sma20","vol_z20","pct_52w","ret_5"]:
        if name in X_row.index:
            val = X_row[name]
            feats.append(f"{name}={val:.2f}")
    return ", ".join(feats)

def score_today(model, X_all, top_n=TOP_N):
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    latest_day = X_all.index.get_level_values(0).max()
    X_latest = X_all.loc[latest_day].dropna()

    proba = model.predict_proba(X_latest)[:,1]
    df = pd.DataFrame({"proba": proba}, index=X_latest.index).sort_values("proba", ascending=False)
    top = df.head(top_n).copy()

    # attach reasons
    rows = []
    for ticker in top.index:
        rc = reason_codes(X_latest.loc[ticker])
        rows.append((ticker, df.loc[ticker, "proba"], rc))
    out = pd.DataFrame(rows, columns=["Ticker","proba","reason"])

    out_path = SIGNALS_DIR / f"signals_{latest_day.date()}.parquet"
    out.to_parquet(out_path, index=False)
    print(f"[score] {latest_day.date()} top-{top_n} saved → {out_path.resolve()}")
    print(out)
    return out_path

def main():
    X_train, y_train, X_test, y_test, X_all = load_xy()
    model = train_or_load(X_train, y_train)
    evaluate(model, X_test, y_test, K=TOP_N)
    score_today(model, X_all, top_n=TOP_N)

if __name__ == "__main__":
    main()
