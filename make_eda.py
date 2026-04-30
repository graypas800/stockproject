# make_eda.py
# Run after build_features.py and build_labels.py have produced their parquet files.
# Optionally also after train_and_score.py has saved the model and a backtest exists.
#
# Outputs PNGs to ./eda/ for use in the presentation.
#
# Usage:
#     python make_eda.py
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
PRICES_PATH = Path("data/raw/prices.parquet")
FEATS_PATH  = Path("data/processed/features.parquet")
LABELS_PATH = Path("data/processed/labels.parquet")
MODEL_PATH  = Path("models/model_lgbm.pkl")
BT_PATH     = Path("signals/backtest_test_period_top5.parquet")
OUT_DIR     = Path("eda")
OUT_DIR.mkdir(exist_ok=True)

# Style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["font.size"] = 10

FEATURE_COLS = ["ret_1","ret_5","ret_20","dist_sma10","dist_sma20","dist_sma50",
                "pct_52w","vol_z20","rsi14"]


# ---------- helpers ----------
def _load_parquet_or_csv(parquet_path: Path):
    """Try parquet first, fall back to the parallel .csv if present."""
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except Exception as e:
            print(f"[warn] parquet read failed for {parquet_path}: {e}")
    csv_path = parquet_path.with_suffix(".csv")
    if csv_path.exists():
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        if {"Date","Ticker"}.issubset(df.columns):
            df = df.set_index(["Date","Ticker"]).sort_index()
        return df
    raise FileNotFoundError(f"Neither {parquet_path} nor {csv_path} found.")


# ---------- 1. Price history ----------
def plot_price_history(prices: pd.DataFrame):
    """Normalized adjusted close per ticker (each starts at 1.0), log scale."""
    close = prices["Close"].unstack("Ticker").sort_index()
    norm = close / close.bfill().iloc[0]  # start each ticker at 1.0

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for tkr in norm.columns:
        ax.plot(norm.index, norm[tkr], label=tkr, linewidth=1.2)
    ax.set_yscale("log")
    ax.set_title("Normalized adjusted close — each ticker starts at 1.0 (log scale)")
    ax.set_xlabel("Date"); ax.set_ylabel("Growth multiple")
    ax.legend(ncol=4, fontsize=8, loc="upper left")
    fig.savefig(OUT_DIR / "01_price_history.png")
    plt.close(fig)
    print("[eda] saved 01_price_history.png")


# ---------- 2. Label balance ----------
def plot_label_balance(labels: pd.DataFrame):
    """Two panels: overall positive rate, and positive rate by year."""
    y = labels["label"]
    overall = y.mean()
    by_year = y.groupby(labels.index.get_level_values(0).year).mean()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].bar(["Negative (0)", "Positive (1)"],
                [(y==0).mean(), (y==1).mean()],
                color=["#bbbbbb", "#3b82f6"])
    axes[0].set_title(f"Overall class balance — positive rate {overall:.1%}")
    axes[0].set_ylabel("Share of rows")
    for i, v in enumerate([(y==0).mean(), (y==1).mean()]):
        axes[0].text(i, v+0.01, f"{v:.1%}", ha="center")

    axes[1].bar(by_year.index.astype(str), by_year.values, color="#3b82f6")
    axes[1].axhline(overall, color="black", linestyle="--", linewidth=1, label=f"Overall {overall:.1%}")
    axes[1].set_title("Positive rate by year")
    axes[1].set_xlabel("Year"); axes[1].set_ylabel("Positive rate")
    axes[1].legend()
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_label_balance.png")
    plt.close(fig)
    print("[eda] saved 02_label_balance.png")


# ---------- 3. Feature correlation ----------
def plot_feature_correlation(features: pd.DataFrame):
    corr = features[FEATURE_COLS].corr()
    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Feature correlation matrix")
    fig.savefig(OUT_DIR / "03_feature_correlation.png")
    plt.close(fig)
    print("[eda] saved 03_feature_correlation.png")


# ---------- 4. Feature distributions split by label ----------
def plot_feature_distributions(features: pd.DataFrame, labels: pd.DataFrame):
    """For each feature, overlay distribution for label=0 vs label=1."""
    df = features.join(labels[["label"]], how="inner").dropna()
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for ax, col in zip(axes.flat, FEATURE_COLS):
        for lbl, color in [(0, "#888888"), (1, "#3b82f6")]:
            data = df.loc[df["label"]==lbl, col]
            ax.hist(data, bins=50, alpha=0.55, density=True, label=f"label={lbl}", color=color)
        ax.set_title(col); ax.legend(fontsize=8)
    fig.suptitle("Feature distributions, split by label", fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_feature_distributions.png")
    plt.close(fig)
    print("[eda] saved 04_feature_distributions.png")


# ---------- 5. Feature importance ----------
def plot_feature_importance():
    if not MODEL_PATH.exists():
        print("[eda] skipping feature_importance — no model at", MODEL_PATH)
        return
    try:
        import joblib
        model = joblib.load(MODEL_PATH)
        imp = pd.Series(model.feature_importances_, index=model.feature_name_).sort_values()
    except Exception as e:
        print(f"[eda] skipping feature_importance — {e}")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp.index, imp.values, color="#3b82f6")
    ax.set_title("LightGBM feature importance (gain)")
    ax.set_xlabel("Importance")
    fig.savefig(OUT_DIR / "05_feature_importance.png")
    plt.close(fig)
    print("[eda] saved 05_feature_importance.png")


# ---------- 6. Backtest precision over time ----------
def plot_backtest_precision():
    if not BT_PATH.exists():
        print("[eda] skipping backtest_precision — no backtest at", BT_PATH)
        print("       run: python evaluate_backtest.py")
        return
    try:
        bt = pd.read_parquet(BT_PATH)
    except Exception as e:
        print(f"[eda] skipping backtest_precision — {e}")
        return

    if "precision_at_N" not in bt.columns:
        print("[eda] backtest file missing precision_at_N column")
        return

    bt = bt.sort_index()
    rolling = bt["precision_at_N"].rolling(20, min_periods=5).mean()
    overall = bt["precision_at_N"].mean()

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(bt.index, bt["precision_at_N"], alpha=0.35, color="#3b82f6", label="daily precision@5")
    ax.plot(rolling.index, rolling.values, color="#1e3a8a", linewidth=2, label="20-day rolling mean")
    ax.axhline(overall, color="black", linestyle=":", linewidth=1, label=f"mean = {overall:.2f}")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Backtest precision@5 across the test period")
    ax.set_xlabel("Date"); ax.set_ylabel("Precision @ 5")
    ax.legend(loc="upper right")
    fig.savefig(OUT_DIR / "06_backtest_precision.png")
    plt.close(fig)
    print("[eda] saved 06_backtest_precision.png")


# ---------- main ----------
def main():
    print("[eda] loading inputs...")
    features = _load_parquet_or_csv(FEATS_PATH)
    labels   = _load_parquet_or_csv(LABELS_PATH)

    # prices is optional — only needed for the price chart
    try:
        prices = _load_parquet_or_csv(PRICES_PATH)
    except FileNotFoundError:
        prices = None
        print("[eda] no prices file — skipping price history plot")

    if prices is not None:
        plot_price_history(prices)
    plot_label_balance(labels)
    plot_feature_correlation(features)
    plot_feature_distributions(features, labels)
    plot_feature_importance()
    plot_backtest_precision()

    print(f"\n[eda] done. {len(list(OUT_DIR.glob('*.png')))} plots written to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
