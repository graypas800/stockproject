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
WF_DAILY    = Path("signals/walkforward_daily_top5.parquet")
WF_FOLDS    = Path("signals/walkforward_folds_top5.parquet")
OUT_DIR     = Path("eda")
OUT_DIR.mkdir(exist_ok=True)

# Style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["font.size"] = 10

# Default features list. If build_features.py has been updated to include the
# new mean-reversion features, importing FEATURE_COLS from it stays in sync.
try:
    from build_features import FEATURE_COLS
except Exception:
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
    # If there are many tickers (expanded universe), use a less busy palette
    # and skip the legend for clarity.
    n = norm.shape[1]
    cmap = plt.get_cmap("tab20" if n > 12 else "tab10")
    for i, tkr in enumerate(norm.columns):
        ax.plot(norm.index, norm[tkr], label=tkr, linewidth=0.9 if n > 12 else 1.2,
                color=cmap(i % cmap.N), alpha=0.85)
    ax.set_yscale("log")
    ax.set_title(f"Normalized adjusted close — {n} tickers, each starts at 1.0 (log scale)")
    ax.set_xlabel("Date"); ax.set_ylabel("Growth multiple")
    if n <= 12:
        ax.legend(ncol=4, fontsize=8, loc="upper left")
    else:
        ax.text(0.99, 0.02, f"{n} tickers (legend hidden)",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color="gray")
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
    cols = [c for c in FEATURE_COLS if c in features.columns]
    corr = features[cols].corr()
    fig, ax = plt.subplots(figsize=(0.6*len(cols)+3, 0.6*len(cols)+2.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True, cbar_kws={"shrink": 0.8},
                annot_kws={"fontsize": 8}, ax=ax)
    ax.set_title("Feature correlation matrix")
    fig.savefig(OUT_DIR / "03_feature_correlation.png")
    plt.close(fig)
    print("[eda] saved 03_feature_correlation.png")


# ---------- 4. Feature distributions split by label ----------
def plot_feature_distributions(features: pd.DataFrame, labels: pd.DataFrame):
    """For each feature, overlay distribution for label=0 vs label=1."""
    df = features.join(labels[["label"]], how="inner").dropna()
    cols = [c for c in FEATURE_COLS if c in df.columns]
    n = len(cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.3*nrows))
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for ax, col in zip(axes_list, cols):
        for lbl, color in [(0, "#888888"), (1, "#3b82f6")]:
            data = df.loc[df["label"]==lbl, col].dropna()
            if len(data):
                ax.hist(data, bins=50, alpha=0.55, density=True,
                        label=f"label={lbl}", color=color)
        ax.set_title(col); ax.legend(fontsize=8)
    for ax in axes_list[n:]:
        ax.axis("off")
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

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35*len(imp))))
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


# ---------- 7. Calibration plot ----------
def _calibration_curve_manual(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10):
    """Quantile-binned calibration curve. Returns (bin_means_pred, bin_means_true, bin_counts).
    Manual implementation avoids needing scikit-learn as a hard dependency.
    """
    y_true  = np.asarray(y_true, dtype=float)
    y_proba = np.asarray(y_proba, dtype=float)
    # quantile bins: each bin holds about the same number of samples
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(y_proba, quantiles))
    if len(edges) < 3:
        # not enough distinct values — fall back to fixed-width bins
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_proba, edges, right=True) - 1, 0, len(edges)-2)
    means_pred, means_true, counts = [], [], []
    for b in range(len(edges)-1):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        means_pred.append(float(y_proba[mask].mean()))
        means_true.append(float(y_true[mask].mean()))
        counts.append(int(mask.sum()))
    return np.array(means_pred), np.array(means_true), np.array(counts)


def plot_calibration():
    """Reliability diagram for the saved LightGBM model on the held-out
    test period. Bins predicted probabilities and plots empirical positive
    rate against mean predicted probability. The diagonal y=x is perfect.
    """
    if not (MODEL_PATH.exists() and FEATS_PATH.exists() and LABELS_PATH.exists()):
        print("[eda] skipping calibration — missing model/features/labels")
        return
    try:
        import joblib
        model = joblib.load(MODEL_PATH)
        X = _load_parquet_or_csv(FEATS_PATH)
        y = _load_parquet_or_csv(LABELS_PATH)["label"]
    except Exception as e:
        print(f"[eda] skipping calibration — {e}")
        return

    df = X.join(y.rename("label"), how="inner").dropna()
    dates = df.index.get_level_values(0).unique().sort_values()
    cut = int(0.80 * len(dates))
    test_dates = dates[cut:]
    test_mask = df.index.get_level_values(0).isin(test_dates)
    if test_mask.sum() == 0:
        print("[eda] skipping calibration — empty test set")
        return

    X_test = df.loc[test_mask].drop(columns=["label"])
    y_test = df.loc[test_mask, "label"].values

    # align column order to whatever the model trained on
    try:
        X_test = X_test[model.feature_name_]
    except Exception:
        pass

    try:
        proba = model.predict_proba(X_test)[:, 1]
    except Exception as e:
        print(f"[eda] skipping calibration — model.predict_proba failed: {e}")
        return

    means_pred, means_true, counts = _calibration_curve_manual(y_test, proba, n_bins=10)
    base_rate = float(np.mean(y_test))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6),
                                   gridspec_kw={"width_ratios": [1.2, 1]})

    # reliability diagram
    ax1.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1, label="Perfect calibration")
    ax1.axhline(base_rate, color="gray", linestyle=":", linewidth=1,
                label=f"Base rate ({base_rate:.2f})")
    ax1.plot(means_pred, means_true, "o-", color="#3b82f6", linewidth=2,
             markersize=7, label="Model")
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    ax1.set_xlabel("Mean predicted probability (bin)")
    ax1.set_ylabel("Empirical positive rate (bin)")
    ax1.set_title("Calibration / reliability diagram (test period)")
    ax1.legend(loc="upper left", fontsize=9)

    # histogram of predicted probabilities
    ax2.hist(proba, bins=30, color="#3b82f6", alpha=0.85)
    ax2.axvline(base_rate, color="gray", linestyle=":", linewidth=1, label=f"Base rate {base_rate:.2f}")
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Number of (Date, Ticker) rows")
    ax2.set_title("Distribution of predicted probabilities")
    ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "07_calibration.png")
    plt.close(fig)
    print("[eda] saved 07_calibration.png")


# ---------- 8. Walk-forward precision ----------
def plot_walkforward():
    if not WF_DAILY.exists():
        print("[eda] skipping walk-forward plot — no file at", WF_DAILY)
        print("       run: python walkforward.py")
        return
    try:
        daily = pd.read_parquet(WF_DAILY).sort_index()
    except Exception as e:
        print(f"[eda] skipping walk-forward plot — {e}")
        return

    if "precision_at_N" not in daily.columns or "test_year" not in daily.columns:
        print("[eda] walk-forward file missing expected columns")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    # per-year bar of mean precision
    per_year = daily.groupby("test_year")["precision_at_N"].mean()
    base = daily["precision_at_N"].mean()
    axes[0].bar(per_year.index.astype(str), per_year.values, color="#3b82f6")
    axes[0].axhline(base, color="black", linestyle="--", linewidth=1,
                    label=f"overall {base:.2f}")
    axes[0].set_title("Walk-forward: mean daily precision@5 per fold")
    axes[0].set_xlabel("Test year"); axes[0].set_ylabel("Precision @ 5")
    axes[0].legend()

    # full daily series stitched together with year boundaries
    rolling = daily["precision_at_N"].rolling(20, min_periods=5).mean()
    axes[1].plot(daily.index, daily["precision_at_N"], alpha=0.3, color="#3b82f6",
                 label="daily precision@5")
    axes[1].plot(rolling.index, rolling.values, color="#1e3a8a", linewidth=2,
                 label="20-day rolling mean")
    for yr in sorted(daily["test_year"].unique()):
        first = daily[daily["test_year"]==yr].index.min()
        axes[1].axvline(first, color="gray", linestyle=":", linewidth=0.8)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("Walk-forward: precision@5 over time, with fold boundaries")
    axes[1].set_xlabel("Date"); axes[1].set_ylabel("Precision @ 5")
    axes[1].legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "08_walkforward.png")
    plt.close(fig)
    print("[eda] saved 08_walkforward.png")


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
    plot_calibration()
    plot_walkforward()

    print(f"\n[eda] done. {len(list(OUT_DIR.glob('*.png')))} plots written to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
