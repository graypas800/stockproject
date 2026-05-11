# app.py
#
# Streamlit dashboard for the Stock ML project.
# Run:    streamlit run app.py
#
# Tabs:
#   1. Today's Picks   — latest signals file with probabilities and reason codes
#   2. Ticker Explorer — pick a ticker + date, view features, label, and model probability
#   3. Backtest        — single-split daily precision and summary stats
#   4. Walk-forward    — fold-by-fold results across years
#   5. EDA Gallery     — every PNG in eda/, with captions
#
# Every section degrades gracefully: if an upstream artifact is missing
# the app tells you which script to run, rather than crashing.

from pathlib import Path
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st


# ---------- paths ----------
ROOT          = Path(__file__).parent
DATA_RAW      = ROOT / "data" / "raw" / "prices.parquet"
DATA_FEATS    = ROOT / "data" / "processed" / "features.parquet"
DATA_LABELS   = ROOT / "data" / "processed" / "labels.parquet"
MODEL_PATH    = ROOT / "models" / "model_lgbm.pkl"
SIGNALS_DIR   = ROOT / "signals"
BACKTEST_PATH = SIGNALS_DIR / "backtest_test_period_top5.parquet"
WF_DAILY      = SIGNALS_DIR / "walkforward_daily_top5.parquet"
WF_FOLDS      = SIGNALS_DIR / "walkforward_folds_top5.parquet"
EDA_DIR       = ROOT / "eda"


# ---------- page setup ----------
st.set_page_config(
    page_title="Stock ML Dashboard",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Stock ML Dashboard")
st.caption(
    "Daily top-5 stock picks from a LightGBM model trained to predict +10% "
    "moves within 20 trading days (with a 15% drawdown cap)."
)


# ---------- helpers ----------
@st.cache_data
def _load_parquet(path: Path) -> pd.DataFrame | None:
    """Load a parquet if it exists, returning None otherwise."""
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:
        st.warning(f"Could not read {path.name}: {e}")
        return None


@st.cache_resource
def _load_model():
    """Load the saved LightGBM model. Returns None if missing or load fails."""
    if not MODEL_PATH.exists():
        return None
    try:
        import joblib
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.warning(f"Could not load model: {e}")
        return None


def _missing(label: str, command: str):
    """Friendly message when an upstream artifact hasn't been generated yet."""
    st.info(f"**{label} not found.** Run this first:\n\n```bash\n{command}\n```")


# ---------- sidebar: pipeline status ----------
with st.sidebar:
    st.header("Pipeline status")
    checks = [
        ("Prices",          DATA_RAW,      "python fetch_prices.py"),
        ("Features",        DATA_FEATS,    "python build_features.py"),
        ("Labels",          DATA_LABELS,   "python build_labels.py"),
        ("Model",           MODEL_PATH,    "python train_and_score.py"),
        ("Backtest",        BACKTEST_PATH, "python evaluate_backtest.py"),
        ("Walk-forward",    WF_DAILY,      "python walkforward.py"),
        ("EDA plots",       EDA_DIR,       "python make_eda.py"),
    ]
    for name, path, _ in checks:
        ok = path.exists() and (path.is_file() or any(path.iterdir()))
        st.write(f"{'✅' if ok else '⛔'} {name}")
    st.divider()
    st.caption(
        "If something says ⛔, that section of the dashboard will show "
        "the command to generate it. Everything else still works."
    )


# ---------- tabs ----------
tab_picks, tab_explorer, tab_backtest, tab_walkforward, tab_eda = st.tabs([
    "🎯 Today's Picks",
    "🔍 Ticker Explorer",
    "📊 Backtest",
    "🚶 Walk-forward",
    "🖼️ EDA Gallery",
])


# ========== TAB 1: Today's Picks ==========
with tab_picks:
    st.subheader("Most recent top-5 picks")

    if not SIGNALS_DIR.exists():
        _missing("Signals directory", "python train_and_score.py")
    else:
        signal_files = sorted(SIGNALS_DIR.glob("signals_*.parquet"), reverse=True)
        if not signal_files:
            _missing("Daily signal files", "python train_and_score.py")
        else:
            # Let the user choose which signal date to view if there's more than one
            file_choice = st.selectbox(
                "Signal date",
                options=signal_files,
                format_func=lambda p: p.stem.replace("signals_", ""),
                index=0,
            )
            picks = _load_parquet(file_choice)
            if picks is None or picks.empty:
                st.warning("That signals file is empty.")
            else:
                # Format proba as percentage
                display = picks.copy()
                if "proba" in display.columns:
                    display["proba"] = display["proba"].map(lambda v: f"{v:.1%}")
                st.dataframe(display, use_container_width=True, hide_index=True)
                st.caption(
                    "Probabilities are the model's confidence that each ticker "
                    "will hit +10% within 20 trading days without first dropping "
                    "more than 15%. Reason codes show the feature values that "
                    "drove each pick."
                )


# ========== TAB 2: Ticker Explorer ==========
with tab_explorer:
    st.subheader("Inspect a single (Date, Ticker)")

    features = _load_parquet(DATA_FEATS)
    labels   = _load_parquet(DATA_LABELS)

    if features is None:
        _missing("Features", "python build_features.py")
    else:
        tickers = sorted(features.index.get_level_values("Ticker").unique())
        col1, col2 = st.columns(2)
        with col1:
            ticker = st.selectbox("Ticker", tickers, index=0)
        with col2:
            ticker_dates = features.xs(ticker, level="Ticker").index.sort_values()
            if len(ticker_dates) == 0:
                st.warning(f"No feature rows for {ticker}")
                date = None
            else:
                date = st.date_input(
                    "Date",
                    value=ticker_dates.max().date(),
                    min_value=ticker_dates.min().date(),
                    max_value=ticker_dates.max().date(),
                )

        if date is not None:
            date_ts = pd.Timestamp(date)
            try:
                row = features.loc[(date_ts, ticker)]
            except KeyError:
                # nearest available trading day
                tdates = ticker_dates
                nearest = tdates[tdates.get_indexer([date_ts], method="nearest")[0]]
                st.warning(
                    f"No row for {ticker} on {date}; showing nearest trading "
                    f"day {nearest.date()}."
                )
                row = features.loc[(nearest, ticker)]
                date_ts = nearest

            st.markdown(f"### {ticker} on {date_ts.date()}")

            # feature values in a tidy grid
            feat_df = (
                row.rename("value")
                   .to_frame()
                   .assign(value=lambda d: d["value"].round(4))
            )
            st.dataframe(feat_df, use_container_width=True)

            # label, if known
            if labels is not None:
                try:
                    lbl = int(labels.loc[(date_ts, ticker), "label"])
                    badge = "✅ Positive (1)" if lbl == 1 else "❌ Negative (0)"
                    st.markdown(f"**Label:** {badge}")
                except KeyError:
                    st.markdown("**Label:** _not available (likely within the look-ahead window)_")

            # model probability, if model loads
            model = _load_model()
            if model is not None:
                try:
                    cols = list(model.feature_name_)
                    x_row = features.loc[[(date_ts, ticker)], cols]
                    proba = float(model.predict_proba(x_row)[0, 1])
                    st.markdown(f"**Model probability of label=1:** {proba:.1%}")
                except Exception as e:
                    st.caption(f"(Model couldn't score this row: {e})")

            # a small chart: ret_1, cumulative, recent window
            recent = (
                features.xs(ticker, level="Ticker")
                        .loc[:date_ts]
                        .tail(120)
            )
            if "ret_1" in recent.columns and not recent.empty:
                cum = (1 + recent["ret_1"]).cumprod()
                st.markdown("##### Last ~6 months (cumulative return)")
                st.line_chart(cum, height=240)


# ========== TAB 3: Backtest ==========
with tab_backtest:
    st.subheader("Single-split backtest — last 20% of dates held out")

    bt = _load_parquet(BACKTEST_PATH)
    if bt is None:
        _missing("Backtest results", "python evaluate_backtest.py")
    else:
        bt = bt.sort_index()
        # summary
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Days evaluated",        f"{len(bt)}")
        c2.metric("Mean precision@5",      f"{bt['precision_at_N'].mean():.2%}")
        c3.metric("Mean ret_end (post-slip)", f"{bt['avg_ret_end_H'].mean():.2%}")
        c4.metric("Worst intra-window dd", f"{bt['worst_dd_H'].min():.2%}")

        # daily precision with rolling mean
        st.markdown("#### Daily precision@5 with 20-day rolling mean")
        rolling = bt["precision_at_N"].rolling(20, min_periods=5).mean()
        plot_df = pd.DataFrame({
            "daily precision@5": bt["precision_at_N"],
            "20-day rolling":   rolling,
        })
        st.line_chart(plot_df, height=320)

        # best / worst days
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 5 best days (precision)")
            st.dataframe(
                bt.nlargest(5, "precision_at_N")[
                    ["precision_at_N", "hits", "avg_ret_end_H"]
                ],
                use_container_width=True,
            )
        with c2:
            st.markdown("#### 5 worst days (precision)")
            st.dataframe(
                bt.nsmallest(5, "precision_at_N")[
                    ["precision_at_N", "hits", "avg_ret_end_H"]
                ],
                use_container_width=True,
            )


# ========== TAB 4: Walk-forward ==========
with tab_walkforward:
    st.subheader("Walk-forward backtest — train on years < Y, test on Y, slide")

    folds = _load_parquet(WF_FOLDS)
    daily = _load_parquet(WF_DAILY)

    if folds is None or daily is None:
        _missing("Walk-forward results", "python walkforward.py")
    else:
        # overall summary
        overall_prec = daily["precision_at_N"].mean(skipna=True)
        overall_ret  = daily["avg_ret_end_H"].mean(skipna=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Folds run",            f"{len(folds)}")
        c2.metric("Overall precision@5",  f"{overall_prec:.2%}")
        c3.metric("Overall ret_end mean", f"{overall_ret:.2%}")

        # per-fold table
        st.markdown("#### Per-fold summary")
        st.dataframe(
            folds.assign(
                precision_at_N_mean=lambda d: d["precision_at_N_mean"].map(lambda v: f"{v:.2%}"),
                avg_ret_max_H_mean= lambda d: d["avg_ret_max_H_mean"].map(lambda v: f"{v:.2%}"),
                avg_ret_end_H_mean= lambda d: d["avg_ret_end_H_mean"].map(lambda v: f"{v:.2%}"),
                worst_dd_H_min=     lambda d: d["worst_dd_H_min"].map(lambda v: f"{v:.2%}"),
            ),
            use_container_width=True,
            hide_index=True,
        )

        # per-fold bar chart
        st.markdown("#### Mean daily precision@5 per fold")
        st.bar_chart(
            folds.set_index("test_year")["precision_at_N_mean"],
            height=280,
        )

        # daily timeline
        st.markdown("#### Daily precision@5 across all folds (rolling)")
        rolling = daily["precision_at_N"].rolling(20, min_periods=5).mean()
        st.line_chart(
            pd.DataFrame({
                "daily":          daily["precision_at_N"],
                "20-day rolling": rolling,
            }),
            height=320,
        )


# ========== TAB 5: EDA Gallery ==========
with tab_eda:
    st.subheader("Exploratory Data Analysis")

    if not EDA_DIR.exists() or not any(EDA_DIR.glob("*.png")):
        _missing("EDA plots", "python make_eda.py")
    else:
        captions = {
            "01_price_history":          "Normalized adjusted close per ticker on a log scale. Each ticker starts at 1.0.",
            "02_label_balance":          "Class balance overall and by year. 2022 has the highest positive rate despite being a bear market — volatility, not direction, drives the label.",
            "03_feature_correlation":    "Correlation matrix. Trend features (returns, SMA distances) cluster together; volume z-score is independent.",
            "04_feature_distributions":  "Each feature, split by label. Positives skew toward LOW RSI and BELOW their SMAs — a mean-reversion signal, not momentum.",
            "05_feature_importance":     "LightGBM gain importance. Which features the model actually used.",
            "06_backtest_precision":     "Daily precision@5 across the single-split test period.",
            "07_calibration":            "Reliability diagram: does the model's predicted probability match the empirical positive rate?",
            "08_walkforward":            "Walk-forward precision per fold (left) and daily across all folds (right).",
        }
        # Sort plots numerically by filename prefix when possible
        png_files = sorted(EDA_DIR.glob("*.png"), key=lambda p: p.stem)
        for png in png_files:
            cap = captions.get(png.stem, "")
            st.markdown(f"#### `{png.name}`")
            if cap:
                st.caption(cap)
            st.image(str(png), use_container_width=True)
            st.divider()


# ---------- footer ----------
st.markdown(
    "<br><div style='text-align:center; color:#999; font-size:0.85em'>"
    "Stock ML Project · LightGBM · yfinance · Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
