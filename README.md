# Stock ML Project

A small machine-learning pipeline that scans a watchlist of liquid US stocks every day and predicts which ones are most likely to deliver a +10% move within the next 20 trading days, without first dropping more than 15%. The output is a daily "top 5" with the model's probability and a short reason for each pick.

It's a class project — not investment advice.

---

## Quick start

```bash
# 1. clone and set up
git clone https://github.com/graypas800/stockproject.git
cd stockproject
python3 -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. run the pipeline in order
python fetch_prices.py             # download OHLCV from Yahoo Finance
python build_features.py           # engineer 13 features
python build_labels.py             # forward-window labels
python train_and_score.py          # train LightGBM, score today's picks
python evaluate_backtest.py        # single-split backtest
python walkforward.py              # honest walk-forward backtest
python tests.py                    # validation tests (should print "all passed")
python make_eda.py                 # generate 8 EDA plots

# 3. launch the dashboard
streamlit run app.py
```

The first run takes a few minutes — most of that is the data download. After that, everything is fast.

---

## What's in here

```
stockProj/
├── fetch_prices.py        # download OHLCV from Yahoo Finance
├── build_features.py      # 13 engineered features per (Date, Ticker)
├── build_labels.py        # +10%-in-20-days labels with drawdown cap
├── train_and_score.py     # train LightGBM, score the latest day's universe
├── evaluate_backtest.py   # 80/20 backtest with next-day-open execution
├── walkforward.py         # rolling year-by-year walk-forward
├── make_eda.py            # generate the EDA plots
├── tests.py               # 17 schema / range / correctness assertions
├── app.py                 # Streamlit dashboard
├── cheatsheet.md          # quick reference for every feature and metric
├── REFLECTION.md          # short writeup of the design process
├── requirements.txt
├── data/
│   ├── raw/               # downloaded prices
│   └── processed/         # features and labels
├── models/                # saved LightGBM models
├── signals/               # daily picks + backtest results
└── eda/                   # generated plots (PNG)
```

---

## ETL workflow

**Extract** — `fetch_prices.py` pulls daily OHLCV bars for 56 large- and mid-cap US tickers from Yahoo Finance via the `yfinance` library, starting from January 2018. Closes are auto-adjusted so splits and dividends don't create artificial jumps. The result is saved as `data/raw/prices.parquet` indexed by `(Date, Ticker)`.

**Transform — features** — `build_features.py` produces 13 features per row:

- Returns over 1, 5, and 20 days
- Distance from the 10-, 20-, 50-, and 200-day simple moving averages
- 52-week percentile position
- 20-day volume z-score
- 14-day RSI scaled to [0, 1]
- Consecutive down-day streak
- Volatility regime (20-day realized vol z-scored against its own 252-day history)
- Cross-sectional momentum rank (where this ticker ranks vs the rest of the universe on `ret_20` today)

Rolling features that can't be computed for the warm-up period are dropped rather than imputed — imputing would invent data. All `inf` values from zero-division corner cases get replaced with `NaN` and dropped.

**Transform — labels** — `build_labels.py` does a forward look. For each `(Date, Ticker)`, label = 1 if the max close over the next 20 trading days reaches +10% **and** the min close stays above -15%. The drawdown cap matters: a stock that hits +10% only after crashing 30% first isn't actually a trade you'd want to take. Rows where the 20-day future window isn't fully known yet (i.e., the last 20 dates per ticker) get dropped.

**Load** — features and labels are written to `data/processed/` as both parquet (compact and fast) and CSV (human-readable for debugging).

---

## The Streamlit dashboard

`streamlit run app.py` opens a five-tab dashboard:

1. **Today's Picks** — the latest top-5 with model probability and the feature values that drove each pick.
2. **Ticker Explorer** — pick any ticker and date, see its feature values, the true label, and what the model would predict.
3. **Backtest** — daily precision@5 over the last 20% of dates with summary stats and best/worst days.
4. **Walk-forward** — fold-by-fold breakdown across years, plus a daily timeline.
5. **EDA Gallery** — all eight generated plots with captions.

The sidebar shows a live status of which pipeline artifacts exist. If something is missing, the relevant tab will tell you exactly which script to run to generate it.

---

## Notes

- **Don't trade off this.** It's a class project to learn ETL, feature engineering, and a basic ML evaluation loop. Backtest precision is real, but live performance would be way harder once you account for transaction costs, taxes, and the fact that 2018–2025 was an unusually friendly period for US tech.
- **The model trains in seconds.** Walk-forward takes longer because it trains one model per test year (5–7 folds depending on data range).
- **Yahoo Finance occasionally fails or returns gaps.** The pipeline drops rows with missing Close, but if a fetch errors out, just run `python fetch_prices.py` again.

---

## Requirements

Python 3.10+. Main packages are pinned in `requirements.txt`:

- pandas, numpy, pyarrow
- yfinance
- lightgbm, scikit-learn
- joblib
- matplotlib, seaborn
- streamlit

---

## License

For coursework. No license claim, no warranty.
