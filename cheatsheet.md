# Stock ML Project Cheat Sheet

## Pipeline order

```
python fetch_prices.py        # download OHLCV → data/raw/prices.parquet
python build_features.py      # engineer features → data/processed/features.parquet
python build_labels.py        # forward-window labels → data/processed/labels.parquet
python train_and_score.py     # train one model + emit today's top-5
python evaluate_backtest.py   # single-split backtest (uses NEXT-DAY OPEN entry)
python walkforward.py         # rolling walk-forward backtest across years
python tests.py               # 14+ assertions on schema / values / label correctness
python make_eda.py            # eight plots into eda/
```

After upgrading code, rerun the full pipeline: the saved `model_lgbm.pkl` was
trained on the old feature columns and will not match the new schema.

---

## OHLCV columns

| Term   | Definition                          | Intuition                          |
|--------|-------------------------------------|------------------------------------|
| Open   | First trade price of the day        | Starting point of trading session  |
| High   | Highest trade price of the day      | Peak price buyers paid             |
| Low    | Lowest trade price of the day       | Lowest price sellers accepted      |
| Close  | Last trade price of the day         | Final market consensus             |
| Volume | Number of shares traded             | How busy the stock was             |

---

## Features (13 total)

### Original 9

| Feature       | Formula                                        | Intuition                                       |
|---------------|------------------------------------------------|-------------------------------------------------|
| ret_1         | Close_t / Close_{t-1} − 1                      | 1-day return                                    |
| ret_5         | Close_t / Close_{t-5} − 1                      | 1-week return                                   |
| ret_20        | Close_t / Close_{t-20} − 1                     | ~1-month return                                 |
| dist_sma10    | (Close − sma10) / sma10                        | Distance from 10-day moving average             |
| dist_sma20    | (Close − sma20) / sma20                        | Distance from 20-day moving average             |
| dist_sma50    | (Close − sma50) / sma50                        | Distance from 50-day moving average             |
| pct_52w       | (Close − min_252d) / (max_252d − min_252d)     | Where today sits in the 1-year range (0=low, 1=high) |
| vol_z20       | (Vol − mean_20d) / std_20d                     | How unusual today's volume is                   |
| rsi14         | Standard RSI(14), rescaled to [0,1]            | Momentum oscillator                             |

### New 4 (added after EDA showed positives skew mean-reverting)

| Feature         | Formula                                                            | Intuition                                                 |
|-----------------|--------------------------------------------------------------------|-----------------------------------------------------------|
| dist_sma200     | (Close − sma200) / sma200                                          | Long-term trend context. Below 200-day = potential dip    |
| down_streak     | Consecutive days of negative ret_1 ending at t                     | "How long has this stock been bleeding?"                  |
| vol_regime      | (vol20 − mean_252_of_vol20) / std_252_of_vol20                     | Are we in a chop regime or a calm regime, relative to history? |
| mom_rank_xs     | Cross-sectional percentile rank of ret_20 within each date         | Where this stock ranks vs the rest of the universe TODAY  |

---

## Labels

| Term            | Definition                                                                  |
|-----------------|-----------------------------------------------------------------------------|
| HORIZON_DAYS (H)| Look-ahead window size (20 trading days)                                    |
| TARGET_RET (T)  | Required return (+10%)                                                       |
| DRAWDOWN_CAP (D)| Maximum tolerated intra-window drop (−15%)                                  |
| Label = 1       | max future close in H ≥ Close_t × (1+T)  AND  min future close ≥ Close_t × (1−D) |
| Label = 0       | otherwise                                                                   |

Positive rate ≈ 18 % overall. Year-by-year varies a lot (2022 was ~40 %).

---

## Universe

Set in `fetch_prices.py` via `UNIVERSE = "small" | "large"`.

| Mode   | Tickers | When to use                                          |
|--------|---------|------------------------------------------------------|
| small  | 11      | Fast iteration / debugging                           |
| large  | 56      | Real training. Tech, financials, consumer, healthcare, industrials, plus SPY |

---

## Model & evaluation

| Term            | Definition                                                                  |
|-----------------|-----------------------------------------------------------------------------|
| LightGBM        | Gradient-boosted decision-tree classifier                                   |
| scale_pos_weight| (#negatives / #positives), used to fight class imbalance                    |
| proba           | Model's predicted probability of label=1                                    |
| precision@N     | Of the top N picks each day, what fraction were label=1                     |
| slippage        | Trading cost; here 5 bps applied to the next-day open entry                 |
| drawdown        | Maximum drop before recovery                                                |
| Single-split    | First 80 % of dates → train, last 20 % → test                                |
| Walk-forward    | Train on years < Y, test on year Y, slide forward. Honest out-of-sample read |

---

## Execution model

`evaluate_backtest.py` and `walkforward.py` both assume:

* signal observed at **close of day D**
* trade entered at **open of day D+1**
* exit window measured on close prices from D+1 through D+H
* 5 bps slippage subtracted once on the entry

This matches what a real desk could actually do.
