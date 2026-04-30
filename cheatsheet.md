Stock ML Project Cheat Sheet

python fetch_prices.py - update raw prices

python build_features.py - compute features

python build_labels.py - compute labels

python train_and_score.py - train, quick precision check, produce today’s top N

python evaluate_backtest.py - periodic deeper evaluation

Term	Formula / Definition	Intuition (plain english)
Open	/ first trade price of the day	/ Starting point of trading session
High	/ highest trade price of the day	/ Peak price buyers paid today
Low	/ lowest trade price of the day	/ Lowest price sellers accepted today
Close	/ last trade price of the day	/ Final market consensus for the day
Volume	/ number of shares traded	/ How busy the stock was

Features
Term	Formula / Definition	Intuition
ret_1	/ (Close_t / Close_{t-1}) − 1	/ 1-day return (yesterday → today)
ret_5	/ (Close_t / Close_{t-5}) − 1	/ Short-term 1-week return
ret_20	/ (Close_t / Close_{t-20}) − 1	/ Medium-term ~1 month return
sma10	/ mean(Close over 10d)	/ Short trendline
sma20	/ mean(Close over 20d)	/ Monthly trendline
sma50	/ mean(Close over 50d)	/ Longer trendline
dist_sma10	/ (Close − sma10)/sma10	/ How far above/below 10d average
dist_sma20	/ (Close − sma20)/sma20	/ Same for 20d
dist_sma50	/ (Close − sma50)/sma50	/ Same for 50d
pct_52w	/ (Close − min_252d) / (max_252d − min_252d)	/ Where today sits in the 1-year range (0=low, 1=high)
vol_z20	/ (Vol − mean_20d)/std_20d	/ How unusual today’s trading volume is
rsi14	/ RSI oscillator scaled 0–1	/ Momentum: near 1 = “overbought,” near 0 = “oversold”

Labels
Term	Formula / Definition	Intuition
HORIZON_DAYS (H)	lookahead window size	/ How far into the future you check
TARGET_RET (T)	required return	/ How much upside qualifies as a “win”
DRAWDOWN_CAP (D)	min allowed drawdown	/ Risk filter: stock must not crash more than D during window
Label = 1	max future close within H ≥ Close_today × (1+T) (and drawdown ≥ −D if used)	Stock hit the target
Label = 0	otherwise	Stock did not meet target

Model / Evaluation
Term	Definition	Intuition
LightGBM	/ Gradient boosting tree model	/ Learns decision rules like “high RSI + high volume = more upside”
proba	/ Model’s predicted probability of label=1	/ Confidence score between 0 and 1
precision@N	/ Of the top N picks, what % were actually winners	/ Accuracy of top signals
slippage	/ Trading cost from imperfect execution	/ We subtract e.g. 0.1% to simulate reality
drawdown	/ Max drop before recovery	/ Pain of holding through a dip
backtest	/ Simulated performance on past data	/ Checks if model generalizes over time