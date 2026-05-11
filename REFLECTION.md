# Reflection

Honestly, this project ended up being a lot more interesting than I expected, mostly because the data kept surprising me.

## What I set out to build

I wanted to make something that picks stocks. The plan was: download prices, build momentum-style features (returns, moving-average distances, RSI), label whether each stock went up 10% in the next month, train LightGBM, take the top-5 picks each day, see how it does. Pretty standard ML project.

## The thing I got wrong

The whole project was set up around the idea that *momentum* — stocks that have been going up — would predict more upside. I built features for that. I expected positives to cluster at high RSI, above their moving averages, near 52-week highs.

When I actually plotted the feature distributions split by label, the opposite was true. Positives skewed toward **low** RSI, *below* their moving averages, and were spread across the 52-week range instead of clumping at the top. The model was learning a mean-reversion / buy-the-dip signal, not a momentum signal.

That changed how I thought about the rest of the project. I added features that explicitly capture mean reversion (`dist_sma200`, `down_streak`, `vol_regime`) and a cross-sectional rank that flags the *worst* recent performers relative to the universe — because those are the ones that bounce. The original feature set kind of accidentally captured this through `dist_sma10/20/50`, but now it's explicit.

## The 2022 thing

When I plotted the positive rate by year, 2022 — the worst year for tech in this whole window — had the *highest* positive rate at 40%+. That made no sense to me at first. Then I realized: "+10% in 20 days" is a volatility measure, not a direction measure. A stock can crash and bounce and crash and bounce, and every bounce triggers the label. The actual trend can be down. Which means the model's "win rate" partly tells you what regime you're in, not just how good the model is. That's why the single 80/20 split is a bad way to evaluate this — the test period happens to be one regime — and why I switched to walk-forward.

## Debugging stuff

- Spent way too long on a Git authentication problem trying to push to GitHub. Turned out my Mac had an SSH key tied to a different account, and overriding it required both flipping the remote to HTTPS and clearing the Keychain. That was a real one-hour rabbit hole.
- An early version of `build_features.py` had a duplicate RSI function (one global, one nested inside `build_features`). The global one shadowed nothing but was wasteful. Caught it during cleanup.
- `pct_52w` blew up with infinities the first time because when a ticker traded flat for a stretch, the 252-day max equalled the 252-day min and I was dividing by zero. Fixed with `+ 1e-9` in the denominator and a final `replace(inf, NaN).dropna()`.
- The label-correctness test in `tests.py` was the most valuable test I wrote. It recomputes 50 random labels from raw prices and compares to what `build_labels.py` produced. If the rolling-window logic ever drifts, this catches it immediately.

## Decisions I'm okay with

- Dropping the warm-up rows instead of imputing missing values. There's no honest way to "fill in" what a 200-day SMA was on day 5.
- Modeling execution at next-day open instead of same-day close. Same-day close is what same-day close would actually cost — you can't sit on a closing-bell signal and get filled at it.
- Class imbalance handled via `scale_pos_weight` rather than oversampling. Simpler, no synthetic rows, and LightGBM handles it well.

## Decisions I'd revisit with more time

- The universe is still tech-heavy. I expanded from 11 to 56 tickers across sectors but tech is over-represented because that's where I added the most names. A balanced selection by market cap and sector would generalize better.
- I never tuned hyperparameters. The LightGBM settings are reasonable defaults but probably leave a few points of precision on the table. A small Optuna sweep would be a natural next step.
- The reason codes attached to each daily pick are just a list of feature values. A more useful version would use SHAP to show which features actually pushed the probability up for *that specific* ticker.

## What I learned

The thing I'll take from this project is that EDA is not optional. I almost skipped the feature-distribution plot because I "knew what the features looked like." The plot is what told me the strategy was actually mean reversion, not momentum, which changed everything downstream — the features I added, the way I evaluate, the way I talk about the project. Looking at the data before trusting the plan would have saved me real time.

Also: write the small tests early. I added `tests.py` partway through and it caught two off-by-one issues I would not have noticed otherwise.
