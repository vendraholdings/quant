# quant: ML stock ranking on Apple Silicon, tested honestly

[![tests](https://github.com/vendraholdings/quant/actions/workflows/tests.yml/badge.svg)](https://github.com/vendraholdings/quant/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.12-blue)
![MLX](https://img.shields.io/badge/GPU-MLX-black)
![license](https://img.shields.io/badge/license-MIT-green)

Every night this project downloads S&P 500 data, builds features, and searches **~2,000 machine-learning models**:
LightGBM runs on the CPU and MLX neural nets run on the Apple GPU at the same time. Each model trades a monthly
long-short portfolio, and the best one is then tested for **whether it beats luck**.

Most backtests on GitHub stop at "look, a high Sharpe". This one asks two harder questions: how much of that Sharpe
comes from trying 2,000 models, and how much comes from testing only on companies that survived to today.

## Results

| run | universe | trials | median Sharpe (net) | best Sharpe (net) | median IC |
|---|---|---|---|---|---|
| original | today's S&P 500 members | 1,979 | 0.80 | 1.03 | 0.036 |
| point-in-time, price features | members on each date, incl. later removals | 993 | -0.17 | 0.07 | 0.010 |
| point-in-time, + earnings features (LightGBM) | same | 863 | -0.03 | 0.29 | 0.013 |
| point-in-time, + earnings features (MLX MLP) | same | 1,750 | -0.13 | 0.17 | 0.009 |

**Finding: the original result was mostly survivorship bias.** On today's index members every one of the 1,979
configurations earned a positive net Sharpe (best 1.03, Deflated Sharpe probability 99.9%). Rerun on the point-in-time
universe, which includes companies that were later removed or went bust, the median configuration loses money after
costs and the best reaches 0.29. Adding earnings features (surprise, timing, post-earnings drift) helps, but not
enough to produce a strategy worth trading. Full comparison: [`reports/ab_compare.md`](reports/ab_compare.md).

The selection-bias side (how much of the best of 1,979 is luck) is analysed in a separate project,
[backtest-overfitting](https://github.com/vendraholdings/backtest-overfitting).

**Earnings event study** (53,531 events, 662 tickers, 2001-2026; [`reports/earnings.md`](reports/earnings.md)):
the top surprise quintile outperforms the bottom by 1.19% over the following 60 trading days before costs, a gap
that has narrowed since 2008. 15% of earnings moves exceed twice the stock's typical move.

Original run, today's members (survivorship-biased): equity curve below; report in [`reports/overnight.md`](reports/overnight.md).

![equity curve, original survivorship-biased run](reports/overnight.png)

## How it works

```
prices (yfinance)  ─────────────► 14 features per stock-day ─► rank-normalise per date
   momentum 1w..12m, 12-1 mom, volatility, 52w-high, max-return, skew, liquidity, size, volume surge
                                     │
                    walk-forward: refit every year on all past data
                    40-day embargo between train and test (no look-ahead)
                                     │
        ┌────────────────────────────┴────────────────────────────┐
  LightGBM random search (CPU, multi-process)       MLX MLP random search (GPU)
        └────────────────────────────┬────────────────────────────┘
          long top 10% / short bottom 10%, rebalance every 21 days, 10 bps per side
                                     │
          Deflated Sharpe  ·  GPU block bootstrap  ·  equity curve vs SPY
```

- **Target:** each stock's next-21-day return, ranked against the other stocks on the same date (a cross-sectional rank).
- **No leakage:** yearly expanding-window refits, with a 40-calendar-day gap so no training target overlaps the test year. A unit test proves it.
- **Costs:** turnover-based, 10 bps per side, charged on both legs.
- **Search:** resume-safe random search that runs until a deadline, with results streamed to JSONL so a crash loses nothing.

## In progress: earnings edge

- **Event study** ([`earnings_study.py`](src/quant/research/earnings_study.py)): earnings-day moves, and post-earnings drift by EPS-surprise quintile.
- **Straddle pricing:** the break-even implied-move markup, from a GPU bootstrap.
- **Earnings features for the ranker:** days to/since earnings, last surprise, last move. Joined *as-of* the date the reaction was complete, never earlier, with a unit test.
- **A/B test:** the same model search with and without earnings features, on the point-in-time universe.
- **Daily option-chain snapshots** for 40 liquid names ([`options_snap.py`](src/quant/data/options_snap.py)). Free historical implied-vol data doesn't exist, so this builds its own.

## Run it

```bash
uv sync                                                    # Python 3.12, Apple Silicon (MLX)
uv run pytest -q                                           # correctness + leakage tests
uv run python -m quant.data.universe                       # S&P 500 tickers
uv run python -m quant.data.prices --tickers-file config/sp500.txt --start 2000-01-01
uv run python -m quant.research.xsec panel                 # feature panel (Parquet)
uv run python -m quant.research.xsec sweep --model lgb --deadline $(( $(date +%s) + 3600 )) --workers 6 --threads 4
uv run python -m quant.research.xsec report                # reports/overnight.md + chart
```

`scripts/overnight.sh` chains everything into a single unattended night run, with a hard stop time.

## Stack

polars (lazy, as-of joins) · LightGBM · MLX (Apple GPU) · NumPy/SciPy · DuckDB/Parquet · yfinance · uv

## Layout

```
src/quant/data/       prices, S&P universe + historical membership, earnings dates, option snapshots
src/quant/research/   xsec.py (features, walk-forward, backtest, Deflated Sharpe, bootstrap), earnings_study.py
scripts/              overnight runners
tests/                leakage, costs, point-in-time and Deflated Sharpe tests
reports/              generated results
```

Research code, not investment advice.
