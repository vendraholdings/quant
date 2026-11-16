# 30-day roadmap

Goal: four solid public quant/ML repos - each with a results README, tests, CI, one-command reproduction and honest statistics
(costs, Deflated Sharpe, stated biases). Heavy compute runs overnight on an Apple Silicon Mac Studio.

| Days | Repo | Ships |
|---|---|---|
| 1-7 | quant (this repo) | ML stock ranker results, CI + leakage tests, earnings features, point-in-time universe, write-up on which signals survive a Deflated Sharpe |
| 8-14 | earnings-vol | Earnings event study, post-earnings drift, realized vs implied earnings moves (own daily option snapshots), MLX Monte Carlo pricer validated against QuantLib |
| 15-21 | mlx-finance | Deep learning for markets on Apple Silicon: MLX time-series / order-book models, MLX vs PyTorch-MPS benchmarks |
| 22-28 | portfolio-risk | HRP / mean-variance / risk parity with costs, PCA factor risk model, VaR / ES backtests (Kupiec, Christoffersen) |
| 29-30 | profile | Pinned repos, CI badges, licences, one-page results summary |

## Log
- Day 1: cross-sectional ranker + overnight random search (LightGBM CPU + MLX GPU).
- Day 2: earnings data, event study (PEAD, straddle break-even via MLX bootstrap), PIT S&P membership, earnings features A/B, daily option snapshots.
