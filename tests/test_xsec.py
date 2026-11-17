"""Correctness and leakage tests for the cross-sectional ranker (synthetic data, no network)."""
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from quant.research import xsec


def _panel(n_days=600, n_names=50, seed=0):
    rng = np.random.default_rng(seed)
    days = [date(2018, 1, 1) + timedelta(days=i) for i in range(int(n_days * 1.45)) if (date(2018, 1, 1) + timedelta(days=i)).weekday() < 5][:n_days]
    rows = [(d, f"T{j:03d}", float(rng.normal())) for d in days for j in range(n_names)]
    return pl.DataFrame(rows, schema=["date", "ticker", "fwd"], orient="row")


def test_perfect_predictor_wins_and_ic_is_one():
    d = _panel(60).with_columns(pl.col("fwd").alias("pred"))
    bt = xsec.backtest(d)
    assert bt["ic"].min() == pytest.approx(1.0)
    assert (bt["gross"] > 0).all()


def test_inverse_predictor_loses():
    d = _panel(60).with_columns((-pl.col("fwd")).alias("pred"))
    assert (xsec.backtest(d)["gross"] < 0).all()


def test_costs_only_charged_on_turnover():
    # same ranking every period -> full cost on the first period only
    d = _panel(20).with_columns(pl.col("ticker").str.slice(1).cast(pl.Int32).cast(pl.Float64).alias("pred"))
    bt = xsec.backtest(d, cost_bps=10)
    assert bt["cost"][0] == pytest.approx(2 * 2 * 10 / 1e4)
    assert bt["cost"][1:].abs().max() == 0


def test_tiny_universe_still_trades_one_name_per_side():
    d = _panel(10, n_names=4).with_columns(pl.col("fwd").alias("pred"))
    bt = xsec.backtest(d)
    assert bt["gross"].is_finite().all() and (bt["gross"] > 0).all()


def test_walk_forward_split_has_no_lookahead():
    d = _panel(1000, n_names=1).with_columns(pl.col("date").rank("dense").cast(pl.Int32).alias("di"))
    P = dict(day=d["date"].cast(pl.Int32).to_numpy(), di=d["di"].to_numpy(), year=d["date"].dt.year().to_numpy())
    for yr in (2019, 2020, 2021):
        tr, te = xsec.split(P, yr)
        assert te.any() and tr.any()
        assert (P["year"][te] == yr).all()
        assert (P["di"][te] % xsec.H == 0).all()
        # the last training target (H trading days ahead) must finish before the test year starts
        last_train_di = P["di"][tr].max()
        target_end_day = P["day"][P["di"] == last_train_di + xsec.H][0]
        assert target_end_day < P["day"][P["year"] == yr].min()


def test_earnings_features_only_use_completed_events(tmp_path):
    ev = pl.DataFrame({"ticker": ["A"], "edate": [date(2020, 1, 10)], "avail_date": [date(2020, 1, 13)],
                       "surprise_pct": [5.0], "ab_move": [0.04], "exp_move": [0.03]})
    ev.write_parquet(tmp_path / "events.parquet")
    (tmp_path / "earn").mkdir()
    ev.select("ticker", "edate").write_parquet(tmp_path / "earn" / "A.parquet")
    days = [date(2020, 1, d) for d in (9, 10, 12, 13, 14)]
    lf = pl.LazyFrame({"date": days, "ticker": ["A"] * 5})
    out = xsec.add_earnings(lf, tmp_path / "events.parquet", tmp_path / "earn").collect().sort("date")
    known = dict(zip(out["date"].to_list(), out["e_last_move"].to_list()))
    assert known[date(2020, 1, 12)] is None          # reaction not complete yet: no peeking
    assert known[date(2020, 1, 13)] == pytest.approx(0.04)
    assert out.filter(pl.col("date") == date(2020, 1, 9))["e_days_to"][0] == 1


def test_point_in_time_filter_drops_non_members(tmp_path):
    pl.DataFrame({"snap_date": [date(2020, 1, 1), date(2020, 1, 1), date(2020, 6, 1)],
                  "ticker": ["A", "B", "A"]}).write_parquet(tmp_path / "m.parquet")
    lf = pl.LazyFrame({"date": [date(2020, 3, 1), date(2020, 3, 1), date(2020, 7, 1), date(2020, 7, 1)],
                       "ticker": ["A", "B", "A", "B"]})
    out = xsec.pit_filter(lf, tmp_path / "m.parquet").collect()
    assert sorted(zip(out["date"].to_list(), out["ticker"].to_list())) == [
        (date(2020, 3, 1), "A"), (date(2020, 3, 1), "B"), (date(2020, 7, 1), "A")]


def test_deflated_sharpe_punishes_data_mining():
    rng = np.random.default_rng(1)
    trials = rng.normal(0, 1, (2000, 120))           # 2000 pure-noise strategies
    srs = trials.mean(1) / trials.std(1, ddof=1)
    luckiest = trials[srs.argmax()]
    p_lucky, _ = xsec.deflated_sharpe(luckiest, srs)
    real = rng.normal(0.8, 1, 120)                    # a genuinely strong strategy
    p_real, _ = xsec.deflated_sharpe(real, np.append(srs, real.mean() / real.std(ddof=1)))
    assert p_lucky < 0.95 < p_real                   # luckiest noise fails the 95% bar, the real one passes
