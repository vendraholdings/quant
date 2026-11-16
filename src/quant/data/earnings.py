"""Earnings dates + EPS surprises per ticker (yfinance, ~100 quarters back), one Parquet per ticker.

    python -m quant.data.earnings --tickers-file config/sp500.txt --out data/earnings
"""
from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
import yfinance as yf

from quant.data.prices import read_tickers

log = logging.getLogger("earnings")
SCHEMA = {"ticker": pl.Utf8, "edate": pl.Date, "hour": pl.Int8,
          "eps_est": pl.Float64, "eps_act": pl.Float64, "surprise_pct": pl.Float64}


def fetch(ticker: str) -> pl.DataFrame:
    e = yf.Ticker(ticker).get_earnings_dates(limit=100)
    if e is None or e.empty:
        return pl.DataFrame(schema=SCHEMA)
    e = e.reset_index()
    ts = e.iloc[:, 0].dt.tz_convert("America/New_York")
    return pl.DataFrame({
        "ticker": ticker, "edate": ts.dt.tz_localize(None).to_numpy().astype("datetime64[D]"), "hour": ts.dt.hour.to_numpy(),
        "eps_est": e["EPS Estimate"].to_numpy(), "eps_act": e["Reported EPS"].to_numpy(),
        "surprise_pct": e["Surprise(%)"].to_numpy()}).cast(SCHEMA).unique("edate").sort("edate")


def update(ticker: str, out: Path, max_age_days: float) -> tuple[str, int, bool]:
    path = out / f"{ticker}.parquet"
    if path.exists() and time.time() - path.stat().st_mtime < max_age_days * 86400:
        return ticker, 0, True  # fresh enough; earnings change quarterly
    for attempt in range(3):
        try:
            df = fetch(ticker)
            if df.height:
                df.write_parquet(path)
            return ticker, df.height, df.height > 0
        except Exception as ex:  # rate limits / missing tickers
            if attempt == 2:
                log.warning("%s failed: %s", ticker, ex)
            time.sleep(2 * (attempt + 1))
    return ticker, 0, False


def load_earnings(out: str | Path = "data/earnings") -> pl.DataFrame:
    return pl.read_parquet(Path(out) / "*.parquet")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers-file", type=Path, default=Path("config/sp500.txt"))
    ap.add_argument("--out", type=Path, default=Path("data/earnings"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-age-days", type=float, default=7)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    a.out.mkdir(parents=True, exist_ok=True)
    tickers = read_tickers(a.tickers_file)
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda t: update(t, a.out, a.max_age_days), tickers))
    bad = [t for t, _, ok in res if not ok]
    log.info("earnings: tickers %d | rows %d | no data %d", len(tickers), sum(n for _, n, _ in res), len(bad))


if __name__ == "__main__":
    main()
