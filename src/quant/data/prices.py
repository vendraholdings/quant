"""Daily OHLCV data lake: one Parquet file per ticker, updated incrementally.

    uv run python -m quant.data.prices --tickers-file config/universe.txt --start 2010-01-01 --out data/prices
"""
from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import yfinance as yf

log = logging.getLogger("prices")

COLS = {"Date": "date", "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Adj Close": "adj_close", "Volume": "volume"}
SCHEMA = {"date": pl.Date, "ticker": pl.Utf8, "open": pl.Float64, "high": pl.Float64,
          "low": pl.Float64, "close": pl.Float64, "adj_close": pl.Float64, "volume": pl.Int64}


def read_tickers(path: Path) -> list[str]:
    out = []
    for line in path.read_text().splitlines():
        t = line.split("#", 1)[0].strip().upper()
        if t:
            out.append(t)
    return list(dict.fromkeys(out))


def fetch(ticker: str, start: date) -> pl.DataFrame:
    raw = yf.download(ticker, start=start.isoformat(), auto_adjust=False,
                      progress=False, multi_level_index=False, threads=False)
    if raw is None or raw.empty:
        return pl.DataFrame(schema=SCHEMA)
    df = pl.from_pandas(raw.reset_index()).rename(COLS, strict=False)
    return (df.with_columns(pl.lit(ticker).alias("ticker"))
              .select([pl.col(c).cast(t) for c, t in SCHEMA.items()])
              .drop_nulls("close"))


def update_ticker(ticker: str, start: date, out: Path) -> tuple[str, int, bool]:
    path = out / f"{ticker.replace('/', '_')}.parquet"
    try:
        old = pl.read_parquet(path) if path.exists() else pl.DataFrame(schema=SCHEMA)
        since = old["date"].max() + timedelta(days=1) if old.height else start
        if since > date.today():
            return ticker, 0, True
        for attempt in range(3):  # yfinance's sqlite cache can throw 'database is locked' under threads
            new = fetch(ticker, since)
            if new.height:
                break
            time.sleep(1 + attempt)
        if new.height == 0:
            # nothing new is normal for an existing file (weekend/holiday); empty first download is a failure
            if old.height == 0:
                log.warning("%s: no data returned", ticker)
            return ticker, 0, old.height > 0
        merged = pl.concat([old, new]).unique("date", keep="last").sort("date")
        tmp = path.with_suffix(".tmp")
        merged.write_parquet(tmp)
        tmp.replace(path)  # atomic: never leave a half-written file
        return ticker, merged.height - old.height, True
    except Exception as e:  # one bad ticker must not stop the batch
        log.warning("%s failed: %s", ticker, e)
        return ticker, 0, False


def load_prices(out: str | Path = "data/prices") -> pl.LazyFrame:
    return pl.scan_parquet(Path(out) / "*.parquet")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tickers-file", type=Path, default=Path("config/universe.txt"))
    ap.add_argument("--start", type=date.fromisoformat, default=date(2010, 1, 1))
    ap.add_argument("--out", type=Path, default=Path("data/prices"))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    a.out.mkdir(parents=True, exist_ok=True)
    tickers = read_tickers(a.tickers_file)
    with ThreadPoolExecutor(a.workers) as ex:
        results = list(ex.map(lambda t: update_ticker(t, a.start, a.out), tickers))

    failed = [t for t, _, ok in results if not ok]
    log.info("tickers %d | rows added %d | failed %d %s",
             len(tickers), sum(n for _, n, _ in results), len(failed), failed or "")


if __name__ == "__main__":
    main()
