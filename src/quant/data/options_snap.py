"""Daily option-chain snapshot -> our own implied-volatility history (free data has none).

    python -m quant.data.options_snap --tickers-file config/options_universe.txt --out data/options

Writes data/options/<UTC date>/<TICKER>.parquet. Run once per US trading day, ideally ~30 min before the close.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import polars as pl
import yfinance as yf

from quant.data.prices import read_tickers

log = logging.getLogger("options_snap")
KEEP = ["contractSymbol", "strike", "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility"]


def snap_ticker(ticker: str, max_expiries: int) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    spot = float(t.fast_info["last_price"])
    parts = []
    for exp in list(t.options)[:max_expiries]:
        ch = t.option_chain(exp)
        for kind, frame in (("C", ch.calls), ("P", ch.puts)):
            if frame is not None and len(frame):
                parts.append(frame[KEEP].assign(ticker=ticker, expiry=exp, type=kind, spot=spot))
    if not parts:
        raise ValueError("no option chain")
    return pd.concat(parts, ignore_index=True)


def snap_with_retry(ticker: str, max_expiries: int, out: Path, snap: datetime) -> int | None:
    for attempt in range(1, 4):
        try:
            df = pl.from_pandas(snap_ticker(ticker, max_expiries)).with_columns(
                pl.lit(snap).alias("snap_utc"), pl.col("volume", "openInterest").cast(pl.Float64))
            df.write_parquet(out / f"{ticker}.parquet")
            return df.height
        except Exception as e:
            if attempt == 3:
                log.warning("%s failed: %s", ticker, e)
            time.sleep(2 * attempt)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers-file", type=Path, default=Path("config/options_universe.txt"))
    ap.add_argument("--out", type=Path, default=Path("data/options"))
    ap.add_argument("--max-expiries", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    snap = datetime.now(timezone.utc).replace(microsecond=0)
    out = a.out / snap.date().isoformat()
    out.mkdir(parents=True, exist_ok=True)
    tickers = read_tickers(a.tickers_file)
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda t: snap_with_retry(t, a.max_expiries, out, snap), tickers))
    ok = [r for r in res if r is not None]
    print(f"snap {snap.date()} tickers ok={len(ok)} failed={len(res) - len(ok)} rows={sum(ok)} -> {out}")
    sys.exit(1 if len(ok) < len(tickers) / 2 else 0)


if __name__ == "__main__":
    main()
