"""Point-in-time S&P 500 membership (fja05680/sp500 history) -> data/members.parquet + all-time ticker list.

    python -m quant.data.members --since 2000-01-01

members.parquet: (snap_date, ticker) for every membership snapshot; a ticker is a member on day D iff it
appears in the latest snapshot on or before D. Many removed names have no free price history, so this
removes index-inclusion bias but only part of survivorship bias.
"""
from __future__ import annotations

import argparse
import io
import urllib.request
from datetime import date
from pathlib import Path

import polars as pl

URL = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
       "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv")


def load_members(path: str | Path = "data/members.parquet") -> pl.DataFrame:
    return pl.read_parquet(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=date.fromisoformat, default=date(2000, 1, 1))
    ap.add_argument("--out", type=Path, default=Path("data/members.parquet"))
    ap.add_argument("--tickers-out", type=Path, default=Path("config/sp500_hist.txt"))
    a = ap.parse_args()
    with urllib.request.urlopen(URL, timeout=60) as r:
        raw = pl.read_csv(io.BytesIO(r.read()), try_parse_dates=True)
    long = (raw.with_columns(pl.col("tickers").str.split(","))
               .explode("tickers")
               .select(pl.col("date").alias("snap_date"),
                       pl.col("tickers").str.strip_chars().str.replace_all(r"\.", "-").alias("ticker"))
               .filter(pl.col("ticker") != ""))
    # keep the last snapshot before `since` so membership on `since` is defined
    first = long.filter(pl.col("snap_date") <= a.since)["snap_date"].max() or long["snap_date"].min()
    long = long.filter(pl.col("snap_date") >= first).sort("snap_date", "ticker")
    if long.group_by("snap_date").len()["len"].median() < 450:
        raise SystemExit("membership file looks wrong, refusing to write")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    long.write_parquet(a.out)
    tickers = sorted(long["ticker"].unique().to_list())
    a.tickers_out.write_text("# every S&P 500 member since %s (point-in-time source: fja05680/sp500)\n%s\n"
                             % (a.since, "\n".join(tickers)))
    print(f"members: {long['snap_date'].n_unique()} snapshots {first} -> {long['snap_date'].max()}, "
          f"{len(tickers)} distinct tickers -> {a.tickers_out}")


if __name__ == "__main__":
    main()
