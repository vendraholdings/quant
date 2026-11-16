"""Write current S&P 500 constituents to config/sp500.txt (yfinance ticker format, e.g. BRK-B).

    python -m quant.data.universe [--out config/sp500.txt]

Note: current members only, so backtests on it carry survivorship bias.
"""
from __future__ import annotations

import argparse
import io
import urllib.request
from pathlib import Path

import polars as pl

URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("config/sp500.txt"))
    a = ap.parse_args()
    with urllib.request.urlopen(URL, timeout=30) as r:
        df = pl.read_csv(io.BytesIO(r.read()))
    syms = sorted({s.strip().replace(".", "-") for s in df["Symbol"].to_list() if s and s.strip()})
    if len(syms) < 450:
        raise SystemExit(f"only {len(syms)} symbols, refusing to overwrite")
    a.out.write_text("# S&P 500 constituents (current), source: datasets/s-and-p-500-companies\n" + "\n".join(syms) + "\n")
    print(f"wrote {len(syms)} tickers to {a.out}")


if __name__ == "__main__":
    main()
