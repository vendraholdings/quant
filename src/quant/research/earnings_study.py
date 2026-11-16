"""Earnings event study: how big are earnings moves, does post-earnings drift (PEAD) exist, and
what would straddles have needed to cost to break even?

    python -m quant.research.earnings_study [--prices-dir data/prices --earnings-dir data/earnings]

Event window: close(t-1) -> close(t+1), t = first trading day on/after the earnings date. This covers both
before-open (reaction on t) and after-close (reaction on t+1) reports without needing the timing.
Abnormal = stock minus SPY over the same window. "Typical move" = mean |abnormal move| of that ticker's
previous 8 events (needs >= 4), i.e. only information available before the event.

Writes data/earnings_events.parquet (used as ranker features) and reports/earnings.md + charts.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import polars as pl

DRIFT = 60          # trading days of post-event drift to track
MARKUPS = np.round(np.arange(0.6, 1.81, 0.1), 2)  # implied move = markup x typical move


def build_events(prices_dir: Path, earnings_dir: Path, min_year: int = 2001) -> tuple[pl.DataFrame, np.ndarray]:
    px = pl.read_parquet(prices_dir / "*.parquet", columns=["ticker", "date", "adj_close"]).drop_nulls().sort("ticker", "date")
    spy = px.filter(pl.col("ticker") == "SPY")
    if spy.height == 0:
        raise SystemExit("SPY prices missing (run prices with config/bench.txt)")
    sd, sp = spy["date"].to_numpy(), spy["adj_close"].to_numpy()
    ev = (pl.read_parquet(earnings_dir / "*.parquet")
            .filter(pl.col("eps_act").is_not_null() & (pl.col("edate").dt.year() >= min_year)))
    evd = {t: g["edate"].to_numpy() for t, g in ev.group_by("ticker", maintain_order=True) for t in [t[0]]}
    evs = {t: g for t, g in ev.group_by("ticker", maintain_order=True) for t in [t[0]]}
    rows, cars = [], []
    for (t,), g in px.group_by("ticker", maintain_order=True):
        if t not in evd:
            continue
        d, p = g["date"].to_numpy(), g["adj_close"].to_numpy()
        e = evs[t]
        idx = np.searchsorted(d, evd[t])
        for j, i in enumerate(idx):
            if i < 1 or i + 1 + DRIFT >= len(d):
                continue
            si = np.searchsorted(sd, d[i - 1:i + 2 + DRIFT])
            if si[-1] >= len(sd) or (sd[si] != d[i - 1:i + 2 + DRIFT]).any():
                continue  # calendar mismatch (e.g. halted stock): skip rather than misalign
            s = sp[si]
            move, smove = p[i + 1] / p[i - 1] - 1, s[2] / s[0] - 1
            car = p[i + 1:i + 2 + DRIFT] / p[i + 1] - s[2:] / s[2]  # cumulative abnormal return after the window
            rows.append((t, e["edate"][j], d[i + 1].item(), move, move - smove, e["surprise_pct"][j]))
            cars.append(car[1:].astype(np.float32))
    df = pl.DataFrame(rows, schema={"ticker": pl.Utf8, "edate": pl.Date, "avail_date": pl.Date, "move": pl.Float64,
                                    "ab_move": pl.Float64, "surprise_pct": pl.Float64}, orient="row")
    df = df.with_row_index("eid").sort("ticker", "edate").with_columns(
        pl.col("ab_move").abs().shift(1).rolling_mean(8, min_samples=4).over("ticker").alias("exp_move"))
    return df, np.stack(cars) if cars else np.zeros((0, DRIFT), np.float32)


def straddle_sim(absm: np.ndarray, expm: np.ndarray, n_boot: int = 1_000_000, size: int = 50, chunk: int = 125_000):
    """Long-straddle return per event ~ (|move| - premium) / premium with premium = markup * typical move.
    Bootstraps random 50-event books on the GPU (MLX) for every markup. Short straddle = -long."""
    import mlx.core as mx
    A, E, n = mx.array(absm.astype(np.float32)), mx.array(expm.astype(np.float32)), len(absm)
    out = {}
    for k in MARKUPS:
        r = (A - k * E) / (k * E)
        means = []
        for _ in range(n_boot // chunk):
            idx = mx.random.randint(0, n, (chunk, size))
            means.append(np.array(r[idx].mean(axis=1)))
        b = np.concatenate(means)
        out[float(k)] = dict(mean=float(np.array(r.mean())), p5=float(np.percentile(b, 5)), p95=float(np.percentile(b, 95)),
                             p_long_wins=float((b > 0).mean()))
    return out


def tstat(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))) if len(x) > 2 else float("nan")


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--prices-dir", type=Path, default=Path("data/prices"))
    ap.add_argument("--earnings-dir", type=Path, default=Path("data/earnings"))
    ap.add_argument("--events-out", type=Path, default=Path("data/earnings_events.parquet"))
    ap.add_argument("--report", type=Path, default=Path("reports/earnings.md"))
    ap.add_argument("--n-boot", type=int, default=1_000_000)
    a = ap.parse_args()

    df, cars = build_events(a.prices_dir, a.earnings_dir)
    a.events_out.parent.mkdir(parents=True, exist_ok=True)
    df.drop("eid").write_parquet(a.events_out)
    cars = cars[df["eid"].to_numpy()]  # align CAR rows with the sorted frame
    absm, expm = df["ab_move"].abs().to_numpy(), df["exp_move"].to_numpy()
    ok = np.isfinite(expm) & (expm > 0)

    # PEAD: surprise quintiles within each calendar quarter
    s = pl.col("surprise_pct").fill_nan(None)  # missing surprises: excluded from rank and from the count
    q = (df.with_columns(pl.col("edate").dt.truncate("1q").alias("qtr"))
           .with_columns(((s.rank() / s.count()).over("qtr") * 5).ceil().clip(1, 5).alias("sq"))["sq"]
           .fill_null(0).to_numpy())
    lines = ["# Earnings event study", "",
             f"{df.height:,} reported earnings events, {df['ticker'].n_unique()} tickers, "
             f"{df['edate'].min()} -> {df['edate'].max()}. Window close(t-1)->close(t+1), abnormal = minus SPY.", "",
             "## How big are earnings moves?", "",
             f"- median |abnormal move| {np.median(absm):.2%}, mean {absm.mean():.2%}, 95th pct {np.percentile(absm, 95):.2%}",
             f"- realized / typical (prior 8 events): median {np.median(absm[ok] / expm[ok]):.2f}, "
             f"share > 2x typical {(absm[ok] / expm[ok] > 2).mean():.1%} (fat tails = short-straddle blowups)", "",
             "## Post-earnings drift by surprise quintile (abnormal return after the window)", "",
             "| quintile | events | +20d | t | +60d | t |", "|---|---|---|---|---|---|"]
    for k in range(1, 6):
        m = q == k
        lines.append(f"| Q{k} | {m.sum():,} | {cars[m, 19].mean():.2%} | {tstat(cars[m, 19]):.1f} | {cars[m, 59].mean():.2%} | {tstat(cars[m, 59]):.1f} |")
    spread = cars[q == 5, 59].mean() - cars[q == 1, 59].mean()
    lines += ["", f"Q5 minus Q1 at +60d: **{spread:.2%}** (before costs). Drift by era:", "",
              "| era | Q5-Q1 +60d |", "|---|---|"]
    yrs = df["edate"].dt.year().to_numpy()
    for lo, hi in ((2001, 2007), (2008, 2013), (2014, 2019), (2020, 2026)):
        e = (yrs >= lo) & (yrs <= hi)
        lines.append(f"| {lo}-{hi} | {cars[e & (q == 5), 59].mean() - cars[e & (q == 1), 59].mean():.2%} |")

    t0 = time.time()
    sim = straddle_sim(absm[ok], expm[ok], a.n_boot)
    secs = time.time() - t0
    be = min(sim, key=lambda k: abs(sim[k]["mean"]))
    lines += ["", "## Straddles: what implied move would have made them fair?", "",
              "Premium = markup x the stock's typical earnings move; return held over the event window "
              f"(ignores bid/ask and vega after the event). {a.n_boot:,} random 50-event books per markup, "
              f"bootstrapped on the GPU with MLX in {secs:.1f}s.", "",
              "| markup | long straddle mean | 5%..95% (50-event book) | P(long book wins) |", "|---|---|---|---|"]
    for k, s in sim.items():
        lines.append(f"| {k:.1f}x | {s['mean']:+.1%} | {s['p5']:+.1%} .. {s['p95']:+.1%} | {s['p_long_wins']:.0%} |")
    lines += ["", f"Break-even markup ~ **{be:.1f}x**: if options price the earnings move above this, selling straddles "
              "had positive expectancy (with fat-tail risk); below it, buying them did. "
              "Next step: compare with real implied moves from the daily option snapshots (data/options).", ""]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for k in range(1, 6):
        ax[0].plot(np.arange(1, DRIFT + 1), cars[q == k].mean(0) * 100, label=f"Q{k}")
    ax[0].set(title="Post-earnings drift by surprise quintile", xlabel="trading days after event", ylabel="CAR %")
    ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].hist(np.clip(absm[ok] / expm[ok], 0, 5), bins=60)
    ax[1].axvline(1, c="k", ls="--"); ax[1].set(title="Realized / typical earnings move", xlabel="ratio")
    a.report.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.report.with_suffix(".png"), dpi=120, bbox_inches="tight")
    lines.append(f"![earnings]({a.report.with_suffix('.png').name})")
    a.report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
