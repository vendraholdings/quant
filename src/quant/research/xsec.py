"""Cross-sectional stock ranking: features -> walk-forward LightGBM / MLX MLP -> long-short backtest.

    python -m quant.research.xsec panel  --tickers-file config/sp500.txt
    python -m quant.research.xsec sweep  --model lgb --deadline <unix ts> --workers 6 --threads 4
    python -m quant.research.xsec sweep  --model mlx --deadline <unix ts> --workers 2
    python -m quant.research.xsec report

Target: next-21-trading-day return, ranked per date. Expanding-window yearly refits with a
40-calendar-day embargo between train and test. Long top decile / short bottom decile, rebalanced
every 21 trading days, 10 bps per side.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import polars as pl

H = 21           # horizon and rebalance period, trading days
EMBARGO = 40     # calendar days between last train target and first test date
FEATS = ["mom_5", "mom_21", "mom_63", "mom_126", "mom_252", "mom_12_1", "vol_21", "vol_63",
         "hi_52", "maxr_21", "skew_63", "dv_surge", "illiq_21", "size"]


# ---------------------------------------------------------------- panel
EARN_FEATS = ["e_days_since", "e_days_to", "e_last_surprise", "e_last_move", "e_exp_move"]
META = ("date", "di", "ticker", "y", "fwd")


def add_earnings(lf: pl.LazyFrame, events: Path, earnings_dir: Path) -> pl.LazyFrame:
    """As-of joins only: last event whose reaction is complete (avail_date <= date) and next scheduled date."""
    ev = (pl.scan_parquet(events).select("ticker", "avail_date", "surprise_pct", "ab_move", "exp_move")
            .rename({"surprise_pct": "e_last_surprise", "ab_move": "e_last_move", "exp_move": "e_exp_move"})
            .sort("avail_date"))
    nxt = pl.scan_parquet(earnings_dir / "*.parquet").select("ticker", pl.col("edate").alias("next_edate")).sort("next_edate")
    days = lambda a, b: (a - b).dt.total_days()  # noqa: E731
    return (lf.sort("date")
              .join_asof(ev, left_on="date", right_on="avail_date", by="ticker", strategy="backward")
              .join_asof(nxt, left_on="date", right_on="next_edate", by="ticker", strategy="forward")
              .with_columns(days(pl.col("date"), pl.col("avail_date")).alias("e_days_since"),
                            days(pl.col("next_edate"), pl.col("date")).alias("e_days_to"))
              .with_columns(pl.when(pl.col("e_days_since") <= 120).then(pl.col(c)).alias(c)
                            for c in ["e_days_since", "e_last_surprise", "e_last_move", "e_exp_move"])
              .with_columns(pl.when(pl.col("e_days_to") <= 100).then(pl.col("e_days_to")).alias("e_days_to"))
              .drop("avail_date", "next_edate"))


def pit_filter(lf: pl.LazyFrame, members: Path) -> pl.LazyFrame:
    """Keep (ticker, date) only if the ticker was an S&P 500 member on that date."""
    m = pl.scan_parquet(members)
    snaps = m.select("snap_date").unique().sort("snap_date")
    return (lf.sort("date").join_asof(snaps, left_on="date", right_on="snap_date", strategy="backward")
              .join(m, on=["snap_date", "ticker"], how="semi").drop("snap_date"))


def build_panel(prices_dir: Path, tickers: list[str], min_names: int, events: Path | None = None,
                earnings_dir: Path | None = None, members: Path | None = None) -> pl.DataFrame:
    px, r, ldv = pl.col("adj_close"), pl.col("r1"), pl.col("ldv")
    lf = (pl.scan_parquet(prices_dir / "*.parquet").filter(pl.col("ticker").is_in(tickers))
          .sort("ticker", "date")
          .with_columns(px.pct_change().over("ticker").alias("r1"),
                        (pl.col("close") * pl.col("volume")).clip(1).log().alias("ldv"))
          .with_columns(
              *[(px / px.shift(n) - 1).over("ticker").alias(f"mom_{n}") for n in (5, 21, 63, 126, 252)],
              (px.shift(21) / px.shift(252) - 1).over("ticker").alias("mom_12_1"),
              r.rolling_std(21).over("ticker").alias("vol_21"),
              r.rolling_std(63).over("ticker").alias("vol_63"),
              (px / px.rolling_max(252) - 1).over("ticker").alias("hi_52"),
              r.rolling_max(21).over("ticker").alias("maxr_21"),
              r.rolling_skew(63).over("ticker").alias("skew_63"),
              (ldv.rolling_mean(5) - ldv.rolling_mean(63)).over("ticker").alias("dv_surge"),
              (r.abs() / ldv.exp()).rolling_mean(21).over("ticker").log().alias("illiq_21"),
              ldv.rolling_mean(63).over("ticker").alias("size"),
              (px.shift(-H) / px - 1).over("ticker").alias("fwd"))
          .filter(pl.col("close") > 3)
          .drop_nulls(FEATS + ["fwd"])
          .filter(pl.all_horizontal([pl.col(f).is_finite() for f in FEATS] + [pl.col("fwd").is_finite()])))
    feats = list(FEATS)
    if members:
        lf = pit_filter(lf, members)
    if events and earnings_dir:
        lf, feats = add_earnings(lf, events, earnings_dir), feats + EARN_FEATS
    # rank-normalise per date to [-0.5, 0.5]; missing earnings info -> 0 (cross-sectional median)
    lf = (lf.filter(pl.len().over("date") >= min_names)
            .with_columns(*[((pl.col(f).rank() / pl.col(f).count()).over("date") - 0.5).fill_null(0.0).fill_nan(0.0)
                            .cast(pl.Float32).alias(f) for f in feats],
                          ((pl.col("fwd").rank() / pl.len()).over("date") - 0.5).cast(pl.Float32).alias("y"))
            .with_columns(pl.col("date").rank("dense").cast(pl.Int32).alias("di"))
            .select("date", "di", "ticker", *feats, "y", "fwd"))
    return lf.collect().sort("date", "ticker")


# ---------------------------------------------------------------- models
LGB_SPACE = dict(lr=[0.01, 0.02, 0.05, 0.1], leaves=[7, 15, 31, 63, 127], min_leaf=[100, 500, 2000, 5000],
                 ff=[0.5, 0.7, 0.9], bf=[0.5, 0.7, 0.9], l2=[0, 1, 10, 100], trees=[100, 300, 600, 1000],
                 obj=["regression", "huber"])
MLX_SPACE = dict(hidden=[64, 128, 256, 512], depth=[2, 3, 4], dropout=[0.0, 0.1, 0.3], lr=[1e-3, 3e-4, 1e-4],
                 wd=[0.0, 1e-4, 1e-2], epochs=[3, 5, 10], batch=[1024, 4096])


def fit_lgb(cfg, Xtr, ytr, Xte, threads):
    import lightgbm as lgb
    p = dict(objective=cfg["obj"], learning_rate=cfg["lr"], num_leaves=cfg["leaves"],
             min_data_in_leaf=cfg["min_leaf"], feature_fraction=cfg["ff"], bagging_fraction=cfg["bf"],
             bagging_freq=1, lambda_l2=cfg["l2"], num_threads=threads, verbose=-1, seed=cfg["seed"])
    return lgb.train(p, lgb.Dataset(Xtr, ytr), cfg["trees"]).predict(Xte, num_threads=threads)


def fit_mlx(cfg, Xtr, ytr, Xte, threads):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    class MLP(nn.Module):
        def __init__(self, d, h, depth, p):
            super().__init__()
            self.layers = [nn.Linear(d if i == 0 else h, h) for i in range(depth)]
            self.drop, self.out = nn.Dropout(p), nn.Linear(h, 1)

        def __call__(self, x):
            for layer in self.layers:
                x = self.drop(nn.gelu(layer(x)))
            return self.out(x).squeeze(-1)

    mx.random.seed(cfg["seed"])
    model = MLP(Xtr.shape[1], cfg["hidden"], cfg["depth"], cfg["dropout"])
    opt = optim.AdamW(learning_rate=cfg["lr"], weight_decay=cfg["wd"])
    step = nn.value_and_grad(model, lambda m, x, y: nn.losses.mse_loss(m(x), y))
    X, Y, n, bs = mx.array(Xtr), mx.array(ytr), len(ytr), cfg["batch"]
    model.train()
    for _ in range(cfg["epochs"]):
        perm = mx.random.permutation(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            _, g = step(model, X[idx], Y[idx])
            opt.update(model, g)
            mx.eval(model.parameters(), opt.state)
    model.eval()
    return np.array(model(mx.array(Xte)))


FIT = {"lgb": fit_lgb, "mlx": fit_mlx}
SPACE = {"lgb": LGB_SPACE, "mlx": MLX_SPACE}


# ---------------------------------------------------------------- backtest
def backtest(d: pl.DataFrame, cost_bps: float = 10.0) -> pl.DataFrame:
    """d: date, ticker, pred, fwd on rebalance dates -> per-period gross/net long-short returns and IC."""
    d = (d.with_columns(pl.col("pred").rank("ordinal").over("date").alias("rk"), pl.len().over("date").alias("n"))
          .with_columns((pl.col("n") * 0.1).floor().clip(lower_bound=1).alias("k")))
    is_long, is_short = pl.col("rk") > pl.col("n") - pl.col("k"), pl.col("rk") <= pl.col("k")
    g = (d.group_by("date").agg(
             pl.col("ticker").filter(is_long).alias("long"),
             pl.col("ticker").filter(is_short).alias("short"),
             (pl.col("fwd").filter(is_long).mean() - pl.col("fwd").filter(is_short).mean()).alias("gross"),
             pl.corr("pred", "fwd", method="spearman").fill_nan(0.0).alias("ic"))  # constant pred -> IC 0
         .sort("date"))
    costs, pl_, ps = [], set(), set()
    for L, S in zip(g["long"].to_list(), g["short"].to_list()):
        L, S = set(L), set(S)
        turn = (len(L - pl_) / max(len(L), 1)) + (len(S - ps) / max(len(S), 1))
        costs.append(2 * turn * cost_bps / 1e4)  # sell old + buy new, both legs
        pl_, ps = L, S
    return g.drop("long", "short").with_columns(pl.Series("cost", costs), (pl.col("gross") - pl.Series(costs)).alias("net"))


def metrics(bt: pl.DataFrame) -> dict:
    ppy = 252 / H
    out = {"periods": bt.height, "mean_ic": float(bt["ic"].mean()), "ic_ir": float(bt["ic"].mean() / bt["ic"].std() * math.sqrt(ppy))}
    for k in ("gross", "net"):
        r = bt[k].to_numpy()
        eq = np.cumprod(1 + r)
        out |= {f"sharpe_{k}": float(r.mean() / r.std(ddof=1) * math.sqrt(ppy)),
                f"cagr_{k}": float(eq[-1] ** (ppy / len(r)) - 1),
                f"maxdd_{k}": float((eq / np.maximum.accumulate(eq) - 1).min())}
    return out


# ---------------------------------------------------------------- sweep
_P: dict | None = None


def _load(panel: str) -> dict:
    global _P
    if _P is None:
        df = pl.read_parquet(panel)
        feats = [c for c in df.columns if c not in META]  # whatever features the panel was built with
        _P = dict(X=df.select(feats).to_numpy().astype(np.float32), y=df["y"].to_numpy(),
                  day=df["date"].cast(pl.Int32).to_numpy(), di=df["di"].to_numpy(),
                  year=df["date"].dt.year().to_numpy(), meta=df.select("date", "ticker", "fwd"))
    return _P


def split(P: dict, yr: int) -> tuple[np.ndarray, np.ndarray]:
    """Walk-forward masks for test year yr. Test = rebalance days in yr. Train = every 5th day ending EMBARGO
    calendar days before yr starts: targets look H trading days ahead, so no train target overlaps the test year."""
    te = (P["year"] == yr) & (P["di"] % H == 0)
    start = P["day"][P["year"] == yr].min()
    tr = (P["day"] < start - EMBARGO) & (P["di"] % 5 == 0)  # every 5th day: fewer overlapping targets
    return tr, te


def run_config(model: str, cfg: dict, panel: str, out_dir: str, first_year: int, deadline: float, threads: int):
    P = _load(panel)
    parts = []
    t0 = time.time()
    for yr in sorted(set(P["year"][P["year"] >= first_year].tolist())):
        if time.time() > deadline:
            return None  # out of time: drop partial config
        tr, te = split(P, yr)
        if te.sum() == 0 or tr.sum() < 1000:
            continue
        pred = FIT[model](cfg, P["X"][tr], P["y"][tr], P["X"][te], threads)
        parts.append(P["meta"].filter(pl.Series(te)).with_columns(pl.Series("pred", pred, dtype=pl.Float64)))
    preds = pl.concat(parts)
    m = metrics(backtest(preds)) | {"model": model, "id": cfg["id"], "cfg": cfg, "secs": round(time.time() - t0, 1)}
    preds.write_parquet(Path(out_dir) / model / f"{cfg['id']}.parquet")
    return m


def sample_cfg(model: str, rng: random.Random) -> dict:
    cfg = {k: rng.choice(v) for k, v in SPACE[model].items()} | {"seed": rng.randrange(10_000)}
    cfg["id"] = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]
    return cfg


def sweep(a) -> None:
    out = Path(a.results)
    (out / a.model).mkdir(parents=True, exist_ok=True)
    log = out / f"{a.model}.jsonl"
    done = {json.loads(line)["id"] for line in log.read_text().splitlines()} if log.exists() else set()
    rng, n_new = random.Random(time.time_ns()), 0
    with ProcessPoolExecutor(a.workers) as ex:
        running = set()
        while True:
            while len(running) < a.workers and time.time() < a.deadline and n_new + len(running) < a.max_configs:
                cfg = sample_cfg(a.model, rng)
                if cfg["id"] not in done:
                    done.add(cfg["id"])
                    running.add(ex.submit(run_config, a.model, cfg, a.panel, a.results, a.first_year, a.deadline, a.threads))
            if not running:
                break
            fin, running = wait(running, return_when=FIRST_COMPLETED)
            for f in fin:
                m = f.result()
                if m:
                    n_new += 1
                    with log.open("a") as fh:
                        fh.write(json.dumps(m) + "\n")
                    print(f"[{a.model}] {n_new} {m['id']} sharpe_net={m['sharpe_net']:.2f} ic={m['mean_ic']:.4f} {m['secs']}s", flush=True)


# ---------------------------------------------------------------- report
def deflated_sharpe(r: np.ndarray, trial_srs: np.ndarray) -> tuple[float, float]:
    """Bailey & Lopez de Prado. r: per-period returns of the chosen strategy; trial_srs: per-period SR of all trials."""
    from scipy.stats import kurtosis, norm, skew
    n, sr = len(trial_srs), r.mean() / r.std(ddof=1)
    g = 0.5772156649
    sr0 = math.sqrt(np.var(trial_srs, ddof=1)) * ((1 - g) * norm.ppf(1 - 1 / n) + g * norm.ppf(1 - 1 / (n * math.e))) if n > 1 else 0.0
    den = math.sqrt(1 - skew(r) * sr + (kurtosis(r, fisher=False) - 1) / 4 * sr ** 2)
    return float(norm.cdf((sr - sr0) * math.sqrt(len(r) - 1) / den)), sr0 * math.sqrt(252 / H)


def mlx_bootstrap(r: np.ndarray, n_boot: int = 1_000_000, block: int = 6, chunk: int = 100_000) -> np.ndarray:
    """Stationary-ish block bootstrap of annualised Sharpe on the GPU."""
    import mlx.core as mx
    R, n = mx.array(r.astype(np.float32)), len(r)
    nb, out = math.ceil(n / block), []
    off = mx.arange(block)
    for _ in range(n_boot // chunk):
        starts = mx.random.randint(0, n - block + 1, (chunk, nb))
        s = R[(starts[..., None] + off).reshape(chunk, -1)[:, :n]]
        out.append(np.array(s.mean(1) / mx.sqrt(s.var(1, ddof=1)) * math.sqrt(252 / H)))
    return np.concatenate(out)


def report(a) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = Path(a.results)
    rows = [json.loads(line) for m in ("lgb", "mlx") if (res / f"{m}.jsonl").exists()
            for line in (res / f"{m}.jsonl").read_text().splitlines()]
    if not rows:
        raise SystemExit("no results")
    tab = pl.DataFrame([{k: v for k, v in r.items() if k != "cfg"} for r in rows]).sort("sharpe_net", descending=True)
    bts, lines = {}, []
    ppy = 252 / H
    trial_srs = tab["sharpe_net"].to_numpy() / math.sqrt(ppy)

    def preds(model, i):
        return pl.read_parquet(res / model / f"{i}.parquet")

    for model in ("lgb", "mlx"):
        sub = tab.filter(pl.col("model") == model)
        if sub.height:
            bts[f"best {model}"] = backtest(preds(model, sub["id"][0]))
    # blend: rank-average predictions of top-5 configs of each model (selection is in-sample, see DSR)
    top = [(r["model"], r["id"]) for r in tab.group_by("model").head(5).iter_rows(named=True)]
    blend = None
    for model, i in top:
        p = preds(model, i).with_columns(((pl.col("pred").rank() / pl.len()).over("date")).alias(i))
        blend = p.select("date", "ticker", "fwd", i) if blend is None else blend.join(p.select("date", "ticker", i), on=["date", "ticker"])
    if blend is not None and len(top) > 1:
        bts["blend top-5s"] = backtest(blend.with_columns(pl.mean_horizontal([i for _, i in top]).alias("pred")))

    spy_path = Path(a.prices_dir) / "SPY.parquet"
    best_name = max(bts, key=lambda k: metrics(bts[k])["sharpe_net"])
    best_r = bts[best_name]["net"].to_numpy()
    dsr, sr0 = deflated_sharpe(best_r, trial_srs)
    t0 = time.time()
    boot = mlx_bootstrap(best_r)
    boot_s = time.time() - t0

    lines += ["# Overnight cross-sectional ML run", "",
              f"Trials: {len(rows)} ({(tab['model'] == 'lgb').sum()} LightGBM, {(tab['model'] == 'mlx').sum()} MLX MLP). "
              f"Universe: current S&P 500 members (survivorship-biased: delisted names missing, so absolute returns are flattering). "
              f"Long top / short bottom decile, {H}-day rebalance, 10 bps per side.", "",
              "## Headline", "", "| strategy | Sharpe net | Sharpe gross | CAGR net | max DD net | mean IC |", "|---|---|---|---|---|---|"]
    for k, bt in bts.items():
        m = metrics(bt)
        lines.append(f"| {k} | {m['sharpe_net']:.2f} | {m['sharpe_gross']:.2f} | {m['cagr_net']:.1%} | {m['maxdd_net']:.1%} | {m['mean_ic']:.4f} |")
    lines += ["", f"## Is the best one real? ({best_name})", "",
              f"- Deflated Sharpe probability: **{dsr:.1%}** (expected best-of-{len(rows)} Sharpe by pure luck: {sr0:.2f})",
              f"- Block-bootstrap Sharpe (1M resamples on GPU in {boot_s:.1f}s): 5% {np.percentile(boot, 5):.2f}, "
              f"median {np.median(boot):.2f}, 95% {np.percentile(boot, 95):.2f}; P(Sharpe <= 0) = {(boot <= 0).mean():.1%}", "",
              "## Top 10 configs", "", "| model | id | Sharpe net | IC | IC IR | maxDD | secs |", "|---|---|---|---|---|---|---|"]
    for r in tab.head(10).iter_rows(named=True):
        lines.append(f"| {r['model']} | {r['id']} | {r['sharpe_net']:.2f} | {r['mean_ic']:.4f} | {r['ic_ir']:.2f} | {r['maxdd_net']:.1%} | {r['secs']} |")
    best_cfgs = {r["id"]: r["cfg"] for r in rows}
    lines += ["", "## Best configs", ""] + [f"- {m} `{i}`: `{json.dumps(best_cfgs[i])}`" for m, i in top[:2]]

    fig, ax = plt.subplots(figsize=(10, 5))
    for k, bt in bts.items():
        ax.plot(bt["date"].to_list(), np.cumprod(1 + bt["net"].to_numpy()), label=k)
    if spy_path.exists():
        dates = bts[best_name]["date"]
        spy = pl.read_parquet(spy_path).sort("date").with_columns((pl.col("adj_close").shift(-H) / pl.col("adj_close") - 1).alias("f"))
        s = dates.to_frame().join(spy.select("date", "f"), on="date", how="left")["f"].fill_null(0).to_numpy()
        ax.plot(dates.to_list(), np.cumprod(1 + s), label="SPY buy & hold", ls="--", c="grey")
    ax.set_yscale("log"); ax.set_title("Growth of $1, net of costs"); ax.legend(); ax.grid(alpha=.3)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=120, bbox_inches="tight")
    lines += ["", f"![equity]({out.with_suffix('.png').name})", ""]
    out.write_text("\n".join(lines))
    print("\n".join(lines[:16]))


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("panel")
    p.add_argument("--tickers-file", default="config/sp500.txt")
    p.add_argument("--prices-dir", default="data/prices")
    p.add_argument("--out", default="data/panel.parquet")
    p.add_argument("--min-names", type=int, default=100)
    p.add_argument("--events", type=Path, help="data/earnings_events.parquet -> adds earnings features")
    p.add_argument("--earnings-dir", type=Path, default=Path("data/earnings"))
    p.add_argument("--members", type=Path, help="data/members.parquet -> point-in-time S&P filter")
    c = sp.add_parser("compare")
    c.add_argument("dirs", nargs="+")
    s = sp.add_parser("sweep")
    s.add_argument("--model", choices=["lgb", "mlx"], required=True)
    s.add_argument("--panel", default="data/panel.parquet")
    s.add_argument("--results", default="data/results")
    s.add_argument("--deadline", type=float, default=time.time() + 3600)
    s.add_argument("--workers", type=int, default=6)
    s.add_argument("--threads", type=int, default=4)
    s.add_argument("--first-year", type=int, default=2006)
    s.add_argument("--max-configs", type=int, default=10**9)
    r = sp.add_parser("report")
    r.add_argument("--results", default="data/results")
    r.add_argument("--prices-dir", default="data/prices")
    r.add_argument("--out", default="reports/overnight.md")
    a = ap.parse_args()
    if a.cmd == "panel":
        from quant.data.prices import read_tickers
        df = build_panel(Path(a.prices_dir), read_tickers(Path(a.tickers_file)), a.min_names,
                         a.events, a.earnings_dir if a.events else None, a.members)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(a.out)
        print(f"panel {df.shape} {df['date'].min()} -> {df['date'].max()} names/date {df.group_by('date').len()['len'].median()}")
    elif a.cmd == "sweep":
        sweep(a)
    elif a.cmd == "compare":  # same search space, different data: compare the whole trial distributions
        print("| results | model | trials | median Sharpe net | 90th pct | best | median IC |\n|---|---|---|---|---|---|---|")
        for d in a.dirs:
            for m in ("lgb", "mlx"):
                f = Path(d) / f"{m}.jsonl"
                if f.exists():
                    t = pl.read_ndjson(f)
                    s = t["sharpe_net"]
                    print(f"| {d} | {m} | {t.height} | {s.median():.2f} | {s.quantile(.9):.2f} | {s.max():.2f} | {t['mean_ic'].median():.4f} |")
    else:
        report(a)


if __name__ == "__main__":
    main()
