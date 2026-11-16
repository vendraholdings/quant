# Quant research on the Mac Studio

Goal: quant finance + ML research on the Mac Studio (M5 Ultra, 30 CPU / 64 GPU cores, 96 GB).
Runs locally on an Apple Silicon Mac; env managed by `uv` (Python 3.12). Setup: `uv sync`.

## Rules
- A local Ollama LLM may be loaded and holding memory. Only pause it for heavy jobs, which run at night.
- Heavy job = executable script in `jobs/queue/`; `scripts/night.sh` runs it with the model paused, then reloads it.
- **Import lightgbm / sklearn before torch** in the same process (both ship libomp; torch-first segfaults).
- Prefer MLX over torch-MPS for GPU work (~2-4x faster here). LightGBM on 30 threads for tabular.
- Data lives in `data/` (git-ignored, Parquet). Disk: ~588 GB free.

## Steps
1. [x] Env + packages (polars, duckdb, lightgbm, xgboost, torch, mlx, numba, cvxpy, vectorbt, yfinance, jupyterlab)
2. [x] Hardware check (`scripts/check_hw.py`)
3. [x] Daily price data lake (`src/quant/data/prices.py`, universe in `config/universe.txt`)
4. [x] Model pause + night runner (`scripts/model_pause.sh`, `scripts/night.sh`)
5. [ ] Schedule night runner (launchd, ~02:00)
6. [ ] Expand universe to S&P 500 + add FRED macro data
7. [ ] Factor research: momentum/value/low-vol, LightGBM cross-sectional ranker, purged walk-forward CV
8. [ ] Portfolio construction (mean-variance, HRP) with transaction costs
9. [ ] Options: Black-Scholes, MLX Monte Carlo, implied-vol surface
10. [ ] Deep learning on crypto order-book data (MLX)

## Checks (prove it works)
    export PATH="$HOME/.local/bin:$PATH"; cd ~/quant
    uv run python scripts/check_hw.py                 # expect ALL CHECKS PASSED
    uv run python -m quant.data.prices                # expect failed 0
    scripts/model_pause.sh run <cmd>                  # pause model, run cmd, reload
    scripts/night.sh                                  # empty queue -> "queue empty", model untouched

## Baseline numbers (2026-11-16)
torch MPS fp32 21 TFLOPS | MLX fp32 48 / fp16 75-87 TFLOPS | LightGBM 500k x 50, 200 trees: 4.7 s
