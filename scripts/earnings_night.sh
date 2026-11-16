#!/bin/bash
# "Earnings edge" night run with the local Ollama model paused.
#   scripts/earnings_night.sh [STOP=06:00] [RELOAD=06:30]
# A/B: same LightGBM search on the point-in-time universe, A = price features, B = + earnings features.
set -u
cd "$(dirname "$0")/.." || exit 1
export PATH="$HOME/.local/bin:$PATH" OMP_WAIT_POLICY=PASSIVE   # passive: idle OpenMP threads sleep, not spin
STOP=${1:-06:00}; RELOAD=${2:-06:30}
mkdir -p jobs/logs reports
LOG="jobs/logs/earnings-night-$(date +%Y%m%d-%H%M).log"
exec >>"$LOG" 2>&1
at_ts() { local t; t=$(date -j -f "%Y-%m-%d %H:%M:%S" "$(date +%Y-%m-%d) $1:00" +%s)
          [ "$t" -le "$(date +%s)" ] && t=$((t + 86400)); echo "$t"; }
STOP_TS=$(at_ts "$STOP"); RELOAD_TS=$(at_ts "$RELOAD")
echo "$(date) start; stop $(date -r "$STOP_TS"); reload by $(date -r "$RELOAD_TS")"

scripts/model_pause.sh off
nohup bash -c "sleep $((RELOAD_TS - $(date +%s))); pkill -f '[q]uant[.]research[.]'; pkill -f '[q]uant/[.]venv/.*spawn_main'; scripts/model_pause.sh on" >>"$LOG" 2>&1 &
SAFETY=$!
trap 'scripts/model_pause.sh on; kill $SAFETY 2>/dev/null' EXIT
caffeinate -i -w $$ &
step() { echo "$(date) == $*"; "$@" || echo "$(date) !! failed: $*"; }
X="uv run python -m"

step $X quant.data.universe
step $X quant.data.members --since 2000-01-01
cat config/sp500.txt config/sp500_hist.txt | grep -v "^#" | sort -u > config/sp500_all.txt
step $X quant.data.prices --tickers-file config/sp500_all.txt --start 2000-01-01
step $X quant.data.prices --tickers-file config/bench.txt --start 2000-01-01
step $X quant.data.earnings --tickers-file config/sp500_all.txt
step $X quant.research.earnings_study
step $X quant.research.xsec panel --tickers-file config/sp500_all.txt --members data/members.parquet --out data/panel_a.parquet
step $X quant.research.xsec panel --tickers-file config/sp500_all.txt --members data/members.parquet \
     --events data/earnings_events.parquet --out data/panel_b.parquet

S="$X quant.research.xsec sweep --deadline $STOP_TS"
step $S --model lgb --panel data/panel_a.parquet --results data/results_a --workers 7 --threads 2 & P1=$!
step $S --model lgb --panel data/panel_b.parquet --results data/results_b --workers 7 --threads 2 & P2=$!
step $S --model mlx --panel data/panel_b.parquet --results data/results_b --workers 2 & P3=$!
wait $P1 $P2 $P3

step $X quant.research.xsec report --results data/results_b --out reports/earnings_ranker.md
$X quant.research.xsec compare data/results data/results_a data/results_b > reports/ab_compare.md
cat reports/ab_compare.md
echo "$(date) done"
