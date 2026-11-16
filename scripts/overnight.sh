#!/bin/bash
# Full-machine overnight research run with the local Ollama model paused.
#   scripts/overnight.sh [STOP=06:00] [RELOAD=06:30]
# STOP: no new configs start after this; RELOAD: hard deadline, jobs killed and model reloaded.
# Run detached:  nohup scripts/overnight.sh 06:00 06:30 >/dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.." || exit 1
export PATH="$HOME/.local/bin:$PATH"
STOP=${1:-06:00}; RELOAD=${2:-06:30}
mkdir -p jobs/logs reports
LOG="jobs/logs/overnight-$(date +%Y%m%d-%H%M).log"
exec >>"$LOG" 2>&1

at_ts() {  # unix time of next HH:MM
  local t; t=$(date -j -f "%Y-%m-%d %H:%M" "$(date +%Y-%m-%d) $1" +%s)
  [ "$t" -le "$(date +%s)" ] && t=$((t + 86400)); echo "$t"
}
STOP_TS=$(at_ts "$STOP"); RELOAD_TS=$(at_ts "$RELOAD")
echo "$(date) start; stop new work $(date -r "$STOP_TS"), reload model by $(date -r "$RELOAD_TS")"

scripts/model_pause.sh off
# independent safety net: even if this script dies, reload the model at RELOAD
# ([q] trick: the pattern must not match this bash -c line itself; spawn_main = pool worker processes)
nohup bash -c "sleep $((RELOAD_TS - $(date +%s))); pkill -f '[q]uant[.]research[.]xsec'; pkill -f '[q]uant/[.]venv/.*spawn_main'; scripts/model_pause.sh on" \
  >>"$LOG" 2>&1 &
SAFETY=$!
trap 'scripts/model_pause.sh on; kill $SAFETY 2>/dev/null' EXIT
caffeinate -i -w $$ &

step() { echo "$(date) == $*"; "$@" || echo "$(date) !! failed: $*"; }

step uv run python -m quant.data.universe
printf "SPY\n" > config/bench.txt
step uv run python -m quant.data.prices --tickers-file config/sp500.txt --start 2000-01-01
step uv run python -m quant.data.prices --tickers-file config/bench.txt --start 2000-01-01
step uv run python -m quant.research.xsec panel

# CPU and GPU sweeps side by side until STOP; leave ~5 min of slack before RELOAD for the report
step uv run python -m quant.research.xsec sweep --model lgb --deadline "$STOP_TS" --workers 6 --threads 4 & P1=$!
step uv run python -m quant.research.xsec sweep --model mlx --deadline "$STOP_TS" --workers 2 & P2=$!
wait $P1 $P2

step uv run python -m quant.research.xsec report
echo "$(date) done"
