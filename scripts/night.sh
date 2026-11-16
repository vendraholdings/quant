#!/bin/bash
# Night runner: runs every executable in jobs/queue/ (in name order) with the Ollama model paused.
# Empty queue -> exits immediately, model untouched. Finished jobs move to jobs/done/, logs in jobs/logs/.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PATH="$HOME/.local/bin:$PATH"
LOCK=/tmp/quant-night.lock
mkdir "$LOCK" 2>/dev/null || { echo "already running"; exit 0; }
trap "rmdir $LOCK" EXIT
jobs=$(find jobs/queue -type f -perm -u+x ! -name ".*" | sort)
[ -n "$jobs" ] || { echo "$(date) queue empty"; exit 0; }
scripts/model_pause.sh off
for j in $jobs; do
  name=$(basename "$j"); log="jobs/logs/$(date +%Y%m%d-%H%M)-$name.log"
  echo "$(date) start $name" | tee -a jobs/logs/night.log
  "$j" > "$log" 2>&1; rc=$?
  echo "$(date) end $name rc=$rc" | tee -a jobs/logs/night.log
  mv "$j" jobs/done/
done
scripts/model_pause.sh on
