#!/bin/bash
# Pause/resume Ollama models to free unified memory for training.
#   scripts/model_pause.sh off   -> remember loaded models, unload them
#   scripts/model_pause.sh on    -> reload the remembered models
#   scripts/model_pause.sh run <cmd...> -> off, run cmd, on (always reloads)
# Reload keep_alive: $RELOAD_KEEP_ALIVE, else Ollama's own OLLAMA_KEEP_ALIVE, else 60m.
set -euo pipefail
API=${OLLAMA_HOST:-http://127.0.0.1:11434}
SAVE="$HOME/.quant_paused_models"
KEEP=${RELOAD_KEEP_ALIVE:-$(launchctl getenv OLLAMA_KEEP_ALIVE 2>/dev/null || true)}
KEEP=${KEEP:-60m}
# Ollama wants a number (seconds; negative = forever) or a duration string like "60m"
if [[ "$KEEP" =~ ^-?[0-9]+$ ]]; then KA=$KEEP; else KA="\"$KEEP\""; fi

off() {
  local names
  names=$(curl -s "$API/api/ps" | /usr/bin/python3 -c "import sys,json;[print(m['name']) for m in json.load(sys.stdin)['models']]")
  # never overwrite a saved list with an empty one (e.g. second 'off' in a row)
  [ -n "$names" ] && echo "$names" > "$SAVE"
  while read -r m; do [ -n "$m" ] || continue
    curl -s "$API/api/generate" -d "{\"model\":\"$m\",\"keep_alive\":0}" >/dev/null && echo "unloaded $m"
  done <<< "$names"
}
on() {
  [ -s "$SAVE" ] || { echo "nothing to reload"; return 0; }
  while read -r m; do [ -n "$m" ] || continue
    curl -s "$API/api/generate" -d "{\"model\":\"$m\",\"keep_alive\":$KA}" >/dev/null && echo "reloaded $m (keep_alive $KEEP)"
  done < "$SAVE"
  rm -f "$SAVE"
}
case "${1:-}" in
  off) off ;; on) on ;;
  run) shift; off; trap on EXIT; "$@" ;;
  *) echo "usage: $0 off|on|run <cmd...>"; exit 1 ;;
esac
