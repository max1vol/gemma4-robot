#!/usr/bin/env bash
set -euo pipefail

# Chrome was too slow for camera preview on the Pi 3. Keep this service as a
# cleanup/idle process so systemd does not relaunch the old Chromium kiosk.

ROOT="${ROOT:-$HOME/gemma4-robot}"
PORT="${PORT:-8765}"
DISPLAY_NUM="${DISPLAY_NUM:-0}"
FOREGROUND=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --foreground)
      FOREGROUND=1
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
  shift
done

HTTP_PID="/tmp/gemma4-kiosk-http.pid"
X_PID="/tmp/gemma4-kiosk-x.pid"

stop_pid_file() {
  local pid_file="$1"
  if [ -f "$pid_file" ]; then
    local pid
    pid="$(cat "$pid_file" 2>/dev/null || true)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      sleep 1
    fi
    rm -f "$pid_file"
  fi
}

cleanup_kiosk() {
  stop_pid_file "$HTTP_PID"
  stop_pid_file "$X_PID"
  pkill -TERM -u "$(id -u)" -f "chromium.*chromium-kiosk-profile" 2>/dev/null || true
  pkill -TERM -u "$(id -u)" -f "openbox" 2>/dev/null || true
  pkill -TERM -u "$(id -u)" -f "Xorg :$DISPLAY_NUM" 2>/dev/null || true
  pkill -TERM -u "$(id -u)" -f "startx /tmp/gemma4-kiosk-xinitrc" 2>/dev/null || true
  pkill -TERM -u "$(id -u)" -f "python3 -m http.server $PORT" 2>/dev/null || true
  sleep 2
  pkill -KILL -u "$(id -u)" -f "chromium.*chromium-kiosk-profile" 2>/dev/null || true
  pkill -KILL -u "$(id -u)" -f "openbox" 2>/dev/null || true
  pkill -KILL -u "$(id -u)" -f "Xorg :$DISPLAY_NUM" 2>/dev/null || true
  pkill -KILL -u "$(id -u)" -f "startx /tmp/gemma4-kiosk-xinitrc" 2>/dev/null || true
  pkill -KILL -u "$(id -u)" -f "python3 -m http.server $PORT" 2>/dev/null || true
  rm -f "/tmp/.X${DISPLAY_NUM}-lock"
}

cleanup_kiosk

echo "chrome_kiosk=disabled"
echo "camera_preview=double-click HAT button"
echo "root=$ROOT"

if [ "$FOREGROUND" -eq 1 ]; then
  trap 'cleanup_kiosk; exit 0' INT TERM
  while true; do
    sleep 3600 &
    wait "$!" || true
  done
fi
