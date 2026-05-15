#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/gemma4-robot}"
KIOSK_DIR="${KIOSK_DIR:-$ROOT/kiosk}"
PORT="${PORT:-8765}"
DISPLAY_NUM="${DISPLAY_NUM:-0}"
URL="${URL:-http://127.0.0.1:$PORT/}"
CHROMIUM="${CHROMIUM:-/usr/lib/chromium/chromium}"
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
HTTP_LOG="/tmp/gemma4-kiosk-http.log"
X_LOG="/tmp/gemma4-kiosk-x.log"
XINITRC="/tmp/gemma4-kiosk-xinitrc"

mkdir -p "$KIOSK_DIR"

if [ ! -f "$KIOSK_DIR/status.json" ]; then
  cat > "$KIOSK_DIR/status.json" <<'JSON'
{
  "state": "idle",
  "turn": 0,
  "input": "",
  "output": "",
  "error": "",
  "updated_at": ""
}
JSON
fi

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
  sleep 2
  pkill -KILL -u "$(id -u)" -f "chromium.*chromium-kiosk-profile" 2>/dev/null || true
  pkill -KILL -u "$(id -u)" -f "openbox" 2>/dev/null || true
  pkill -KILL -u "$(id -u)" -f "Xorg :$DISPLAY_NUM" 2>/dev/null || true
  rm -f "/tmp/.X${DISPLAY_NUM}-lock"
}

cleanup_kiosk

cat > "$XINITRC" <<EOF
#!/usr/bin/env bash
xset s off
xset -dpms
xset s noblank
unclutter -idle 0.2 -root &
openbox &
exec "$CHROMIUM" \\
  --kiosk \\
  --window-size=1024,600 \\
  --disable-gpu \\
  --disable-gpu-compositing \\
  --disable-gpu-rasterization \\
  --disable-accelerated-2d-canvas \\
  --disable-accelerated-video-decode \\
  --disable-vulkan \\
  --use-gl=swiftshader \\
  --no-first-run \\
  --noerrdialogs \\
  --disable-infobars \\
  --disable-session-crashed-bubble \\
  --disable-features=TranslateUI \\
  --autoplay-policy=no-user-gesture-required \\
  --user-data-dir="$ROOT/chromium-kiosk-profile" \\
  "$URL"
EOF
chmod +x "$XINITRC"

nohup python3 -m http.server "$PORT" --bind 127.0.0.1 --directory "$KIOSK_DIR" > "$HTTP_LOG" 2>&1 &
echo "$!" > "$HTTP_PID"

nohup startx "$XINITRC" -- ":$DISPLAY_NUM" > "$X_LOG" 2>&1 &
echo "$!" > "$X_PID"

echo "kiosk_url=$URL"
echo "http_pid=$(cat "$HTTP_PID")"
echo "x_pid=$(cat "$X_PID")"
echo "http_log=$HTTP_LOG"
echo "x_log=$X_LOG"

if [ "$FOREGROUND" -eq 1 ]; then
  trap 'cleanup_kiosk; exit 0' INT TERM
  wait -n "$(cat "$HTTP_PID")" "$(cat "$X_PID")"
  status="$?"
  cleanup_kiosk
  exit "$status"
fi
