#!/usr/bin/env bash
set -euo pipefail

# Chrome was too slow for camera preview on the Pi 3. This launcher cleans up
# the old Chromium/X kiosk and starts the direct framebuffer Pi camera overlay.

ROOT="${ROOT:-$HOME/gemma4-robot}"
PORT="${PORT:-8765}"
DISPLAY_NUM="${DISPLAY_NUM:-0}"
KIOSK_RENDERER="${KIOSK_RENDERER:-python-overlay}"
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
VISION_PID="/tmp/gemma4-vision-overlay.pid"
VISION_LOG="${VISION_LOG:-/tmp/gemma4-vision-overlay.log}"

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
  stop_pid_file "$VISION_PID"
  pkill -TERM -u "$(id -u)" -f "chromium.*chromium-kiosk-profile" 2>/dev/null || true
  pkill -TERM -u "$(id -u)" -f "openbox" 2>/dev/null || true
  pkill -TERM -u "$(id -u)" -f "Xorg :$DISPLAY_NUM" 2>/dev/null || true
  pkill -TERM -u "$(id -u)" -f "startx /tmp/gemma4-kiosk-xinitrc" 2>/dev/null || true
  pkill -TERM -u "$(id -u)" -f "python3 -m http.server $PORT" 2>/dev/null || true
  pkill -TERM -u "$(id -u)" -f "pose_overlay.py" 2>/dev/null || true
  pkill -TERM -u "$(id -u)" -f "rpicam-hello" 2>/dev/null || true
  pkill -TERM -u "$(id -u)" -f "rpicam-vid" 2>/dev/null || true
  sleep 2
  pkill -KILL -u "$(id -u)" -f "chromium.*chromium-kiosk-profile" 2>/dev/null || true
  pkill -KILL -u "$(id -u)" -f "openbox" 2>/dev/null || true
  pkill -KILL -u "$(id -u)" -f "Xorg :$DISPLAY_NUM" 2>/dev/null || true
  pkill -KILL -u "$(id -u)" -f "startx /tmp/gemma4-kiosk-xinitrc" 2>/dev/null || true
  pkill -KILL -u "$(id -u)" -f "python3 -m http.server $PORT" 2>/dev/null || true
  pkill -KILL -u "$(id -u)" -f "pose_overlay.py" 2>/dev/null || true
  pkill -KILL -u "$(id -u)" -f "rpicam-hello" 2>/dev/null || true
  pkill -KILL -u "$(id -u)" -f "rpicam-vid" 2>/dev/null || true
  rm -f "/tmp/.X${DISPLAY_NUM}-lock"
}

hide_console_cursor() {
  # The framebuffer preview covers the screen, but fbcon can still blink its
  # text cursor over the top-left pixels unless the active VT cursor is hidden.
  if command -v setterm >/dev/null 2>&1; then
    setterm -cursor off -blank 0 -powersave off -powerdown 0 < /dev/tty1 > /dev/tty1 2>/dev/null || true
  fi
  printf '\033[?25l' > /dev/tty1 2>/dev/null || true
  if [ -w /sys/class/graphics/fbcon/cursor_blink ]; then
    printf '0\n' > /sys/class/graphics/fbcon/cursor_blink 2>/dev/null || true
  fi
}

cleanup_kiosk
hide_console_cursor

echo "chrome_kiosk=disabled"
echo "camera_preview=pi-camera-$KIOSK_RENDERER"
echo "root=$ROOT"
rm -f "$ROOT/kiosk/vision_command.json"

VISION_ARGS=(
  --width 256
  --height 144
  --framerate 30
  --native-preview
  --preview-width 1024
  --preview-height 504
  --status-height 96
  --status-fps 2
  --pose-width 256
  --pose-height 144
  --pose-fps 2
)

if [ "$KIOSK_RENDERER" = "rpicam-preview" ]; then
  RPICAM_ARGS=(
    -t 0
    --camera 0
    --fullscreen
    --info-text ""
    --viewfinder-width 1024
    --viewfinder-height 576
    --framerate 30
  )
  if [ "$FOREGROUND" -eq 1 ]; then
    trap 'cleanup_kiosk; exit 0' INT TERM
    exec rpicam-hello "${RPICAM_ARGS[@]}"
  fi
  nohup rpicam-hello "${RPICAM_ARGS[@]}" > "$VISION_LOG" 2>&1 &
  echo "$!" > "$VISION_PID"
  exit 0
fi

if [ "$FOREGROUND" -eq 1 ]; then
  trap 'cleanup_kiosk; exit 0' INT TERM
  exec python3 "$ROOT/scripts/vision/pi_camera_pose_overlay.py" "${VISION_ARGS[@]}"
fi

nohup python3 "$ROOT/scripts/vision/pi_camera_pose_overlay.py" "${VISION_ARGS[@]}" > "$VISION_LOG" 2>&1 &
echo "$!" > "$VISION_PID"
