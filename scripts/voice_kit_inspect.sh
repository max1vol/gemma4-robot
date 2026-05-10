#!/usr/bin/env bash
set -euo pipefail

echo "== OS =="
uname -a
cat /etc/os-release 2>/dev/null || true

echo
echo "== ALSA playback devices =="
aplay -l 2>&1 || true

echo
echo "== ALSA capture devices =="
arecord -l 2>&1 || true

echo
echo "== ALSA cards =="
cat /proc/asound/cards 2>/dev/null || true

echo
echo "== ALSA PCM names =="
aplay -L 2>/dev/null | sed -n '1,160p' || true

echo
echo "== AIY Python modules =="
python3 - <<'PY'
import importlib.util

for name in ("aiy", "aiy.voice.audio", "aiy.board", "aiy.leds", "aiy.pins"):
    spec = importlib.util.find_spec(name)
    print(f"{name}: {spec.origin if spec else 'not installed'}")
PY

echo
echo "== AIY directories =="
ls -la /opt/aiy 2>/dev/null || true
ls -la ~/AIY-projects-python 2>/dev/null || true

echo
echo "== Relevant boot config =="
grep -R "google\\|aiy\\|voice\\|dtoverlay\\|i2s\\|snd" \
  /boot/config.txt /boot/firmware/config.txt 2>/dev/null || true
