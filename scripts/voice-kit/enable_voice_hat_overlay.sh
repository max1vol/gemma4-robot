#!/bin/sh
set -eu

CONFIG=/boot/firmware/config.txt
OVERLAY='dtoverlay=googlevoicehat-soundcard'

sudo test -f "$CONFIG"
sudo cp "$CONFIG" "$CONFIG.bak.$(date +%Y%m%d-%H%M%S)"

if sudo grep -qxF "$OVERLAY" "$CONFIG"; then
  echo "$OVERLAY is already present in $CONFIG"
else
  echo "$OVERLAY" | sudo tee -a "$CONFIG" >/dev/null
  echo "Added $OVERLAY to $CONFIG"
fi

echo "Reboot with: sudo reboot"
