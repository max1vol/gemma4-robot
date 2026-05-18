# Raspberry Pi 3 Rebuild Runbook

This project uses a Raspberry Pi 3 reachable as `max@pi3` over Tailscale. The
Pi drives the Google AIY Voice Kit HAT locally and uses Google-hosted Gemma for
the voice agent.

Do not commit secrets. Keep `GEMINI_API_KEY` only in a local file on the Pi.

## 1. Install Raspberry Pi OS

Use Raspberry Pi OS Lite, 64-bit if available. Configure:

- hostname: `pi3`
- user: `max`
- SSH enabled
- Wi-Fi configured if Ethernet is not used

After first boot, update packages:

```sh
sudo apt-get update
sudo apt-get upgrade -y
sudo reboot
```

## 2. Install Tailscale

```sh
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up --ssh
```

From the Mac, verify:

```sh
tailscale ssh max@pi3 'hostname; date; uname -a'
```

## 3. Enable AIY Voice HAT Audio

Install the Voice HAT overlay:

```sh
sudo sh -c "grep -qxF 'dtoverlay=googlevoicehat-soundcard' /boot/firmware/config.txt || echo dtoverlay=googlevoicehat-soundcard >> /boot/firmware/config.txt"
sudo reboot
```

After reboot, verify the sound card:

```sh
aplay -l
arecord -l
cat /proc/asound/cards
```

The expected card looks like:

```text
snd_rpi_googlevoicehat_soundcard
```

On the current setup the HAT is `plughw:1,0` for both playback and capture.
Always verify on a fresh image.

Quick capture test:

```sh
arecord -q -D plughw:1,0 -f S16_LE -r 16000 -c 1 -d 1 /tmp/voicehat-capture-test.wav
file /tmp/voicehat-capture-test.wav
```

## 4. Install Runtime Packages

```sh
sudo apt-get update
sudo apt-get install -y \
  git python3 gpiozero python3-lgpio python3-rpi.gpio \
  alsa-utils chromium xserver-xorg xinit openbox unclutter
```

## 5. Copy Or Clone This Repo

If using GitHub:

```sh
mkdir -p ~/gemma4-robot
git clone <repo-url> ~/gemma4-robot
```

If copying from the Mac:

```sh
tailscale ssh max@pi3 'mkdir -p ~/gemma4-robot'
rsync -av --exclude .git --exclude .env --exclude out ./ max@pi3:~/gemma4-robot/
```

## 6. Add The Gemini API Key

Create the Pi-local env file:

```sh
umask 077
cat > ~/gemma4-robot/.env <<'EOF'
GEMINI_API_KEY=replace-this-on-the-pi
EOF
chmod 600 ~/gemma4-robot/.env
```

This file is intentionally ignored by git.

## 7. Build The Rust Agent On The Mac

Do not compile the Rust agent on the Pi. Build the Linux arm64 binary on the Mac
with Apple container:

```sh
scripts/build_agent_harness_container.sh
```

Then sync `bin/gemma-agent-harness` to the Pi along with the repo files.

## 8. Install Boot Services

Install and enable the voice bot and HDMI camera overlay services:

```sh
sudo cp ~/gemma4-robot/scripts/voice-kit/systemd/gemma-voice-bot.service /etc/systemd/system/
sudo cp ~/gemma4-robot/scripts/voice-kit/systemd/gemma-voice-kiosk.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable gemma-voice-bot.service gemma-voice-kiosk.service
sudo systemctl start gemma-voice-bot.service gemma-voice-kiosk.service
```

Check them:

```sh
systemctl status gemma-voice-bot.service gemma-voice-kiosk.service --no-pager
journalctl -u gemma-voice-bot.service -u gemma-voice-kiosk.service -n 100 --no-pager
```

These services auto-start after reboot.

## 9. Manual Voice Agent Start

```sh
nohup ~/gemma4-robot/bin/gemma-agent-harness \
  --env-file ~/gemma4-robot/.env \
  --model gemma-4-31b-it \
  voice-bot \
  --playback-device plughw:1,0 \
  --capture-device plughw:1,0 \
  > /tmp/gemma_voice_agent.log 2>&1 &
echo $! > /tmp/gemma_voice_agent.pid
```

Check it:

```sh
tail -80 /tmp/gemma_voice_agent.log
ps -fp "$(cat /tmp/gemma_voice_agent.pid)"
```

Interaction model:

- hold the AIY HAT button to record
- release to send the recorded audio to `gemma-4-31b-it`
- tap shorter than `0.35s` to reset the conversation
- boot is silent; model output is written to the display status JSON

The agent writes live display state to:

```text
~/gemma4-robot/kiosk/status.json
```

## 10. Manual HDMI Kiosk Start

Check HDMI mode:

```sh
cat /sys/class/graphics/fb0/virtual_size
kmsprint
```

The current screen is `1024x600`.

Start the kiosk:

```sh
~/gemma4-robot/scripts/voice-kit/start_voice_kiosk.sh
```

Logs:

```text
/tmp/gemma4-vision-overlay.log
```

The kiosk draws directly to `/dev/fb0`; it does not run Chrome or X. It displays:

- the live CSI Pi camera feed from `rpicam-vid --codec yuv420`
- MediaPipe pose landmarks returned by the iPhone bridge
- latest model output from `~/gemma4-robot/kiosk/status.json`
- squat count and coaching state from `~/gemma4-robot/kiosk/vision_state.json`

## 11. Stop Or Restart

With systemd:

```sh
sudo systemctl restart gemma-voice-bot.service gemma-voice-kiosk.service
sudo systemctl stop gemma-voice-bot.service gemma-voice-kiosk.service
```

If started manually, stop the bot:

```sh
kill "$(cat /tmp/gemma_voice_agent.pid)"
```

Stop the kiosk:

```sh
kill "$(cat /tmp/gemma4-kiosk-x.pid)" 2>/dev/null || true
kill "$(cat /tmp/gemma4-kiosk-http.pid)" 2>/dev/null || true
```

Start both again with the commands above.
