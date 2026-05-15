#!/usr/bin/env python3
"""Button-triggered sound, record, and playback demo for Google AIY Voice HAT."""

from __future__ import annotations

import argparse
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from gpiozero import Button


DEFAULT_BASE_DIR = Path.home() / "voice-kit-button-demo"
DEFAULT_EFFECT = DEFAULT_BASE_DIR / "sounds" / "ringingsound.wav"
DEFAULT_RECORDING = DEFAULT_BASE_DIR / "recordings" / "latest.wav"
VOICE_CARD_RE = re.compile(r"(google|voice|aiy|sndrpigoogle)", re.IGNORECASE)
CARD_LINE_RE = re.compile(
    r"^card\s+(?P<card>\d+):\s+(?P<short>[^\[]+)\[(?P<long>[^\]]+)\],"
    r"\s+device\s+(?P<device>\d+):\s+(?P<device_name>.+)$"
)


@dataclass(frozen=True)
class AlsaDevice:
    card: int
    device: int
    short: str
    long: str
    device_name: str

    @property
    def plughw(self) -> str:
        return f"plughw:{self.card},{self.device}"

    @property
    def searchable_text(self) -> str:
        return " ".join([self.short, self.long, self.device_name])


def run_text(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        return exc.output


def list_alsa_devices(command: str) -> list[AlsaDevice]:
    output = run_text([command, "-l"])
    devices: list[AlsaDevice] = []
    for line in output.splitlines():
        match = CARD_LINE_RE.match(line.strip())
        if not match:
            continue
        devices.append(
            AlsaDevice(
                card=int(match.group("card")),
                device=int(match.group("device")),
                short=match.group("short").strip(),
                long=match.group("long").strip(),
                device_name=match.group("device_name").strip(),
            )
        )
    return devices


def choose_alsa_device(kind: str, override: str | None) -> str | None:
    if override:
        return override

    command = "aplay" if kind == "playback" else "arecord"
    devices = list_alsa_devices(command)
    if not devices:
        return None

    for device in devices:
        if VOICE_CARD_RE.search(device.searchable_text):
            return device.plughw

    # If there is only one capture device, use it. For playback, avoid silently
    # using HDMI/headphones when the Voice HAT has not appeared.
    if kind == "capture" and len(devices) == 1:
        return devices[0].plughw
    return None


def explain_audio_blocker() -> None:
    print("No Voice HAT ALSA capture/playback pair found yet.", flush=True)
    print("Enable the kernel overlay, reboot, then run this script again:", flush=True)
    print(
        "  sudo sh -c \"grep -qxF 'dtoverlay=googlevoicehat-soundcard' "
        "/boot/firmware/config.txt || echo dtoverlay=googlevoicehat-soundcard "
        ">> /boot/firmware/config.txt\"",
        flush=True,
    )
    print("  sudo reboot", flush=True)


def wait_for_audio(args: argparse.Namespace) -> tuple[str, str]:
    last_notice = 0.0
    while True:
        playback_device = choose_alsa_device("playback", args.playback_device)
        capture_device = choose_alsa_device("capture", args.capture_device)
        if playback_device and capture_device:
            return playback_device, capture_device

        if not args.wait_audio:
            explain_audio_blocker()
            sys.exit(2)

        now = time.monotonic()
        if now - last_notice > 30:
            explain_audio_blocker()
            last_notice = now
        time.sleep(2)


def play_wav(path: Path, playback_device: str) -> None:
    subprocess.run(["aplay", "-q", "-D", playback_device, str(path)], check=True)


def record_wav(path: Path, seconds: int, capture_device: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    attempts = [
        ["-f", "S16_LE", "-r", "16000", "-c", "1"],
        ["-f", "S16_LE", "-r", "16000", "-c", "2"],
        ["-f", "cd"],
    ]

    errors: list[str] = []
    for params in attempts:
        command = ["arecord", "-q", "-D", capture_device, "-d", str(seconds), *params, str(path)]
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode == 0:
            return
        errors.append((result.stderr or result.stdout or "").strip())

    raise RuntimeError("arecord failed: " + " | ".join(error for error in errors if error))


def make_press_handler(args: argparse.Namespace, playback_device: str, capture_device: str):
    busy = threading.Lock()
    effect = Path(args.effect).expanduser()
    recording = Path(args.recording).expanduser()

    def sequence() -> None:
        if not busy.acquire(blocking=False):
            print("Already recording or playing; ignoring button press.", flush=True)
            return
        try:
            print("Button pressed: playing effect.", flush=True)
            play_wav(effect, playback_device)
            print(f"Recording {args.seconds} seconds to {recording}.", flush=True)
            record_wav(recording, args.seconds, capture_device)
            print("Playing recording.", flush=True)
            play_wav(recording, playback_device)
            print("Ready.", flush=True)
        except Exception as exc:
            print(f"Error during button sequence: {exc}", file=sys.stderr, flush=True)
        finally:
            busy.release()

    def start_sequence() -> None:
        threading.Thread(target=sequence, daemon=True).start()

    return start_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--button-gpio", type=int, default=23, help="BCM GPIO for the AIY button")
    parser.add_argument("--seconds", type=int, default=10, help="seconds of audio to record")
    parser.add_argument("--effect", default=str(DEFAULT_EFFECT), help="WAV file played before recording")
    parser.add_argument("--recording", default=str(DEFAULT_RECORDING), help="WAV output path")
    parser.add_argument("--playback-device", help="explicit ALSA playback device, e.g. plughw:2,0")
    parser.add_argument("--capture-device", help="explicit ALSA capture device, e.g. plughw:2,0")
    parser.add_argument("--wait-audio", action="store_true", help="wait until HAT audio appears")
    parser.add_argument("--once", action="store_true", help="run the sequence once without waiting for button")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    effect = Path(args.effect).expanduser()
    if not effect.exists():
        raise SystemExit(f"Effect WAV does not exist: {effect}")

    playback_device, capture_device = wait_for_audio(args)
    print(f"Using playback={playback_device} capture={capture_device}", flush=True)

    handler = make_press_handler(args, playback_device, capture_device)
    if args.once:
        handler()
        while threading.active_count() > 1:
            time.sleep(0.1)
        return

    button = Button(args.button_gpio, pull_up=True, bounce_time=0.05)
    button.when_pressed = handler
    print(f"Listening for button on BCM GPIO {args.button_gpio}. Press Ctrl-C to stop.", flush=True)
    signal.pause()


if __name__ == "__main__":
    main()
