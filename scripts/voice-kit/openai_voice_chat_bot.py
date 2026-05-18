#!/usr/bin/env python3
"""Compatibility launcher for the current Rust voice agent.

This wrapper execs the local Rust harness that records with the USB microphone,
sends raw audio to iPhone-hosted Gemma, uses iPhone Piper TTS, and
plays through HDMI.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    root = Path.home() / "gemma4-robot"
    harness = root / "bin" / "gemma-agent-harness"
    env_file = root / ".env"
    os.execv(
        str(harness),
        [
            str(harness),
            "--env-file",
            str(env_file),
            "voice-bot",
            "--button-source",
            "microbit-serial",
            "--microbit-device",
            "auto",
            "--led-source",
            "none",
            "--playback-device",
            "plughw:vc4hdmi,0",
            "--capture-device",
            "plughw:Camera,0",
            "--sample-rate",
            "48000",
            "--channels",
            "2",
            "--transcription-provider",
            "none",
            "--tts-provider",
            "iphone",
            "--iphone-tts-backend",
            "piper-ryan-high",
            "--startup-greeting",
            "",
        ],
    )
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
