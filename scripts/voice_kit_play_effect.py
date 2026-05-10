#!/usr/bin/env python3
"""Generate and play a short local sound effect through ALSA."""

import argparse
import math
import struct
import subprocess
import wave


def write_effect(path: str, rate: int = 44100) -> None:
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)

        frames = []
        duration = 0.5
        total = int(rate * duration)
        for i in range(total):
            t = i / rate
            freq = 660 if t < duration / 2 else 880
            envelope = min(1.0, i / (rate * 0.03), (total - i) / (rate * 0.05))
            sample = int(13000 * envelope * math.sin(2 * math.pi * freq * t))
            frames.append(struct.pack("<h", sample))

        wav.writeframes(b"".join(frames))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", "-D", default="default", help="ALSA PCM device, e.g. default or plughw:0,0")
    parser.add_argument("--path", default="/tmp/voice-kit-test.wav", help="WAV path to create and play")
    args = parser.parse_args()

    write_effect(args.path)
    subprocess.run(["aplay", "-D", args.device, args.path], check=True)


if __name__ == "__main__":
    main()
