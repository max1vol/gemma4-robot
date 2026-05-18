from __future__ import annotations

import argparse
import json
import wave
from dataclasses import dataclass
from pathlib import Path


SAMPLE_RATE = 24_000


@dataclass(frozen=True)
class Phrase:
    phrase_id: str
    text: str
    filename: Path
    sample_rate: int


class PhraseBank:
    def __init__(self, artifact_dir: Path):
        self.artifact_dir = artifact_dir
        manifest_path = artifact_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        self.by_id = {
            entry["phrase_id"]: Phrase(
                phrase_id=entry["phrase_id"],
                text=entry["text"],
                filename=artifact_dir / entry["filename"],
                sample_rate=int(entry["sample_rate"]),
            )
            for entry in manifest
        }

    def select_for_count(self, count: int) -> Phrase:
        if count == 2 and "count_02_nice_rhythm" in self.by_id:
            return self.by_id["count_02_nice_rhythm"]
        if count == 5 and "count_05_keep_going" in self.by_id:
            return self.by_id["count_05_keep_going"]
        if count == 8 and "count_08_good_control" in self.by_id:
            return self.by_id["count_08_good_control"]
        if count == 10 and "set_10_great_set" in self.by_id:
            return self.by_id["set_10_great_set"]

        phrase_id = f"count_{count:02d}"
        if phrase_id not in self.by_id:
            raise KeyError(f"No phrase for count {count}: {phrase_id}")
        return self.by_id[phrase_id]


def read_wav_mono_16(path: Path) -> bytes:
    with wave.open(str(path), "rb") as reader:
        if reader.getframerate() != SAMPLE_RATE:
            raise ValueError(f"{path} has sample rate {reader.getframerate()}, expected {SAMPLE_RATE}")
        if reader.getnchannels() != 1:
            raise ValueError(f"{path} has {reader.getnchannels()} channels, expected mono")
        if reader.getsampwidth() != 2:
            raise ValueError(f"{path} has sample width {reader.getsampwidth()}, expected 16-bit")
        return reader.readframes(reader.getnframes())


def write_demo_set(bank: PhraseBank, output_path: Path, silence_s: float = 0.75) -> list[dict]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    silence = b"\x00\x00" * int(SAMPLE_RATE * silence_s)
    events: list[dict] = []

    with wave.open(str(output_path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE)
        for count in range(1, 11):
            phrase = bank.select_for_count(count)
            audio = read_wav_mono_16(phrase.filename)
            start_s = writer.getnframes() / SAMPLE_RATE
            writer.writeframes(audio)
            writer.writeframes(silence)
            events.append(
                {
                    "count": count,
                    "phrase_id": phrase.phrase_id,
                    "text": phrase.text,
                    "start_s": start_s,
                    "duration_s": len(audio) / 2 / SAMPLE_RATE,
                    "filename": str(phrase.filename),
                }
            )
    return events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phrase-bank", type=Path, required=True)
    parser.add_argument("--demo-wav", type=Path, required=True)
    args = parser.parse_args()

    bank = PhraseBank(args.phrase_bank)
    events = write_demo_set(bank, args.demo_wav)
    sidecar = args.demo_wav.with_suffix(".json")
    sidecar.write_text(json.dumps(events, indent=2) + "\n")
    print(json.dumps({"demo_wav": str(args.demo_wav), "events": events}, indent=2))


if __name__ == "__main__":
    main()
