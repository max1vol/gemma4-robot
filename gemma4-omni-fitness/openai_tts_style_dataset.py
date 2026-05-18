from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import math
import os
import random
import re
import time
import wave
from pathlib import Path
from typing import Any

import requests


OPENAI_TTS_MODEL = "gpt-4o-mini-tts"
OPENAI_WHISPER_MODEL = "whisper-1"
DEFAULT_OUT_DIR = Path("out/gemma4_omni_openai_tts_style_dataset")
SAMPLE_RATE = 24_000
OPENAI_VOICES = [
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "nova",
    "onyx",
    "sage",
    "shimmer",
    "verse",
    "marin",
    "cedar",
]

NUMBER_WORDS = [
    "One",
    "Two",
    "Three",
    "Four",
    "Five",
    "Six",
    "Seven",
    "Eight",
    "Nine",
    "Ten",
    "Eleven",
    "Twelve",
    "Thirteen",
    "Fourteen",
    "Fifteen",
    "Sixteen",
    "Seventeen",
    "Eighteen",
    "Nineteen",
    "Twenty",
]

STYLE_PROFILES: dict[str, dict[str, str]] = {
    "neutral": {
        "style": "neutral",
        "speed": "normal",
        "loudness": "normal",
        "voice": "friendly",
        "target_instruction": "Speak as a friendly home fitness coach in a normal indoor voice.",
        "input_text": "Can you count my reps?",
        "input_instruction": "Speak naturally in a normal indoor voice.",
    },
    "whispered": {
        "style": "whispered",
        "speed": "normal",
        "loudness": "quiet",
        "voice": "breathy",
        "target_instruction": "Whisper softly and quietly, close to the microphone, while staying intelligible.",
        "input_text": "I'm whispering, please stay quiet.",
        "input_instruction": "Whisper softly and quietly, close to the microphone.",
    },
    "projected": {
        "style": "projected",
        "speed": "normal",
        "loudness": "loud",
        "voice": "firm",
        "target_instruction": "Project clearly and loudly like a coach across a room. Sound firm, not angry.",
        "input_text": "Can you talk louder?",
        "input_instruction": "Ask clearly from across a room, as a direct request.",
    },
    "calm_slow": {
        "style": "calm",
        "speed": "slow",
        "loudness": "normal",
        "voice": "gentle",
        "target_instruction": "Speak slowly and calmly with a relaxed supportive tone and clear pauses.",
        "input_text": "Slow it down please.",
        "input_instruction": "Speak calmly, asking for a slower coaching pace.",
    },
    "fast_upbeat": {
        "style": "upbeat",
        "speed": "fast",
        "loudness": "normal",
        "voice": "upbeat",
        "target_instruction": "Speak fast and upbeat with positive workout energy while staying understandable.",
        "input_text": "Speed up the coaching.",
        "input_instruction": "Speak quickly with upbeat energy, as a short request.",
    },
    "warm": {
        "style": "warm",
        "speed": "normal",
        "loudness": "normal",
        "voice": "reassuring",
        "target_instruction": "Speak warmly and reassuringly, like a supportive home fitness coach.",
        "input_text": "Can you use a warmer voice?",
        "input_instruction": "Ask naturally for a warmer, more reassuring voice.",
    },
}

SUFFIXES = [
    "",
    " Nice rhythm.",
    " Strong rep.",
    " Keep going.",
    " Great set.",
    " Good control.",
    " Quiet.",
    " Steady rep.",
]

COACH_LINES = [
    "Breathe. One steady rep.",
    "Last one. Strong finish.",
    "Good control. Keep going.",
    "Nice rhythm. Stay with it.",
    "Strong rep. Reset tall.",
    "Great set. Breathe.",
]


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"").strip("'"))


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def normalize_transcript(value: str) -> str:
    number_map = {
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine",
        "10": "ten",
        "11": "eleven",
        "12": "twelve",
        "13": "thirteen",
        "14": "fourteen",
        "15": "fifteen",
        "16": "sixteen",
        "17": "seventeen",
        "18": "eighteen",
        "19": "nineteen",
        "20": "twenty",
    }
    lowered = value.lower()
    for digit, word in sorted(number_map.items(), key=lambda item: -len(item[0])):
        lowered = re.sub(rf"\b{re.escape(digit)}\b", word, lowered)
    return " ".join(re.findall(r"[a-z']+", lowered)).replace("'", "")


def wav_info(path: Path) -> dict[str, Any]:
    content = path.read_bytes()
    channels = int.from_bytes(content[22:24], "little")
    sample_rate = int.from_bytes(content[24:28], "little")
    sample_width = int.from_bytes(content[34:36], "little") // 8
    data_pos = content.find(b"data")
    if data_pos < 0:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            frames = wav.getnframes()
            raw = wav.readframes(frames)
    else:
        declared_data_bytes = int.from_bytes(content[data_pos + 4 : data_pos + 8], "little")
        actual_data_bytes = len(content) - (data_pos + 8)
        data_bytes = min(declared_data_bytes, actual_data_bytes)
        if declared_data_bytes in {0xFFFFFFFF, 0x7FFFFFFF} or declared_data_bytes > actual_data_bytes:
            data_bytes = actual_data_bytes
        raw = content[data_pos + 8 : data_pos + 8 + data_bytes]
        sample_count = len(raw) // max(sample_width, 1)
        frames = sample_count // max(channels, 1)
    samples = []
    if sample_width == 2:
        for offset in range(0, len(raw) - 1, 2):
            samples.append(int.from_bytes(raw[offset : offset + 2], "little", signed=True) / 32768.0)
    rms = math.sqrt(sum(sample * sample for sample in samples) / max(len(samples), 1))
    peak = max((abs(sample) for sample in samples), default=0.0)
    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "frames": frames,
        "duration_s": frames / sample_rate,
        "rms_dbfs": 20 * math.log10(max(rms, 1e-9)),
        "peak_dbfs": 20 * math.log10(max(peak, 1e-9)),
    }


def request_with_backoff(
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    data: Any = None,
    files: dict[str, Any] | None = None,
    timeout: int = 120,
    retries: int = 5,
) -> requests.Response:
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = requests.request(method, url, headers=headers, data=data, files=files, timeout=timeout)
        except requests.RequestException as exc:
            last_error = exc
            response = None
        if response is not None and response.status_code < 400:
            return response
        retryable = response is None or response.status_code in {408, 409, 429, 500, 502, 503, 504}
        if not retryable or attempt >= retries:
            if response is not None:
                raise RuntimeError(f"request failed: {response.status_code} {response.text[:1000]}")
            raise RuntimeError(f"request failed: {type(last_error).__name__}: {last_error}")
        retry_after = 0.0
        if response is not None:
            try:
                retry_after = float(response.headers.get("retry-after", "0"))
            except ValueError:
                retry_after = 0.0
        wait_s = retry_after if retry_after > 0 else min(2 ** attempt, 30) + random.random()
        time.sleep(wait_s)
    raise AssertionError("unreachable")


def generate_speech(api_key: str, text: str, voice: str, instructions: str, speed: float, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path = path.with_suffix(".instructions.txt")
    prompt_path.write_text(instructions + "\n")
    text_path = path.with_suffix(".text.txt")
    text_path.write_text(text + "\n")
    if path.exists() and path.stat().st_size > 44:
        return {
            "source_model": f"openai:{OPENAI_TTS_MODEL}",
            "voice": voice,
            "instructions": instructions,
            "speed": speed,
            "wav_path": str(path),
            "reused_existing_wav": True,
            **wav_info(path),
        }
    payload = {
        "model": OPENAI_TTS_MODEL,
        "voice": voice,
        "input": text,
        "instructions": instructions,
        "speed": speed,
        "response_format": "wav",
    }
    response = request_with_backoff(
        "POST",
        "https://api.openai.com/v1/audio/speech",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        data=json.dumps(payload),
    )
    path.write_bytes(response.content)
    return {
        "source_model": f"openai:{OPENAI_TTS_MODEL}",
        "voice": voice,
        "instructions": instructions,
        "speed": speed,
        "wav_path": str(path),
        "content_type": response.headers.get("content-type", ""),
        "reused_existing_wav": False,
        **wav_info(path),
    }


def transcribe_with_whisper(api_key: str, path: Path, expected_text: str) -> dict[str, Any]:
    response = request_with_backoff(
        "POST",
        "https://api.openai.com/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {api_key}"},
        data={"model": OPENAI_WHISPER_MODEL},
        files={"file": (path.name, path.read_bytes(), "audio/wav")},
        timeout=120,
    )
    payload = response.json()
    heard = payload.get("text", "")
    return {
        "model": f"openai:{OPENAI_WHISPER_MODEL}",
        "expected": expected_text,
        "heard": heard,
        "expected_normalized": normalize_transcript(expected_text),
        "heard_normalized": normalize_transcript(heard),
        "matches_expected": normalize_transcript(expected_text) == normalize_transcript(heard),
        "raw": payload,
    }


def judge_audio_no_hints(api_key: str, path: Path, model: str, expected_text: str) -> dict[str, Any]:
    audio_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    prompt = (
        "The audio is attached in this message. Listen to it directly now; do not ask me to "
        "play or upload it again. Return compact JSON with keys: transcript, style, speed, "
        "loudness, voice_notes, problems. Do not infer missing words from context; report "
        "only what you can hear."
    )
    if "audio" in model:
        payload = {
            "model": model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an audio evaluator. Output only compact JSON.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_b64,
                                "format": "wav",
                            },
                        },
                    ],
                },
            ],
        }
        try:
            response = request_with_backoff(
                "POST",
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=120,
                retries=3,
            )
        except Exception as exc:
            return {
                "ok": False,
                "model": f"openai:{model}",
                "error": f"{type(exc).__name__}: {exc}"[:1000],
            }
        body = response.json()
        raw_text = body.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        response_id = body.get("id")
    else:
        payload = {
            "model": model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_b64,
                                "format": "wav",
                            },
                        },
                    ],
                }
            ],
        }
        try:
            response = request_with_backoff(
                "POST",
                "https://api.openai.com/v1/responses",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=120,
                retries=3,
            )
        except Exception as exc:
            return {
                "ok": False,
                "model": f"openai:{model}",
                "error": f"{type(exc).__name__}: {exc}"[:1000],
            }
        body = response.json()
        texts = []
        for item in body.get("output", []):
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"}:
                    texts.append(content.get("text", ""))
        raw_text = "\n".join(texts).strip()
        response_id = body.get("id")
    transcript = raw_text
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            transcript = str(parsed.get("transcript", raw_text))
    except json.JSONDecodeError:
        parsed = {"raw_text": raw_text}
    return {
        "ok": True,
        "model": f"openai:{model}",
        "response_id": response_id,
        "raw_text": raw_text,
        "json": parsed,
        "heard": transcript,
        "heard_normalized": normalize_transcript(transcript),
        "expected": expected_text,
        "expected_normalized": normalize_transcript(expected_text),
        "matches_expected": normalize_transcript(transcript) == normalize_transcript(expected_text),
    }


def style_for_index(split: str, index: int) -> str:
    styles = list(STYLE_PROFILES)
    offset = {"train": 0, "validation": 2, "test": 4}.get(split, 0)
    return styles[(index + offset) % len(styles)]


def voice_for_index(split: str, index: int) -> str:
    offset = {"train": 0, "validation": 3, "test": 6}.get(split, 0)
    return OPENAI_VOICES[(index * 3 + offset) % len(OPENAI_VOICES)]


def target_text_for_index(split: str, index: int) -> str:
    offset = {"train": 0, "validation": 7, "test": 13}.get(split, 0)
    group = index % 10
    number = NUMBER_WORDS[(index + offset) % len(NUMBER_WORDS)]
    if group < 6:
        return f"{number}."
    if group < 9:
        suffix = SUFFIXES[(index // 3 + offset) % len(SUFFIXES)]
        return f"{number}.{suffix}" if suffix else f"{number}."
    return COACH_LINES[(index + offset) % len(COACH_LINES)]


def speed_for_style(style_key: str, index: int) -> float:
    if style_key == "calm_slow":
        return 0.82 if index % 2 else 0.9
    if style_key == "fast_upbeat":
        return 1.25 if index % 2 else 1.15
    return 1.0


def build_example(split: str, index: int, target_only: bool) -> dict[str, Any]:
    style_key = style_for_index(split, index)
    style = STYLE_PROFILES[style_key]
    voice = voice_for_index(split, index)
    target_text = target_text_for_index(split, index)
    speed = speed_for_style(style_key, index)
    row_id = f"openai_{split}_{index:05d}_{slugify(style_key)}_{slugify(target_text)[:48]}"
    target_instruction = (
        f"{style['target_instruction']} Say exactly the transcript provided. "
        "Do not add extra words. Keep it suitable for a single-person home workout demo."
    )
    input_instruction = (
        f"{style['input_instruction']} Say exactly the transcript provided. "
        "Do not add extra words."
    )
    return {
        "id": row_id,
        "split": split,
        "exercise": "squat",
        "event": "rep_complete",
        "home_demo_trigger": f"single-person home workout; {style['style']} style; example {index}",
        "input_text": style["input_text"],
        "input_voice": voice,
        "input_direction": input_instruction,
        "target_text": target_text,
        "target_voice": voice,
        "target_direction": target_instruction,
        "target_speed": speed,
        "expected": {
            "transcript": target_text,
            "style": style["style"],
            "speed": style["speed"],
            "loudness": style["loudness"],
            "voice": style["voice"],
        },
        "target_only": target_only,
    }


def example_from_existing_row(row: dict[str, Any], target_only: bool) -> dict[str, Any]:
    labels = row["labels"]
    return {
        "id": row["id"],
        "split": row.get("split", ""),
        "exercise": row.get("exercise", "squat"),
        "event": row.get("event", "rep_complete"),
        "home_demo_trigger": row.get("home_demo_trigger", ""),
        "input_text": labels["input_text"],
        "input_voice": labels["input_voice"],
        "input_direction": labels["input_direction"],
        "target_text": labels["target_text"],
        "target_voice": labels["target_voice"],
        "target_direction": labels["target_direction"],
        "target_speed": float(labels.get("target_speed", 1.0)),
        "expected": labels["expected"],
        "target_only": target_only,
    }


def should_verify(split: str, index: int, mode: str, sample_rate: float) -> bool:
    if mode == "none":
        return False
    if mode == "all":
        return True
    if mode == "sample":
        return (index % max(int(1 / max(sample_rate, 1e-9)), 1)) == 0
    if mode == "split-default":
        if split in {"validation", "test"}:
            return True
        return (index % max(int(1 / max(sample_rate, 1e-9)), 1)) == 0
    raise ValueError(f"unknown verify mode: {mode}")


def make_row(api_key: str, args: argparse.Namespace, local_index: int) -> dict[str, Any]:
    index = args.start_index + local_index
    example = build_example(args.split, index, args.target_only)
    previous_row = getattr(args, "_previous_rows_by_id", {}).get(example["id"])
    if previous_row:
        example = example_from_existing_row(previous_row, args.target_only)
    row_dir = args.out_dir / example["id"]
    target_wav = row_dir / "target.wav"
    input_wav = row_dir / "input.wav"
    target_meta = generate_speech(
        api_key,
        example["target_text"],
        example["target_voice"],
        example["target_direction"],
        example["target_speed"],
        target_wav,
    )
    input_meta = None
    if not args.target_only:
        input_meta = generate_speech(
            api_key,
            example["input_text"],
            example["input_voice"],
            example["input_direction"],
            1.0,
            input_wav,
        )
    verification: dict[str, Any] = {}
    if should_verify(args.split, local_index, args.verify, args.verify_sample_rate):
        verification["target_whisper"] = transcribe_with_whisper(api_key, target_wav, example["target_text"])
    if input_meta is not None and args.verify_input:
        verification["input_whisper"] = transcribe_with_whisper(api_key, input_wav, example["input_text"])
    if args.judge_model and should_verify(args.split, local_index, args.judge, args.judge_sample_rate):
        verification["target_audio_judge"] = judge_audio_no_hints(
            api_key,
            target_wav,
            args.judge_model,
            example["target_text"],
        )
    row = {
        "id": example["id"],
        "split": args.split,
        "exercise": example["exercise"],
        "event": example["event"],
        "home_demo_trigger": example["home_demo_trigger"],
        "labels": {
            "input_text": example["input_text"],
            "input_direction": example["input_direction"],
            "input_voice": example["input_voice"],
            "target_text": example["target_text"],
            "target_direction": example["target_direction"],
            "target_voice": example["target_voice"],
            "target_speed": example["target_speed"],
            "expected": example["expected"],
        },
        "audio": {
            "target": target_meta,
        },
        "verification": verification,
        "_local_index": local_index,
    }
    if input_meta is not None:
        row["audio"]["input"] = input_meta
    if previous_row and previous_row.get("verification") and not row["verification"]:
        row["verification"] = previous_row["verification"]
    return row


def write_manifests(out_dir: Path, manifest: dict[str, Any]) -> None:
    rows = [{k: v for k, v in row.items() if k != "_local_index"} for row in manifest["pairs"]]
    (out_dir / "rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    summary = {**manifest, "pairs": rows}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate style-control audio data for Gemma omni-fitness.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--target-only", action="store_true", help="Generate only target coach audio.")
    parser.add_argument("--verify", choices=["split-default", "none", "sample", "all"], default="split-default")
    parser.add_argument("--verify-sample-rate", type=float, default=0.05)
    parser.add_argument("--verify-input", action="store_true")
    parser.add_argument("--judge-model", default="", help="Optional audio-capable model, e.g. gpt-5.5.")
    parser.add_argument("--judge", choices=["split-default", "none", "sample", "all"], default="none")
    parser.add_argument("--judge-sample-rate", type=float, default=0.05)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sleep-between-requests", type=float, default=0.0)
    return parser


def main() -> None:
    load_dotenv(Path(".env"))
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY or add it to .env")
    args = build_arg_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    previous_summary_path = args.out_dir / "summary.json"
    previous_generated_at = None
    previous_rows_by_id = {}
    if previous_summary_path.exists():
        try:
            previous_summary = json.loads(previous_summary_path.read_text())
            previous_generated_at = previous_summary.get("generated_at_unix")
            previous_rows_by_id = {row["id"]: row for row in previous_summary.get("pairs", [])}
        except json.JSONDecodeError:
            previous_generated_at = None
    args._previous_rows_by_id = previous_rows_by_id

    manifest = {
        "generated_at_unix": previous_generated_at or time.time(),
        "generator": "openai_tts_style_dataset.py",
        "source_tts": f"openai:{OPENAI_TTS_MODEL}",
        "whisper_model": f"openai:{OPENAI_WHISPER_MODEL}",
        "sample_rate": SAMPLE_RATE,
        "split": args.split,
        "limit": args.limit,
        "target_only": args.target_only,
        "verify": args.verify,
        "verify_sample_rate": args.verify_sample_rate,
        "verify_input": args.verify_input,
        "judge_model": args.judge_model or None,
        "judge": args.judge,
        "judge_sample_rate": args.judge_sample_rate,
        "concurrency": args.concurrency,
        "purpose": "home-fitness style-control speech targets for Gemma audio-output-head training",
        "pairs": [],
    }

    rows_by_index: dict[int, dict[str, Any]] = {}
    concurrency = max(1, args.concurrency)
    if concurrency == 1:
        for local_index in range(args.limit):
            row = make_row(api_key, args, local_index)
            rows_by_index[local_index] = row
            manifest["pairs"] = [rows_by_index[key] for key in sorted(rows_by_index)]
            write_manifests(args.out_dir, manifest)
            status = {
                "index": local_index + 1,
                "limit": args.limit,
                "id": row["id"],
                "target": row["labels"]["target_text"],
                "verified": bool(row["verification"]),
                "target_whisper_match": row["verification"].get("target_whisper", {}).get("matches_expected"),
                "target_judge_match": row["verification"].get("target_audio_judge", {}).get("matches_expected"),
            }
            print(json.dumps(status), flush=True)
            if args.sleep_between_requests > 0:
                time.sleep(args.sleep_between_requests)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_index = {
                executor.submit(make_row, api_key, args, local_index): local_index
                for local_index in range(args.limit)
            }
            completed = 0
            for future in concurrent.futures.as_completed(future_to_index):
                local_index = future_to_index[future]
                row = future.result()
                rows_by_index[local_index] = row
                completed += 1
                manifest["pairs"] = [rows_by_index[key] for key in sorted(rows_by_index)]
                write_manifests(args.out_dir, manifest)
                status = {
                    "completed": completed,
                    "limit": args.limit,
                    "index": local_index + 1,
                    "id": row["id"],
                    "target": row["labels"]["target_text"],
                    "verified": bool(row["verification"]),
                    "target_whisper_match": row["verification"].get("target_whisper", {}).get("matches_expected"),
                    "target_judge_match": row["verification"].get("target_audio_judge", {}).get("matches_expected"),
                }
                print(json.dumps(status), flush=True)

    write_manifests(args.out_dir, manifest)
    total_target_s = sum(row["audio"]["target"]["duration_s"] for row in manifest["pairs"])
    total_input_s = sum(row["audio"].get("input", {}).get("duration_s", 0.0) for row in manifest["pairs"])
    verified_targets = [
        row["verification"]["target_whisper"]
        for row in manifest["pairs"]
        if "target_whisper" in row.get("verification", {})
    ]
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "split": args.split,
                "pairs": len(manifest["pairs"]),
                "target_audio_min": total_target_s / 60.0,
                "input_audio_min": total_input_s / 60.0,
                "verified_targets": len(verified_targets),
                "verified_target_matches": sum(1 for item in verified_targets if item["matches_expected"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
