from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
import wave
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types


DEFAULT_TTS_MODELS = [
    "gemini-3.1-flash-tts-preview",
    "gemini-2.5-flash-preview-tts",
]
DEFAULT_VERIFIER_MODEL = "gemini-flash-latest"
DEFAULT_OUT_DIR = Path("out/gemma4_omni_gemini_tts_style_dataset")
SAMPLE_RATE = 24_000


STYLE_PAIRS = [
    {
        "id": "normal_default",
        "home_demo_trigger": "user speaks normally before a squat set",
        "input_text": "Hello, can you count my squats?",
        "input_voice": "Achird",
        "input_direction": "Say naturally in a normal indoor speaking voice.",
        "target_text": "One.",
        "target_voice": "Achird",
        "target_direction": "Say it as a friendly fitness coach in a normal indoor speaking voice.",
        "expected": {
            "transcript": "One.",
            "style": "neutral",
            "speed": "normal",
            "loudness": "normal",
            "voice": "friendly",
        },
    },
    {
        "id": "whisper_match",
        "home_demo_trigger": "user whispers because someone nearby is sleeping",
        "input_text": "Can you count quietly?",
        "input_voice": "Enceladus",
        "input_direction": "Whisper softly and quietly, close to the microphone.",
        "target_text": "Two. Nice rhythm.",
        "target_voice": "Enceladus",
        "target_direction": "Whisper softly and quietly, matching a user who whispered.",
        "expected": {
            "transcript": "Two. Nice rhythm.",
            "style": "whispered",
            "speed": "normal",
            "loudness": "quiet",
            "voice": "breathy",
        },
    },
    {
        "id": "late_night_quiet",
        "home_demo_trigger": "user asks the coach not to wake other people",
        "input_text": "Please keep it down.",
        "input_voice": "Achernar",
        "input_direction": "Say softly and cautiously, like a quiet late-night request.",
        "target_text": "Three.",
        "target_voice": "Achernar",
        "target_direction": "Say it very quietly and gently, but still intelligibly.",
        "expected": {
            "transcript": "Three.",
            "style": "gentle",
            "speed": "normal",
            "loudness": "quiet",
            "voice": "soft",
        },
    },
    {
        "id": "asked_louder",
        "home_demo_trigger": "user asks the model to talk louder from across the room",
        "input_text": "Can you talk louder?",
        "input_voice": "Kore",
        "input_direction": "Say clearly in a normal voice, as a direct request.",
        "target_text": "Four. Louder now.",
        "target_voice": "Kore",
        "target_direction": "Project the voice clearly and loudly like a coach across a room. Do not sound angry.",
        "expected": {
            "transcript": "Four. Louder now.",
            "style": "projected",
            "speed": "normal",
            "loudness": "loud",
            "voice": "firm",
        },
    },
    {
        "id": "hype_yell_home_safe",
        "home_demo_trigger": "user asks for high-energy motivation for the last rep",
        "input_text": "Hype me up for the last rep.",
        "input_voice": "Puck",
        "input_direction": "Say with excited anticipation, like asking for encouragement.",
        "target_text": "Last one! Strong finish!",
        "target_voice": "Fenrir",
        "target_direction": "Give a short high-energy workout shout, loud and excited but positive, not angry.",
        "expected": {
            "transcript": "Last one! Strong finish!",
            "style": "excited",
            "speed": "fast",
            "loudness": "loud",
            "voice": "excitable",
        },
    },
    {
        "id": "slow_calm",
        "home_demo_trigger": "user asks the coach to slow down",
        "input_text": "Slow it down please.",
        "input_voice": "Vindemiatrix",
        "input_direction": "Say calmly, with a relaxed request tone.",
        "target_text": "Breathe. One steady rep.",
        "target_voice": "Vindemiatrix",
        "target_direction": "Speak slowly, gently, and calmly with clear pauses between phrases.",
        "expected": {
            "transcript": "Breathe. One steady rep.",
            "style": "calm",
            "speed": "slow",
            "loudness": "normal",
            "voice": "gentle",
        },
    },
    {
        "id": "fast_energized",
        "home_demo_trigger": "user asks for faster coaching during jumping jacks",
        "input_text": "Speed up the coaching.",
        "input_voice": "Laomedeia",
        "input_direction": "Say with upbeat energy, as a quick request.",
        "target_text": "Five. Keep moving.",
        "target_voice": "Laomedeia",
        "target_direction": "Speak fast, upbeat, and energetic while staying understandable.",
        "expected": {
            "transcript": "Five. Keep moving.",
            "style": "upbeat",
            "speed": "fast",
            "loudness": "normal",
            "voice": "upbeat",
        },
    },
    {
        "id": "warmer_voice_switch",
        "home_demo_trigger": "user asks for a warmer coach voice",
        "input_text": "Can you use a warmer voice?",
        "input_voice": "Achird",
        "input_direction": "Say naturally, asking for a different tone.",
        "target_text": "Six. You're doing well.",
        "target_voice": "Sulafat",
        "target_direction": "Speak warmly and reassuringly, like a supportive home fitness coach.",
        "expected": {
            "transcript": "Six. You're doing well.",
            "style": "warm",
            "speed": "normal",
            "loudness": "normal",
            "voice": "warmer than input voice",
        },
    },
]


HELDOUT_STYLE_PAIRS = [
    {
        "id": "heldout_whisper_seven",
        "home_demo_trigger": "user whispers during a home workout and asks the coach to stay quiet",
        "input_text": "I'm whispering, please stay quiet.",
        "input_voice": "Enceladus",
        "input_direction": "Whisper softly and quietly, close to the microphone.",
        "target_text": "Seven. Quiet.",
        "target_voice": "Enceladus",
        "target_direction": "Whisper softly and quietly, matching a user who whispered.",
        "expected": {
            "transcript": "Seven. Quiet.",
            "style": "whispered",
            "speed": "normal",
            "loudness": "quiet",
            "voice": "breathy",
        },
    },
    {
        "id": "heldout_louder_eight",
        "home_demo_trigger": "user asks the coach to project more clearly across the room",
        "input_text": "Speak louder for the next one.",
        "input_voice": "Kore",
        "input_direction": "Say clearly in a normal voice, as a direct request.",
        "target_text": "Eight. Strong rep.",
        "target_voice": "Kore",
        "target_direction": "Project the voice clearly and loudly like a coach across a room. Do not sound angry.",
        "expected": {
            "transcript": "Eight. Strong rep.",
            "style": "projected",
            "speed": "normal",
            "loudness": "loud",
            "voice": "firm",
        },
    },
]


NUMBER_WORDS = {
    1: "One",
    2: "Two",
    3: "Three",
    4: "Four",
    5: "Five",
    6: "Six",
    7: "Seven",
    8: "Eight",
    9: "Nine",
    10: "Ten",
    11: "Eleven",
    12: "Twelve",
}


STYLE_TEMPLATES = {
    "neutral_count": {
        "input_text": "Count this rep.",
        "input_voice": "Achird",
        "input_direction": "Say naturally in a normal indoor speaking voice.",
        "target_voice": "Achird",
        "target_direction": "Say it as a friendly fitness coach in a normal indoor speaking voice.",
        "expected": {
            "style": "neutral",
            "speed": "normal",
            "loudness": "normal",
            "voice": "friendly",
        },
    },
    "whisper_quiet": {
        "input_text": "Please keep coaching quietly.",
        "input_voice": "Enceladus",
        "input_direction": "Whisper softly and quietly, close to the microphone.",
        "target_voice": "Enceladus",
        "target_direction": "Whisper softly and quietly, matching a user who whispered.",
        "expected": {
            "style": "whispered",
            "speed": "normal",
            "loudness": "quiet",
            "voice": "breathy",
        },
    },
    "projected_strong": {
        "input_text": "Speak louder for the next count.",
        "input_voice": "Kore",
        "input_direction": "Say clearly in a normal voice, as a direct request.",
        "target_voice": "Kore",
        "target_direction": "Project the voice clearly and loudly like a coach across a room. Do not sound angry.",
        "expected": {
            "style": "projected",
            "speed": "normal",
            "loudness": "loud",
            "voice": "firm",
        },
    },
    "upbeat_rhythm": {
        "input_text": "Give me a little encouragement.",
        "input_voice": "Laomedeia",
        "input_direction": "Say with upbeat energy, as a quick request.",
        "target_voice": "Laomedeia",
        "target_direction": "Speak upbeat and energetic while staying understandable.",
        "expected": {
            "style": "upbeat",
            "speed": "fast",
            "loudness": "normal",
            "voice": "upbeat",
        },
    },
}


def make_count_style_pair(number: int, suffix: str, style_key: str, split_name: str) -> dict[str, Any]:
    style = STYLE_TEMPLATES[style_key]
    number_word = NUMBER_WORDS[number]
    target_text = f"{number_word}{suffix}"
    phrase_slug = (
        suffix.lower()
        .replace(".", "")
        .replace("!", "")
        .replace("'", "")
        .replace(" ", "_")
        .strip("_")
        or "count"
    )
    return {
        "id": f"{split_name}_{style_key}_{number:02d}_{phrase_slug}",
        "home_demo_trigger": f"single-person home workout count {number} with {style_key.replace('_', ' ')} delivery",
        "input_text": style["input_text"],
        "input_voice": style["input_voice"],
        "input_direction": style["input_direction"],
        "target_text": target_text,
        "target_voice": style["target_voice"],
        "target_direction": style["target_direction"],
        "expected": {
            "transcript": target_text,
            **style["expected"],
        },
    }


def count_grid_train_pairs() -> list[dict[str, Any]]:
    pairs = []
    for number in range(1, 13):
        pairs.append(make_count_style_pair(number, ".", "neutral_count", "count_train"))
    for number in [2, 4, 6, 8, 10, 12]:
        pairs.append(make_count_style_pair(number, ". Quiet.", "whisper_quiet", "count_train"))
    for number in [1, 3, 5, 7, 9, 11]:
        pairs.append(make_count_style_pair(number, ". Strong rep.", "projected_strong", "count_train"))
    for number in [2, 5, 8, 11]:
        pairs.append(make_count_style_pair(number, ". Nice rhythm.", "upbeat_rhythm", "count_train"))
    return pairs


def count_grid_heldout_pairs() -> list[dict[str, Any]]:
    return [
        make_count_style_pair(7, ". Quiet.", "whisper_quiet", "count_heldout"),
        make_count_style_pair(8, ". Strong rep.", "projected_strong", "count_heldout"),
        make_count_style_pair(11, ". Quiet.", "whisper_quiet", "count_heldout"),
        make_count_style_pair(12, ". Strong rep.", "projected_strong", "count_heldout"),
    ]


def select_style_pairs(scenario_set: str) -> list[dict[str, Any]]:
    if scenario_set == "train":
        return STYLE_PAIRS
    if scenario_set == "heldout":
        return HELDOUT_STYLE_PAIRS
    if scenario_set == "all":
        return STYLE_PAIRS + HELDOUT_STYLE_PAIRS
    if scenario_set == "count-grid-train":
        return count_grid_train_pairs()
    if scenario_set == "count-grid-heldout":
        return count_grid_heldout_pairs()
    if scenario_set == "count-grid-all":
        return count_grid_train_pairs() + count_grid_heldout_pairs()
    raise ValueError(f"unknown scenario set: {scenario_set}")


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"").strip("'"))


def write_wav(path: Path, pcm: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm)
    frame_count = len(pcm) // 2
    samples = [
        int.from_bytes(pcm[offset : offset + 2], "little", signed=True) / 32768.0
        for offset in range(0, len(pcm) - 1, 2)
    ]
    rms = math.sqrt(sum(sample * sample for sample in samples) / max(len(samples), 1))
    peak = max((abs(sample) for sample in samples), default=0.0)
    return {
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
        "sample_width_bytes": 2,
        "frames": frame_count,
        "duration_s": frame_count / SAMPLE_RATE,
        "rms_dbfs": 20 * math.log10(max(rms, 1e-9)),
        "peak_dbfs": 20 * math.log10(max(peak, 1e-9)),
    }


def inspect_wav(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        raw = wav.readframes(frames)
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


def tts_prompt(pair: dict[str, Any], side: str) -> str:
    text = pair[f"{side}_text"]
    direction = pair[f"{side}_direction"]
    return (
        "# AUDIO PROFILE\n"
        "A single speaker in a normal home workout setting.\n\n"
        "# DIRECTOR'S NOTES\n"
        f"{direction}\n"
        "Keep the delivery safe for a home demo and do not add extra words.\n\n"
        "# TRANSCRIPT\n"
        f"{text}"
    )


def generate_tts(
    client: genai.Client,
    models: list[str],
    prompt: str,
    voice: str,
    retries: int = 3,
) -> tuple[bytes, dict[str, Any]]:
    errors = []
    for model in models:
        for attempt in range(retries + 1):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=voice,
                                )
                            )
                        ),
                    ),
                )
                parts = response.candidates[0].content.parts if response.candidates[0].content else None
                if not parts or not getattr(parts[0], "inline_data", None):
                    raise RuntimeError("TTS response did not include inline audio data")
                data = parts[0].inline_data.data
                meta = {
                    "model": model,
                    "voice": voice,
                    "response_id": getattr(response, "response_id", None),
                }
                if errors:
                    meta["fallback_errors"] = errors
                return data, meta
            except Exception as exc:  # SDK exception types vary by release.
                message = f"{type(exc).__name__}: {exc}"[:1000]
                retry_after_match = re.search(r"retry in ([0-9.]+)s", message, flags=re.IGNORECASE)
                is_rate_limit = "429" in message or "RESOURCE_EXHAUSTED" in message
                if is_rate_limit and attempt < retries:
                    wait_s = float(retry_after_match.group(1)) + 1.0 if retry_after_match else 65.0
                    errors.append({"model": model, "attempt": attempt + 1, "retry_after_s": wait_s, "error": message})
                    time.sleep(wait_s)
                    continue
                errors.append({"model": model, "attempt": attempt + 1, "error": message})
                break
    raise RuntimeError(json.dumps({"tts_errors": errors}, indent=2))


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {"raw_text": text, "parse_error": "no JSON object found"}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            return {"raw_text": text, "parse_error": str(exc)}
    if isinstance(parsed, dict):
        return parsed
    return {"items": parsed}


def verify_audio(
    client: genai.Client,
    model: str,
    wav_path: Path,
    pair: dict[str, Any],
    side: str,
) -> dict[str, Any]:
    expected = pair["expected"] if side == "target" else {
        "transcript": pair["input_text"],
        "style_hint": pair["input_direction"],
    }
    prompt = (
        "Listen to this home-fitness audio sample and return only JSON with keys "
        "transcript, style, speed, loudness, voice_notes, matches_expected, problems. "
        f"Expected: {json.dumps(expected, ensure_ascii=False)}. "
        "matches_expected should be true only if both the words and the requested vocal characteristic are clear."
    )
    try:
        response = client.models.generate_content(
            model=model,
            contents=[
                prompt,
                types.Part.from_bytes(
                    data=wav_path.read_bytes(),
                    mime_type="audio/wav",
                ),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
    except Exception as exc:
        return {
            "ok": False,
            "model": model,
            "error": f"{type(exc).__name__}: {exc}"[:1000],
        }
    parsed = parse_json_object(response.text or "")
    return {
        "ok": "parse_error" not in parsed,
        "model": model,
        "json": parsed,
        "raw_text": response.text,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Gemini TTS input/target style pairs for Gemma omni-fitness audio-head training."
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--tts-model",
        action="append",
        default=[],
        help="TTS model to try. Can be repeated. Defaults to Gemini 3.1 Flash TTS, then Gemini 2.5 Flash TTS fallback.",
    )
    parser.add_argument("--verifier-model", default=DEFAULT_VERIFIER_MODEL)
    parser.add_argument(
        "--scenario-set",
        choices=[
            "train",
            "heldout",
            "all",
            "count-grid-train",
            "count-grid-heldout",
            "count-grid-all",
        ],
        default="train",
        help="Which built-in home-demo style scenarios to generate.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit generated pairs. 0 means all selected pairs.")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--tts-retries", type=int, default=4)
    return parser


def main() -> None:
    load_dotenv(Path(".env"))
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY or GOOGLE_API_KEY, or add it to .env")

    args = build_arg_parser().parse_args()
    tts_models = args.tts_model or DEFAULT_TTS_MODELS
    client = genai.Client(api_key=api_key)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "generated_at_unix": time.time(),
        "tts_models_tried": tts_models,
        "verifier_model": None if args.skip_verify else args.verifier_model,
        "sample_rate": SAMPLE_RATE,
        "scenario_set": args.scenario_set,
        "purpose": "small home-demo style-control pairs for Gemma audio-output-head training/evaluation",
        "pairs": [],
    }

    selected_pairs = select_style_pairs(args.scenario_set)
    if args.limit:
        selected_pairs = selected_pairs[: args.limit]

    for pair in selected_pairs:
        pair_dir = args.out_dir / pair["id"]
        pair_dir.mkdir(parents=True, exist_ok=True)
        pair_record = {
            "id": pair["id"],
            "home_demo_trigger": pair["home_demo_trigger"],
            "labels": {
                "input_text": pair["input_text"],
                "input_direction": pair["input_direction"],
                "input_voice": pair["input_voice"],
                "target_text": pair["target_text"],
                "target_direction": pair["target_direction"],
                "target_voice": pair["target_voice"],
                "expected": pair["expected"],
            },
            "audio": {},
            "verification": {},
        }
        for side in ["input", "target"]:
            wav_path = pair_dir / f"{side}.wav"
            prompt_path = pair_dir / f"{side}_prompt.txt"
            prompt_text = tts_prompt(pair, side)
            if wav_path.exists():
                wav_meta = inspect_wav(wav_path)
                tts_meta = {
                    "model": "existing",
                    "voice": pair[f"{side}_voice"],
                    "reused_existing_wav": True,
                }
                if not prompt_path.exists():
                    prompt_path.write_text(prompt_text + "\n")
            else:
                pcm, tts_meta = generate_tts(
                    client=client,
                    models=tts_models,
                    prompt=prompt_text,
                    voice=pair[f"{side}_voice"],
                    retries=args.tts_retries,
                )
                wav_meta = write_wav(wav_path, pcm)
                prompt_path.write_text(prompt_text + "\n")
            pair_record["audio"][side] = {
                "wav_path": str(wav_path),
                "prompt_path": str(prompt_path),
                "tts": tts_meta,
                "wav": wav_meta,
            }
            if not args.skip_verify:
                pair_record["verification"][side] = verify_audio(
                    client=client,
                    model=args.verifier_model,
                    wav_path=wav_path,
                    pair=pair,
                    side=side,
                )
        manifest["pairs"].append(pair_record)

    summary_path = args.out_dir / "summary.json"
    jsonl_path = args.out_dir / "pairs.jsonl"
    summary_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    jsonl_path.write_text(
        "\n".join(json.dumps(pair, ensure_ascii=False) for pair in manifest["pairs"]) + "\n"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
