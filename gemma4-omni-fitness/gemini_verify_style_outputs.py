from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types


DEFAULT_MODEL = "gemini-flash-latest"
NUMBER_WORDS = {
    "0": "zero",
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


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"").strip("'"))


def parse_json_value(text: str) -> dict[str, Any]:
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


def normalized_match(parsed: dict[str, Any]) -> bool:
    if parsed.get("matches_expected") is True:
        return True
    items = parsed.get("items")
    if isinstance(items, list) and items and isinstance(items[0], dict):
        return items[0].get("matches_expected") is True
    return False


def normalize_transcript(text: str) -> str:
    text = text.lower()
    for digit, word in NUMBER_WORDS.items():
        text = re.sub(rf"\b{re.escape(digit)}\b", word, text)
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def whisper_match(evaluation: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    expected_text = str(expected.get("transcript") or evaluation.get("reference_text") or "")
    heard_text = str((evaluation.get("openai_transcription") or {}).get("text") or "")
    expected_normalized = normalize_transcript(expected_text)
    heard_normalized = normalize_transcript(heard_text)
    return {
        "expected": expected_text,
        "heard": heard_text,
        "expected_normalized": expected_normalized,
        "heard_normalized": heard_normalized,
        "matches_expected": bool(expected_normalized and expected_normalized == heard_normalized),
    }


def verify_one(client: genai.Client, model: str, wav_path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    prompt = (
        "Listen to this generated fitness-coach audio sample and return only JSON with keys "
        "transcript, style, speed, loudness, voice_notes, matches_expected, problems. "
        f"Expected: {json.dumps(expected, ensure_ascii=False)}. "
        "matches_expected must be true only if both the words and requested vocal characteristic are clear."
    )
    try:
        response = client.models.generate_content(
            model=model,
            contents=[
                prompt,
                types.Part.from_bytes(data=wav_path.read_bytes(), mime_type="audio/wav"),
            ],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
    except Exception as exc:
        return {
            "ok": False,
            "model": model,
            "wav_path": str(wav_path),
            "error": f"{type(exc).__name__}: {exc}"[:1000],
        }
    parsed = parse_json_value(response.text or "")
    return {
        "ok": "parse_error" not in parsed,
        "model": model,
        "wav_path": str(wav_path),
        "json": parsed,
        "matches_expected": normalized_match(parsed),
        "raw_text": response.text,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify generated Gemma style-head WAVs with Gemini audio understanding.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--summary-name", default="summary.json")
    return parser


def main() -> None:
    load_dotenv(Path(".env"))
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY or GOOGLE_API_KEY, or add it to .env")
    args = build_arg_parser().parse_args()
    summary_path = args.run_dir / args.summary_name
    summary = json.loads(summary_path.read_text())
    client = genai.Client(api_key=api_key)

    verifications = []
    for evaluation in summary.get("generated_evaluations", []):
        eval_id = evaluation["eval_id"]
        wav_path = args.run_dir / "generated_audio" / f"{eval_id}.wav"
        expected = evaluation.get("expected") or {
            "transcript": evaluation.get("reference_text", ""),
        }
        verifications.append(
            {
                "eval_id": eval_id,
                "expected": expected,
                "openai_whisper": whisper_match(evaluation, expected),
                "full_audio": verify_one(client, args.model, wav_path, expected),
            }
        )

    all_gemini_matches = all(item["full_audio"].get("matches_expected") for item in verifications)
    all_whisper_matches = all(item["openai_whisper"].get("matches_expected") for item in verifications)
    result = {
        "model": args.model,
        "run_dir": str(args.run_dir),
        "count": len(verifications),
        "all_full_audio_matches": all_gemini_matches,
        "all_whisper_transcript_matches": all_whisper_matches,
        "all_full_audio_and_whisper_matches": all_gemini_matches and all_whisper_matches,
        "verifications": verifications,
    }
    out_path = args.run_dir / "gemini_style_verification.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    summary["gemini_style_verification"] = {
        "path": str(out_path),
        "model": args.model,
        "count": result["count"],
        "all_full_audio_matches": result["all_full_audio_matches"],
        "all_whisper_transcript_matches": result["all_whisper_transcript_matches"],
        "all_full_audio_and_whisper_matches": result["all_full_audio_and_whisper_matches"],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
