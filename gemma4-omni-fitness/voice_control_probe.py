from __future__ import annotations

import base64
import math
import json
import os
import time
from pathlib import Path

import requests


OPENAI_TTS_MODEL = "gpt-4o-mini-tts"
OPENAI_VERIFIER_MODEL = os.environ.get("VOICE_CONTROL_VERIFIER_MODEL", "gpt-5.4-mini")
FALLBACK_VERIFIER_MODEL = os.environ.get("VOICE_CONTROL_FALLBACK_VERIFIER_MODEL", "gpt-4o-audio-preview")
OUT_DIR = Path("out/gemma4_omni_voice_control_probe")


SCENARIOS = [
    {
        "id": "neutral_alloy",
        "voice": "alloy",
        "input": "Two. Nice rhythm.",
        "instructions": "Speak in a normal conversational fitness-coach voice.",
        "speed": 1.0,
        "expected": {
            "style": "neutral",
            "relative_speed": "normal",
            "relative_loudness": "normal",
            "same_words": "Two. Nice rhythm.",
        },
    },
    {
        "id": "whisper_alloy",
        "voice": "alloy",
        "input": "Two. Nice rhythm.",
        "instructions": "Whisper the line softly and quietly, as if responding to a user who whispered.",
        "speed": 1.0,
        "expected": {
            "style": "whispered",
            "relative_speed": "normal",
            "relative_loudness": "quiet",
            "same_words": "Two. Nice rhythm.",
        },
    },
    {
        "id": "loud_alloy",
        "voice": "alloy",
        "input": "Two. Nice rhythm.",
        "instructions": "Speak loudly and clearly, like a coach projecting across a gym.",
        "speed": 1.0,
        "expected": {
            "style": "projected",
            "relative_speed": "normal",
            "relative_loudness": "loud",
            "same_words": "Two. Nice rhythm.",
        },
    },
    {
        "id": "slow_calm_instruction_alloy",
        "voice": "alloy",
        "input": "Two. Nice rhythm.",
        "instructions": "Speak slowly and calmly with a relaxed supportive tone. Stretch the phrase noticeably.",
        "speed": 1.0,
        "expected": {
            "style": "calm",
            "relative_speed": "slow",
            "relative_loudness": "normal",
            "same_words": "Two. Nice rhythm.",
        },
    },
    {
        "id": "slow_speed_075_alloy",
        "voice": "alloy",
        "input": "Two. Nice rhythm.",
        "instructions": "Speak in a normal conversational fitness-coach voice.",
        "speed": 0.75,
        "expected": {
            "style": "neutral",
            "relative_speed": "slow",
            "relative_loudness": "normal",
            "same_words": "Two. Nice rhythm.",
        },
    },
    {
        "id": "fast_urgent_instruction_alloy",
        "voice": "alloy",
        "input": "Two. Nice rhythm.",
        "instructions": "Speak quickly with upbeat urgency, but keep it understandable.",
        "speed": 1.0,
        "expected": {
            "style": "upbeat",
            "relative_speed": "fast",
            "relative_loudness": "normal",
            "same_words": "Two. Nice rhythm.",
        },
    },
    {
        "id": "fast_speed_135_alloy",
        "voice": "alloy",
        "input": "Two. Nice rhythm.",
        "instructions": "Speak in a normal conversational fitness-coach voice.",
        "speed": 1.35,
        "expected": {
            "style": "neutral",
            "relative_speed": "fast",
            "relative_loudness": "normal",
            "same_words": "Two. Nice rhythm.",
        },
    },
    {
        "id": "neutral_verse",
        "voice": "verse",
        "input": "Two. Nice rhythm.",
        "instructions": "Speak in a normal conversational fitness-coach voice.",
        "speed": 1.0,
        "expected": {
            "style": "neutral",
            "relative_speed": "normal",
            "relative_loudness": "normal",
            "same_words": "Two. Nice rhythm.",
            "voice": "different from alloy",
        },
    },
]


def request_openai_speech(api_key: str, scenario: dict) -> tuple[bytes, dict]:
    payload = {
        "model": OPENAI_TTS_MODEL,
        "voice": scenario["voice"],
        "input": scenario["input"],
        "response_format": "wav",
        "instructions": scenario["instructions"],
        "speed": scenario["speed"],
    }
    response = requests.post(
        "https://api.openai.com/v1/audio/speech",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=120,
    )
    if response.status_code >= 400:
        fallback_payload = {
            k: v
            for k, v in payload.items()
            if k not in {"instructions", "speed"}
        }
        fallback_response = requests.post(
            "https://api.openai.com/v1/audio/speech",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            data=json.dumps(fallback_payload),
            timeout=120,
        )
        if fallback_response.status_code >= 400:
            raise RuntimeError(
                f"TTS failed for {scenario['id']}: {response.status_code} {response.text[:600]}"
            )
        return fallback_response.content, {
            "requested_payload": {k: v for k, v in payload.items() if k != "model"},
            "effective_payload": {k: v for k, v in fallback_payload.items() if k != "model"},
            "fallback_reason": response.text[:600],
            "content_type": fallback_response.headers.get("content-type", ""),
        }
    return response.content, {
        "requested_payload": {k: v for k, v in payload.items() if k != "model"},
        "effective_payload": {k: v for k, v in payload.items() if k != "model"},
        "content_type": response.headers.get("content-type", ""),
    }


def wav_info(path: Path) -> dict:
    content = path.read_bytes()
    channels = int.from_bytes(content[22:24], "little")
    sample_rate = int.from_bytes(content[24:28], "little")
    sample_width_bytes = int.from_bytes(content[34:36], "little") // 8
    data_pos = content.find(b"data")
    if data_pos < 0:
        raise ValueError(f"{path} has no data chunk")
    declared_data_bytes = int.from_bytes(content[data_pos + 4 : data_pos + 8], "little")
    actual_data_bytes = len(content) - (data_pos + 8)
    data_bytes = min(declared_data_bytes, actual_data_bytes)
    if declared_data_bytes in {0xFFFFFFFF, 0x7FFFFFFF} or declared_data_bytes > actual_data_bytes:
        data_bytes = actual_data_bytes
    pcm = content[data_pos + 8 : data_pos + 8 + data_bytes]
    sample_count = data_bytes // sample_width_bytes
    frame_count = sample_count // max(channels, 1)
    samples = []
    if sample_width_bytes == 2:
        usable = sample_count * 2
        for offset in range(0, usable, 2):
            samples.append(int.from_bytes(pcm[offset : offset + 2], "little", signed=True) / 32768.0)
    rms = math.sqrt(sum(sample * sample for sample in samples) / max(len(samples), 1))
    peak = max((abs(sample) for sample in samples), default=0.0)
    return {
        "channels": channels,
        "sample_width_bytes": sample_width_bytes,
        "sample_rate": sample_rate,
        "frames": frame_count,
        "duration_s": frame_count / sample_rate,
        "rms_dbfs": 20 * math.log10(max(rms, 1e-9)),
        "peak_dbfs": 20 * math.log10(max(peak, 1e-9)),
    }


def verify_with_openai_responses(api_key: str, wav_path: Path, scenario: dict, model: str) -> dict:
    audio_b64 = base64.b64encode(wav_path.read_bytes()).decode("ascii")
    prompt = (
        "You are evaluating a generated fitness-coach audio sample. "
        "Listen to the audio and return compact JSON with these keys: "
        "transcript, style, relative_speed, relative_loudness, voice_identity_notes, "
        "matches_expected, problems. "
        f"Expected properties: {json.dumps(scenario['expected'], ensure_ascii=False)}. "
        "For matches_expected, use true only if the words and requested vocal characteristic are both clear."
    )
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
    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=120,
    )
    if response.status_code >= 400:
        return {
            "ok": False,
            "model": model,
            "status": response.status_code,
            "error": response.text[:1000],
        }
    body = response.json()
    texts = []
    for item in body.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                texts.append(content.get("text", ""))
    return {
        "ok": True,
        "model": model,
        "text": "\n".join(texts).strip(),
        "response_id": body.get("id"),
    }


def verify_with_openai_chat_audio(api_key: str, wav_path: Path, scenario: dict, model: str) -> dict:
    audio_b64 = base64.b64encode(wav_path.read_bytes()).decode("ascii")
    prompt = (
        "The audio is attached in this message. Listen to it directly now; do not ask me to play or upload it again. "
        "Return only strict JSON with these keys: "
        "transcript, style, relative_speed, relative_loudness, voice_identity_notes, "
        "matches_expected, problems. "
        f"Expected properties: {json.dumps(scenario['expected'], ensure_ascii=False)}. "
        "For matches_expected, use true only if the words and requested vocal characteristic are both clear."
    )
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": "You are an audio evaluator. You must listen to attached audio and output only JSON.",
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
            }
        ],
    }
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=120,
    )
    if response.status_code >= 400:
        return {
            "ok": False,
            "model": model,
            "status": response.status_code,
            "error": response.text[:1000],
        }
    body = response.json()
    content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
    return {
        "ok": True,
        "model": model,
        "text": content,
        "response_id": body.get("id"),
        "usage": body.get("usage"),
    }


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale_wav in OUT_DIR.glob("*.wav"):
        stale_wav.unlink()
    manifest = {
        "tts_model": OPENAI_TTS_MODEL,
        "primary_verifier_model": OPENAI_VERIFIER_MODEL,
        "fallback_verifier_model": FALLBACK_VERIFIER_MODEL,
        "generated_at_unix": time.time(),
        "scenarios": [],
    }

    for scenario in SCENARIOS:
        wav_bytes, tts_meta = request_openai_speech(api_key, scenario)
        wav_path = OUT_DIR / f"{scenario['id']}.wav"
        wav_path.write_bytes(wav_bytes)
        primary = verify_with_openai_responses(api_key, wav_path, scenario, OPENAI_VERIFIER_MODEL)
        fallback = None
        if not primary.get("ok") and FALLBACK_VERIFIER_MODEL:
            fallback = verify_with_openai_chat_audio(api_key, wav_path, scenario, FALLBACK_VERIFIER_MODEL)
        manifest["scenarios"].append(
            {
                "id": scenario["id"],
                "wav_path": str(wav_path),
                "expected": scenario["expected"],
                "tts": tts_meta,
                "wav": wav_info(wav_path),
                "verification": {
                    "primary": primary,
                    "fallback": fallback,
                },
            }
        )

    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
