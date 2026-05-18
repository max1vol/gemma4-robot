from __future__ import annotations

import io
import json
import os
import re
import time
import zipfile
from pathlib import Path
from typing import Any

import modal


APP_NAME = "gemma4-omni-llasa-xcodec2-prior-smoke"
LLASA_MODEL_ID = os.environ.get(
    "GEMMA4_OMNI_LLASA_MODEL",
    os.environ.get("GEMMA4_OMNI_LLASSA_MODEL", "HKUSTAudio/Llasa-1B"),
)
XCODEC2_MODEL_ID = "HKUSTAudio/xcodec2"
GPU = os.environ.get("GEMMA4_OMNI_LLASSA_GPU", "L4")
XCODEC2_SAMPLE_RATE = 16_000
STREAM_FIRST_TOKENS = 20
MODAL_GPU_COST_PER_SECOND = {
    "T4": 0.000164,
    "L4": 0.000222,
    "A10": 0.000306,
    "A100-40GB": 0.000583,
    "A100": 0.000694,
    "H100": 0.001097,
}


image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libsndfile1", "git")
    .pip_install(
        "torch==2.5.0",
        "torchaudio==2.5.0",
        "torchao==0.7.0",
        "transformers==4.49.0",
        "accelerate",
        "soundfile",
        "requests",
        "hf_xet",
        "xcodec2==0.1.5",
    )
)

cache_volume = modal.Volume.from_name("fitness-omni-gemma-audio-cache", create_if_missing=True)
app = modal.App(APP_NAME)


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"").strip("'"))


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


def load_manifest_pairs(summary_path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(summary_path.read_text())
    pairs = manifest.get("pairs")
    if isinstance(pairs, list):
        return pairs
    rows_path = summary_path.with_name("rows.jsonl")
    if not rows_path.exists():
        raise ValueError(f"{summary_path} has no pair list and {rows_path} does not exist")
    return [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]


def verification_matches(pair: dict[str, Any], key: str) -> bool:
    verification = pair.get("verification") or {}
    result = verification.get(key) or {}
    return result.get("matches_expected") is True


def normalize_expected_json(value: dict[str, Any]) -> dict[str, Any]:
    expected = value.get("expected") or {}
    return {
        "transcript": str(expected.get("transcript") or value.get("target_text") or ""),
        "style": str(expected.get("style") or ""),
        "speed": str(expected.get("speed") or ""),
        "loudness": str(expected.get("loudness") or ""),
        "voice": str(expected.get("voice") or ""),
    }


def load_style_rows(
    summary_path: Path,
    split: str,
    limit: int = 0,
    offset: int = 0,
    require_target_whisper_match: bool = False,
) -> list[dict[str, Any]]:
    rows = []
    for pair in load_manifest_pairs(summary_path):
        if require_target_whisper_match and not verification_matches(pair, "target_whisper"):
            continue
        labels = pair["labels"]
        rows.append(
            {
                "split": split,
                "example_id": pair["id"],
                "target_text": labels["target_text"],
                "target_direction": labels.get("target_direction", ""),
                "target_voice": labels.get("target_voice", ""),
                "home_demo_trigger": pair.get("home_demo_trigger", ""),
                "expected": normalize_expected_json(labels),
            }
        )
    rows = rows[offset:]
    return rows[:limit] if limit else rows


def safe_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")


@app.function(
    image=image,
    volumes={"/cache": cache_volume},
    secrets=[modal.Secret.from_local_environ(["OPENAI_API_KEY"])],
    gpu=GPU,
    cpu=6.0,
    memory=49_152,
    timeout=3600,
)
def run_llasa_xcodec2_prior_smoke(
    rows: list[dict[str, Any]],
    temperature: float = 0.9,
    top_p: float = 0.95,
    max_new_tokens: int = 220,
    do_sample: bool = True,
) -> dict[str, Any]:
    import numpy as np
    import requests
    import soundfile as sf
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from xcodec2.modeling_xcodec2 import XCodec2Model

    started = time.perf_counter()
    device = torch.device("cuda")

    def extract_speech_ids(token_strings: list[str]) -> list[int]:
        speech_ids = []
        for token_str in token_strings:
            if token_str.startswith("<|s_") and token_str.endswith("|>"):
                speech_ids.append(int(token_str[4:-2]))
        return speech_ids

    def wav_bytes_from_array(audio_np: np.ndarray, sample_rate: int = XCODEC2_SAMPLE_RATE) -> bytes:
        buffer = io.BytesIO()
        sf.write(buffer, audio_np.reshape(-1), sample_rate, format="WAV", subtype="PCM_16")
        return buffer.getvalue()

    def decode_tokens_to_wav(codec_model: Any, tokens: list[int]) -> tuple[bytes, float, float]:
        if not tokens:
            return wav_bytes_from_array(np.zeros((0,), dtype=np.float32)), 0.0, 0.0
        decode_started = time.perf_counter()
        speech_tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0).unsqueeze(0)
        with torch.inference_mode():
            decoded = codec_model.decode_code(speech_tokens).detach().float().cpu().numpy().reshape(-1)
        decode_ms = (time.perf_counter() - decode_started) * 1000
        decoded = np.clip(decoded, -1.0, 1.0).astype(np.float32)
        return wav_bytes_from_array(decoded), len(decoded) / XCODEC2_SAMPLE_RATE, decode_ms

    def openai_transcribe_wav(name: str, wav_bytes: bytes, expected_text: str) -> dict[str, Any]:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return {"ok": False, "error": "OPENAI_API_KEY is not set"}
        response = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            data={"model": "whisper-1"},
            files={"file": (name, wav_bytes, "audio/wav")},
            timeout=120,
        )
        if response.status_code >= 400:
            return {"ok": False, "status": response.status_code, "error": response.text[:1000]}
        payload = response.json()
        heard = payload.get("text", "")
        return {
            "ok": True,
            "model": "openai:whisper-1",
            "expected": expected_text,
            "heard": heard,
            "expected_normalized": normalize_transcript(expected_text),
            "heard_normalized": normalize_transcript(heard),
            "matches_expected": normalize_transcript(expected_text) == normalize_transcript(heard),
            "raw": payload,
        }

    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(LLASA_MODEL_ID, cache_dir="/cache/huggingface")
    model = AutoModelForCausalLM.from_pretrained(
        LLASA_MODEL_ID,
        cache_dir="/cache/huggingface",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    codec_model = XCodec2Model.from_pretrained(XCODEC2_MODEL_ID, cache_dir="/cache/huggingface")
    codec_model.eval().to(device)
    load_s = time.perf_counter() - load_started
    speech_end_id = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")

    evaluations = []
    artifact_files: list[tuple[str, bytes]] = []
    for row_index, row in enumerate(rows):
        target_text = row["target_text"]
        formatted_text = f"<|TEXT_UNDERSTANDING_START|>{target_text}<|TEXT_UNDERSTANDING_END|>"
        chat = [
            {"role": "user", "content": "Convert the text to speech:" + formatted_text},
            {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>"},
        ]
        input_ids = tokenizer.apply_chat_template(
            chat,
            tokenize=True,
            return_tensors="pt",
            continue_final_message=True,
        ).to(model.device)
        generate_started = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                max_length=int(input_ids.shape[-1]) + max_new_tokens,
                eos_token_id=speech_end_id,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
            )
        generation_ms = (time.perf_counter() - generate_started) * 1000
        generated_ids = outputs[0][input_ids.shape[1] :]
        if generated_ids.numel() and int(generated_ids[-1]) == speech_end_id:
            generated_ids = generated_ids[:-1]
        speech_token_strings = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        speech_ids = extract_speech_ids(speech_token_strings)
        generated_wav, generated_audio_s, decode_ms = decode_tokens_to_wav(codec_model, speech_ids)
        transcription = openai_transcribe_wav(
            f"{safe_filename(row['example_id'])}_llasa_generated.wav",
            generated_wav,
            target_text,
        )

        first_tokens = speech_ids[:STREAM_FIRST_TOKENS]
        first_wav, first_audio_s, first_decode_ms = decode_tokens_to_wav(codec_model, first_tokens)
        first_transcription = openai_transcribe_wav(
            f"{safe_filename(row['example_id'])}_llasa_first.wav",
            first_wav,
            target_text,
        )
        generated_name = f"generated_audio/{safe_filename(row['example_id'])}.wav"
        first_name = f"first_chunks/{safe_filename(row['example_id'])}.wav"
        artifact_files.append((generated_name, generated_wav))
        artifact_files.append((first_name, first_wav))
        evaluation = {
            "split": row["split"],
            "example_id": row["example_id"],
            "target_text": target_text,
            "expected": row["expected"],
            "llasa_model_id": LLASA_MODEL_ID,
            "xcodec2_model_id": XCODEC2_MODEL_ID,
            "input_token_count": int(input_ids.shape[-1]),
            "generated_token_count": int(generated_ids.numel()),
            "speech_token_count": len(speech_ids),
            "generated_audio_s": generated_audio_s,
            "generation_ms": generation_ms,
            "decode_ms": decode_ms,
            "generation_tokens_per_second": len(speech_ids) / max(generation_ms / 1000, 1e-9),
            "streaming": {
                "first_token_count": len(first_tokens),
                "first_audio_s": first_audio_s,
                "first_decode_ms": first_decode_ms,
                "first_transcription": first_transcription,
                "note": "first_ready_ms excludes incremental LM generation because this smoke uses model.generate.",
            },
            "openai_transcription": transcription,
            "generated_audio_path": generated_name,
            "first_chunk_path": first_name,
        }
        evaluations.append(evaluation)
        print(
            json.dumps(
                {
                    "index": row_index + 1,
                    "count": len(rows),
                    "example_id": row["example_id"],
                    "target_text": target_text,
                    "speech_tokens": len(speech_ids),
                    "generated_audio_s": generated_audio_s,
                    "whisper_match": transcription.get("matches_expected"),
                    "whisper_heard": transcription.get("heard"),
                    "generation_ms": generation_ms,
                    "decode_ms": decode_ms,
                }
            ),
            flush=True,
        )

    artifact_buffer = io.BytesIO()
    with zipfile.ZipFile(artifact_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in artifact_files:
            zf.writestr(name, content)
        zf.writestr(
            "llasa_xcodec2_prior_smoke_manifest.json",
            json.dumps(
                {
                    "llasa_model_id": LLASA_MODEL_ID,
                    "xcodec2_model_id": XCODEC2_MODEL_ID,
                    "sample_rate": XCODEC2_SAMPLE_RATE,
                    "gpu": GPU,
                    "evaluations": evaluations,
                },
                indent=2,
            ),
        )

    elapsed_s = time.perf_counter() - started
    cache_volume.commit()
    return {
        "elapsed_s": elapsed_s,
        "model_load_s": load_s,
        "llasa_model_id": LLASA_MODEL_ID,
        "xcodec2_model_id": XCODEC2_MODEL_ID,
        "xcodec2_sample_rate": XCODEC2_SAMPLE_RATE,
        "gpu": GPU,
        "modal_gpu_cost_per_second": MODAL_GPU_COST_PER_SECOND.get(GPU),
        "estimated_gpu_cost_usd": (
            elapsed_s * MODAL_GPU_COST_PER_SECOND[GPU]
            if GPU in MODAL_GPU_COST_PER_SECOND
            else None
        ),
        "example_count": len(evaluations),
        "whisper_verified_count": sum(1 for item in evaluations if item["openai_transcription"].get("ok")),
        "whisper_match_count": sum(
            1 for item in evaluations if item["openai_transcription"].get("matches_expected") is True
        ),
        "first_chunk_whisper_match_count": sum(
            1 for item in evaluations if item["streaming"]["first_transcription"].get("matches_expected") is True
        ),
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "evaluations": evaluations,
        "artifact_zip": artifact_buffer.getvalue(),
    }


@app.local_entrypoint()
def main(
    dataset_summary: str = "out/gemma4_omni_validation_40/summary.json",
    out_dir: str = "out/gemma4_omni_llasa_xcodec2_prior_smoke",
    split: str = "validation",
    limit: int = 4,
    offset: int = 0,
    require_target_whisper_match: bool = True,
    temperature: float = 0.9,
    top_p: float = 0.95,
    max_new_tokens: int = 220,
    do_sample: bool = True,
):
    load_dotenv(Path(".env"))
    rows = load_style_rows(
        Path(dataset_summary),
        split=split,
        limit=limit,
        offset=offset,
        require_target_whisper_match=require_target_whisper_match,
    )
    result = run_llasa_xcodec2_prior_smoke.remote(
        rows=rows,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
    )
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    zip_path = target / "gemma4_omni_llasa_xcodec2_prior_smoke_artifact.zip"
    zip_path.write_bytes(result.pop("artifact_zip"))
    (target / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith(".json") or name.startswith("generated_audio/") or name.startswith("first_chunks/"):
                zf.extract(name, target)
    print(json.dumps({**result, "zip_path": str(zip_path), "out_dir": str(target)}, indent=2))
