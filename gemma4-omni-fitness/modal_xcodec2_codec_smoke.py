from __future__ import annotations

import io
import json
import math
import os
import re
import time
import zipfile
from pathlib import Path
from typing import Any

import modal


APP_NAME = "gemma4-omni-xcodec2-codec-smoke"
XCODEC2_MODEL_ID = "HKUSTAudio/xcodec2"
XCODEC2_GPU = os.environ.get("GEMMA4_OMNI_XCODEC2_GPU", "L4")
SOURCE_SAMPLE_RATE = 24_000
XCODEC2_SAMPLE_RATE = 16_000
MODAL_GPU_COST_PER_SECOND = {
    "T4": 0.000164,
    "L4": 0.000222,
    "A10": 0.000306,
    "A100-40GB": 0.000583,
    "A100": 0.000694,
    "H100": 0.001097,
}


xcodec2_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libsndfile1", "git")
    .pip_install(
        "torch==2.5.0",
        "torchaudio==2.5.0",
        "torchao==0.7.0",
        "transformers==4.49.0",
        "accelerate",
        "soundfile",
        "librosa",
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


def resolve_manifest_path(summary_path: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return summary_path.parent / path


def load_codec_rows(
    summary_path: Path,
    split: str,
    limit: int = 0,
    require_target_whisper_match: bool = False,
) -> list[dict[str, Any]]:
    rows = []
    for pair in load_manifest_pairs(summary_path):
        if require_target_whisper_match and not verification_matches(pair, "target_whisper"):
            continue
        labels = pair["labels"]
        target_audio_path = resolve_manifest_path(summary_path, pair["audio"]["target"]["wav_path"])
        if target_audio_path is None:
            raise ValueError(f"missing target wav path for {pair['id']}")
        rows.append(
            {
                "split": split,
                "example_id": pair["id"],
                "target_text": labels["target_text"],
                "expected": labels.get("expected") or {"transcript": labels["target_text"]},
                "target_direction": labels.get("target_direction", ""),
                "target_voice": labels.get("target_voice", ""),
                "home_demo_trigger": pair.get("home_demo_trigger", ""),
                "source_path": str(target_audio_path),
                "target_wav_bytes": target_audio_path.read_bytes(),
                "source_audio_meta": pair["audio"]["target"],
            }
        )
    return rows[:limit] if limit else rows


def rms_and_peak(audio: list[float]) -> dict[str, float]:
    if not audio:
        return {"rms_dbfs": -180.0, "peak_dbfs": -180.0}
    rms = math.sqrt(sum(sample * sample for sample in audio) / len(audio))
    peak = max(abs(sample) for sample in audio)
    return {
        "rms_dbfs": 20 * math.log10(max(rms, 1e-9)),
        "peak_dbfs": 20 * math.log10(max(peak, 1e-9)),
    }


def safe_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")


@app.function(
    image=xcodec2_image,
    volumes={"/cache": cache_volume},
    secrets=[modal.Secret.from_local_environ(["OPENAI_API_KEY"])],
    gpu=XCODEC2_GPU,
    cpu=6.0,
    memory=49_152,
    timeout=3600,
)
def run_xcodec2_codec_smoke(rows: list[dict[str, Any]], verify_with_whisper: bool = True) -> dict[str, Any]:
    import numpy as np
    import requests
    import soundfile as sf
    import torch
    from xcodec2.modeling_xcodec2 import XCodec2Model

    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def wav_bytes_from_array(audio_np: np.ndarray, sample_rate: int) -> bytes:
        buffer = io.BytesIO()
        sf.write(buffer, audio_np.reshape(-1), sample_rate, format="WAV", subtype="PCM_16")
        return buffer.getvalue()

    def read_wav_mono(wav_bytes: bytes) -> tuple[np.ndarray, int]:
        audio_np, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
        audio_np = np.asarray(audio_np, dtype=np.float32)
        if audio_np.ndim == 2:
            audio_np = audio_np.mean(axis=1)
        return np.clip(audio_np.reshape(-1), -1.0, 1.0), int(sample_rate)

    def resample_to_xcodec2(audio_np: np.ndarray, source_sample_rate: int) -> np.ndarray:
        if source_sample_rate == XCODEC2_SAMPLE_RATE:
            return audio_np.astype(np.float32)
        import librosa

        return librosa.resample(
            audio_np.astype(np.float32),
            orig_sr=source_sample_rate,
            target_sr=XCODEC2_SAMPLE_RATE,
        ).astype(np.float32)

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
    model = XCodec2Model.from_pretrained(XCODEC2_MODEL_ID, cache_dir="/cache/huggingface")
    model.eval().to(device)
    load_s = time.perf_counter() - load_started

    evaluations = []
    artifact_files: list[tuple[str, bytes]] = []
    for row_index, row in enumerate(rows):
        source_np, source_rate = read_wav_mono(row["target_wav_bytes"])
        input_np = resample_to_xcodec2(source_np, source_rate)
        input_tensor = torch.from_numpy(input_np).float().unsqueeze(0).to(device)
        encode_started = time.perf_counter()
        with torch.inference_mode():
            vq_code = model.encode_code(input_waveform=input_tensor)
        encode_ms = (time.perf_counter() - encode_started) * 1000
        decode_started = time.perf_counter()
        with torch.inference_mode():
            reconstructed_tensor = model.decode_code(vq_code).detach().float().cpu()
        decode_ms = (time.perf_counter() - decode_started) * 1000
        reconstructed_np = reconstructed_tensor.numpy().reshape(-1).astype(np.float32)
        reconstructed_np = np.clip(reconstructed_np, -1.0, 1.0)
        reconstructed_wav = wav_bytes_from_array(reconstructed_np, XCODEC2_SAMPLE_RATE)
        input_16k_wav = wav_bytes_from_array(input_np, XCODEC2_SAMPLE_RATE)
        token_count = int(vq_code.detach().cpu().reshape(-1).numel())
        source_duration_s = len(source_np) / source_rate
        reconstructed_duration_s = len(reconstructed_np) / XCODEC2_SAMPLE_RATE
        eval_id = safe_filename(row["example_id"])
        whisper = (
            openai_transcribe_wav(f"{eval_id}_xcodec2_reconstructed.wav", reconstructed_wav, row["target_text"])
            if verify_with_whisper
            else {"ok": False, "skipped": True}
        )
        artifact_files.append((f"source_16k/{eval_id}.wav", input_16k_wav))
        artifact_files.append((f"reconstructed_audio/{eval_id}.wav", reconstructed_wav))
        evaluation = {
            "split": row["split"],
            "example_id": row["example_id"],
            "target_text": row["target_text"],
            "expected": row["expected"],
            "source_path": row["source_path"],
            "source_sample_rate": source_rate,
            "source_duration_s": source_duration_s,
            "xcodec2_sample_rate": XCODEC2_SAMPLE_RATE,
            "xcodec2_input_duration_s": len(input_np) / XCODEC2_SAMPLE_RATE,
            "reconstructed_duration_s": reconstructed_duration_s,
            "vq_code_shape": list(vq_code.shape),
            "vq_token_count": token_count,
            "vq_tokens_per_second": token_count / max(source_duration_s, 1e-9),
            "encode_ms": encode_ms,
            "decode_ms": decode_ms,
            "source_audio": rms_and_peak(source_np.tolist()),
            "reconstructed_audio": rms_and_peak(reconstructed_np.tolist()),
            "whisper": whisper,
        }
        evaluations.append(evaluation)
        print(
            json.dumps(
                {
                    "index": row_index + 1,
                    "count": len(rows),
                    "example_id": row["example_id"],
                    "target_text": row["target_text"],
                    "tokens": token_count,
                    "tokens_per_second": evaluation["vq_tokens_per_second"],
                    "whisper_match": whisper.get("matches_expected"),
                    "whisper_heard": whisper.get("heard"),
                    "encode_ms": encode_ms,
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
            "xcodec2_codec_smoke_manifest.json",
            json.dumps(
                {
                    "xcodec2_model_id": XCODEC2_MODEL_ID,
                    "sample_rate": XCODEC2_SAMPLE_RATE,
                    "gpu": XCODEC2_GPU,
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
        "xcodec2_model_id": XCODEC2_MODEL_ID,
        "source_sample_rate": SOURCE_SAMPLE_RATE,
        "xcodec2_sample_rate": XCODEC2_SAMPLE_RATE,
        "gpu": XCODEC2_GPU,
        "modal_gpu_cost_per_second": MODAL_GPU_COST_PER_SECOND.get(XCODEC2_GPU),
        "estimated_gpu_cost_usd": (
            elapsed_s * MODAL_GPU_COST_PER_SECOND[XCODEC2_GPU]
            if XCODEC2_GPU in MODAL_GPU_COST_PER_SECOND
            else None
        ),
        "example_count": len(evaluations),
        "whisper_verified_count": sum(1 for item in evaluations if item["whisper"].get("ok")),
        "whisper_match_count": sum(1 for item in evaluations if item["whisper"].get("matches_expected") is True),
        "evaluations": evaluations,
        "artifact_zip": artifact_buffer.getvalue(),
    }


@app.local_entrypoint()
def main(
    dataset_summary: str = "out/gemma4_omni_validation_40/summary.json",
    out_dir: str = "out/gemma4_omni_xcodec2_codec_smoke",
    split: str = "validation",
    limit: int = 8,
    require_target_whisper_match: bool = True,
    verify_with_whisper: bool = True,
):
    load_dotenv(Path(".env"))
    rows = load_codec_rows(
        Path(dataset_summary),
        split=split,
        limit=limit,
        require_target_whisper_match=require_target_whisper_match,
    )
    result = run_xcodec2_codec_smoke.remote(rows=rows, verify_with_whisper=verify_with_whisper)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    zip_path = target / "gemma4_omni_xcodec2_codec_smoke_artifact.zip"
    zip_path.write_bytes(result.pop("artifact_zip"))
    (target / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith(".json") or name.startswith("source_16k/") or name.startswith("reconstructed_audio/"):
                zf.extract(name, target)
    print(json.dumps({**result, "zip_path": str(zip_path), "out_dir": str(target)}, indent=2))
