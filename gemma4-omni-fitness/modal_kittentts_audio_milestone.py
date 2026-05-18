from __future__ import annotations

import io
import json
import re
import time
import zipfile
from pathlib import Path

import modal


APP_NAME = "fitness-omni-audio-milestone"
MODEL_ID = "KittenML/kitten-tts-mini-0.8"
KITTENTTS_COMMIT = "9f3e0d8b6600b56ebe1b4d7b6d8e1e020077d1f2"
KITTENTTS_INSTALL = f"git+https://github.com/KittenML/KittenTTS.git@{KITTENTTS_COMMIT}"
VOICE = "Jasper"
SAMPLE_RATE = 24_000

COUNT_PHRASES = [
    ("count_01", "One."),
    ("count_02", "Two."),
    ("count_03", "Three."),
    ("count_04", "Four."),
    ("count_05", "Five."),
    ("count_06", "Six."),
    ("count_07", "Seven."),
    ("count_08", "Eight."),
    ("count_09", "Nine."),
    ("count_10", "Ten."),
    ("count_11", "Eleven."),
    ("count_12", "Twelve."),
    ("count_13", "Thirteen."),
    ("count_14", "Fourteen."),
    ("count_15", "Fifteen."),
    ("count_16", "Sixteen."),
    ("count_17", "Seventeen."),
    ("count_18", "Eighteen."),
    ("count_19", "Nineteen."),
    ("count_20", "Twenty."),
]

COACH_PHRASES = [
    ("count_02_nice_rhythm", "Two. Nice rhythm."),
    ("count_03_nice_rhythm", "Three. Nice rhythm."),
    ("count_05_keep_going", "Five. Keep going."),
    ("count_08_good_control", "Eight. Good control."),
    ("set_10_great_set", "That's ten. Great set."),
    ("keep_going", "Keep going."),
    ("nice_rhythm", "Nice rhythm."),
    ("good_control", "Good control."),
    ("great_set", "Great set."),
]

STREAM_BENCHMARK = [
    ("count_01", "One."),
    ("count_02_nice_rhythm", "Two. Nice rhythm."),
    ("set_10_great_set", "That's ten. Great set."),
    ("keep_going", "Keep going."),
]


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("espeak-ng", "git", "libsndfile1")
    .pip_install(
        KITTENTTS_INSTALL,
        "hf_xet",
    )
)

cache_volume = modal.Volume.from_name("fitness-omni-kittentts-cache", create_if_missing=True)
app = modal.App(APP_NAME)


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_")


@app.function(
    image=image,
    volumes={"/cache": cache_volume},
    cpu=4.0,
    memory=4096,
    timeout=900,
)
def render_phrase_bank() -> dict:
    import numpy as np
    import soundfile as sf
    from kittentts import KittenTTS

    started = time.perf_counter()
    model = KittenTTS(MODEL_ID, cache_dir="/cache/huggingface")
    load_s = time.perf_counter() - started

    zip_buffer = io.BytesIO()
    phrase_metrics: list[dict] = []
    manifest: list[dict] = []

    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for index, (phrase_id, text) in enumerate(COUNT_PHRASES + COACH_PHRASES, start=1):
            synth_started = time.perf_counter()
            audio = model.generate(text=text, voice=VOICE, speed=1.0)
            synth_s = time.perf_counter() - synth_started

            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            audio_s = len(audio) / SAMPLE_RATE
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")

            filename = f"phrase_bank/{index:03d}_{_safe_name(phrase_id)}.wav"
            zf.writestr(filename, wav_buffer.getvalue())

            row = {
                "phrase_id": phrase_id,
                "text": text,
                "filename": filename,
                "sample_rate": SAMPLE_RATE,
                "samples": int(len(audio)),
                "audio_s": audio_s,
                "synthesis_s": synth_s,
                "real_time_factor": synth_s / audio_s if audio_s else None,
                "voice": VOICE,
                "model": MODEL_ID,
            }
            manifest.append(row)
            phrase_metrics.append(row)

        stream_metrics: list[dict] = []
        has_stream = hasattr(model, "generate_stream")
        if has_stream:
            for phrase_id, text in STREAM_BENCHMARK:
                stream_started = time.perf_counter()
                first_chunk_s = None
                chunks = []
                for chunk in model.generate_stream(text=text, voice=VOICE, speed=1.0):
                    now = time.perf_counter()
                    if first_chunk_s is None:
                        first_chunk_s = now - stream_started
                    chunks.append(np.asarray(chunk, dtype=np.float32).reshape(-1))

                stream_total_s = time.perf_counter() - stream_started
                total_samples = int(sum(len(chunk) for chunk in chunks))
                audio_s = total_samples / SAMPLE_RATE
                stream_metrics.append(
                    {
                        "phrase_id": phrase_id,
                        "text": text,
                        "chunk_count": len(chunks),
                        "first_chunk_s": first_chunk_s,
                        "stream_total_s": stream_total_s,
                        "samples": total_samples,
                        "audio_s": audio_s,
                        "stream_real_time_factor": stream_total_s / audio_s if audio_s else None,
                    }
                )

        metrics = {
            "app": APP_NAME,
            "model": MODEL_ID,
            "kittentts_install": KITTENTTS_INSTALL,
            "kittentts_commit": KITTENTTS_COMMIT,
            "voice": VOICE,
            "sample_rate": SAMPLE_RATE,
            "model_load_s": load_s,
            "phrase_count": len(manifest),
            "phrase_metrics": phrase_metrics,
            "streaming_supported": has_stream,
            "stream_metrics": stream_metrics,
            "notes": [
                "Phrase-bank playback has near-zero synthesis latency in the live loop.",
                "Live synthesis latency is measured on Modal CPU, not on the eventual Pi or iPhone runtime.",
            ],
        }
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        zf.writestr("metrics.json", json.dumps(metrics, indent=2))

    cache_volume.commit()
    return {"metrics": metrics, "artifact_zip": zip_buffer.getvalue()}


@app.local_entrypoint()
def main(out_dir: str = "out/gemma4_omni_audio_milestone"):
    result = render_phrase_bank.remote()
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)

    zip_path = target / "gemma4_omni_kittentts_phrase_bank.zip"
    zip_path.write_bytes(result["artifact_zip"])

    with zipfile.ZipFile(io.BytesIO(result["artifact_zip"])) as zf:
        zf.extractall(target)

    metrics_path = target / "metrics.json"
    metrics_path.write_text(json.dumps(result["metrics"], indent=2) + "\n")

    summary = {
        "out_dir": str(target),
        "zip_path": str(zip_path),
        "model": result["metrics"]["model"],
        "voice": result["metrics"]["voice"],
        "model_load_s": result["metrics"]["model_load_s"],
        "phrase_count": result["metrics"]["phrase_count"],
        "streaming_supported": result["metrics"]["streaming_supported"],
        "stream_metrics": result["metrics"]["stream_metrics"],
    }
    print(json.dumps(summary, indent=2))
