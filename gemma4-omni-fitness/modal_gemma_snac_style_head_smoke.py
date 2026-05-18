from __future__ import annotations

import io
import json
import os
import time
import zipfile
from pathlib import Path
from typing import Any

import modal


APP_NAME = "gemma4-omni-snac-style-head-smoke"
GEMMA_MODEL_ID = "google/gemma-4-E2B-it"
SNAC_MODEL_ID = "hubertsiuzdak/snac_24khz"
TRAIN_GPU = os.environ.get("GEMMA4_OMNI_TRAIN_GPU", "L4")
MODAL_GPU_COST_PER_SECOND = {
    "T4": 0.000164,
    "L4": 0.000222,
    "A10": 0.000306,
    "A100-40GB": 0.000583,
    "A100": 0.000694,
    "H100": 0.001097,
}
SAMPLE_RATE = 24_000
SNAC_CODEBOOK_SIZE = 4096
STREAM_FIRST_LEVEL0_TOKENS = 20
STREAM_NEXT_LEVEL0_TOKENS = 40
STREAM_OVERLAP_LEVEL0_TOKENS = 10
RUNTIME_MODULE_PATH = Path(__file__).with_name("fitness_omni_audio_runtime.py")


train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libsndfile1")
    .pip_install(
        "torch",
        "transformers>=4.57.0",
        "accelerate",
        "bitsandbytes",
        "snac",
        "soundfile",
        "requests",
        "hf_xet",
    )
    .add_local_file(RUNTIME_MODULE_PATH, "/root/fitness_omni_audio_runtime.py")
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


def normalize_expected_json(value: dict[str, Any]) -> dict[str, Any]:
    expected = value.get("expected") or {}
    return {
        "transcript": str(expected.get("transcript") or value.get("target_text") or ""),
        "style": str(expected.get("style") or ""),
        "speed": str(expected.get("speed") or ""),
        "loudness": str(expected.get("loudness") or ""),
        "voice": str(expected.get("voice") or ""),
    }


def load_style_dataset(summary_path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    manifest = json.loads(summary_path.read_text())
    rows = []
    for pair in manifest.get("pairs", []):
        labels = pair["labels"]
        target_audio_path = Path(pair["audio"]["target"]["wav_path"])
        input_audio_path = Path(pair["audio"]["input"]["wav_path"])
        expected = normalize_expected_json(labels)
        rows.append(
            {
                "example_id": pair["id"],
                "home_demo_trigger": pair.get("home_demo_trigger", ""),
                "input_text": labels["input_text"],
                "input_direction": labels["input_direction"],
                "input_voice": labels["input_voice"],
                "target_text": labels["target_text"],
                "target_direction": labels["target_direction"],
                "target_voice": labels["target_voice"],
                "expected": expected,
                "input_wav_bytes": input_audio_path.read_bytes(),
                "target_wav_bytes": target_audio_path.read_bytes(),
                "input_source_path": str(input_audio_path),
                "target_source_path": str(target_audio_path),
            }
        )
    return rows[:limit] if limit else rows


def make_style_prompt(row: dict[str, Any]) -> str:
    expected = row["expected"]
    return (
        "<fitness_state>\n"
        "exercise: squat\n"
        "event: rep_complete\n"
        f"home_demo_trigger: {row['home_demo_trigger']}\n"
        f"user_text: {row['input_text']}\n"
        f"user_vocal_condition: {row['input_direction']}\n"
        f"target_text: {row['target_text']}\n"
        f"target_style: {expected['style']}\n"
        f"target_speed: {expected['speed']}\n"
        f"target_loudness: {expected['loudness']}\n"
        f"target_voice: {expected['voice']}\n"
        f"target_voice_id: {row['target_voice']}\n"
        "</fitness_state>\n"
        "Emit the target coach response as speech audio through the audio-output head.\n"
    )


@app.function(
    image=train_image,
    volumes={"/cache": cache_volume},
    secrets=[modal.Secret.from_local_environ(["OPENAI_API_KEY"])],
    gpu=TRAIN_GPU,
    cpu=6.0,
    memory=32_768,
    timeout=3600,
)
def train_style_projection_head(style_rows: list[dict[str, Any]], max_steps: int = 300) -> dict:
    import sys

    import numpy as np
    import requests
    import soundfile as sf
    import torch
    import torch.nn.functional as F
    from snac import SNAC
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    sys.path.insert(0, "/root")
    from fitness_omni_audio_runtime import (
        SNACProjectionHead,
        StreamingConfig,
        generate_streaming_chunks,
        predict_snac_codes,
        simulate_playback_queue,
    )

    started = time.perf_counter()
    device = torch.device("cuda")

    def wav_bytes_from_array(audio_np: np.ndarray) -> bytes:
        buffer = io.BytesIO()
        sf.write(buffer, audio_np.reshape(-1), SAMPLE_RATE, format="WAV", subtype="PCM_16")
        return buffer.getvalue()

    def wav_bytes_to_array(wav_bytes: bytes) -> np.ndarray:
        audio_np, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"expected {SAMPLE_RATE} Hz WAV, got {sample_rate}")
        audio_np = np.asarray(audio_np, dtype=np.float32)
        if audio_np.ndim == 2:
            audio_np = audio_np.mean(axis=1)
        return np.clip(audio_np.reshape(-1), -1.0, 1.0).astype(np.float32)

    def openai_transcribe_wav(name: str, wav_bytes: bytes) -> dict:
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
        return {"ok": True, "text": payload.get("text", ""), "raw": payload}

    snac = SNAC.from_pretrained(SNAC_MODEL_ID, cache_dir="/cache/huggingface").eval().to(device)

    train_rows = []
    max_level_lengths = [0, 0, 0]
    for source_row in style_rows:
        audio_np = wav_bytes_to_array(source_row["target_wav_bytes"])
        audio = torch.from_numpy(audio_np).to(device).view(1, 1, -1)
        with torch.inference_mode():
            codes = snac.encode(audio)
        code_values = []
        code_shapes = []
        for level, codebook in enumerate(codes):
            values = codebook.detach().cpu().reshape(-1).long()
            code_values.append(values)
            code_shapes.append(list(codebook.shape))
            max_level_lengths[level] = max(max_level_lengths[level], int(values.numel()))
        train_rows.append(
            {
                **{k: v for k, v in source_row.items() if k not in {"input_wav_bytes", "target_wav_bytes"}},
                "prompt": make_style_prompt(source_row),
                "code_values": code_values,
                "snac_code_shapes": code_shapes,
                "audio_samples": int(len(audio_np)),
                "audio_s": len(audio_np) / SAMPLE_RATE,
                "audio_np": audio_np,
            }
        )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(GEMMA_MODEL_ID, cache_dir="/cache/huggingface")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    gemma = AutoModelForCausalLM.from_pretrained(
        GEMMA_MODEL_ID,
        cache_dir="/cache/huggingface",
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    gemma.eval()

    cond_vectors = []
    for row in train_rows:
        encoded = tokenizer(row["prompt"], return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = gemma(**encoded, output_hidden_states=True, use_cache=False)
        cond_vectors.append(outputs.hidden_states[-1][0, -1].detach().float().cpu())

    del gemma
    torch.cuda.empty_cache()

    cond = torch.stack(cond_vectors).to(device)
    hidden_size = cond.shape[-1]
    labels_by_level = []
    mask_by_level = []
    for level, max_len in enumerate(max_level_lengths):
        labels = torch.full((len(train_rows), max_len), -100, dtype=torch.long)
        mask = torch.zeros((len(train_rows), max_len), dtype=torch.bool)
        for row_idx, row in enumerate(train_rows):
            values = row["code_values"][level]
            labels[row_idx, : values.numel()] = values
            mask[row_idx, : values.numel()] = True
        labels_by_level.append(labels.to(device))
        mask_by_level.append(mask.to(device))

    head = SNACProjectionHead(hidden_size=hidden_size, max_level_lengths=max_level_lengths).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    losses = []
    accuracies = []
    for step in range(max_steps):
        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), device=device)
        total_correct = 0
        total_count = 0
        for level in range(len(max_level_lengths)):
            logits = head.forward_level(cond, level)
            labels = labels_by_level[level]
            loss = F.cross_entropy(
                logits.reshape(-1, SNAC_CODEBOOK_SIZE),
                labels.reshape(-1),
                ignore_index=-100,
            )
            total_loss = total_loss + loss
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                mask = mask_by_level[level]
                total_correct += int((pred[mask] == labels[mask]).sum().item())
                total_count += int(mask.sum().item())
        total_loss.backward()
        optimizer.step()
        losses.append(float(total_loss.detach().cpu()))
        accuracies.append(total_correct / max(total_count, 1))
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == max_steps:
            print(
                json.dumps(
                    {
                        "train_step": step + 1,
                        "max_steps": max_steps,
                        "loss": losses[-1],
                        "token_accuracy": accuracies[-1],
                        "elapsed_s": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )

    head.eval()

    def decode_codes_to_audio(decode_codes: list[torch.Tensor]) -> tuple[np.ndarray, float]:
        decode_started = time.perf_counter()
        with torch.inference_mode():
            decoded_value = snac.decode(decode_codes).detach().float().cpu().numpy().reshape(-1)
        decode_ms_value = (time.perf_counter() - decode_started) * 1000
        decoded_value = np.clip(decoded_value, -1.0, 1.0).astype(np.float32)
        return decoded_value, decode_ms_value

    streaming_config = StreamingConfig(
        first_level0_tokens=STREAM_FIRST_LEVEL0_TOKENS,
        next_level0_tokens=STREAM_NEXT_LEVEL0_TOKENS,
        overlap_level0_tokens=STREAM_OVERLAP_LEVEL0_TOKENS,
    )

    generated_evaluations = []
    for row_index, reference_row in enumerate(train_rows):
        eval_id = reference_row["example_id"]
        head_started = time.perf_counter()
        decode_codes, level_token_counts, predicted_values_by_level = predict_snac_codes(
            head,
            cond[row_index : row_index + 1],
            reference_row["snac_code_shapes"],
        )
        head_inference_ms = (time.perf_counter() - head_started) * 1000
        decoded, decode_ms = decode_codes_to_audio(decode_codes)
        wav_bytes = wav_bytes_from_array(decoded)
        transcription = openai_transcribe_wav(f"{eval_id}.wav", wav_bytes)
        token_correct = 0
        token_total = 0
        for level, predicted in enumerate(predicted_values_by_level):
            target = reference_row["code_values"][level].tolist()
            token_total += len(target)
            token_correct += sum(1 for a, b in zip(predicted, target) if a == b)

        stream_chunks, stitched_streaming_audio, streaming_summary = generate_streaming_chunks(
            head=head,
            cond_vector=cond[row_index : row_index + 1],
            snac_code_shapes=reference_row["snac_code_shapes"],
            decode_codes_to_audio=decode_codes_to_audio,
            wav_bytes_from_array=wav_bytes_from_array,
            config=streaming_config,
            perf_counter=time.perf_counter,
        )
        playback_queue = simulate_playback_queue(stream_chunks)
        stitched_streaming_wav_bytes = wav_bytes_from_array(stitched_streaming_audio)
        stitched_streaming_transcription = openai_transcribe_wav(
            f"{eval_id}_streaming_stitched.wav",
            stitched_streaming_wav_bytes,
        )
        chunk_transcriptions = [
            openai_transcribe_wav(f"{eval_id}_chunk_{chunk['index']:02d}.wav", chunk["wav_bytes"])
            for chunk in stream_chunks
        ]
        generated_evaluations.append(
            {
                "eval_id": eval_id,
                "prompt": reference_row["prompt"],
                "reference_text": reference_row["target_text"],
                "expected": reference_row["expected"],
                "target_voice": reference_row["target_voice"],
                "target_direction": reference_row["target_direction"],
                "input_text": reference_row["input_text"],
                "input_direction": reference_row["input_direction"],
                "home_demo_trigger": reference_row["home_demo_trigger"],
                "generated_audio_s": len(decoded) / SAMPLE_RATE,
                "target_audio_s": reference_row["audio_s"],
                "level_token_counts": level_token_counts,
                "token_accuracy": token_correct / max(token_total, 1),
                "head_inference_ms": head_inference_ms,
                "snac_decode_ms": decode_ms,
                "streaming": {
                    "chunk_count": streaming_summary["chunk_count"],
                    "chunks": [
                        {k: v for k, v in chunk.items() if k != "wav_bytes"}
                        for chunk in stream_chunks
                    ],
                    "chunk_transcriptions": chunk_transcriptions,
                    "stitched_audio_s": streaming_summary["stitched_audio_s"],
                    "stitched_transcription": stitched_streaming_transcription,
                    "playback_queue": playback_queue,
                    "max_chunk_ready_ms": streaming_summary["max_chunk_ready_ms"],
                    "total_emitted_audio_s": streaming_summary["total_emitted_audio_s"],
                },
                "openai_transcription": transcription,
                "wav_bytes": wav_bytes,
                "streaming_wav_bytes": stitched_streaming_wav_bytes,
                "chunk_wav_bytes": [chunk["wav_bytes"] for chunk in stream_chunks],
            }
        )

    artifact_dir = Path("/tmp/gemma4_omni_style_projection_head")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "hidden_size": hidden_size,
            "max_level_lengths": max_level_lengths,
            "snac_codebook_size": SNAC_CODEBOOK_SIZE,
            "gemma_model_id": GEMMA_MODEL_ID,
            "snac_model_id": SNAC_MODEL_ID,
        },
        artifact_dir / "style_projection_head.pt",
    )

    manifest_rows = []
    for row in train_rows:
        manifest_rows.append(
            {
                "example_id": row["example_id"],
                "input_text": row["input_text"],
                "target_text": row["target_text"],
                "expected": row["expected"],
                "target_voice": row["target_voice"],
                "target_direction": row["target_direction"],
                "prompt": row["prompt"],
                "snac_code_shapes": row["snac_code_shapes"],
                "audio_samples": row["audio_samples"],
                "audio_s": row["audio_s"],
            }
        )

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(artifact_dir / "style_projection_head.pt", "style_projection_head.pt")
        for row in train_rows:
            zf.writestr(f"input_audio/{row['example_id']}.wav", next(
                source["input_wav_bytes"] for source in style_rows if source["example_id"] == row["example_id"]
            ))
            zf.writestr(f"target_audio/{row['example_id']}.wav", wav_bytes_from_array(row["audio_np"]))
        for evaluation in generated_evaluations:
            zf.writestr(f"generated_audio/{evaluation['eval_id']}.wav", evaluation["wav_bytes"])
            zf.writestr(
                f"generated_audio_streaming/{evaluation['eval_id']}_stitched.wav",
                evaluation["streaming_wav_bytes"],
            )
            for chunk_index, chunk_wav_bytes in enumerate(evaluation["chunk_wav_bytes"]):
                zf.writestr(
                    f"generated_audio_chunks/{evaluation['eval_id']}_chunk_{chunk_index:02d}.wav",
                    chunk_wav_bytes,
                )
        zf.writestr(
            "gemma4_omni_style_projection_head_manifest.json",
            json.dumps(
                {
                    "gemma_model_id": GEMMA_MODEL_ID,
                    "snac_model_id": SNAC_MODEL_ID,
                    "sample_rate": SAMPLE_RATE,
                    "streaming_config": {
                        "first_level0_tokens": STREAM_FIRST_LEVEL0_TOKENS,
                        "next_level0_tokens": STREAM_NEXT_LEVEL0_TOKENS,
                        "overlap_level0_tokens": STREAM_OVERLAP_LEVEL0_TOKENS,
                    },
                    "train_gpu": TRAIN_GPU,
                    "modal_gpu_cost_per_second": MODAL_GPU_COST_PER_SECOND.get(TRAIN_GPU),
                    "max_steps": max_steps,
                    "max_level_lengths": max_level_lengths,
                    "loss_first": losses[0] if losses else None,
                    "loss_last": losses[-1] if losses else None,
                    "token_accuracy_last": accuracies[-1] if accuracies else None,
                    "style_examples": manifest_rows,
                    "generated_evaluations": [
                        {
                            k: v
                            for k, v in evaluation.items()
                            if k not in {"wav_bytes", "streaming_wav_bytes", "chunk_wav_bytes"}
                        }
                        for evaluation in generated_evaluations
                    ],
                },
                indent=2,
            ),
        )

    elapsed_s = time.perf_counter() - started
    cache_volume.commit()
    return {
        "elapsed_s": elapsed_s,
        "gemma_model_id": GEMMA_MODEL_ID,
        "snac_model_id": SNAC_MODEL_ID,
        "streaming_config": {
            "first_level0_tokens": STREAM_FIRST_LEVEL0_TOKENS,
            "next_level0_tokens": STREAM_NEXT_LEVEL0_TOKENS,
            "overlap_level0_tokens": STREAM_OVERLAP_LEVEL0_TOKENS,
        },
        "train_gpu": TRAIN_GPU,
        "modal_gpu_cost_per_second": MODAL_GPU_COST_PER_SECOND.get(TRAIN_GPU),
        "max_steps": max_steps,
        "train_example_count": len(train_rows),
        "max_level_lengths": max_level_lengths,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "token_accuracy_last": accuracies[-1] if accuracies else None,
        "generated_evaluations": [
            {
                k: v
                for k, v in evaluation.items()
                if k not in {"wav_bytes", "streaming_wav_bytes", "chunk_wav_bytes"}
            }
            for evaluation in generated_evaluations
        ],
        "artifact_zip": zip_buffer.getvalue(),
    }


@app.local_entrypoint()
def main(
    dataset_summary: str = "out/gemma4_omni_gemini_tts_style_dataset/summary.json",
    out_dir: str = "out/gemma4_omni_style_projection_head_l4",
    max_steps: int = 300,
    limit: int = 0,
):
    load_dotenv(Path(".env"))
    style_rows = load_style_dataset(Path(dataset_summary), limit=limit or None)
    result = train_style_projection_head.remote(style_rows=style_rows, max_steps=max_steps)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    zip_path = target / "gemma4_omni_style_projection_head_artifact.zip"
    zip_path.write_bytes(result.pop("artifact_zip"))
    (target / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if (
                name.endswith(".json")
                or name.startswith("input_audio/")
                or name.startswith("target_audio/")
                or name.startswith("generated_audio/")
                or name.startswith("generated_audio_chunks/")
                or name.startswith("generated_audio_streaming/")
            ):
                zf.extract(name, target)
    print(json.dumps({**result, "zip_path": str(zip_path), "out_dir": str(target)}, indent=2))
