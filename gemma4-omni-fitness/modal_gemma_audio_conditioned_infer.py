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


APP_NAME = "gemma4-omni-audio-conditioned-infer"
DEFAULT_ARTIFACT = Path(
    "out/gemma4_omni_audio_conditioned_style_head_h100_200/"
    "gemma4_omni_audio_conditioned_style_head_artifact.zip"
)
GEMMA_MODEL_ID = "google/gemma-4-E2B-it"
SNAC_MODEL_ID = "hubertsiuzdak/snac_24khz"
INFER_GPU = os.environ.get("GEMMA4_OMNI_INFER_GPU", "H100")
MODAL_GPU_COST_PER_SECOND = {
    "T4": 0.000164,
    "L4": 0.000222,
    "A10": 0.000306,
    "A100-40GB": 0.000583,
    "A100": 0.000694,
    "H100": 0.001097,
}
SAMPLE_RATE = 24_000
STREAM_FIRST_LEVEL0_TOKENS = 20
STREAM_NEXT_LEVEL0_TOKENS = 40
STREAM_OVERLAP_LEVEL0_TOKENS = 10
RUNTIME_MODULE_PATH = Path(__file__).with_name("fitness_omni_audio_runtime.py")


infer_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libsndfile1")
    .pip_install(
        "torch",
        "transformers>=4.57.0",
        "accelerate",
        "snac",
        "soundfile",
        "requests",
        "hf_xet",
        "librosa",
        "torchvision",
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


def safe_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")


def select_examples(manifest: dict[str, Any], example_ids: str) -> list[dict[str, Any]]:
    examples = manifest.get("style_examples", [])
    if not example_ids:
        return examples
    wanted = [item.strip() for item in example_ids.split(",") if item.strip()]
    by_id = {example["example_id"]: example for example in examples}
    missing = [example_id for example_id in wanted if example_id not in by_id]
    if missing:
        raise ValueError(f"unknown example ids: {missing}; available: {sorted(by_id)}")
    return [by_id[example_id] for example_id in wanted]


def normalize_expected_json(value: dict[str, Any]) -> dict[str, Any]:
    expected = value.get("expected") or {}
    return {
        "transcript": str(expected.get("transcript") or value.get("target_text") or ""),
        "style": str(expected.get("style") or ""),
        "speed": str(expected.get("speed") or ""),
        "loudness": str(expected.get("loudness") or ""),
        "voice": str(expected.get("voice") or ""),
    }


def make_audio_conditioning_text(row: dict[str, Any], include_style_labels: bool = False) -> str:
    expected = row["expected"]
    lines = [
        "<fitness_state>",
        "exercise: squat",
        "event: rep_complete",
        f"home_demo_trigger: {row['home_demo_trigger']}",
        "audio_prompt: attached user-side audio contains the user's request and vocal delivery",
        f"target_text: {row['target_text']}",
    ]
    if include_style_labels:
        lines.extend(
            [
                f"target_style: {expected['style']}",
                f"target_speed: {expected['speed']}",
                f"target_loudness: {expected['loudness']}",
                f"target_voice: {expected['voice']}",
                f"target_voice_id: {row['target_voice']}",
            ]
        )
    lines.extend(
        [
            "</fitness_state>",
            "Use the attached audio prompt as conditioning for how the coach should speak.",
            "Emit the target coach response as speech audio through the audio-output head.",
        ]
    )
    return "\n".join(lines) + "\n"


def load_external_style_rows(
    summary_path: Path,
    limit: int = 0,
    example_ids: str = "",
) -> list[dict[str, Any]]:
    manifest = json.loads(summary_path.read_text())
    wanted = [item.strip() for item in example_ids.split(",") if item.strip()]
    rows = []
    for pair in manifest.get("pairs", []):
        if wanted and pair["id"] not in wanted:
            continue
        labels = pair["labels"]
        row = {
            "example_id": pair["id"],
            "home_demo_trigger": pair.get("home_demo_trigger", ""),
            "input_text": labels["input_text"],
            "input_direction": labels["input_direction"],
            "input_voice": labels["input_voice"],
            "target_text": labels["target_text"],
            "target_direction": labels["target_direction"],
            "target_voice": labels["target_voice"],
            "expected": normalize_expected_json(labels),
            "input_wav_bytes": Path(pair["audio"]["input"]["wav_path"]).read_bytes(),
            "target_wav_bytes": Path(pair["audio"]["target"]["wav_path"]).read_bytes(),
            "input_source_path": pair["audio"]["input"]["wav_path"],
            "target_source_path": pair["audio"]["target"]["wav_path"],
        }
        row["conditioning_text"] = make_audio_conditioning_text(row, include_style_labels=False)
        rows.append(row)
    if wanted:
        found = {row["example_id"] for row in rows}
        missing = [example_id for example_id in wanted if example_id not in found]
        if missing:
            raise ValueError(f"unknown external example ids: {missing}; available in {summary_path}")
    return rows[:limit] if limit else rows


@app.function(
    image=infer_image,
    volumes={"/cache": cache_volume},
    secrets=[modal.Secret.from_local_environ(["OPENAI_API_KEY"])],
    gpu=INFER_GPU,
    cpu=6.0,
    memory=49_152,
    timeout=1800,
)
def infer_audio_conditioned(
    artifact_zip_bytes: bytes,
    example_ids: str = "",
    input_audio_overrides: dict[str, bytes] | None = None,
    external_style_rows: list[dict[str, Any]] | None = None,
) -> dict:
    import sys

    import numpy as np
    import requests
    import soundfile as sf
    import torch
    from snac import SNAC
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    sys.path.insert(0, "/root")
    from fitness_omni_audio_runtime import (
        SNACMemorySequenceHead,
        SNACProjectionHead,
        SNACSequenceHead,
        StreamingConfig,
        generate_streaming_chunks_memory,
        generate_streaming_chunks,
        generate_streaming_chunks_sequence,
        predict_snac_codes_memory,
        predict_snac_codes,
        predict_snac_codes_sequence,
        simulate_playback_queue,
    )

    started = time.perf_counter()
    device = torch.device("cuda")
    artifact_dir = Path("/tmp/gemma4_omni_infer_artifact")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(artifact_zip_bytes)) as zf:
        zf.extractall(artifact_dir)

    memory_manifest_path = artifact_dir / "gemma4_omni_audio_conditioned_memory_sequence_head_manifest.json"
    sequence_manifest_path = artifact_dir / "gemma4_omni_audio_conditioned_sequence_head_manifest.json"
    projection_manifest_path = artifact_dir / "gemma4_omni_audio_conditioned_style_head_manifest.json"
    if memory_manifest_path.exists():
        manifest_path = memory_manifest_path
        checkpoint_path = artifact_dir / "audio_conditioned_memory_sequence_head.pt"
    elif sequence_manifest_path.exists():
        manifest_path = sequence_manifest_path
        checkpoint_path = artifact_dir / "audio_conditioned_sequence_head.pt"
    else:
        manifest_path = projection_manifest_path
        checkpoint_path = artifact_dir / "audio_conditioned_style_head.pt"
    manifest = json.loads(manifest_path.read_text())
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    head_type = str(checkpoint.get("head_type") or manifest.get("head_type") or "snac_projection_mlp")
    input_audio_overrides = input_audio_overrides or {}
    external_style_rows = external_style_rows or []

    def wav_bytes_from_array(audio_np: np.ndarray) -> bytes:
        buffer = io.BytesIO()
        sf.write(buffer, audio_np.reshape(-1), SAMPLE_RATE, format="WAV", subtype="PCM_16")
        return buffer.getvalue()

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

    processor = AutoProcessor.from_pretrained(GEMMA_MODEL_ID, cache_dir="/cache/huggingface", padding_side="left")
    gemma = AutoModelForMultimodalLM.from_pretrained(
        GEMMA_MODEL_ID,
        cache_dir="/cache/huggingface",
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    gemma.eval()
    snac = SNAC.from_pretrained(SNAC_MODEL_ID, cache_dir="/cache/huggingface").eval().to(device)

    if head_type == "snac_memory_sequence_gru_attention":
        head_cls = SNACMemorySequenceHead
    elif head_type == "snac_sequence_gru":
        head_cls = SNACSequenceHead
    else:
        head_cls = SNACProjectionHead
    head = head_cls(
        hidden_size=int(checkpoint["hidden_size"]),
        max_level_lengths=[int(v) for v in checkpoint["max_level_lengths"]],
        codebook_size=int(checkpoint["snac_codebook_size"]),
    ).to(device)
    head.load_state_dict(checkpoint["state_dict"])
    head.eval()

    examples = []
    if not external_style_rows:
        for example in select_examples(manifest, example_ids):
            examples.append({**example, "source": "packaged_train_example"})

    def wav_bytes_to_array(wav_bytes: bytes) -> np.ndarray:
        audio_np, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"expected {SAMPLE_RATE} Hz WAV, got {sample_rate}")
        audio_np = np.asarray(audio_np, dtype=np.float32)
        if audio_np.ndim == 2:
            audio_np = audio_np.mean(axis=1)
        return np.clip(audio_np.reshape(-1), -1.0, 1.0).astype(np.float32)

    for row in external_style_rows:
        audio_np = wav_bytes_to_array(row["target_wav_bytes"])
        audio = torch.from_numpy(audio_np).to(device).view(1, 1, -1)
        with torch.inference_mode():
            target_codes = snac.encode(audio)
        snac_code_shapes = [list(codebook.shape) for codebook in target_codes]
        for level, shape in enumerate(snac_code_shapes):
            required = int(np.prod(shape))
            max_allowed = int(head.max_level_lengths[level])
            if required > max_allowed:
                raise ValueError(
                    f"{row['example_id']} target has {required} level-{level} tokens, "
                    f"but packaged head supports only {max_allowed}"
                )
        examples.append(
            {
                "source": "external_style_dataset",
                "example_id": row["example_id"],
                "conditioning_text": row["conditioning_text"],
                "target_text": row["target_text"],
                "expected": row["expected"],
                "target_direction": row["target_direction"],
                "input_text": row["input_text"],
                "input_direction": row["input_direction"],
                "input_wav_bytes": row["input_wav_bytes"],
                "target_audio_s": len(audio_np) / SAMPLE_RATE,
                "snac_code_shapes": snac_code_shapes,
            }
        )

    streaming_config = StreamingConfig(
        first_level0_tokens=STREAM_FIRST_LEVEL0_TOKENS,
        next_level0_tokens=STREAM_NEXT_LEVEL0_TOKENS,
        overlap_level0_tokens=STREAM_OVERLAP_LEVEL0_TOKENS,
    )

    def decode_codes_to_audio(decode_codes: list[torch.Tensor]) -> tuple[np.ndarray, float]:
        decode_started = time.perf_counter()
        with torch.inference_mode():
            decoded_value = snac.decode(decode_codes).detach().float().cpu().numpy().reshape(-1)
        decode_ms_value = (time.perf_counter() - decode_started) * 1000
        decoded_value = np.clip(decoded_value, -1.0, 1.0).astype(np.float32)
        return decoded_value, decode_ms_value

    generated_evaluations = []
    input_dir = Path("/tmp/gemma4_omni_infer_inputs")
    input_dir.mkdir(parents=True, exist_ok=True)
    for example in examples:
        eval_id = example["example_id"]
        input_wav_bytes = input_audio_overrides.get(eval_id)
        if input_wav_bytes is None:
            input_wav_bytes = example.get("input_wav_bytes")
        if input_wav_bytes is None:
            input_wav_bytes = (artifact_dir / "input_audio" / f"{eval_id}.wav").read_bytes()
        input_path = input_dir / f"{safe_filename(eval_id)}_input.wav"
        input_path.write_bytes(input_wav_bytes)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": str(input_path)},
                    {"type": "text", "text": example["conditioning_text"]},
                ],
            }
        ]
        condition_started = time.perf_counter()
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(gemma.device)
        with torch.inference_mode():
            outputs = gemma(**inputs, output_hidden_states=True, use_cache=False)
        hidden = outputs.hidden_states[-1][0].detach().float()
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            valid_mask = attention_mask[0].detach().bool().to(hidden.device)
            hidden = hidden[valid_mask]
        cond = hidden[-1].to(device).unsqueeze(0)
        memory = hidden.to(device).unsqueeze(0)
        memory_mask = torch.ones((1, memory.shape[1]), dtype=torch.bool, device=device)
        condition_ms = (time.perf_counter() - condition_started) * 1000

        head_started = time.perf_counter()
        if head_type == "snac_memory_sequence_gru_attention":
            decode_codes, level_token_counts, _ = predict_snac_codes_memory(
                head,
                memory,
                memory_mask,
                example["snac_code_shapes"],
            )
        elif head_type == "snac_sequence_gru":
            decode_codes, level_token_counts, _ = predict_snac_codes_sequence(head, cond, example["snac_code_shapes"])
        else:
            decode_codes, level_token_counts, _ = predict_snac_codes(head, cond, example["snac_code_shapes"])
        head_inference_ms = (time.perf_counter() - head_started) * 1000
        decoded, decode_ms = decode_codes_to_audio(decode_codes)
        wav_bytes = wav_bytes_from_array(decoded)
        transcription = openai_transcribe_wav(f"{eval_id}.wav", wav_bytes)

        if head_type == "snac_memory_sequence_gru_attention":
            stream_chunks, stitched_audio, streaming_summary = generate_streaming_chunks_memory(
                head=head,
                memory_vectors=memory,
                memory_mask=memory_mask,
                snac_code_shapes=example["snac_code_shapes"],
                decode_codes_to_audio=decode_codes_to_audio,
                wav_bytes_from_array=wav_bytes_from_array,
                config=streaming_config,
                perf_counter=time.perf_counter,
            )
        elif head_type == "snac_sequence_gru":
            stream_chunks, stitched_audio, streaming_summary = generate_streaming_chunks_sequence(
                head=head,
                cond_vector=cond,
                snac_code_shapes=example["snac_code_shapes"],
                decode_codes_to_audio=decode_codes_to_audio,
                wav_bytes_from_array=wav_bytes_from_array,
                config=streaming_config,
                perf_counter=time.perf_counter,
            )
        else:
            stream_chunks, stitched_audio, streaming_summary = generate_streaming_chunks(
                head=head,
                cond_vector=cond,
                snac_code_shapes=example["snac_code_shapes"],
                decode_codes_to_audio=decode_codes_to_audio,
                wav_bytes_from_array=wav_bytes_from_array,
                config=streaming_config,
                perf_counter=time.perf_counter,
            )
        playback_queue = simulate_playback_queue(stream_chunks)
        stitched_wav_bytes = wav_bytes_from_array(stitched_audio)
        stitched_transcription = openai_transcribe_wav(f"{eval_id}_streaming_stitched.wav", stitched_wav_bytes)
        chunk_transcriptions = [
            openai_transcribe_wav(f"{eval_id}_chunk_{chunk['index']:02d}.wav", chunk["wav_bytes"])
            for chunk in stream_chunks
        ]

        generated_evaluations.append(
            {
                "eval_id": eval_id,
                "split": example.get("split"),
                "source": example.get("source", "unknown"),
                "conditioning_text": example["conditioning_text"],
                "reference_text": example["target_text"],
                "expected": example["expected"],
                "target_direction": example["target_direction"],
                "input_text": example["input_text"],
                "input_direction": example["input_direction"],
                "condition_ms": condition_ms,
                "input_token_count": int(inputs["input_ids"].shape[-1]),
                "generated_audio_s": len(decoded) / SAMPLE_RATE,
                "level_token_counts": level_token_counts,
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
                    "stitched_transcription": stitched_transcription,
                    "playback_queue": playback_queue,
                    "max_chunk_ready_ms": streaming_summary["max_chunk_ready_ms"],
                    "total_emitted_audio_s": streaming_summary["total_emitted_audio_s"],
                },
                "openai_transcription": transcription,
                "wav_bytes": wav_bytes,
                "streaming_wav_bytes": stitched_wav_bytes,
                "chunk_wav_bytes": [chunk["wav_bytes"] for chunk in stream_chunks],
            }
        )

    elapsed_s = time.perf_counter() - started
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "gemma4_omni_audio_conditioned_infer_manifest.json",
            json.dumps(
                {
                    "gemma_model_id": GEMMA_MODEL_ID,
                    "snac_model_id": SNAC_MODEL_ID,
                    "source_artifact_conditioning": checkpoint.get("conditioning"),
                    "source_artifact_include_style_labels": checkpoint.get("include_style_labels"),
                    "source_artifact_head_type": head_type,
                    "sample_rate": SAMPLE_RATE,
                    "infer_gpu": INFER_GPU,
                    "modal_gpu_cost_per_second": MODAL_GPU_COST_PER_SECOND.get(INFER_GPU),
                    "streaming_config": {
                        "first_level0_tokens": STREAM_FIRST_LEVEL0_TOKENS,
                        "next_level0_tokens": STREAM_NEXT_LEVEL0_TOKENS,
                        "overlap_level0_tokens": STREAM_OVERLAP_LEVEL0_TOKENS,
                    },
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

    return {
        "elapsed_s": elapsed_s,
        "gemma_model_id": GEMMA_MODEL_ID,
        "snac_model_id": SNAC_MODEL_ID,
        "source_artifact_conditioning": checkpoint.get("conditioning"),
        "source_artifact_include_style_labels": checkpoint.get("include_style_labels"),
        "source_artifact_head_type": head_type,
        "infer_gpu": INFER_GPU,
        "modal_gpu_cost_per_second": MODAL_GPU_COST_PER_SECOND.get(INFER_GPU),
        "example_count": len(examples),
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
    artifact_zip: str = str(DEFAULT_ARTIFACT),
    out_dir: str = "out/gemma4_omni_audio_conditioned_infer",
    example_ids: str = "",
    dataset_summary: str = "",
    limit: int = 0,
):
    load_dotenv(Path(".env"))
    artifact_bytes = Path(artifact_zip).read_bytes()
    external_rows = (
        load_external_style_rows(Path(dataset_summary), limit=limit, example_ids=example_ids)
        if dataset_summary
        else None
    )
    result = infer_audio_conditioned.remote(
        artifact_zip_bytes=artifact_bytes,
        example_ids="" if external_rows else example_ids,
        external_style_rows=external_rows,
    )
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    zip_path = target / "gemma4_omni_audio_conditioned_infer_artifact.zip"
    zip_path.write_bytes(result.pop("artifact_zip"))
    (target / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if (
                name.endswith(".json")
                or name.startswith("generated_audio/")
                or name.startswith("generated_audio_chunks/")
                or name.startswith("generated_audio_streaming/")
            ):
                zf.extract(name, target)
    print(json.dumps({**result, "zip_path": str(zip_path), "out_dir": str(target)}, indent=2))
