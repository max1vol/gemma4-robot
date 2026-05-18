from __future__ import annotations

import io
import json
import os
import time
import zipfile
from pathlib import Path

import modal


APP_NAME = "gemma4-omni-snac-projection-head-smoke"
GEMMA_MODEL_ID = "google/gemma-4-E2B-it"
SNAC_MODEL_ID = "hubertsiuzdak/snac_24khz"
KITTENTTS_MODEL_ID = "KittenML/kitten-tts-mini-0.8"
KITTENTTS_COMMIT = "9f3e0d8b6600b56ebe1b4d7b6d8e1e020077d1f2"
KITTENTTS_INSTALL = f"git+https://github.com/KittenML/KittenTTS.git@{KITTENTTS_COMMIT}"
TRAIN_GPU = os.environ.get("GEMMA4_OMNI_TRAIN_GPU", "L4")
AUDIO_GPU = os.environ.get("GEMMA4_OMNI_AUDIO_GPU", "H100")
MODAL_GPU_COST_PER_SECOND = {
    "T4": 0.000164,
    "L4": 0.000222,
    "A10": 0.000306,
    "A100-40GB": 0.000583,
    "A100": 0.000694,
    "H100": 0.001097,
}
VOICE = "Jasper"
SAMPLE_RATE = 24_000
SNAC_CODEBOOK_SIZE = 4096
STREAM_FIRST_LEVEL0_TOKENS = 20
STREAM_NEXT_LEVEL0_TOKENS = 40
STREAM_OVERLAP_LEVEL0_TOKENS = 10
RUNTIME_MODULE_PATH = Path(__file__).with_name("fitness_omni_audio_runtime.py")

TRAIN_PHRASES = [
    ("count_01", "One."),
    ("count_02_nice_rhythm", "Two. Nice rhythm."),
    ("count_03", "Three."),
    ("count_05_keep_going", "Five. Keep going."),
    ("set_10_great_set", "That's ten. Great set."),
]


base_train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("espeak-ng", "git", "libsndfile1")
    .pip_install(
        "torch",
        "transformers>=4.57.0",
        "accelerate",
        "peft",
        "bitsandbytes",
        "snac",
        "soundfile",
        "requests",
        "hf_xet",
        KITTENTTS_INSTALL,
    )
)
train_image = base_train_image.add_local_file(RUNTIME_MODULE_PATH, "/root/fitness_omni_audio_runtime.py")
audio_probe_image = base_train_image.pip_install("librosa", "torchvision").add_local_file(
    RUNTIME_MODULE_PATH,
    "/root/fitness_omni_audio_runtime.py",
)

cache_volume = modal.Volume.from_name("fitness-omni-gemma-audio-cache", create_if_missing=True)
app = modal.App(APP_NAME)


def make_prompt(phrase_id: str, text: str) -> str:
    return (
        "<fitness_state>\n"
        "exercise: squat\n"
        "event: rep_complete\n"
        f"phrase_id: {phrase_id}\n"
        f"spoken_text: {text}\n"
        "</fitness_state>\n"
        "Emit speech audio through the audio-output head.\n"
    )


@app.function(
    image=audio_probe_image,
    volumes={"/cache": cache_volume},
    secrets=[modal.Secret.from_local_environ(["OPENAI_API_KEY"])],
    gpu=AUDIO_GPU,
    cpu=4.0,
    memory=32_768,
    timeout=900,
)
def audio_input_probe_remote(prompt_text: str = "Hello, count my squats.") -> dict:
    import numpy as np
    import requests
    import soundfile as sf
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    def canonicalize_wav(content: bytes) -> tuple[bytes, float] | None:
        if content[:4] != b"RIFF" or content[8:12] != b"WAVE":
            return None
        channels = int.from_bytes(content[22:24], "little")
        sample_rate = int.from_bytes(content[24:28], "little")
        bits_per_sample = int.from_bytes(content[34:36], "little")
        data_pos = content.find(b"data")
        if channels <= 0 or sample_rate <= 0 or bits_per_sample != 16 or data_pos < 0:
            return None
        pcm = content[data_pos + 8 :]
        sample_count = len(pcm) // 2
        usable = sample_count - (sample_count % channels)
        samples = np.frombuffer(pcm[: usable * 2], dtype="<i2").reshape(-1, channels)
        audio = samples.astype(np.float32) / 32768.0
        min_samples = int(sample_rate * 2.4)
        if audio.shape[0] < min_samples:
            audio = np.pad(audio, ((0, min_samples - audio.shape[0]), (0, 0)))
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
        return buffer.getvalue(), float(audio.shape[0] / sample_rate)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {"ok": False, "error": "OPENAI_API_KEY is not set"}

    tts_response = requests.post(
        "https://api.openai.com/v1/audio/speech",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        data=json.dumps(
            {
                "model": "gpt-4o-mini-tts",
                "voice": "alloy",
                "input": prompt_text,
                "response_format": "wav",
            }
        ),
        timeout=120,
    )
    if tts_response.status_code >= 400:
        return {"ok": False, "status": tts_response.status_code, "error": tts_response.text[:1000]}
    canonical = canonicalize_wav(tts_response.content)
    if not canonical:
        return {"ok": False, "error": "OpenAI TTS did not return canonicalizable WAV"}

    wav_bytes, duration_s = canonical
    audio_path = Path("/tmp/openai_tts_audio_input.wav")
    audio_path.write_bytes(wav_bytes)

    processor = AutoProcessor.from_pretrained(GEMMA_MODEL_ID, cache_dir="/cache/huggingface", padding_side="left")
    model = AutoModelForMultimodalLM.from_pretrained(
        GEMMA_MODEL_ID,
        cache_dir="/cache/huggingface",
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": str(audio_path)},
                {
                    "type": "text",
                    "text": "Transcribe the speech exactly. Output only the spoken words, with no commentary.",
                },
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    transcript = processor.decode(output[0][input_len:], skip_special_tokens=True).strip()
    return {
        "ok": True,
        "prompt_text": prompt_text,
        "transcript": transcript,
        "content_type": tts_response.headers.get("content-type", ""),
        "extension": "wav",
        "duration_s": duration_s,
        "audio": wav_bytes,
        "audio_gpu": AUDIO_GPU,
    }


@app.function(
    image=train_image,
    volumes={"/cache": cache_volume},
    secrets=[modal.Secret.from_local_environ(["OPENAI_API_KEY"])],
    gpu=TRAIN_GPU,
    cpu=6.0,
    memory=32_768,
    timeout=3600,
)
def train_projection_head(
    max_steps: int = 300,
    audio_input_probe: dict | None = None,
) -> dict:
    import sys

    import numpy as np
    import requests
    import soundfile as sf
    import torch
    import torch.nn.functional as F
    from kittentts import KittenTTS
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

    tts = KittenTTS(KITTENTTS_MODEL_ID, cache_dir="/cache/huggingface")
    snac = SNAC.from_pretrained(SNAC_MODEL_ID, cache_dir="/cache/huggingface").eval().to(device)

    phrase_rows = []
    max_level_lengths = [0, 0, 0]
    for phrase_id, text in TRAIN_PHRASES:
        audio_np = np.asarray(tts.generate(text=text, voice=VOICE, speed=1.0), dtype=np.float32).reshape(-1)
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
        phrase_rows.append(
            {
                "example_id": phrase_id,
                "phrase_id": phrase_id,
                "text": text,
                "prompt": make_prompt(phrase_id, text),
                "code_values": code_values,
                "snac_code_shapes": code_shapes,
                "audio_samples": int(len(audio_np)),
                "audio_s": len(audio_np) / SAMPLE_RATE,
                "audio_np": audio_np,
            }
        )

    if audio_input_probe and audio_input_probe.get("ok"):
        transcript = str(audio_input_probe.get("transcript") or "").strip()
        phrase_id, text = TRAIN_PHRASES[1]
        source = next(row for row in phrase_rows if row["phrase_id"] == phrase_id)
        phrase_rows.append(
            {
                **{k: v for k, v in source.items() if k != "example_id"},
                "example_id": "audio_input_to_audio_output",
                "prompt": source["prompt"] + f"Audio input transcript from Gemma audio frontend: {transcript}\n",
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
    for row in phrase_rows:
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
        labels = torch.full((len(phrase_rows), max_len), -100, dtype=torch.long)
        mask = torch.zeros((len(phrase_rows), max_len), dtype=torch.bool)
        for row_idx, row in enumerate(phrase_rows):
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
    generated_evaluations = []
    eval_ids = [row["example_id"] for row in phrase_rows]

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

    for eval_id in eval_ids:
        row_index = next(i for i, row in enumerate(phrase_rows) if row["example_id"] == eval_id)
        reference_row = phrase_rows[row_index]
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
        first_chunk = stream_chunks[0]
        first_chunk_audio, _ = sf.read(io.BytesIO(first_chunk["wav_bytes"]), dtype="float32")
        first_chunk_audio = np.asarray(first_chunk_audio, dtype=np.float32).reshape(-1)
        first_chunk_wav_bytes = wav_bytes_from_array(first_chunk_audio)
        first_chunk_transcription = openai_transcribe_wav(f"{eval_id}_first_chunk.wav", first_chunk_wav_bytes)
        stitched_streaming_wav_bytes = wav_bytes_from_array(stitched_streaming_audio)
        stitched_streaming_transcription = openai_transcribe_wav(
            f"{eval_id}_streaming_stitched.wav",
            stitched_streaming_wav_bytes,
        )
        chunk_transcriptions = []
        for chunk in stream_chunks:
            chunk_transcriptions.append(
                openai_transcribe_wav(
                    f"{eval_id}_chunk_{chunk['index']:02d}.wav",
                    chunk["wav_bytes"],
                )
            )
        generated_evaluations.append(
            {
                "eval_id": eval_id,
                "prompt": reference_row["prompt"],
                "reference_phrase_id": reference_row["phrase_id"],
                "reference_text": reference_row["text"],
                "generated_audio_s": len(decoded) / SAMPLE_RATE,
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
                    "first_chunk_level0_tokens": first_chunk["level0_tokens"],
                    "next_chunk_level0_tokens": STREAM_NEXT_LEVEL0_TOKENS,
                    "overlap_level0_tokens": STREAM_OVERLAP_LEVEL0_TOKENS,
                    "first_chunk_audio_s": len(first_chunk_audio) / SAMPLE_RATE,
                    "first_chunk_level_token_counts": first_chunk["level_token_counts"],
                    "first_chunk_head_ms": first_chunk["head_ms"],
                    "first_chunk_snac_decode_ms": first_chunk["snac_decode_ms"],
                    "first_chunk_ready_ms": first_chunk["ready_ms"],
                    "first_chunk_transcription": first_chunk_transcription,
                },
                "openai_transcription": transcription,
                "wav_bytes": wav_bytes,
                "first_chunk_wav_bytes": first_chunk_wav_bytes,
                "streaming_wav_bytes": stitched_streaming_wav_bytes,
                "chunk_wav_bytes": [chunk["wav_bytes"] for chunk in stream_chunks],
            }
        )

    artifact_dir = Path("/tmp/gemma4_omni_projection_head")
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
        artifact_dir / "projection_head.pt",
    )

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(artifact_dir / "projection_head.pt", "projection_head.pt")
        if audio_input_probe and audio_input_probe.get("audio"):
            zf.writestr("input_audio/hello_count_my_squats.wav", audio_input_probe["audio"])
        manifest_rows = []
        for row in phrase_rows:
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, row["audio_np"], SAMPLE_RATE, format="WAV", subtype="PCM_16")
            zf.writestr(f"target_audio/{row['example_id']}.wav", wav_buffer.getvalue())
            manifest_rows.append(
                {
                    "example_id": row["example_id"],
                    "phrase_id": row["phrase_id"],
                    "text": row["text"],
                    "prompt": row["prompt"],
                    "snac_code_shapes": row["snac_code_shapes"],
                    "audio_samples": row["audio_samples"],
                    "audio_s": row["audio_s"],
                }
            )
        for evaluation in generated_evaluations:
            zf.writestr(f"generated_audio/{evaluation['eval_id']}.wav", evaluation["wav_bytes"])
            zf.writestr(
                f"generated_audio_chunks/{evaluation['eval_id']}_first_chunk.wav",
                evaluation["first_chunk_wav_bytes"],
            )
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
            "gemma4_omni_projection_head_manifest.json",
            json.dumps(
                {
                    "gemma_model_id": GEMMA_MODEL_ID,
                    "snac_model_id": SNAC_MODEL_ID,
                    "kittentts_model_id": KITTENTTS_MODEL_ID,
                    "voice": VOICE,
                    "sample_rate": SAMPLE_RATE,
                    "streaming_config": {
                        "first_level0_tokens": STREAM_FIRST_LEVEL0_TOKENS,
                        "next_level0_tokens": STREAM_NEXT_LEVEL0_TOKENS,
                        "overlap_level0_tokens": STREAM_OVERLAP_LEVEL0_TOKENS,
                    },
                    "train_gpu": TRAIN_GPU,
                    "audio_gpu": AUDIO_GPU,
                    "modal_gpu_cost_per_second": {
                        "train_gpu": MODAL_GPU_COST_PER_SECOND.get(TRAIN_GPU),
                        "audio_gpu": MODAL_GPU_COST_PER_SECOND.get(AUDIO_GPU),
                    },
                    "max_steps": max_steps,
                    "max_level_lengths": max_level_lengths,
                    "loss_first": losses[0] if losses else None,
                    "loss_last": losses[-1] if losses else None,
                    "token_accuracy_last": accuracies[-1] if accuracies else None,
                    "audio_input_probe": {k: v for k, v in (audio_input_probe or {}).items() if k != "audio"},
                    "phrases": manifest_rows,
                    "generated_evaluations": [
                        {
                            k: v
                            for k, v in evaluation.items()
                            if k not in {"wav_bytes", "first_chunk_wav_bytes", "streaming_wav_bytes", "chunk_wav_bytes"}
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
        "audio_gpu": AUDIO_GPU,
        "modal_gpu_cost_per_second": {
            "train_gpu": MODAL_GPU_COST_PER_SECOND.get(TRAIN_GPU),
            "audio_gpu": MODAL_GPU_COST_PER_SECOND.get(AUDIO_GPU),
        },
        "max_steps": max_steps,
        "train_example_count": len(phrase_rows),
        "max_level_lengths": max_level_lengths,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "token_accuracy_last": accuracies[-1] if accuracies else None,
        "audio_input_probe": {k: v for k, v in (audio_input_probe or {}).items() if k != "audio"},
        "generated_evaluations": [
            {
                k: v
                for k, v in evaluation.items()
                if k not in {"wav_bytes", "first_chunk_wav_bytes", "streaming_wav_bytes", "chunk_wav_bytes"}
            }
            for evaluation in generated_evaluations
        ],
        "artifact_zip": zip_buffer.getvalue(),
    }


@app.local_entrypoint()
def main(
    out_dir: str = "out/gemma4_omni_projection_head_smoke",
    max_steps: int = 300,
    run_audio_input_probe: bool = True,
    cached_audio_input_transcript: str = "",
):
    if cached_audio_input_transcript:
        probe = {
            "ok": True,
            "cached": True,
            "prompt_text": "Hello, count my squats.",
            "transcript": cached_audio_input_transcript,
        }
    elif run_audio_input_probe:
        probe = audio_input_probe_remote.remote()
    else:
        probe = {"ok": False, "skipped": True}
    result = train_projection_head.remote(max_steps=max_steps, audio_input_probe=probe)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    zip_path = target / "gemma4_omni_projection_head_artifact.zip"
    zip_path.write_bytes(result.pop("artifact_zip"))
    (target / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if (
                name.endswith(".json")
                or name.startswith("target_audio/")
                or name.startswith("generated_audio/")
                or name.startswith("generated_audio_chunks/")
                or name.startswith("generated_audio_streaming/")
                or name.startswith("input_audio/")
            ):
                zf.extract(name, target)
    print(json.dumps({**result, "zip_path": str(zip_path), "out_dir": str(target)}, indent=2))
