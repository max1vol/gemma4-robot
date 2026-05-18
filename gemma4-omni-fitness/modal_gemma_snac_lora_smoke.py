from __future__ import annotations

import io
import json
import os
import re
import time
import zipfile
from pathlib import Path

import modal


APP_NAME = "fitness-omni-gemma-snac-lora-smoke"
GEMMA_MODEL_ID = "google/gemma-4-E2B-it"
SNAC_MODEL_ID = "hubertsiuzdak/snac_24khz"
KITTENTTS_MODEL_ID = "KittenML/kitten-tts-mini-0.8"
KITTENTTS_COMMIT = "9f3e0d8b6600b56ebe1b4d7b6d8e1e020077d1f2"
KITTENTTS_INSTALL = f"git+https://github.com/KittenML/KittenTTS.git@{KITTENTTS_COMMIT}"
GPU_TYPE = os.environ.get("GEMMA4_OMNI_MODAL_GPU") or os.environ.get("FITNESS_OMNI_MODAL_GPU", "H100")
VOICE = "Jasper"
SAMPLE_RATE = 24_000

TRAIN_PHRASES = [
    ("count_01", "One."),
    ("count_02_nice_rhythm", "Two. Nice rhythm."),
    ("count_03", "Three."),
    ("count_05_keep_going", "Five. Keep going."),
    ("set_10_great_set", "That's ten. Great set."),
]


image = (
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
        "librosa",
        "torchvision",
        "hf_xet",
        KITTENTTS_INSTALL,
    )
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
        "Emit the speech audio as SNAC codec tokens only.\n"
    )


def token_to_text(level: int, token: int) -> str:
    return f"<snac_{level}_{int(token)}>"


def text_to_token(value: str) -> tuple[int, int]:
    _, level, token = value.strip("<>").split("_")
    return int(level), int(token)


def parse_audio_tokens(value: str) -> list[tuple[int, int]]:
    return [(int(level), int(token)) for level, token in re.findall(r"<snac_(\d+)_(\d+)>", value)]


@app.function(
    image=image,
    volumes={"/cache": cache_volume},
    secrets=[modal.Secret.from_local_environ(["OPENAI_API_KEY"])],
    gpu=GPU_TYPE,
    cpu=6.0,
    memory=32_768,
    timeout=3600,
)
def train_smoke(max_steps: int = 1, run_audio_input_probe: bool = True) -> dict:
    import numpy as np
    import requests
    import soundfile as sf
    import torch
    from kittentts import KittenTTS
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from snac import SNAC
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig

    device = torch.device("cuda")
    started = time.perf_counter()

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

    def openai_tts_audio(text: str) -> dict:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return {"ok": False, "error": "OPENAI_API_KEY is not set"}

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

        payload = {
            "model": "gpt-4o-mini-tts",
            "voice": "alloy",
            "input": text,
            "response_format": "wav",
        }
        response = requests.post(
            "https://api.openai.com/v1/audio/speech",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=120,
        )
        if response.status_code >= 400 and "response_format" in payload:
            payload.pop("response_format", None)
            response = requests.post(
                "https://api.openai.com/v1/audio/speech",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=120,
            )
        if response.status_code >= 400:
            return {"ok": False, "status": response.status_code, "error": response.text[:1000]}
        content_type = response.headers.get("content-type", "")
        canonical = canonicalize_wav(response.content)
        if canonical:
            audio, duration_s = canonical
            return {
                "ok": True,
                "audio": audio,
                "content_type": content_type,
                "extension": "wav",
                "duration_s": duration_s,
            }
        extension = "mp3"
        return {"ok": True, "audio": response.content, "content_type": content_type, "extension": extension}

    def wav_bytes_from_array(audio_np: np.ndarray) -> bytes:
        buffer = io.BytesIO()
        sf.write(buffer, audio_np.reshape(-1), SAMPLE_RATE, format="WAV", subtype="PCM_16")
        return buffer.getvalue()

    def decode_generated_tokens(generated_text: str, reference_row: dict) -> dict:
        parsed = parse_audio_tokens(generated_text)
        by_level: dict[int, list[int]] = {}
        for level, token in parsed:
            by_level.setdefault(level, []).append(token)

        codes = []
        level_counts = {}
        padded_or_truncated = False
        for level, shape in enumerate(reference_row["snac_code_shapes"]):
            required = int(np.prod(shape))
            values = by_level.get(level, [])
            level_counts[level] = len(values)
            if len(values) < required:
                values = values + [0] * (required - len(values))
                padded_or_truncated = True
            elif len(values) > required:
                values = values[:required]
                padded_or_truncated = True
            codes.append(torch.tensor(values, dtype=torch.long, device=device).view(*shape))

        with torch.inference_mode():
            decoded = snac.decode(codes).detach().float().cpu().numpy().reshape(-1)
        decoded = np.clip(decoded, -1.0, 1.0).astype(np.float32)
        return {
            "audio_np": decoded,
            "parsed_token_count": len(parsed),
            "level_token_counts": level_counts,
            "padded_or_truncated": padded_or_truncated,
        }

    def gemma_audio_input_probe(input_audio_path: Path) -> dict:
        if not run_audio_input_probe:
            return {"ok": False, "skipped": True, "reason": "run_audio_input_probe=false"}
        try:
            from transformers import AutoModelForMultimodalLM

            processor = AutoProcessor.from_pretrained(GEMMA_MODEL_ID, cache_dir="/cache/huggingface", padding_side="left")
            audio_model = AutoModelForMultimodalLM.from_pretrained(
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
                        {"type": "audio", "audio": str(input_audio_path)},
                        {
                            "type": "text",
                            "text": (
                                "Transcribe the speech exactly. Output only the spoken words, "
                                "with no commentary."
                            ),
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
            ).to(audio_model.device)
            input_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                output = audio_model.generate(**inputs, max_new_tokens=80, do_sample=False)
            response = processor.decode(output[0][input_len:], skip_special_tokens=True)
            parsed = response.strip()
            del audio_model
            torch.cuda.empty_cache()
            return {"ok": True, "transcript": parsed}
        except Exception as exc:
            torch.cuda.empty_cache()
            return {"ok": False, "error": repr(exc)}

    tts = KittenTTS(KITTENTTS_MODEL_ID, cache_dir="/cache/huggingface")
    snac = SNAC.from_pretrained(SNAC_MODEL_ID, cache_dir="/cache/huggingface").eval().to(device)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    openai_audio_input = openai_tts_audio("Hello, count my squats.")
    openai_audio_input_path = Path(f"/tmp/openai_audio_input.{openai_audio_input.get('extension', 'wav')}")
    if openai_audio_input.get("ok"):
        openai_audio_input_path.write_bytes(openai_audio_input["audio"])
    audio_input_probe = (
        gemma_audio_input_probe(openai_audio_input_path)
        if openai_audio_input.get("ok")
        else {"ok": False, "error": "OpenAI TTS input generation failed", "tts_result": openai_audio_input}
    )

    phrase_rows = []
    all_audio_token_texts: set[str] = set()
    for phrase_id, text in TRAIN_PHRASES:
        audio_np = np.asarray(tts.generate(text=text, voice=VOICE, speed=1.0), dtype=np.float32).reshape(-1)
        audio = torch.from_numpy(audio_np).to(device).view(1, 1, -1)
        with torch.inference_mode():
            codes = snac.encode(audio)

        flattened = []
        code_shapes = []
        for level, codebook in enumerate(codes):
            values = codebook.detach().cpu().reshape(-1).tolist()
            code_shapes.append(list(codebook.shape))
            for value in values:
                token_text = token_to_text(level, value)
                all_audio_token_texts.add(token_text)
                flattened.append(token_text)

        phrase_rows.append(
            {
                "phrase_id": phrase_id,
                "text": text,
                "prompt": make_prompt(phrase_id, text),
                "audio_tokens": flattened,
                "audio_token_count": len(flattened),
                "snac_code_shapes": code_shapes,
                "audio_samples": int(len(audio_np)),
                "audio_s": len(audio_np) / SAMPLE_RATE,
                "audio_np": audio_np,
            }
        )

    tokenizer = AutoTokenizer.from_pretrained(GEMMA_MODEL_ID, cache_dir="/cache/huggingface")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    audio_specials = sorted(all_audio_token_texts)
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                "<audio_start>",
                "<audio_end>",
                *audio_specials,
            ]
        }
    )
    audio_token_ids = tokenizer.convert_tokens_to_ids(["<audio_start>", "<audio_end>", *audio_specials])

    model = AutoModelForCausalLM.from_pretrained(
        GEMMA_MODEL_ID,
        cache_dir="/cache/huggingface",
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    # Gemma 4's default mean/covariance embedding initialization allocates
    # several extra GiB during vocab expansion. Random init is enough for this
    # smoke run because the added SNAC tokens are trained immediately.
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        ensure_weight_tying=True,
        target_modules=[
            "q_proj.linear",
            "k_proj.linear",
            "v_proj.linear",
            "o_proj.linear",
            "gate_proj.linear",
            "up_proj.linear",
            "down_proj.linear",
        ],
        trainable_token_indices=audio_token_ids,
    )
    model = get_peft_model(model, lora_config)
    try:
        model.print_trainable_parameters()
    except Exception as exc:
        print(f"Could not print trainable parameter summary: {exc!r}", flush=True)
    model.train()

    examples = []
    for row in phrase_rows:
        target = "<audio_start> " + " ".join(row["audio_tokens"]) + " <audio_end>"
        full_text = row["prompt"] + target
        prompt_ids = tokenizer(row["prompt"], add_special_tokens=False).input_ids
        full = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=2048)
        input_ids = torch.tensor(full.input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[: len(prompt_ids)] = -100
        examples.append({"input_ids": input_ids, "labels": labels, "phrase_id": row["phrase_id"]})

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    losses = []
    for step in range(max_steps):
        example = examples[step % len(examples)]
        input_ids = example["input_ids"].unsqueeze(0).to(device)
        labels = example["labels"].unsqueeze(0).to(device)
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == max_steps:
            print(
                json.dumps(
                    {
                        "train_step": step + 1,
                        "max_steps": max_steps,
                        "loss": losses[-1],
                        "elapsed_s": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )

    model.eval()
    generated_evaluations = []
    eval_requests = [
        ("text_state_count_01", make_prompt("count_01", "One."), phrase_rows[0]),
    ]
    if audio_input_probe.get("ok"):
        inferred = str(audio_input_probe.get("transcript") or "").strip()
        eval_requests.append(
            (
                "audio_input_to_audio_output",
                make_prompt("count_02_nice_rhythm", "Two. Nice rhythm.")
                + f"Audio input transcript from Gemma audio frontend: {inferred}\n",
                phrase_rows[1],
            )
        )

    for eval_id, test_prompt, reference_row in eval_requests:
        encoded = tokenizer(test_prompt + "<audio_start>", return_tensors="pt").to(device)
        max_new_tokens = min(reference_row["audio_token_count"] + 1, 1024)
        min_new_tokens = min(reference_row["audio_token_count"], max_new_tokens)
        allowed_decode_token_ids = sorted(
            set(tokenizer.convert_tokens_to_ids(["<audio_end>", *audio_specials]))
        )

        def prefix_allowed_tokens_fn(_batch_id, _input_ids):
            return allowed_decode_token_ids

        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.convert_tokens_to_ids("<audio_end>"),
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            )
        generated_text = tokenizer.decode(generated[0][encoded.input_ids.shape[1] :], skip_special_tokens=False)
        decoded = decode_generated_tokens(generated_text, reference_row)
        wav_bytes = wav_bytes_from_array(decoded["audio_np"])
        transcription = openai_transcribe_wav(f"{eval_id}.wav", wav_bytes)
        generated_evaluations.append(
            {
                "eval_id": eval_id,
                "prompt": test_prompt,
                "reference_phrase_id": reference_row["phrase_id"],
                "reference_text": reference_row["text"],
                "generated_text": generated_text,
                "generated_audio_s": len(decoded["audio_np"]) / SAMPLE_RATE,
                "parsed_token_count": decoded["parsed_token_count"],
                "level_token_counts": decoded["level_token_counts"],
                "padded_or_truncated": decoded["padded_or_truncated"],
                "openai_transcription": transcription,
                "wav_bytes": wav_bytes,
            }
        )

    adapter_dir = Path("/tmp/fitness_omni_gemma_snac_lora")
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in adapter_dir.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(adapter_dir))
        manifest = []
        for row in phrase_rows:
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, row["audio_np"], SAMPLE_RATE, format="WAV", subtype="PCM_16")
            zf.writestr(f"target_audio/{row['phrase_id']}.wav", wav_buffer.getvalue())
            clean = {k: v for k, v in row.items() if k != "audio_np"}
            manifest.append(clean)
        if openai_audio_input.get("ok"):
            zf.writestr(f"input_audio/hello_count_my_squats.{openai_audio_input['extension']}", openai_audio_input["audio"])
        for evaluation in generated_evaluations:
            zf.writestr(f"generated_audio/{evaluation['eval_id']}.wav", evaluation["wav_bytes"])
        zf.writestr(
            "gemma4_omni_audio_training_manifest.json",
            json.dumps(
                {
                    "gemma_model_id": GEMMA_MODEL_ID,
                    "snac_model_id": SNAC_MODEL_ID,
                    "kittentts_model_id": KITTENTTS_MODEL_ID,
                    "voice": VOICE,
                    "sample_rate": SAMPLE_RATE,
                    "gpu_type": GPU_TYPE,
                    "max_steps": max_steps,
                    "phrases": manifest,
                    "losses": losses,
                    "openai_audio_input": {k: v for k, v in openai_audio_input.items() if k != "audio"},
                    "gemma_audio_input_probe": audio_input_probe,
                    "generated_evaluations": [
                        {k: v for k, v in evaluation.items() if k != "wav_bytes"}
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
        "gpu_type": GPU_TYPE,
        "train_phrase_count": len(phrase_rows),
        "unique_audio_tokens": len(audio_specials),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "losses": losses,
        "openai_audio_input": {k: v for k, v in openai_audio_input.items() if k != "audio"},
        "gemma_audio_input_probe": audio_input_probe,
        "generated_evaluations": [
            {k: v for k, v in evaluation.items() if k != "wav_bytes"}
            for evaluation in generated_evaluations
        ],
        "artifact_zip": zip_buffer.getvalue(),
    }


@app.local_entrypoint()
def main(
    out_dir: str = "out/gemma4_omni_gemma_snac_lora_smoke",
    max_steps: int = 1,
    run_audio_input_probe: bool = True,
):
    result = train_smoke.remote(max_steps=max_steps, run_audio_input_probe=run_audio_input_probe)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    zip_path = target / "gemma_snac_lora_smoke_artifact.zip"
    zip_path.write_bytes(result.pop("artifact_zip"))
    (target / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        for name in names:
            if (
                name.endswith(".json")
                or name.startswith("target_audio/")
                or name.startswith("generated_audio/")
                or name.startswith("input_audio/")
            ):
                zf.extract(name, target)
    print(json.dumps({**result, "zip_path": str(zip_path), "out_dir": str(target)}, indent=2))
