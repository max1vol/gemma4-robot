from __future__ import annotations

import io
import json
import os
import random
import re
import time
import zipfile
from pathlib import Path
from typing import Any

import modal


APP_NAME = "gemma4-omni-llasa-lora-audio-head-smoke"
LLASA_MODEL_ID = os.environ.get(
    "GEMMA4_OMNI_LLASA_MODEL",
    os.environ.get("GEMMA4_OMNI_LLASSA_MODEL", "HKUSTAudio/Llasa-1B"),
)
XCODEC2_MODEL_ID = "HKUSTAudio/xcodec2"
GPU = os.environ.get("GEMMA4_OMNI_LLASA_LORA_GPU", "L4")
XCODEC2_SAMPLE_RATE = 16_000
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
        "peft==0.14.0",
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
        target_audio_path = resolve_manifest_path(summary_path, pair["audio"]["target"]["wav_path"])
        if target_audio_path is None:
            raise ValueError(f"missing target wav path for {pair['id']}")
        rows.append(
            {
                "split": split,
                "example_id": pair["id"],
                "target_text": labels["target_text"],
                "target_direction": labels.get("target_direction", ""),
                "target_voice": labels.get("target_voice", ""),
                "home_demo_trigger": pair.get("home_demo_trigger", ""),
                "expected": normalize_expected_json(labels),
                "target_wav_bytes": target_audio_path.read_bytes(),
                "target_source_path": str(target_audio_path),
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
    timeout=7200,
)
def train_llasa_lora_audio_head(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    max_steps: int = 80,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    lora_rank: int = 8,
    eval_train_limit: int = 4,
    eval_validation_limit: int = 4,
    temperature: float = 0.9,
    top_p: float = 0.95,
    max_new_tokens: int = 220,
) -> dict[str, Any]:
    import librosa
    import numpy as np
    import requests
    import soundfile as sf
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from xcodec2.modeling_xcodec2 import XCodec2Model

    started = time.perf_counter()
    device = torch.device("cuda")

    def ids_to_speech_tokens(speech_ids: list[int]) -> list[str]:
        return [f"<|s_{speech_id}|>" for speech_id in speech_ids]

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

    def read_wav_mono(wav_bytes: bytes) -> tuple[np.ndarray, int]:
        audio_np, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
        audio_np = np.asarray(audio_np, dtype=np.float32)
        if audio_np.ndim == 2:
            audio_np = audio_np.mean(axis=1)
        return np.clip(audio_np.reshape(-1), -1.0, 1.0), int(sample_rate)

    def resample_to_xcodec2(audio_np: np.ndarray, source_sample_rate: int) -> np.ndarray:
        if source_sample_rate == XCODEC2_SAMPLE_RATE:
            return audio_np.astype(np.float32)
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

    def make_prompt_ids(tokenizer: Any, target_text: str) -> torch.Tensor:
        formatted_text = f"<|TEXT_UNDERSTANDING_START|>{target_text}<|TEXT_UNDERSTANDING_END|>"
        chat = [
            {"role": "user", "content": "Convert the text to speech:" + formatted_text},
            {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>"},
        ]
        return tokenizer.apply_chat_template(
            chat,
            tokenize=True,
            return_tensors="pt",
            continue_final_message=True,
        )[0]

    def make_training_ids(tokenizer: Any, target_text: str, speech_ids: list[int]) -> torch.Tensor:
        formatted_text = f"<|TEXT_UNDERSTANDING_START|>{target_text}<|TEXT_UNDERSTANDING_END|>"
        speech_text = "<|SPEECH_GENERATION_START|>" + "".join(ids_to_speech_tokens(speech_ids)) + "<|SPEECH_GENERATION_END|>"
        chat = [
            {"role": "user", "content": "Convert the text to speech:" + formatted_text},
            {"role": "assistant", "content": speech_text},
        ]
        return torch.tensor(tokenizer.apply_chat_template(chat, tokenize=True), dtype=torch.long)

    def decode_tokens_to_wav(codec_model: Any, speech_ids: list[int]) -> tuple[bytes, float, float]:
        if not speech_ids:
            return wav_bytes_from_array(np.zeros((0,), dtype=np.float32)), 0.0, 0.0
        decode_started = time.perf_counter()
        tokens = torch.tensor(speech_ids, dtype=torch.long, device=device).unsqueeze(0).unsqueeze(0)
        with torch.inference_mode():
            decoded = codec_model.decode_code(tokens).detach().float().cpu().numpy().reshape(-1)
        decode_ms = (time.perf_counter() - decode_started) * 1000
        decoded = np.clip(decoded, -1.0, 1.0).astype(np.float32)
        return wav_bytes_from_array(decoded), len(decoded) / XCODEC2_SAMPLE_RATE, decode_ms

    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(LLASA_MODEL_ID, cache_dir="/cache/huggingface", padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        LLASA_MODEL_ID,
        cache_dir="/cache/huggingface",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device)
    model.config.use_cache = False
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 4,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    codec_model = XCodec2Model.from_pretrained(XCODEC2_MODEL_ID, cache_dir="/cache/huggingface")
    codec_model.eval().to(device)
    load_s = time.perf_counter() - load_started

    all_rows = train_rows + validation_rows
    prepared = []
    max_length = 0
    speech_generation_start_id = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")
    for row in all_rows:
        audio_np, source_rate = read_wav_mono(row["target_wav_bytes"])
        audio_16k = resample_to_xcodec2(audio_np, source_rate)
        audio_tensor = torch.from_numpy(audio_16k).float().unsqueeze(0).to(device)
        with torch.inference_mode():
            speech_ids = codec_model.encode_code(input_waveform=audio_tensor).detach().cpu().reshape(-1).long().tolist()
        input_ids = make_training_ids(tokenizer, row["target_text"], speech_ids)
        labels = torch.full_like(input_ids, -100)
        speech_start_positions = (input_ids == speech_generation_start_id).nonzero(as_tuple=True)[0]
        if speech_start_positions.numel() == 0:
            raise ValueError(f"no speech generation start token in {row['example_id']}")
        labels[int(speech_start_positions[0]) :] = input_ids[int(speech_start_positions[0]) :]
        max_length = max(max_length, int(input_ids.numel()))
        prepared.append(
            {
                **{k: v for k, v in row.items() if k not in {"target_wav_bytes"}},
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": torch.ones_like(input_ids),
                "speech_token_count": len(speech_ids),
                "source_audio_s": len(audio_np) / source_rate,
                "target_xcodec2_audio_s": len(audio_16k) / XCODEC2_SAMPLE_RATE,
            }
        )
    codec_model.to("cpu")
    del audio_tensor
    torch.cuda.empty_cache()

    def pad_batch(items: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(int(item["input_ids"].numel()) for item in items)
        input_ids = torch.full((len(items), max_len), tokenizer.pad_token_id, dtype=torch.long)
        labels = torch.full((len(items), max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(items), max_len), dtype=torch.long)
        for index, item in enumerate(items):
            length = int(item["input_ids"].numel())
            input_ids[index, :length] = item["input_ids"]
            labels[index, :length] = item["labels"]
            attention_mask[index, :length] = item["attention_mask"]
        return {
            "input_ids": input_ids.to(device),
            "labels": labels.to(device),
            "attention_mask": attention_mask.to(device),
        }

    train_items = [row for row in prepared if row["split"] == "train"]
    validation_items = [row for row in prepared if row["split"] == "validation"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    rng = random.Random(1234)
    losses = []
    model.train()
    for step in range(max_steps):
        batch = rng.sample(train_items, min(batch_size, len(train_items)))
        tensors = pad_batch(batch)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**tensors)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
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

    speech_end_id = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
    artifact_files: list[tuple[str, bytes]] = []
    evaluations = []
    model.eval()
    codec_model.to(device)
    eval_items = train_items[:eval_train_limit] + validation_items[:eval_validation_limit]
    for item in eval_items:
        prompt_ids = make_prompt_ids(tokenizer, item["target_text"]).unsqueeze(0).to(device)
        generate_started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                prompt_ids,
                max_length=int(prompt_ids.shape[-1]) + max_new_tokens,
                eos_token_id=speech_end_id,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
            )
        generation_ms = (time.perf_counter() - generate_started) * 1000
        generated_ids = generated[0][prompt_ids.shape[1] :]
        if generated_ids.numel() and int(generated_ids[-1]) == speech_end_id:
            generated_ids = generated_ids[:-1]
        speech_token_strings = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        generated_speech_ids = extract_speech_ids(speech_token_strings)
        generated_wav, generated_audio_s, decode_ms = decode_tokens_to_wav(codec_model, generated_speech_ids)
        transcription = openai_transcribe_wav(
            f"{safe_filename(item['example_id'])}_llasa_lora.wav",
            generated_wav,
            item["target_text"],
        )
        generated_name = f"generated_audio/{safe_filename(item['example_id'])}.wav"
        artifact_files.append((generated_name, generated_wav))
        evaluations.append(
            {
                "split": item["split"],
                "example_id": item["example_id"],
                "target_text": item["target_text"],
                "expected": item["expected"],
                "speech_token_count": len(generated_speech_ids),
                "target_speech_token_count": item["speech_token_count"],
                "generated_audio_s": generated_audio_s,
                "target_audio_s": item["target_xcodec2_audio_s"],
                "generation_ms": generation_ms,
                "decode_ms": decode_ms,
                "generation_tokens_per_second": len(generated_speech_ids) / max(generation_ms / 1000, 1e-9),
                "openai_transcription": transcription,
                "generated_audio_path": generated_name,
            }
        )
        print(
            json.dumps(
                {
                    "eval_id": item["example_id"],
                    "split": item["split"],
                    "target_text": item["target_text"],
                    "speech_tokens": len(generated_speech_ids),
                    "whisper_match": transcription.get("matches_expected"),
                    "whisper_heard": transcription.get("heard"),
                }
            ),
            flush=True,
        )

    adapter_dir = Path("/tmp/llasa_lora_audio_head")
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir / "adapter")
    tokenizer.save_pretrained(adapter_dir / "adapter")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in artifact_files:
            zf.writestr(name, content)
        for file_path in (adapter_dir / "adapter").rglob("*"):
            if file_path.is_file():
                zf.write(file_path, f"adapter/{file_path.relative_to(adapter_dir / 'adapter')}")
        zf.writestr(
            "llasa_lora_audio_head_manifest.json",
            json.dumps(
                {
                    "llasa_model_id": LLASA_MODEL_ID,
                    "xcodec2_model_id": XCODEC2_MODEL_ID,
                    "gpu": GPU,
                    "train_examples": len(train_items),
                    "validation_examples": len(validation_items),
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
        "train_example_count": len(train_items),
        "validation_example_count": len(validation_items),
        "max_steps": max_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "lora_rank": lora_rank,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "whisper_verified_count": sum(1 for item in evaluations if item["openai_transcription"].get("ok")),
        "train_whisper_match_count": sum(
            1 for item in evaluations if item["split"] == "train" and item["openai_transcription"].get("matches_expected")
        ),
        "train_eval_count": sum(1 for item in evaluations if item["split"] == "train"),
        "validation_whisper_match_count": sum(
            1
            for item in evaluations
            if item["split"] == "validation" and item["openai_transcription"].get("matches_expected")
        ),
        "validation_eval_count": sum(1 for item in evaluations if item["split"] == "validation"),
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "evaluations": evaluations,
        "artifact_zip": zip_buffer.getvalue(),
    }


@app.local_entrypoint()
def main(
    train_dataset_summary: str = "out/gemma4_omni_train_200/summary.json",
    validation_dataset_summary: str = "out/gemma4_omni_validation_40/summary.json",
    out_dir: str = "out/gemma4_omni_llasa_lora_audio_head_smoke",
    train_limit: int = 32,
    validation_limit: int = 4,
    validation_offset: int = 6,
    max_steps: int = 80,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    lora_rank: int = 8,
    eval_train_limit: int = 4,
    eval_validation_limit: int = 4,
    require_target_whisper_match: bool = True,
    temperature: float = 0.9,
    top_p: float = 0.95,
    max_new_tokens: int = 220,
    allow_current_dataset_training: bool = False,
):
    load_dotenv(Path(".env"))
    if not allow_current_dataset_training:
        raise SystemExit(
            "Training is disabled by default because the current experimental dataset failed manual quality review. "
            "Curate or regenerate the dataset first; pass --allow-current-dataset-training only for a deliberate smoke run."
        )
    train_rows = load_style_rows(
        Path(train_dataset_summary),
        split="train",
        limit=train_limit,
        require_target_whisper_match=require_target_whisper_match,
    )
    validation_rows = load_style_rows(
        Path(validation_dataset_summary),
        split="validation",
        limit=validation_limit,
        offset=validation_offset,
        require_target_whisper_match=require_target_whisper_match,
    )
    result = train_llasa_lora_audio_head.remote(
        train_rows=train_rows,
        validation_rows=validation_rows,
        max_steps=max_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        lora_rank=lora_rank,
        eval_train_limit=eval_train_limit,
        eval_validation_limit=eval_validation_limit,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    zip_path = target / "gemma4_omni_llasa_lora_audio_head_artifact.zip"
    zip_path.write_bytes(result.pop("artifact_zip"))
    (target / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith(".json") or name.startswith("generated_audio/") or name.startswith("adapter/"):
                zf.extract(name, target)
    print(json.dumps({**result, "zip_path": str(zip_path), "out_dir": str(target)}, indent=2))
