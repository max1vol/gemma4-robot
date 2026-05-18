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


APP_NAME = "gemma4-omni-audio-conditioned-sequence-head-smoke"
GEMMA_MODEL_ID = "google/gemma-4-E2B-it"
SNAC_MODEL_ID = "hubertsiuzdak/snac_24khz"
AUDIO_CONDITION_GPU = os.environ.get("GEMMA4_OMNI_AUDIO_CONDITION_GPU", "H100")
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


audio_condition_image = (
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


def normalize_expected_json(value: dict[str, Any]) -> dict[str, Any]:
    expected = value.get("expected") or {}
    return {
        "transcript": str(expected.get("transcript") or value.get("target_text") or ""),
        "style": str(expected.get("style") or ""),
        "speed": str(expected.get("speed") or ""),
        "loudness": str(expected.get("loudness") or ""),
        "voice": str(expected.get("voice") or ""),
    }


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


def load_style_dataset(
    summary_path: Path,
    split: str,
    limit: int = 0,
    require_target_whisper_match: bool = False,
    require_input_whisper_match: bool = False,
) -> list[dict[str, Any]]:

    def resolve_manifest_path(value: str | None) -> Path | None:
        if not value:
            return None
        path = Path(value)
        if path.is_absolute() or path.exists():
            return path
        return summary_path.parent / path

    rows = []
    for pair in load_manifest_pairs(summary_path):
        if require_target_whisper_match and not verification_matches(pair, "target_whisper"):
            continue
        if require_input_whisper_match and pair.get("audio", {}).get("input") and not verification_matches(
            pair,
            "input_whisper",
        ):
            continue
        labels = pair["labels"]
        input_audio_value = pair.get("audio", {}).get("input", {}).get("wav_path")
        input_audio_path = resolve_manifest_path(input_audio_value)
        target_audio_path = resolve_manifest_path(pair["audio"]["target"]["wav_path"])
        if target_audio_path is None:
            raise ValueError(f"missing target wav path for {pair['id']}")
        rows.append(
            {
                "split": split,
                "example_id": pair["id"],
                "home_demo_trigger": pair.get("home_demo_trigger", ""),
                "input_text": labels["input_text"],
                "input_direction": labels["input_direction"],
                "input_voice": labels["input_voice"],
                "target_text": labels["target_text"],
                "target_direction": labels["target_direction"],
                "target_voice": labels["target_voice"],
                "expected": normalize_expected_json(labels),
                "input_wav_bytes": input_audio_path.read_bytes() if input_audio_path else None,
                "target_wav_bytes": target_audio_path.read_bytes(),
                "input_source_path": str(input_audio_path) if input_audio_path else None,
                "target_source_path": str(target_audio_path),
            }
        )
    return rows[:limit] if limit else rows


def make_audio_conditioning_text(row: dict[str, Any], include_style_labels: bool) -> str:
    expected = row["expected"]
    audio_prompt = (
        "attached user-side audio contains the user's request and vocal delivery"
        if row.get("input_wav_bytes")
        else "no user-side audio is attached for this target-only training row"
    )
    lines = [
        "<fitness_state>",
        "exercise: squat",
        "event: rep_complete",
        f"home_demo_trigger: {row['home_demo_trigger']}",
        f"audio_prompt: {audio_prompt}",
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


def make_text_scaffold_style_conditioning_text(row: dict[str, Any], include_style_labels: bool) -> str:
    expected = row["expected"]
    audio_prompt = (
        "attached user-side audio contains the user's request and vocal delivery"
        if row.get("input_wav_bytes")
        else "no user-side audio is attached for this target-only training row"
    )
    lines = [
        "<fitness_state>",
        "exercise: squat",
        "event: rep_complete",
        f"home_demo_trigger: {row['home_demo_trigger']}",
        f"audio_prompt: {audio_prompt}",
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
            "Extract the requested speaking style from the audio prompt and compact fitness state.",
            "The exact coach utterance text is provided separately to the audio-output head.",
        ]
    )
    return "\n".join(lines) + "\n"


def make_target_text_scaffold(row: dict[str, Any]) -> str:
    return (
        "<speech_text>\n"
        f"{row['target_text']}\n"
        "</speech_text>\n"
        "Represent this exact coach utterance for direct speech-audio generation."
    )


def safe_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")


@app.function(
    image=audio_condition_image,
    volumes={"/cache": cache_volume},
    secrets=[modal.Secret.from_local_environ(["OPENAI_API_KEY"])],
    gpu=AUDIO_CONDITION_GPU,
    cpu=6.0,
    memory=49_152,
    timeout=3600,
)
def train_audio_conditioned_sequence_head(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    max_steps: int = 200,
    include_style_labels: bool = False,
    condition_mode: str = "final",
    eval_train_limit: int = 0,
    eval_validation_limit: int = 0,
    free_run_loss_weight: float = 0.0,
    free_run_warmup_steps: int = 0,
    free_run_interval: int = 1,
    free_run_level0_only: bool = False,
    max_input_audio_s: float = 0.0,
    transcribe_chunks: bool = False,
) -> dict:
    import sys

    import numpy as np
    import requests
    import soundfile as sf
    import torch
    import torch.nn.functional as F
    from snac import SNAC
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    sys.path.insert(0, "/root")
    from fitness_omni_audio_runtime import (
        SNACMemorySequenceHead,
        SNACSequenceHead,
        SNACTextCoarseToFineSequenceHead,
        SNACTextScaffoldProjectionHead,
        SNACTextScaffoldSequenceHead,
        StreamingConfig,
        generate_streaming_chunks_memory,
        generate_streaming_chunks_sequence,
        generate_streaming_chunks_text_coarse_to_fine,
        generate_streaming_chunks_text_scaffold_projection,
        generate_streaming_chunks_text_scaffold,
        predict_snac_codes_memory,
        predict_snac_codes_sequence,
        predict_snac_codes_text_coarse_to_fine,
        predict_snac_codes_text_scaffold_projection,
        predict_snac_codes_text_scaffold,
        simulate_playback_queue,
    )

    started = time.perf_counter()
    device = torch.device("cuda")
    if condition_mode not in {"final", "memory", "text", "text_projection", "text_coarse"}:
        raise ValueError(f"unknown condition_mode: {condition_mode}")

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

    all_rows = train_rows + validation_rows
    snac = SNAC.from_pretrained(SNAC_MODEL_ID, cache_dir="/cache/huggingface").eval().to(device)
    prepared_rows = []
    max_level_lengths = [0, 0, 0]
    for source_row in all_rows:
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
        prepared_rows.append(
            {
                **{k: v for k, v in source_row.items() if k not in {"target_wav_bytes"}},
                "conditioning_text": (
                    make_text_scaffold_style_conditioning_text(source_row, include_style_labels)
                    if condition_mode in {"text", "text_projection", "text_coarse"}
                    else make_audio_conditioning_text(source_row, include_style_labels)
                ),
                "target_text_scaffold": make_target_text_scaffold(source_row),
                "code_values": code_values,
                "snac_code_shapes": code_shapes,
                "audio_samples": int(len(audio_np)),
                "audio_s": len(audio_np) / SAMPLE_RATE,
                "audio_np": audio_np,
            }
        )

    processor = AutoProcessor.from_pretrained(GEMMA_MODEL_ID, cache_dir="/cache/huggingface", padding_side="left")
    gemma = AutoModelForMultimodalLM.from_pretrained(
        GEMMA_MODEL_ID,
        cache_dir="/cache/huggingface",
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    gemma.eval()

    cond_vectors = []
    memory_vectors = []
    text_memory_vectors = []
    audio_input_records = []
    input_dir = Path("/tmp/gemma_audio_condition_sequence_inputs")
    input_dir.mkdir(parents=True, exist_ok=True)
    for row in prepared_rows:
        content = []
        input_audio_s = None
        input_audio_original_s = None
        input_audio_truncated = False
        if row.get("input_wav_bytes"):
            input_audio_np = wav_bytes_to_array(row["input_wav_bytes"])
            input_audio_original_s = len(input_audio_np) / SAMPLE_RATE
            if max_input_audio_s > 0:
                max_input_samples = int(max_input_audio_s * SAMPLE_RATE)
                if len(input_audio_np) > max_input_samples:
                    input_audio_np = input_audio_np[:max_input_samples]
                    input_audio_truncated = True
            input_audio_s = len(input_audio_np) / SAMPLE_RATE
            row["conditioned_input_wav_bytes"] = wav_bytes_from_array(input_audio_np)
            input_path = input_dir / f"{safe_filename(row['example_id'])}_input.wav"
            input_path.write_bytes(row["conditioned_input_wav_bytes"])
            content.append({"type": "audio", "audio": str(input_path)})
        content.append({"type": "text", "text": row["conditioning_text"]})
        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(gemma.device)
        try:
            with torch.inference_mode():
                outputs = gemma(**inputs, output_hidden_states=True, use_cache=False)
        except Exception as exc:
            raise RuntimeError(
                f"Gemma conditioning failed for {row['example_id']} "
                f"(input_audio_s={input_audio_s}, original_input_audio_s={input_audio_original_s})"
            ) from exc
        hidden = outputs.hidden_states[-1][0].detach().float().cpu()
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            valid_mask = attention_mask[0].detach().bool().cpu()
            hidden = hidden[valid_mask]
        cond_vectors.append(hidden[-1])
        memory_vectors.append(hidden)
        text_token_count = None
        if condition_mode in {"text", "text_projection", "text_coarse"}:
            text_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": row["target_text_scaffold"]},
                    ],
                }
            ]
            text_inputs = processor.apply_chat_template(
                text_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(gemma.device)
            with torch.inference_mode():
                text_outputs = gemma(**text_inputs, output_hidden_states=True, use_cache=False)
            text_hidden = text_outputs.hidden_states[-1][0].detach().float().cpu()
            text_attention_mask = text_inputs.get("attention_mask")
            if text_attention_mask is not None:
                text_valid_mask = text_attention_mask[0].detach().bool().cpu()
                text_hidden = text_hidden[text_valid_mask]
            text_memory_vectors.append(text_hidden)
            text_token_count = int(text_hidden.shape[0])
        audio_input_records.append(
            {
                "split": row["split"],
                "example_id": row["example_id"],
                "input_audio_s": input_audio_s,
                "input_audio_original_s": input_audio_original_s,
                "input_audio_truncated": input_audio_truncated,
                "conditioning_text": row["conditioning_text"],
                "target_text_scaffold": (
                    row["target_text_scaffold"]
                    if condition_mode in {"text", "text_projection", "text_coarse"}
                    else None
                ),
                "input_token_count": int(inputs["input_ids"].shape[-1]),
                "memory_token_count": int(hidden.shape[0]),
                "text_scaffold_token_count": text_token_count,
            }
        )

    del gemma
    torch.cuda.empty_cache()

    cond = torch.stack(cond_vectors).to(device)
    hidden_size = cond.shape[-1]
    max_memory_len = max(int(memory.shape[0]) for memory in memory_vectors)
    memory = torch.zeros((len(memory_vectors), max_memory_len, hidden_size), dtype=torch.float32)
    memory_mask = torch.zeros((len(memory_vectors), max_memory_len), dtype=torch.bool)
    for row_idx, row_memory in enumerate(memory_vectors):
        length = int(row_memory.shape[0])
        memory[row_idx, :length] = row_memory
        memory_mask[row_idx, :length] = True
    memory = memory.to(device)
    memory_mask = memory_mask.to(device)
    text_memory = None
    text_memory_mask = None
    if condition_mode in {"text", "text_projection", "text_coarse"}:
        max_text_memory_len = max(int(text_memory.shape[0]) for text_memory in text_memory_vectors)
        text_memory = torch.zeros((len(text_memory_vectors), max_text_memory_len, hidden_size), dtype=torch.float32)
        text_memory_mask = torch.zeros((len(text_memory_vectors), max_text_memory_len), dtype=torch.bool)
        for row_idx, row_text_memory in enumerate(text_memory_vectors):
            length = int(row_text_memory.shape[0])
            text_memory[row_idx, :length] = row_text_memory
            text_memory_mask[row_idx, :length] = True
        text_memory = text_memory.to(device)
        text_memory_mask = text_memory_mask.to(device)
    train_indices = [idx for idx, row in enumerate(prepared_rows) if row["split"] == "train"]
    validation_indices = [idx for idx, row in enumerate(prepared_rows) if row["split"] == "validation"]
    train_cond = cond[train_indices]
    train_memory = memory[train_indices]
    train_memory_mask = memory_mask[train_indices]
    train_text_memory = text_memory[train_indices] if text_memory is not None else None
    train_text_memory_mask = text_memory_mask[train_indices] if text_memory_mask is not None else None

    labels_by_level = []
    mask_by_level = []
    for level, max_len in enumerate(max_level_lengths):
        labels = torch.full((len(train_indices), max_len), -100, dtype=torch.long)
        mask = torch.zeros((len(train_indices), max_len), dtype=torch.bool)
        for out_idx, row_idx in enumerate(train_indices):
            values = prepared_rows[row_idx]["code_values"][level]
            labels[out_idx, : values.numel()] = values
            mask[out_idx, : values.numel()] = True
        labels_by_level.append(labels.to(device))
        mask_by_level.append(mask.to(device))

    head_type = {
        "final": "snac_sequence_gru",
        "memory": "snac_memory_sequence_gru_attention",
        "text": "snac_text_scaffold_sequence_gru_attention",
        "text_projection": "snac_text_scaffold_projection",
        "text_coarse": "snac_text_coarse_to_fine_sequence_gru_attention",
    }[condition_mode]
    if condition_mode == "memory":
        head = SNACMemorySequenceHead(hidden_size=hidden_size, max_level_lengths=max_level_lengths).to(device)
    elif condition_mode == "text":
        head = SNACTextScaffoldSequenceHead(hidden_size=hidden_size, max_level_lengths=max_level_lengths).to(device)
    elif condition_mode == "text_coarse":
        head = SNACTextCoarseToFineSequenceHead(hidden_size=hidden_size, max_level_lengths=max_level_lengths).to(device)
    elif condition_mode == "text_projection":
        head = SNACTextScaffoldProjectionHead(hidden_size=hidden_size, max_level_lengths=max_level_lengths).to(device)
    else:
        head = SNACSequenceHead(hidden_size=hidden_size, max_level_lengths=max_level_lengths).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    losses = []
    accuracies = []
    free_run_losses = []
    for step in range(max_steps):
        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), device=device)
        total_free_run_loss = torch.zeros((), device=device)
        total_correct = 0
        total_count = 0
        use_free_run_loss = (
            condition_mode == "text"
            and free_run_loss_weight > 0
            and step + 1 > free_run_warmup_steps
            and free_run_interval > 0
            and (step + 1 - free_run_warmup_steps) % free_run_interval == 0
        )
        for level in range(len(max_level_lengths)):
            labels = labels_by_level[level]
            if condition_mode == "memory":
                logits = head.forward_level_teacher(train_memory, train_memory_mask, level, labels)
            elif condition_mode == "text":
                logits = head.forward_level_teacher(train_cond, train_text_memory, train_text_memory_mask, level, labels)
                if use_free_run_loss and (not free_run_level0_only or level == 0):
                    free_logits = head.forward_level_free_running(
                        train_cond,
                        train_text_memory,
                        train_text_memory_mask,
                        level,
                        labels.shape[1],
                    )
                    total_free_run_loss = total_free_run_loss + F.cross_entropy(
                        free_logits.reshape(-1, SNAC_CODEBOOK_SIZE),
                        labels.reshape(-1),
                        ignore_index=-100,
                    )
            elif condition_mode == "text_coarse":
                coarse_labels = labels_by_level[level - 1] if level > 0 else None
                logits = head.forward_level_teacher(
                    train_cond,
                    train_text_memory,
                    train_text_memory_mask,
                    level,
                    labels,
                    coarse_tokens=coarse_labels,
                )
            elif condition_mode == "text_projection":
                logits = head.forward_level(train_cond, train_text_memory, train_text_memory_mask, level, labels.shape[1])
            else:
                logits = head.forward_level_teacher(train_cond, level, labels)
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
        if use_free_run_loss:
            total_loss = total_loss + total_free_run_loss * free_run_loss_weight
        total_loss.backward()
        optimizer.step()
        losses.append(float(total_loss.detach().cpu()))
        free_run_losses.append(float(total_free_run_loss.detach().cpu()) if use_free_run_loss else None)
        accuracies.append(total_correct / max(total_count, 1))
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == max_steps:
            print(
                json.dumps(
                    {
                        "train_step": step + 1,
                        "max_steps": max_steps,
                        "loss": losses[-1],
                        "free_run_loss": free_run_losses[-1],
                        "teacher_forced_token_accuracy": accuracies[-1],
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
    eval_train_seen = 0
    eval_validation_seen = 0
    for row_index, reference_row in enumerate(prepared_rows):
        if reference_row["split"] == "train":
            eval_train_seen += 1
            if eval_train_limit and eval_train_seen > eval_train_limit:
                continue
        if reference_row["split"] == "validation":
            eval_validation_seen += 1
            if eval_validation_limit and eval_validation_seen > eval_validation_limit:
                continue
        eval_id = reference_row["example_id"]
        head_started = time.perf_counter()
        if condition_mode == "memory":
            decode_codes, level_token_counts, predicted_values_by_level = predict_snac_codes_memory(
                head,
                memory[row_index : row_index + 1],
                memory_mask[row_index : row_index + 1],
                reference_row["snac_code_shapes"],
            )
        elif condition_mode == "text":
            decode_codes, level_token_counts, predicted_values_by_level = predict_snac_codes_text_scaffold(
                head,
                cond[row_index : row_index + 1],
                text_memory[row_index : row_index + 1],
                text_memory_mask[row_index : row_index + 1],
                reference_row["snac_code_shapes"],
            )
        elif condition_mode == "text_coarse":
            decode_codes, level_token_counts, predicted_values_by_level = predict_snac_codes_text_coarse_to_fine(
                head,
                cond[row_index : row_index + 1],
                text_memory[row_index : row_index + 1],
                text_memory_mask[row_index : row_index + 1],
                reference_row["snac_code_shapes"],
            )
        elif condition_mode == "text_projection":
            decode_codes, level_token_counts, predicted_values_by_level = predict_snac_codes_text_scaffold_projection(
                head,
                cond[row_index : row_index + 1],
                text_memory[row_index : row_index + 1],
                text_memory_mask[row_index : row_index + 1],
                reference_row["snac_code_shapes"],
            )
        else:
            decode_codes, level_token_counts, predicted_values_by_level = predict_snac_codes_sequence(
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

        if condition_mode == "memory":
            stream_chunks, stitched_streaming_audio, streaming_summary = generate_streaming_chunks_memory(
                head=head,
                memory_vectors=memory[row_index : row_index + 1],
                memory_mask=memory_mask[row_index : row_index + 1],
                snac_code_shapes=reference_row["snac_code_shapes"],
                decode_codes_to_audio=decode_codes_to_audio,
                wav_bytes_from_array=wav_bytes_from_array,
                config=streaming_config,
                perf_counter=time.perf_counter,
            )
        elif condition_mode == "text":
            stream_chunks, stitched_streaming_audio, streaming_summary = generate_streaming_chunks_text_scaffold(
                head=head,
                style_vectors=cond[row_index : row_index + 1],
                text_vectors=text_memory[row_index : row_index + 1],
                text_mask=text_memory_mask[row_index : row_index + 1],
                snac_code_shapes=reference_row["snac_code_shapes"],
                decode_codes_to_audio=decode_codes_to_audio,
                wav_bytes_from_array=wav_bytes_from_array,
                config=streaming_config,
                perf_counter=time.perf_counter,
            )
        elif condition_mode == "text_coarse":
            stream_chunks, stitched_streaming_audio, streaming_summary = generate_streaming_chunks_text_coarse_to_fine(
                head=head,
                style_vectors=cond[row_index : row_index + 1],
                text_vectors=text_memory[row_index : row_index + 1],
                text_mask=text_memory_mask[row_index : row_index + 1],
                snac_code_shapes=reference_row["snac_code_shapes"],
                decode_codes_to_audio=decode_codes_to_audio,
                wav_bytes_from_array=wav_bytes_from_array,
                config=streaming_config,
                perf_counter=time.perf_counter,
            )
        elif condition_mode == "text_projection":
            stream_chunks, stitched_streaming_audio, streaming_summary = generate_streaming_chunks_text_scaffold_projection(
                head=head,
                style_vectors=cond[row_index : row_index + 1],
                text_vectors=text_memory[row_index : row_index + 1],
                text_mask=text_memory_mask[row_index : row_index + 1],
                snac_code_shapes=reference_row["snac_code_shapes"],
                decode_codes_to_audio=decode_codes_to_audio,
                wav_bytes_from_array=wav_bytes_from_array,
                config=streaming_config,
                perf_counter=time.perf_counter,
            )
        else:
            stream_chunks, stitched_streaming_audio, streaming_summary = generate_streaming_chunks_sequence(
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
        chunk_transcriptions = []
        if transcribe_chunks:
            chunk_transcriptions = [
                openai_transcribe_wav(f"{eval_id}_chunk_{chunk['index']:02d}.wav", chunk["wav_bytes"])
                for chunk in stream_chunks
            ]
        generated_evaluations.append(
            {
                "split": reference_row["split"],
                "eval_id": eval_id,
                "conditioning_text": reference_row["conditioning_text"],
                "reference_text": reference_row["target_text"],
                "expected": reference_row["expected"],
                "input_text": reference_row["input_text"],
                "input_direction": reference_row["input_direction"],
                "target_direction": reference_row["target_direction"],
                "home_demo_trigger": reference_row["home_demo_trigger"],
                "generated_audio_s": len(decoded) / SAMPLE_RATE,
                "target_audio_s": reference_row["audio_s"],
                "level_token_counts": level_token_counts,
                "greedy_token_accuracy": token_correct / max(token_total, 1),
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

    artifact_dir = Path("/tmp/gemma4_omni_audio_conditioned_sequence_head")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "head_type": head_type,
            "condition_mode": condition_mode,
            "hidden_size": hidden_size,
            "max_level_lengths": max_level_lengths,
            "snac_codebook_size": SNAC_CODEBOOK_SIZE,
            "gemma_model_id": GEMMA_MODEL_ID,
            "snac_model_id": SNAC_MODEL_ID,
            "conditioning": "Gemma AutoModelForMultimodalLM hidden state over attached input WAV + compact fitness state",
            "include_style_labels": include_style_labels,
            "uses_text_scaffold": condition_mode in {"text", "text_projection", "text_coarse"},
            "free_run_loss_weight": free_run_loss_weight,
            "free_run_warmup_steps": free_run_warmup_steps,
            "free_run_interval": free_run_interval,
            "free_run_level0_only": free_run_level0_only,
            "max_input_audio_s": max_input_audio_s,
            "transcribe_chunks": transcribe_chunks,
        },
        artifact_dir / "audio_conditioned_sequence_head.pt",
    )

    manifest_rows = []
    for row in prepared_rows:
        manifest_rows.append(
            {
                "split": row["split"],
                "example_id": row["example_id"],
                "input_text": row["input_text"],
                "input_direction": row["input_direction"],
                "target_text": row["target_text"],
                "target_direction": row["target_direction"],
                "expected": row["expected"],
                "conditioning_text": row["conditioning_text"],
                "target_text_scaffold": row["target_text_scaffold"],
                "snac_code_shapes": row["snac_code_shapes"],
                "audio_samples": row["audio_samples"],
                "audio_s": row["audio_s"],
            }
        )

    zip_buffer = io.BytesIO()
    checkpoint_name = (
        "audio_conditioned_memory_sequence_head.pt"
        if condition_mode == "memory"
        else "audio_conditioned_text_scaffold_projection_head.pt"
        if condition_mode == "text_projection"
        else "audio_conditioned_text_coarse_to_fine_sequence_head.pt"
        if condition_mode == "text_coarse"
        else "audio_conditioned_text_scaffold_sequence_head.pt"
        if condition_mode == "text"
        else "audio_conditioned_sequence_head.pt"
    )
    manifest_name = (
        "gemma4_omni_audio_conditioned_memory_sequence_head_manifest.json"
        if condition_mode == "memory"
        else "gemma4_omni_audio_conditioned_text_scaffold_projection_head_manifest.json"
        if condition_mode == "text_projection"
        else "gemma4_omni_audio_conditioned_text_coarse_to_fine_sequence_head_manifest.json"
        if condition_mode == "text_coarse"
        else "gemma4_omni_audio_conditioned_text_scaffold_sequence_head_manifest.json"
        if condition_mode == "text"
        else "gemma4_omni_audio_conditioned_sequence_head_manifest.json"
    )
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(artifact_dir / "audio_conditioned_sequence_head.pt", checkpoint_name)
        for row in prepared_rows:
            if row.get("input_wav_bytes"):
                zf.writestr(
                    f"input_audio/{row['example_id']}.wav",
                    row.get("conditioned_input_wav_bytes", row["input_wav_bytes"]),
                )
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
            manifest_name,
            json.dumps(
                {
                    "gemma_model_id": GEMMA_MODEL_ID,
                    "snac_model_id": SNAC_MODEL_ID,
                    "sample_rate": SAMPLE_RATE,
                    "head_type": head_type,
                    "condition_mode": condition_mode,
                    "conditioning": (
                        "Gemma native audio frontend hidden state over input WAV plus compact fitness state; "
                        "separate Gemma hidden-state scaffold over target utterance text"
                        if condition_mode in {"text", "text_projection", "text_coarse"}
                        else "Gemma native audio frontend hidden state over input WAV plus compact fitness state"
                    ),
                    "include_style_labels": include_style_labels,
                    "uses_text_scaffold": condition_mode in {"text", "text_projection", "text_coarse"},
                    "streaming_config": {
                        "first_level0_tokens": STREAM_FIRST_LEVEL0_TOKENS,
                        "next_level0_tokens": STREAM_NEXT_LEVEL0_TOKENS,
                        "overlap_level0_tokens": STREAM_OVERLAP_LEVEL0_TOKENS,
                    },
                    "audio_condition_gpu": AUDIO_CONDITION_GPU,
                    "modal_gpu_cost_per_second": MODAL_GPU_COST_PER_SECOND.get(AUDIO_CONDITION_GPU),
                    "max_steps": max_steps,
                    "free_run_loss_weight": free_run_loss_weight,
                    "free_run_warmup_steps": free_run_warmup_steps,
                    "free_run_interval": free_run_interval,
                    "free_run_level0_only": free_run_level0_only,
                    "max_input_audio_s": max_input_audio_s,
                    "transcribe_chunks": transcribe_chunks,
                    "train_example_count": len(train_indices),
                    "validation_example_count": len(validation_indices),
                    "eval_train_limit": eval_train_limit,
                    "eval_validation_limit": eval_validation_limit,
                    "max_level_lengths": max_level_lengths,
                    "loss_first": losses[0] if losses else None,
                    "loss_last": losses[-1] if losses else None,
                    "free_run_loss_last": next((value for value in reversed(free_run_losses) if value is not None), None),
                    "teacher_forced_token_accuracy_last": accuracies[-1] if accuracies else None,
                    "audio_inputs": audio_input_records,
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
        "head_type": head_type,
        "condition_mode": condition_mode,
        "conditioning": (
            "Gemma native audio frontend hidden state over input WAV plus compact fitness state; "
            "separate Gemma hidden-state scaffold over target utterance text"
            if condition_mode in {"text", "text_projection", "text_coarse"}
            else "Gemma native audio frontend hidden state over input WAV plus compact fitness state"
        ),
        "include_style_labels": include_style_labels,
        "uses_text_scaffold": condition_mode in {"text", "text_projection", "text_coarse"},
        "streaming_config": {
            "first_level0_tokens": STREAM_FIRST_LEVEL0_TOKENS,
            "next_level0_tokens": STREAM_NEXT_LEVEL0_TOKENS,
            "overlap_level0_tokens": STREAM_OVERLAP_LEVEL0_TOKENS,
        },
        "audio_condition_gpu": AUDIO_CONDITION_GPU,
        "modal_gpu_cost_per_second": MODAL_GPU_COST_PER_SECOND.get(AUDIO_CONDITION_GPU),
        "max_steps": max_steps,
        "free_run_loss_weight": free_run_loss_weight,
        "free_run_warmup_steps": free_run_warmup_steps,
        "free_run_interval": free_run_interval,
        "free_run_level0_only": free_run_level0_only,
        "max_input_audio_s": max_input_audio_s,
        "transcribe_chunks": transcribe_chunks,
        "train_example_count": len(train_indices),
        "validation_example_count": len(validation_indices),
        "eval_train_limit": eval_train_limit,
        "eval_validation_limit": eval_validation_limit,
        "max_level_lengths": max_level_lengths,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "free_run_loss_last": next((value for value in reversed(free_run_losses) if value is not None), None),
        "teacher_forced_token_accuracy_last": accuracies[-1] if accuracies else None,
        "audio_inputs": audio_input_records,
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
    train_dataset_summary: str = "out/gemma4_omni_gemini_tts_style_dataset/summary.json",
    validation_dataset_summary: str = "out/gemma4_omni_gemini_tts_style_heldout/summary.json",
    out_dir: str = "out/gemma4_omni_audio_conditioned_sequence_head",
    max_steps: int = 200,
    train_limit: int = 0,
    validation_limit: int = 0,
    include_style_labels: bool = False,
    condition_mode: str = "final",
    eval_train_limit: int = 0,
    eval_validation_limit: int = 0,
    free_run_loss_weight: float = 0.0,
    free_run_warmup_steps: int = 0,
    free_run_interval: int = 1,
    free_run_level0_only: bool = False,
    require_target_whisper_match: bool = False,
    require_input_whisper_match: bool = False,
    max_input_audio_s: float = 0.0,
    input_audio_mode: str = "include",
    transcribe_chunks: bool = False,
):
    load_dotenv(Path(".env"))
    train_rows = load_style_dataset(
        Path(train_dataset_summary),
        split="train",
        limit=train_limit,
        require_target_whisper_match=require_target_whisper_match,
        require_input_whisper_match=False,
    )
    validation_rows = load_style_dataset(
        Path(validation_dataset_summary),
        split="validation",
        limit=validation_limit,
        require_target_whisper_match=require_target_whisper_match,
        require_input_whisper_match=require_input_whisper_match,
    )
    if input_audio_mode not in {"include", "none"}:
        raise ValueError(f"unknown input_audio_mode: {input_audio_mode}")
    if input_audio_mode == "none":
        for row in validation_rows:
            row["input_wav_bytes"] = None
            row["input_source_path"] = None
    result = train_audio_conditioned_sequence_head.remote(
        train_rows=train_rows,
        validation_rows=validation_rows,
        max_steps=max_steps,
        include_style_labels=include_style_labels,
        condition_mode=condition_mode,
        eval_train_limit=eval_train_limit,
        eval_validation_limit=eval_validation_limit,
        free_run_loss_weight=free_run_loss_weight,
        free_run_warmup_steps=free_run_warmup_steps,
        free_run_interval=free_run_interval,
        free_run_level0_only=free_run_level0_only,
        max_input_audio_s=max_input_audio_s,
        transcribe_chunks=transcribe_chunks,
    )
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    zip_path = target / "gemma4_omni_audio_conditioned_sequence_head_artifact.zip"
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
