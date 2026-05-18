from __future__ import annotations

import io
import json
import math
import os
import random
import re
import time
import zipfile
from pathlib import Path
from typing import Any

import modal


APP_NAME = "gemma4-omni-xcodec2-sequence-head-smoke"
GEMMA_MODEL_ID = "google/gemma-4-E2B-it"
XCODEC2_MODEL_ID = "HKUSTAudio/xcodec2"
TRAIN_GPU = os.environ.get("GEMMA4_OMNI_XCODEC2_TRAIN_GPU", "L4")
XCODEC2_SAMPLE_RATE = 16_000
XCODEC2_CODEBOOK_SIZE = 65_536
STREAM_FIRST_TOKENS = 20
MODAL_GPU_COST_PER_SECOND = {
    "T4": 0.000164,
    "L4": 0.000222,
    "A10": 0.000306,
    "A100-40GB": 0.000583,
    "A100": 0.000694,
    "H100": 0.001097,
}


gemma_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libsndfile1")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "hf_xet",
        "sentencepiece",
        "torchvision",
    )
)

xcodec2_train_image = (
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


def normalize_expected_json(value: dict[str, Any]) -> dict[str, Any]:
    expected = value.get("expected") or {}
    return {
        "transcript": str(expected.get("transcript") or value.get("target_text") or ""),
        "style": str(expected.get("style") or ""),
        "speed": str(expected.get("speed") or ""),
        "loudness": str(expected.get("loudness") or ""),
        "voice": str(expected.get("voice") or ""),
    }


def load_style_dataset(
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
                "home_demo_trigger": pair.get("home_demo_trigger", ""),
                "target_text": labels["target_text"],
                "target_direction": labels.get("target_direction", ""),
                "target_voice": labels.get("target_voice", ""),
                "expected": normalize_expected_json(labels),
                "target_wav_bytes": target_audio_path.read_bytes(),
                "target_source_path": str(target_audio_path),
            }
        )
    return rows[:limit] if limit else rows


def make_conditioning_text(row: dict[str, Any], include_style_labels: bool) -> str:
    expected = row["expected"]
    lines = [
        "<fitness_state>",
        "exercise: squat",
        "event: rep_complete",
        f"home_demo_trigger: {row['home_demo_trigger']}",
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
            "Represent this exact coach utterance as speech-audio tokens.",
        ]
    )
    return "\n".join(lines) + "\n"


def safe_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")


@app.function(
    image=gemma_image,
    volumes={"/cache": cache_volume},
    gpu=TRAIN_GPU,
    cpu=6.0,
    memory=49_152,
    timeout=3600,
)
def extract_gemma_conditioning(
    rows: list[dict[str, Any]],
    include_style_labels: bool = False,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(GEMMA_MODEL_ID, cache_dir="/cache/huggingface", padding_side="left")
    gemma = AutoModelForMultimodalLM.from_pretrained(
        GEMMA_MODEL_ID,
        cache_dir="/cache/huggingface",
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    gemma.eval()
    records = []
    for row in rows:
        conditioning_text = make_conditioning_text(row, include_style_labels)
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": conditioning_text}],
            }
        ]
        encoded = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(gemma.device)
        with torch.inference_mode():
            outputs = gemma(**encoded, output_hidden_states=True, use_cache=False)
        hidden = outputs.hidden_states[-1][0].detach().float().cpu()
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            hidden = hidden[attention_mask[0].detach().bool().cpu()]
        vector = hidden[-1]
        records.append(
            {
                "example_id": row["example_id"],
                "split": row["split"],
                "conditioning_text": conditioning_text,
                "token_count": int(encoded["input_ids"].shape[-1]),
                "cond_vector": vector.tolist(),
            }
        )
        print(
            json.dumps(
                {
                    "conditioned": len(records),
                    "count": len(rows),
                    "example_id": row["example_id"],
                    "token_count": int(encoded["input_ids"].shape[-1]),
                }
            ),
            flush=True,
        )
    cache_volume.commit()
    return {
        "elapsed_s": time.perf_counter() - started,
        "gemma_model_id": GEMMA_MODEL_ID,
        "gpu": TRAIN_GPU,
        "example_count": len(records),
        "records": records,
    }


@app.function(
    image=xcodec2_train_image,
    volumes={"/cache": cache_volume},
    secrets=[modal.Secret.from_local_environ(["OPENAI_API_KEY"])],
    gpu=TRAIN_GPU,
    cpu=6.0,
    memory=49_152,
    timeout=3600,
)
def train_gemma_xcodec2_sequence_head(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    condition_records: list[dict[str, Any]],
    gemma_conditioning_summary: dict[str, Any],
    max_steps: int = 200,
    batch_size: int = 16,
    eval_train_limit: int = 0,
    eval_validation_limit: int = 0,
    learning_rate: float = 1e-3,
    width: int = 384,
) -> dict[str, Any]:
    import numpy as np
    import requests
    import soundfile as sf
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from xcodec2.modeling_xcodec2 import XCodec2Model

    started = time.perf_counter()
    device = torch.device("cuda")

    class XCodec2SequenceHead(nn.Module):
        def __init__(self, hidden_size: int, max_length: int, head_width: int, codebook_size: int):
            super().__init__()
            self.max_length = max_length
            self.codebook_size = codebook_size
            self.bos_token_id = codebook_size
            self.layers = 2
            self.width = head_width
            self.cond = nn.Sequential(
                nn.Linear(hidden_size, head_width),
                nn.SiLU(),
                nn.LayerNorm(head_width),
            )
            self.cond_to_hidden = nn.Linear(head_width, self.layers * head_width)
            self.token_embed = nn.Embedding(codebook_size + 1, head_width)
            self.pos_embed = nn.Embedding(max_length, head_width)
            self.gru = nn.GRU(head_width, head_width, num_layers=self.layers, batch_first=True)
            self.out = nn.Linear(head_width, codebook_size)

        def _initial_hidden(self, cond_vectors: torch.Tensor) -> torch.Tensor:
            batch = cond_vectors.shape[0]
            cond = self.cond(cond_vectors)
            hidden = self.cond_to_hidden(cond).view(batch, self.layers, self.width)
            return hidden.transpose(0, 1).contiguous()

        def forward_teacher(self, cond_vectors: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
            clean_targets = target_tokens.masked_fill(target_tokens < 0, 0)
            bos = torch.full(
                (target_tokens.shape[0], 1),
                self.bos_token_id,
                dtype=torch.long,
                device=target_tokens.device,
            )
            inputs = torch.cat([bos, clean_targets[:, :-1]], dim=1)
            positions = torch.arange(target_tokens.shape[1], device=target_tokens.device)
            x = self.token_embed(inputs.clamp(0, self.bos_token_id)) + self.pos_embed(positions).unsqueeze(0)
            out, _ = self.gru(x, self._initial_hidden(cond_vectors))
            return self.out(out)

        def generate(self, cond_vector: torch.Tensor, length: int) -> torch.Tensor:
            batch = cond_vector.shape[0]
            length = min(length, self.max_length)
            hidden = self._initial_hidden(cond_vector)
            token = torch.full((batch, 1), self.bos_token_id, dtype=torch.long, device=cond_vector.device)
            generated = []
            for index in range(length):
                position = torch.tensor([index], device=cond_vector.device)
                x = self.token_embed(token) + self.pos_embed(position).unsqueeze(0)
                out, hidden = self.gru(x, hidden)
                token = self.out(out[:, -1]).argmax(dim=-1, keepdim=True)
                generated.append(token)
            return torch.cat(generated, dim=1) if generated else torch.empty((batch, 0), device=cond_vector.device)

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

    all_rows = train_rows + validation_rows
    condition_by_id = {record["example_id"]: record for record in condition_records}
    xcodec2 = XCodec2Model.from_pretrained(XCODEC2_MODEL_ID, cache_dir="/cache/huggingface")
    xcodec2.eval().to(device)

    prepared_rows = []
    max_length = 0
    max_observed_token = 0
    for source_row in all_rows:
        source_np, source_rate = read_wav_mono(source_row["target_wav_bytes"])
        input_np = resample_to_xcodec2(source_np, source_rate)
        input_tensor = torch.from_numpy(input_np).float().unsqueeze(0).to(device)
        with torch.inference_mode():
            code = xcodec2.encode_code(input_waveform=input_tensor).detach().cpu().reshape(-1).long()
        max_length = max(max_length, int(code.numel()))
        max_observed_token = max(max_observed_token, int(code.max().item()) if code.numel() else 0)
        condition_record = condition_by_id[source_row["example_id"]]
        prepared_rows.append(
            {
                **{k: v for k, v in source_row.items() if k not in {"target_wav_bytes"}},
                "conditioning_text": condition_record["conditioning_text"],
                "cond_vector": torch.tensor(condition_record["cond_vector"], dtype=torch.float32),
                "conditioning_token_count": condition_record["token_count"],
                "xcodec2_code": code,
                "xcodec2_code_shape": [1, 1, int(code.numel())],
                "source_sample_rate": source_rate,
                "source_audio_s": len(source_np) / source_rate,
                "audio_np_16k": input_np,
            }
        )

    if max_observed_token >= XCODEC2_CODEBOOK_SIZE:
        raise ValueError(f"observed XCodec2 token {max_observed_token} exceeds configured codebook")

    cond = torch.stack([row["cond_vector"] for row in prepared_rows]).to(device)
    hidden_size = int(cond.shape[-1])
    train_indices = [idx for idx, row in enumerate(prepared_rows) if row["split"] == "train"]
    validation_indices = [idx for idx, row in enumerate(prepared_rows) if row["split"] == "validation"]
    labels = torch.full((len(prepared_rows), max_length), -100, dtype=torch.long)
    masks = torch.zeros((len(prepared_rows), max_length), dtype=torch.bool)
    for row_idx, row in enumerate(prepared_rows):
        code = row["xcodec2_code"]
        labels[row_idx, : code.numel()] = code
        masks[row_idx, : code.numel()] = True
    labels = labels.to(device)
    masks = masks.to(device)

    head = XCodec2SequenceHead(
        hidden_size=hidden_size,
        max_length=max_length,
        head_width=width,
        codebook_size=XCODEC2_CODEBOOK_SIZE,
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=0.01)
    losses = []
    accuracies = []
    rng = random.Random(1234)
    train_indices_local = list(train_indices)
    for step in range(max_steps):
        batch_indices = rng.sample(train_indices_local, min(batch_size, len(train_indices_local)))
        batch_index_tensor = torch.tensor(batch_indices, dtype=torch.long, device=device)
        batch_cond = cond.index_select(0, batch_index_tensor)
        batch_labels = labels.index_select(0, batch_index_tensor)
        batch_masks = masks.index_select(0, batch_index_tensor)
        optimizer.zero_grad(set_to_none=True)
        logits = head.forward_teacher(batch_cond, batch_labels)
        loss = F.cross_entropy(
            logits.reshape(-1, XCODEC2_CODEBOOK_SIZE),
            batch_labels.reshape(-1),
            ignore_index=-100,
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            accuracy = float((pred[batch_masks] == batch_labels[batch_masks]).float().mean().item())
        losses.append(float(loss.detach().cpu()))
        accuracies.append(accuracy)
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == max_steps:
            print(
                json.dumps(
                    {
                        "train_step": step + 1,
                        "max_steps": max_steps,
                        "loss": losses[-1],
                        "batch_teacher_forced_token_accuracy": accuracy,
                        "elapsed_s": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )

    head.eval()

    def decode_tokens_to_audio(tokens: torch.Tensor) -> tuple[np.ndarray, float]:
        decode_started = time.perf_counter()
        code = tokens.detach().long().view(1, 1, -1).to(device)
        with torch.inference_mode():
            decoded_value = xcodec2.decode_code(code).detach().float().cpu().numpy().reshape(-1)
        decode_ms = (time.perf_counter() - decode_started) * 1000
        return np.clip(decoded_value, -1.0, 1.0).astype(np.float32), decode_ms

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
        target_length = int(reference_row["xcodec2_code"].numel())
        head_started = time.perf_counter()
        with torch.inference_mode():
            generated_tokens = head.generate(cond[row_index : row_index + 1], target_length)
        head_inference_ms = (time.perf_counter() - head_started) * 1000
        generated_audio, decode_ms = decode_tokens_to_audio(generated_tokens[0])
        generated_wav = wav_bytes_from_array(generated_audio)
        transcription = openai_transcribe_wav(
            f"{safe_filename(reference_row['example_id'])}_xcodec2_generated.wav",
            generated_wav,
            reference_row["target_text"],
        )
        first_token_count = min(STREAM_FIRST_TOKENS, target_length)
        first_started = time.perf_counter()
        with torch.inference_mode():
            first_tokens = head.generate(cond[row_index : row_index + 1], first_token_count)
        first_head_ms = (time.perf_counter() - first_started) * 1000
        first_audio, first_decode_ms = decode_tokens_to_audio(first_tokens[0])
        first_wav = wav_bytes_from_array(first_audio)
        first_transcription = openai_transcribe_wav(
            f"{safe_filename(reference_row['example_id'])}_xcodec2_first.wav",
            first_wav,
            reference_row["target_text"],
        )
        target_tokens = reference_row["xcodec2_code"].tolist()
        predicted_tokens = generated_tokens.detach().cpu().reshape(-1).tolist()
        token_correct = sum(1 for a, b in zip(predicted_tokens, target_tokens) if a == b)
        generated_evaluations.append(
            {
                "split": reference_row["split"],
                "eval_id": reference_row["example_id"],
                "conditioning_text": reference_row["conditioning_text"],
                "reference_text": reference_row["target_text"],
                "expected": reference_row["expected"],
                "generated_audio_s": len(generated_audio) / XCODEC2_SAMPLE_RATE,
                "target_audio_s": reference_row["source_audio_s"],
                "xcodec2_token_count": target_length,
                "xcodec2_tokens_per_second": target_length / max(reference_row["source_audio_s"], 1e-9),
                "greedy_token_accuracy": token_correct / max(len(target_tokens), 1),
                "head_inference_ms": head_inference_ms,
                "xcodec2_decode_ms": decode_ms,
                "streaming": {
                    "first_token_count": first_token_count,
                    "first_audio_s": len(first_audio) / XCODEC2_SAMPLE_RATE,
                    "first_head_ms": first_head_ms,
                    "first_xcodec2_decode_ms": first_decode_ms,
                    "first_ready_ms": first_head_ms + first_decode_ms,
                    "first_transcription": first_transcription,
                },
                "openai_transcription": transcription,
                "wav_bytes": generated_wav,
                "first_wav_bytes": first_wav,
            }
        )

    artifact_dir = Path("/tmp/gemma4_omni_xcodec2_sequence_head")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "hidden_size": hidden_size,
            "max_length": max_length,
            "xcodec2_codebook_size": XCODEC2_CODEBOOK_SIZE,
            "xcodec2_model_id": XCODEC2_MODEL_ID,
            "gemma_model_id": GEMMA_MODEL_ID,
            "width": width,
            "stream_first_tokens": STREAM_FIRST_TOKENS,
        },
        artifact_dir / "xcodec2_sequence_head.pt",
    )

    manifest_rows = []
    for row in prepared_rows:
        manifest_rows.append(
            {
                "split": row["split"],
                "example_id": row["example_id"],
                "target_text": row["target_text"],
                "expected": row["expected"],
                "conditioning_text": row["conditioning_text"],
                "xcodec2_code_shape": row["xcodec2_code_shape"],
                "source_audio_s": row["source_audio_s"],
                "target_source_path": row["target_source_path"],
            }
        )

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(artifact_dir / "xcodec2_sequence_head.pt", "xcodec2_sequence_head.pt")
        for row in prepared_rows:
            zf.writestr(f"target_audio_16k/{row['example_id']}.wav", wav_bytes_from_array(row["audio_np_16k"]))
        for evaluation in generated_evaluations:
            zf.writestr(f"generated_audio/{evaluation['eval_id']}.wav", evaluation["wav_bytes"])
            zf.writestr(f"generated_audio_first/{evaluation['eval_id']}_first.wav", evaluation["first_wav_bytes"])
        zf.writestr(
            "gemma4_omni_xcodec2_sequence_head_manifest.json",
            json.dumps(
                {
                    "gemma_model_id": GEMMA_MODEL_ID,
                    "xcodec2_model_id": XCODEC2_MODEL_ID,
                    "sample_rate": XCODEC2_SAMPLE_RATE,
                    "codebook_size": XCODEC2_CODEBOOK_SIZE,
                    "stream_first_tokens": STREAM_FIRST_TOKENS,
                    "train_gpu": TRAIN_GPU,
                    "modal_gpu_cost_per_second": MODAL_GPU_COST_PER_SECOND.get(TRAIN_GPU),
                    "max_steps": max_steps,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "width": width,
                    "include_style_labels": gemma_conditioning_summary.get("include_style_labels"),
                    "gemma_conditioning": gemma_conditioning_summary,
                    "train_example_count": len(train_indices),
                    "validation_example_count": len(validation_indices),
                    "max_length": max_length,
                    "max_observed_token": max_observed_token,
                    "loss_first": losses[0] if losses else None,
                    "loss_last": losses[-1] if losses else None,
                    "batch_teacher_forced_token_accuracy_last": accuracies[-1] if accuracies else None,
                    "style_examples": manifest_rows,
                    "generated_evaluations": [
                        {
                            k: v
                            for k, v in evaluation.items()
                            if k not in {"wav_bytes", "first_wav_bytes"}
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
        "xcodec2_model_id": XCODEC2_MODEL_ID,
        "sample_rate": XCODEC2_SAMPLE_RATE,
        "codebook_size": XCODEC2_CODEBOOK_SIZE,
        "stream_first_tokens": STREAM_FIRST_TOKENS,
        "train_gpu": TRAIN_GPU,
        "modal_gpu_cost_per_second": MODAL_GPU_COST_PER_SECOND.get(TRAIN_GPU),
        "estimated_gpu_cost_usd": (
            elapsed_s * MODAL_GPU_COST_PER_SECOND[TRAIN_GPU] if TRAIN_GPU in MODAL_GPU_COST_PER_SECOND else None
        ),
        "max_steps": max_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "width": width,
        "include_style_labels": gemma_conditioning_summary.get("include_style_labels"),
        "gemma_conditioning": gemma_conditioning_summary,
        "train_example_count": len(train_indices),
        "validation_example_count": len(validation_indices),
        "eval_train_limit": eval_train_limit,
        "eval_validation_limit": eval_validation_limit,
        "max_length": max_length,
        "max_observed_token": max_observed_token,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "batch_teacher_forced_token_accuracy_last": accuracies[-1] if accuracies else None,
        "generated_evaluations": [
            {
                k: v
                for k, v in evaluation.items()
                if k not in {"wav_bytes", "first_wav_bytes"}
            }
            for evaluation in generated_evaluations
        ],
        "artifact_zip": zip_buffer.getvalue(),
    }


@app.local_entrypoint()
def main(
    train_dataset_summary: str = "out/gemma4_omni_train_200/summary.json",
    validation_dataset_summary: str = "out/gemma4_omni_validation_40/summary.json",
    out_dir: str = "out/gemma4_omni_xcodec2_sequence_head",
    max_steps: int = 200,
    batch_size: int = 16,
    train_limit: int = 0,
    validation_limit: int = 0,
    include_style_labels: bool = False,
    eval_train_limit: int = 0,
    eval_validation_limit: int = 0,
    learning_rate: float = 1e-3,
    width: int = 384,
    require_target_whisper_match: bool = True,
):
    load_dotenv(Path(".env"))
    train_rows = load_style_dataset(
        Path(train_dataset_summary),
        split="train",
        limit=train_limit,
        require_target_whisper_match=require_target_whisper_match,
    )
    validation_rows = load_style_dataset(
        Path(validation_dataset_summary),
        split="validation",
        limit=validation_limit,
        require_target_whisper_match=require_target_whisper_match,
    )
    compact_rows = [
        {k: v for k, v in row.items() if k != "target_wav_bytes"}
        for row in (train_rows + validation_rows)
    ]
    conditioning_result = extract_gemma_conditioning.remote(
        rows=compact_rows,
        include_style_labels=include_style_labels,
    )
    condition_records = conditioning_result.pop("records")
    conditioning_result["include_style_labels"] = include_style_labels
    result = train_gemma_xcodec2_sequence_head.remote(
        train_rows=train_rows,
        validation_rows=validation_rows,
        condition_records=condition_records,
        gemma_conditioning_summary=conditioning_result,
        max_steps=max_steps,
        batch_size=batch_size,
        eval_train_limit=eval_train_limit,
        eval_validation_limit=eval_validation_limit,
        learning_rate=learning_rate,
        width=width,
    )
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    zip_path = target / "gemma4_omni_xcodec2_sequence_head_artifact.zip"
    zip_path.write_bytes(result.pop("artifact_zip"))
    (target / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if (
                name.endswith(".json")
                or name.startswith("target_audio_16k/")
                or name.startswith("generated_audio/")
                or name.startswith("generated_audio_first/")
            ):
                zf.extract(name, target)
    print(json.dumps({**result, "zip_path": str(zip_path), "out_dir": str(target)}, indent=2))
