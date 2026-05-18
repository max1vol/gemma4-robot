from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from pathlib import Path
from typing import Any


DEFAULT_SUMMARIES = [
    "out/gemma4_omni_train_200/summary.json",
    "out/gemma4_omni_validation_40/summary.json",
]

REVIEW_PASS = {"pass", "approved", "keep", "good"}
REVIEW_FAIL = {"fail", "reject", "quarantine", "bad"}


def normalize_text(value: str) -> str:
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


def word_count(value: str) -> int:
    return len(re.findall(r"[a-zA-Z0-9']+", value))


def expected_duration_bounds(text: str) -> tuple[float, float]:
    # Short coach phrases need generous bounds because "Eight." and
    # "That's ten. Great set." have very different natural pacing.
    words = max(word_count(text), 1)
    minimum = max(0.28, 0.14 * words)
    maximum = min(7.5, max(1.25, 0.95 + 0.72 * words))
    return minimum, maximum


def load_pairs(summary_path: Path) -> list[dict[str, Any]]:
    summary = json.loads(summary_path.read_text())
    if isinstance(summary.get("pairs"), list):
        return summary["pairs"]
    rows_path = summary_path.with_name("rows.jsonl")
    if not rows_path.exists():
        raise ValueError(f"{summary_path} has no pairs array and {rows_path} does not exist")
    return [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]


def resolve_manifest_path(summary_path: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return summary_path.parent / path


def load_manual_reviews(path: Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    reviews: dict[str, dict[str, Any]] = {}
    if not path.exists():
        raise FileNotFoundError(path)
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        sample_id = item.get("id") or item.get("sample_id") or item.get("example_id")
        if not sample_id:
            raise ValueError(f"{path}:{line_no} has no id/sample_id/example_id")
        verdict = str(item.get("review") or item.get("verdict") or item.get("status") or "").strip().lower()
        if verdict not in REVIEW_PASS | REVIEW_FAIL:
            raise ValueError(f"{path}:{line_no} has unsupported review verdict {verdict!r}")
        reviews[str(sample_id)] = {
            "verdict": "pass" if verdict in REVIEW_PASS else "quarantine",
            "notes": item.get("notes") or item.get("note") or "",
            "reviewer": item.get("reviewer") or "",
            "raw": item,
        }
    return reviews


def audio_reasons(audio: dict[str, Any] | None, text: str, path: Path | None) -> list[str]:
    reasons: list[str] = []
    if not audio:
        return ["missing_target_audio_metadata"]
    if path is None:
        reasons.append("missing_target_wav_path")
    elif not path.exists():
        reasons.append("target_wav_file_missing")
    duration = audio.get("duration_s")
    if not isinstance(duration, (int, float)) or not math.isfinite(float(duration)):
        reasons.append("target_duration_missing")
    else:
        minimum, maximum = expected_duration_bounds(text)
        if duration < minimum:
            reasons.append("target_duration_too_short")
        if duration > maximum:
            reasons.append("target_duration_too_long")
    rms = audio.get("rms_dbfs")
    if not isinstance(rms, (int, float)) or not math.isfinite(float(rms)):
        reasons.append("target_rms_missing")
    else:
        if rms < -42:
            reasons.append("target_rms_too_quiet")
        if rms > -6:
            reasons.append("target_rms_too_hot")
    peak = audio.get("peak_dbfs")
    if not isinstance(peak, (int, float)) or not math.isfinite(float(peak)):
        reasons.append("target_peak_missing")
    else:
        if peak < -35:
            reasons.append("target_peak_too_quiet")
        if peak > -0.2:
            reasons.append("target_peak_near_clipping")
    return reasons


def verification_reasons(pair: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    verification = pair.get("verification") or {}
    labels = pair.get("labels") or {}
    target_expected = labels.get("target_text") or labels.get("expected", {}).get("transcript") or ""
    target_whisper = verification.get("target_whisper")
    if not target_whisper:
        reasons.append("target_whisper_missing")
    elif target_whisper.get("matches_expected") is not True:
        heard = normalize_text(str(target_whisper.get("heard") or ""))
        expected = normalize_text(str(target_whisper.get("expected") or target_expected))
        reasons.append("target_whisper_mismatch")
        if heard and expected and heard in {"you", "thank you for watching", "mbc"}:
            reasons.append("target_whisper_generic_failure")
    if pair.get("audio", {}).get("input"):
        input_whisper = verification.get("input_whisper")
        if not input_whisper:
            reasons.append("input_whisper_missing")
        elif input_whisper.get("matches_expected") is not True:
            reasons.append("input_whisper_mismatch")
    return reasons


def build_row(summary_path: Path, pair: dict[str, Any], manual_reviews: dict[str, dict[str, Any]]) -> dict[str, Any]:
    labels = pair.get("labels") or {}
    expected = labels.get("expected") or {}
    audio = pair.get("audio") or {}
    target_audio = audio.get("target") or {}
    target_path = resolve_manifest_path(summary_path, target_audio.get("wav_path"))
    target_text = labels.get("target_text") or expected.get("transcript") or ""
    reasons = verification_reasons(pair)
    reasons.extend(audio_reasons(target_audio, target_text, target_path))
    auto_status = "auto_quarantine" if reasons else "auto_pass_needs_manual_review"
    manual = manual_reviews.get(pair["id"])
    if manual and manual["verdict"] == "pass" and not reasons:
        training_status = "approved_for_training"
    elif manual and manual["verdict"] == "pass" and reasons:
        training_status = "blocked_manual_pass_but_auto_failed"
    elif manual and manual["verdict"] == "quarantine":
        training_status = "quarantined_by_manual_review"
    elif reasons:
        training_status = "quarantined_by_auto_gate"
    else:
        training_status = "blocked_pending_manual_review"
    duration = target_audio.get("duration_s")
    minimum, maximum = expected_duration_bounds(target_text)
    return {
        "id": pair["id"],
        "split": pair.get("split"),
        "summary_path": str(summary_path),
        "target_text": target_text,
        "style": expected.get("style") or "",
        "speed": expected.get("speed") or "",
        "loudness": expected.get("loudness") or "",
        "voice_role": expected.get("voice") or "",
        "target_voice": labels.get("target_voice") or target_audio.get("voice") or "",
        "target_wav_path": target_audio.get("wav_path") or "",
        "target_wav_abs_path": str(target_path) if target_path else "",
        "target_duration_s": duration,
        "target_duration_min_s": minimum,
        "target_duration_max_s": maximum,
        "target_rms_dbfs": target_audio.get("rms_dbfs"),
        "target_peak_dbfs": target_audio.get("peak_dbfs"),
        "target_whisper_expected": (pair.get("verification") or {}).get("target_whisper", {}).get("expected"),
        "target_whisper_heard": (pair.get("verification") or {}).get("target_whisper", {}).get("heard"),
        "target_whisper_match": (pair.get("verification") or {}).get("target_whisper", {}).get("matches_expected"),
        "input_text": labels.get("input_text") or "",
        "input_wav_path": (audio.get("input") or {}).get("wav_path") or "",
        "input_whisper_heard": (pair.get("verification") or {}).get("input_whisper", {}).get("heard"),
        "input_whisper_match": (pair.get("verification") or {}).get("input_whisper", {}).get("matches_expected"),
        "auto_status": auto_status,
        "training_status": training_status,
        "reasons": reasons,
        "manual_review": manual,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    by_reason: dict[str, int] = {}
    by_split: dict[str, int] = {}
    for row in rows:
        by_status[row["training_status"]] = by_status.get(row["training_status"], 0) + 1
        by_split[str(row.get("split") or "unknown")] = by_split.get(str(row.get("split") or "unknown"), 0) + 1
        for reason in row["reasons"]:
            by_reason[reason] = by_reason.get(reason, 0) + 1
    return {
        "total": len(rows),
        "by_split": dict(sorted(by_split.items())),
        "auto_pass_needs_manual_review": sum(1 for row in rows if row["auto_status"] == "auto_pass_needs_manual_review"),
        "auto_quarantine": sum(1 for row in rows if row["auto_status"] == "auto_quarantine"),
        "approved_for_training": by_status.get("approved_for_training", 0),
        "blocked_or_quarantined": len(rows) - by_status.get("approved_for_training", 0),
        "by_training_status": dict(sorted(by_status.items())),
        "by_reason": dict(sorted(by_reason.items())),
    }


def write_outputs(rows: list[dict[str, Any]], out_dir: Path, summaries: list[Path], manual_review_path: Path | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated_at_unix": time.time(),
        "purpose": "Dataset quality gate. Auto checks are necessary but not sufficient; manual approval is required before training.",
        "source_summaries": [str(path) for path in summaries],
        "manual_review_path": str(manual_review_path) if manual_review_path else None,
        **summarize(rows),
    }
    (out_dir / "quality_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (out_dir / "quality_rows.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    queue_fields = [
        "id",
        "split",
        "target_text",
        "style",
        "speed",
        "loudness",
        "target_voice",
        "target_duration_s",
        "target_rms_dbfs",
        "target_peak_dbfs",
        "target_whisper_heard",
        "auto_status",
        "training_status",
        "reasons",
        "target_wav_path",
    ]
    with (out_dir / "manual_review_queue.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=queue_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: ";".join(row[field]) if field == "reasons" else row.get(field) for field in queue_fields})

    approved = [row for row in rows if row["training_status"] == "approved_for_training"]
    with (out_dir / "approved_rows.jsonl").open("w") as handle:
        for row in approved:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    (out_dir / "README.md").write_text(
        "# TTS Dataset Quality Gate\n\n"
        "This directory is generated by `gemma4-omni-fitness/tts_dataset_quality_gate.py`.\n\n"
        "Auto checks are necessary but not sufficient. A row is not approved for training unless it passes auto checks and has a manual review verdict of `pass`.\n\n"
        "Files:\n\n"
        "- `quality_summary.json`: counts and source paths.\n"
        "- `quality_rows.jsonl`: full per-sample status.\n"
        "- `manual_review_queue.csv`: compact spreadsheet for human review.\n"
        "- `approved_rows.jsonl`: only rows explicitly approved for training.\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a no-GPU quality gate for TTS training data.")
    parser.add_argument("--summary", action="append", default=[], help="Dataset summary.json path. Can be repeated.")
    parser.add_argument("--manual-review", default="", help="Optional JSONL manual review verdicts.")
    parser.add_argument("--out-dir", default="out/gemma4_omni_tts_quality_gate")
    parser.add_argument("--fail-on-unapproved", action="store_true")
    args = parser.parse_args()

    summary_paths = [Path(value) for value in (args.summary or DEFAULT_SUMMARIES)]
    manual_review_path = Path(args.manual_review) if args.manual_review else None
    manual_reviews = load_manual_reviews(manual_review_path)
    rows: list[dict[str, Any]] = []
    for summary_path in summary_paths:
        for pair in load_pairs(summary_path):
            rows.append(build_row(summary_path, pair, manual_reviews))
    write_outputs(rows, Path(args.out_dir), summary_paths, manual_review_path)
    summary = summarize(rows)
    print(json.dumps(summary, indent=2))
    if args.fail_on_unapproved and summary["approved_for_training"] < summary["total"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
