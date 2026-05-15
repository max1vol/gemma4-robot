#!/usr/bin/env python3
"""Benchmark MediaPipe Pose Landmarker Lite on Raspberry Pi camera frames."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def ensure_import_stubs() -> None:
    """Let mediapipe import without OpenCV/matplotlib when drawing is unused."""
    stub_dir = Path(__file__).resolve().parent / ".mediapipe_import_stubs"
    stub_dir.mkdir(exist_ok=True)
    cv2_stub = stub_dir / "cv2.py"
    if not cv2_stub.exists():
        cv2_stub.write_text(
            "FONT_HERSHEY_SIMPLEX = 0\n"
            "FONT_HERSHEY_DUPLEX = 2\n"
            "LINE_AA = 16\n"
            "def circle(*args, **kwargs): raise RuntimeError('cv2 stub')\n"
            "def rectangle(*args, **kwargs): raise RuntimeError('cv2 stub')\n"
            "def line(*args, **kwargs): raise RuntimeError('cv2 stub')\n"
            "def putText(*args, **kwargs): raise RuntimeError('cv2 stub')\n"
        )
    mpl = stub_dir / "matplotlib"
    mpl.mkdir(exist_ok=True)
    (mpl / "__init__.py").touch()
    (mpl / "pyplot.py").touch()
    sys.path.insert(0, str(stub_dir))


def yuv420_to_rgb(frame: bytes, width: int, height: int) -> np.ndarray:
    y_size = width * height
    uv_size = y_size // 4
    expected = y_size + 2 * uv_size
    if len(frame) != expected:
        raise ValueError(f"expected {expected} bytes, got {len(frame)}")

    raw = np.frombuffer(frame, dtype=np.uint8)
    y = raw[:y_size].reshape((height, width)).astype(np.int16)
    u = raw[y_size : y_size + uv_size].reshape((height // 2, width // 2)).astype(np.int16)
    v = raw[y_size + uv_size :].reshape((height // 2, width // 2)).astype(np.int16)
    u = np.repeat(np.repeat(u, 2, axis=0), 2, axis=1)
    v = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1)

    c = y - 16
    d = u - 128
    e = v - 128
    r = (298 * c + 409 * e + 128) >> 8
    g = (298 * c - 100 * d - 208 * e + 128) >> 8
    b = (298 * c + 516 * d + 128) >> 8
    return np.dstack(
        (
            np.clip(r, 0, 255),
            np.clip(g, 0, 255),
            np.clip(b, 0, 255),
        )
    ).astype(np.uint8)


def read_exact(stream, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return ordered[idx]


def summarize_ms(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": percentile(values, 0.90),
        "min": min(values),
        "max": max(values),
    }


def make_landmarker(model: str):
    ensure_import_stubs()
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    return mp, vision.PoseLandmarker.create_from_options(options)


def run_camera(args: argparse.Namespace) -> dict[str, object]:
    frame_size = args.width * args.height * 3 // 2
    cmd = [
        "rpicam-vid",
        "--nopreview",
        "--codec",
        "yuv420",
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--framerate",
        str(args.camera_fps),
        "--timeout",
        str(args.timeout_ms),
        "--output",
        "-",
    ]

    init_start = time.perf_counter()
    mp, detector = make_landmarker(args.model)
    init_s = time.perf_counter() - init_start

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    infer_ms: list[float] = []
    convert_ms: list[float] = []
    read_ms: list[float] = []
    pose_frames = 0
    first_frame_at = None
    bench_start = time.perf_counter()

    try:
        for frame_idx in range(args.frames):
            read_start = time.perf_counter()
            frame = read_exact(proc.stdout, frame_size)
            read_ms.append((time.perf_counter() - read_start) * 1000)
            if len(frame) != frame_size:
                break
            if first_frame_at is None:
                first_frame_at = time.perf_counter()

            convert_start = time.perf_counter()
            rgb = yuv420_to_rgb(frame, args.width, args.height)
            convert_ms.append((time.perf_counter() - convert_start) * 1000)

            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_idx * 1000 / args.camera_fps)
            infer_start = time.perf_counter()
            result = detector.detect_for_video(image, timestamp_ms)
            infer_ms.append((time.perf_counter() - infer_start) * 1000)
            if result.pose_landmarks:
                pose_frames += 1
    finally:
        detector.close()
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
        stderr = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""

    bench_end = time.perf_counter()
    frames = len(infer_ms)
    infer_total_s = sum(infer_ms) / 1000
    active_wall_s = bench_end - (first_frame_at or bench_start)
    return {
        "source": "rpicam-vid-yuv420",
        "model": args.model,
        "width": args.width,
        "height": args.height,
        "requested_camera_fps": args.camera_fps,
        "frames": frames,
        "pose_frames": pose_frames,
        "model_init_s": init_s,
        "active_wall_s": active_wall_s,
        "pure_inference_fps": frames / infer_total_s if infer_total_s else 0.0,
        "pipeline_fps_after_first_frame": frames / active_wall_s if active_wall_s else 0.0,
        "read_ms": summarize_ms(read_ms[:frames]),
        "convert_ms": summarize_ms(convert_ms),
        "infer_ms": summarize_ms(infer_ms),
        "rpicam_stderr_tail": "\n".join(stderr.splitlines()[-20:]),
    }


def run_synthetic(args: argparse.Namespace) -> dict[str, object]:
    init_start = time.perf_counter()
    mp, detector = make_landmarker(args.model)
    init_s = time.perf_counter() - init_start
    rgb = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    infer_ms: list[float] = []
    pose_frames = 0
    for frame_idx in range(args.frames + args.warmup):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        infer_start = time.perf_counter()
        result = detector.detect_for_video(image, frame_idx * 33)
        elapsed = (time.perf_counter() - infer_start) * 1000
        if frame_idx >= args.warmup:
            infer_ms.append(elapsed)
            if result.pose_landmarks:
                pose_frames += 1
    detector.close()
    infer_total_s = sum(infer_ms) / 1000
    return {
        "source": "synthetic-black",
        "model": args.model,
        "width": args.width,
        "height": args.height,
        "frames": len(infer_ms),
        "pose_frames": pose_frames,
        "model_init_s": init_s,
        "pure_inference_fps": len(infer_ms) / infer_total_s if infer_total_s else 0.0,
        "infer_ms": summarize_ms(infer_ms),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="pose_landmarker_lite.task")
    parser.add_argument("--source", choices=["camera", "synthetic"], default="camera")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--timeout-ms", type=int, default=8000)
    args = parser.parse_args()

    if args.width % 2 or args.height % 2:
        raise SystemExit("width and height must be even for YUV420")
    if not Path(args.model).exists():
        raise SystemExit(f"model not found: {args.model}")

    if args.source == "camera":
        result = run_camera(args)
    else:
        result = run_synthetic(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
