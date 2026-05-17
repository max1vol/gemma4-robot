#!/usr/bin/env python3
"""Camera exercise mode for the AIY Voice HAT kiosk.

The deployed pose model is a MediaPipe Pose Landmarker. It reports body
landmarks, not activity labels, so squats are counted from hip/knee/ankle
geometry.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path.home() / "gemma4-robot"
DEFAULT_KIOSK_DIR = ROOT / "kiosk"
DEFAULT_STATUS_FILE = DEFAULT_KIOSK_DIR / "exercise_status.json"
DEFAULT_FRAME_FILE = DEFAULT_KIOSK_DIR / "exercise_frame.rgb"
DEFAULT_RUNTIME = ROOT / "out" / "pose_neon_runtime_aarch64_ofast"
DEFAULT_DATA_DIR = ROOT / "out" / "pose_runtime_data"

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28


class SquatCounter:
    def __init__(self, down_angle: float, up_angle: float) -> None:
        self.down_angle = down_angle
        self.up_angle = up_angle
        self.count = 0
        self.phase = "unknown"

    def update(self, knee_angle: float | None) -> None:
        if knee_angle is None:
            return

        if self.phase == "unknown":
            if knee_angle >= self.up_angle:
                self.phase = "standing"
            elif knee_angle <= self.down_angle:
                self.phase = "down"
            return

        if self.phase == "standing" and knee_angle <= self.down_angle:
            self.phase = "down"
            return

        if self.phase == "down" and knee_angle >= self.up_angle:
            self.count += 1
            self.phase = "standing"


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def now_label() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def run_command(command: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )


def capture_bgr_frame(args: argparse.Namespace, path: Path) -> None:
    command = [
        args.rpicam_still,
        "-n",
        "--immediate",
        "-t",
        str(args.capture_timeout_ms),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "-e",
        "rgb",
        "-o",
        str(path),
    ]
    if args.camera_mode:
        command.extend(["--mode", args.camera_mode])
    result = run_command(command, args.capture_timeout_ms / 1000.0 + 8.0)
    if result.returncode != 0:
        raise RuntimeError(f"camera capture failed ({result.returncode}): {result.stdout.strip()[-900:]}")
    expected = args.width * args.height * 3
    size = path.stat().st_size if path.exists() else 0
    if size != expected:
        raise RuntimeError(f"camera frame has {size} bytes, expected {expected}")


def bgr_file_to_rgb_bytes(path: Path) -> bytes:
    data = bytearray(path.read_bytes())
    data[0::3], data[2::3] = data[2::3], data[0::3]
    return bytes(data)


def run_pose(args: argparse.Namespace, rgb_path: Path, pose_json: Path) -> dict[str, Any]:
    command = [
        str(args.pose_runtime),
        "pipeline-rgb",
        str(args.pose_data_dir),
        str(rgb_path),
        str(args.width),
        str(args.height),
        str(pose_json),
        str(args.threads),
        "1",
    ]
    result = run_command(command, args.pose_timeout_seconds)
    if result.returncode != 0:
        raise RuntimeError(f"pose runtime failed ({result.returncode}): {result.stdout.strip()[-900:]}")
    return json.loads(pose_json.read_text())


def landmark_quality(landmark: dict[str, Any]) -> float:
    visibility = float(landmark.get("visibility", 0.0))
    presence = float(landmark.get("presence", 0.0))
    return min(visibility, presence)


def usable(landmarks: list[dict[str, Any]], index: int, min_quality: float) -> bool:
    if index >= len(landmarks):
        return False
    return landmark_quality(landmarks[index]) >= min_quality


def angle_degrees(a: dict[str, Any], b: dict[str, Any], c: dict[str, Any]) -> float | None:
    bax = float(a["x"]) - float(b["x"])
    bay = float(a["y"]) - float(b["y"])
    bcx = float(c["x"]) - float(b["x"])
    bcy = float(c["y"]) - float(b["y"])
    mag_ba = math.hypot(bax, bay)
    mag_bc = math.hypot(bcx, bcy)
    if mag_ba <= 1e-6 or mag_bc <= 1e-6:
        return None
    cosine = max(-1.0, min(1.0, (bax * bcx + bay * bcy) / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cosine))


def average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def extract_landmarks(pose: dict[str, Any]) -> list[dict[str, Any]]:
    groups = pose.get("pose_landmarks")
    if not isinstance(groups, list) or not groups:
        return []
    first = groups[0]
    return first if isinstance(first, list) else []


def squat_metrics(pose: dict[str, Any], min_quality: float) -> dict[str, Any]:
    landmarks = extract_landmarks(pose)
    if not landmarks:
        return {"landmarks": [], "knee_angle": None, "torso_angle": None}

    knee_angles: list[float] = []
    if all(usable(landmarks, idx, min_quality) for idx in [LEFT_HIP, LEFT_KNEE, LEFT_ANKLE]):
        value = angle_degrees(landmarks[LEFT_HIP], landmarks[LEFT_KNEE], landmarks[LEFT_ANKLE])
        if value is not None:
            knee_angles.append(value)
    if all(usable(landmarks, idx, min_quality) for idx in [RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE]):
        value = angle_degrees(landmarks[RIGHT_HIP], landmarks[RIGHT_KNEE], landmarks[RIGHT_ANKLE])
        if value is not None:
            knee_angles.append(value)

    torso_angle = None
    if all(usable(landmarks, idx, min_quality) for idx in [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]):
        shoulder_x = (float(landmarks[LEFT_SHOULDER]["x"]) + float(landmarks[RIGHT_SHOULDER]["x"])) * 0.5
        shoulder_y = (float(landmarks[LEFT_SHOULDER]["y"]) + float(landmarks[RIGHT_SHOULDER]["y"])) * 0.5
        hip_x = (float(landmarks[LEFT_HIP]["x"]) + float(landmarks[RIGHT_HIP]["x"])) * 0.5
        hip_y = (float(landmarks[LEFT_HIP]["y"]) + float(landmarks[RIGHT_HIP]["y"])) * 0.5
        torso_angle = abs(math.degrees(math.atan2(shoulder_x - hip_x, shoulder_y - hip_y)))

    return {
        "landmarks": landmarks,
        "knee_angle": average(knee_angles),
        "torso_angle": torso_angle,
    }


def compact_landmarks(landmarks: list[dict[str, Any]]) -> list[dict[str, float]]:
    compact: list[dict[str, float]] = []
    for landmark in landmarks[:33]:
        compact.append(
            {
                "x": float(landmark.get("x", 0.0)),
                "y": float(landmark.get("y", 0.0)),
                "visibility": float(landmark.get("visibility", 0.0)),
                "presence": float(landmark.get("presence", 0.0)),
            }
        )
    return compact


def write_status(
    args: argparse.Namespace,
    state: str,
    frame_seq: int,
    counter: SquatCounter,
    *,
    pose: dict[str, Any] | None = None,
    landmarks: list[dict[str, Any]] | None = None,
    knee_angle: float | None = None,
    torso_angle: float | None = None,
    fps: float | None = None,
    error: str = "",
) -> None:
    payload = {
        "mode": "exercise",
        "state": state,
        "updated_at": now_label(),
        "frame_seq": frame_seq,
        "frame_width": args.width,
        "frame_height": args.height,
        "frame_path": args.frame_file.name,
        "squats": counter.count,
        "squat_phase": counter.phase,
        "knee_angle": knee_angle,
        "torso_angle": torso_angle,
        "fps": fps,
        "pose_count": int(pose.get("pose_count", 0)) if pose else 0,
        "pose_presence": float(pose.get("pose_presence", 0.0)) if pose else 0.0,
        "landmarks": compact_landmarks(landmarks or []),
        "error": error,
    }
    atomic_write_json(args.status_file, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-file", type=Path, default=DEFAULT_STATUS_FILE)
    parser.add_argument("--frame-file", type=Path, default=DEFAULT_FRAME_FILE)
    parser.add_argument("--pose-runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--pose-data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--rpicam-still", default="rpicam-still")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--camera-mode", default="")
    parser.add_argument("--capture-timeout-ms", type=int, default=1)
    parser.add_argument("--pose-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--min-quality", type=float, default=0.35)
    parser.add_argument("--down-angle", type=float, default=115.0)
    parser.add_argument("--up-angle", type=float, default=160.0)
    parser.add_argument("--target-period", type=float, default=0.05)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.status_file = args.status_file.expanduser()
    args.frame_file = args.frame_file.expanduser()
    args.pose_runtime = args.pose_runtime.expanduser()
    args.pose_data_dir = args.pose_data_dir.expanduser()

    if not args.pose_runtime.exists():
        raise SystemExit(f"pose runtime missing: {args.pose_runtime}")
    if not args.pose_data_dir.exists():
        raise SystemExit(f"pose data dir missing: {args.pose_data_dir}")

    stop = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    work_dir = args.status_file.parent / "exercise_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    bgr_path = work_dir / "frame.bgr"
    rgb_path = work_dir / "frame.rgb"
    pose_json = work_dir / "pose.json"

    counter = SquatCounter(args.down_angle, args.up_angle)
    write_status(args, "starting", 0, counter)

    frame_seq = 0
    last_frame_time = 0.0
    while not stop:
        frame_start = time.monotonic()
        try:
            capture_bgr_frame(args, bgr_path)
            rgb = bgr_file_to_rgb_bytes(bgr_path)
            atomic_write_bytes(rgb_path, rgb)
            atomic_write_bytes(args.frame_file, rgb)
            pose = run_pose(args, rgb_path, pose_json)
            pose_present = (
                int(pose.get("pose_count", 0)) > 0
                and float(pose.get("pose_presence", 0.0)) >= args.min_quality
            )
            metrics = squat_metrics(pose, args.min_quality) if pose_present else {
                "landmarks": [],
                "knee_angle": None,
                "torso_angle": None,
            }
            knee_angle = metrics["knee_angle"]
            if pose_present:
                counter.update(knee_angle)
            frame_seq += 1
            now = time.monotonic()
            fps = 1.0 / (now - last_frame_time) if last_frame_time else None
            last_frame_time = now
            state = "active" if pose_present else "no_pose"
            write_status(
                args,
                state,
                frame_seq,
                counter,
                pose=pose,
                landmarks=metrics["landmarks"],
                knee_angle=knee_angle,
                torso_angle=metrics["torso_angle"],
                fps=fps,
            )
        except Exception as exc:
            frame_seq += 1
            write_status(args, "error", frame_seq, counter, error=str(exc))
            print(f"exercise mode error: {exc}", file=sys.stderr, flush=True)
            time.sleep(1.0)

        if args.once:
            break

        elapsed = time.monotonic() - frame_start
        if elapsed < args.target_period:
            time.sleep(args.target_period - elapsed)

    write_status(args, "stopped", frame_seq, counter)


if __name__ == "__main__":
    main()
