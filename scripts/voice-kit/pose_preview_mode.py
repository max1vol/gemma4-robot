#!/usr/bin/env python3
"""Framebuffer camera preview with pose skeleton overlay.

This intentionally avoids Chromium and Python image libraries. It streams raw
YUV frames from rpicam-vid, draws a lightweight color preview to /dev/fb0,
and runs the existing pose runtime on sampled frames with a two-thread limit.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import zlib
from pathlib import Path
from typing import Any


ROOT = Path.home() / "gemma4-robot"
DEFAULT_RUNTIME = ROOT / "out" / "pose_neon_runtime_aarch64_ofast"
DEFAULT_DATA_DIR = ROOT / "out" / "pose_runtime_data"
DEFAULT_LATENCY_LOG = Path("/tmp/gemma4-pose-preview-latencies.jsonl")
DEFAULT_IPHONE_POSE_BRIDGE = "http://127.0.0.1:8765"

CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]


def parse_fbset() -> tuple[int, int, int]:
    output = subprocess.check_output(["fbset", "-i"], text=True, stderr=subprocess.DEVNULL)
    width = height = line_length = 0
    for line in output.splitlines():
        parts = line.strip().split()
        if parts[:1] == ["geometry"] and len(parts) >= 3:
            width = int(parts[1])
            height = int(parts[2])
        if parts[:1] == ["LineLength"]:
            for part in reversed(parts[1:]):
                if part.isdigit():
                    line_length = int(part)
                    break
    if width <= 0 or height <= 0 or line_length <= 0:
        raise RuntimeError("could not parse framebuffer geometry")
    return width, height, line_length


def read_exact(stream: Any, size: int) -> bytes | None:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def clip_u8(value: int) -> int:
    if value < 0:
        return 0
    if value > 255:
        return 255
    return value


def yuv_to_rgb(y: int, u: int, v: int) -> tuple[int, int, int]:
    c = y - 16
    d = u - 128
    e = v - 128
    r = (298 * c + 409 * e + 128) >> 8
    g = (298 * c - 100 * d - 208 * e + 128) >> 8
    b = (298 * c + 516 * d + 128) >> 8
    return clip_u8(r), clip_u8(g), clip_u8(b)


def rgb565_bytes(r: int, g: int, b: int) -> bytes:
    pixel = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)
    return pixel.to_bytes(2, "little")


def yuv420_to_rgb24(yuv: bytes, width: int, height: int) -> bytes:
    frame = width * height
    y_plane = yuv[:frame]
    u_plane = yuv[frame:frame + frame // 4]
    v_plane = yuv[frame + frame // 4:frame + frame // 2]
    rgb = bytearray(frame * 3)
    out = 0
    chroma_width = width // 2
    for row in range(height):
        y_row = row * width
        uv_row = (row // 2) * chroma_width
        for col in range(width):
            uv_index = uv_row + (col // 2)
            r, g, b = yuv_to_rgb(y_plane[y_row + col], u_plane[uv_index], v_plane[uv_index])
            rgb[out] = r
            rgb[out + 1] = g
            rgb[out + 2] = b
            out += 3
    return bytes(rgb)


def raw_deflate(data: bytes, level: int) -> bytes:
    compressor = zlib.compressobj(level, zlib.DEFLATED, -15)
    return compressor.compress(data) + compressor.flush()


def iphone_pose_payload(yuv: bytes, args: argparse.Namespace) -> tuple[str, bytes, float]:
    started = time.monotonic()
    if args.iphone_pose_format == "yuv420":
        return "yuv420", yuv, (time.monotonic() - started) * 1000
    if args.iphone_pose_format == "deflate_yuv420":
        return "deflate_yuv420", raw_deflate(yuv, args.iphone_zlib_level), (time.monotonic() - started) * 1000

    rgb = yuv420_to_rgb24(yuv, args.width, args.height)
    if args.iphone_pose_format == "rgb24":
        return "rgb24", rgb, (time.monotonic() - started) * 1000
    if args.iphone_pose_format == "deflate_rgb24":
        return "deflate_rgb24", raw_deflate(rgb, args.iphone_zlib_level), (time.monotonic() - started) * 1000
    raise ValueError(f"unsupported iPhone pose format: {args.iphone_pose_format}")


def run_iphone_pose(yuv: bytes, args: argparse.Namespace, record: dict[str, Any]) -> dict[str, Any]:
    frame_format, payload, encode_ms = iphone_pose_payload(yuv, args)
    record["pose_format"] = frame_format
    record["encode_ms"] = round(encode_ms, 2)
    record["payload_bytes"] = len(payload)
    query = urllib.parse.urlencode(
        {
            "format": frame_format,
            "width": args.width,
            "height": args.height,
            "pose_backend": args.iphone_pose_backend,
            "pose_model": args.iphone_pose_model,
            "timeout": str(args.iphone_pose_timeout),
        }
    )
    request = urllib.request.Request(
        f"{args.iphone_pose_bridge.rstrip('/')}/pose-frame?{query}",
        data=payload,
        headers={"Content-Type": "application/octet-stream"},
        method="POST",
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=args.iphone_pose_timeout + 5) as response:
        body = response.read()
    record["http_ms"] = round((time.monotonic() - started) * 1000, 2)
    pose = json.loads(body.decode("utf-8"))
    record["iphone_decode_ms"] = round(float(pose.get("decode_seconds", 0.0)) * 1000, 2)
    record["iphone_inference_ms"] = round(float(pose.get("inference_seconds", 0.0)) * 1000, 2)
    record["iphone_total_ms"] = round(float(pose.get("total_seconds", 0.0)) * 1000, 2)
    return pose


def extract_landmarks(pose: dict[str, Any], min_quality: float) -> list[dict[str, float]]:
    groups = pose.get("pose_landmarks")
    if not isinstance(groups, list) or not groups or not isinstance(groups[0], list):
        return []
    landmarks = []
    for landmark in groups[0][:33]:
        visibility = float(landmark.get("visibility", 0.0))
        presence = float(landmark.get("presence", 0.0))
        if min(visibility, presence) < min_quality:
            landmarks.append({"x": 0.0, "y": 0.0, "ok": 0.0})
        else:
            landmarks.append({
                "x": float(landmark.get("x", 0.0)),
                "y": float(landmark.get("y", 0.0)),
                "ok": 1.0,
            })
    return landmarks


class SharedPose:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.landmarks: list[dict[str, float]] = []
        self.pose_count = 0
        self.pose_presence = 0.0
        self.model_running = False
        self.last_model_start = 0.0
        self.last_model_ms: float | None = None


def draw_line(buf: bytearray, fb_w: int, fb_h: int, stride: int, x0: int, y0: int, x1: int, y1: int, color: bytes) -> None:
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x = x0
    y = y0
    while True:
        for oy in (-1, 0, 1):
            yy = y + oy
            if 0 <= yy < fb_h:
                for ox in (-1, 0, 1):
                    xx = x + ox
                    if 0 <= xx < fb_w:
                        off = yy * stride + xx * 2
                        buf[off:off + 2] = color
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def draw_skeleton(buf: bytearray, fb_w: int, fb_h: int, stride: int, ox: int, oy: int, dw: int, dh: int, landmarks: list[dict[str, float]]) -> None:
    if len(landmarks) < 33:
        return
    yellow = (0xffe0).to_bytes(2, "little")
    cyan = (0x07ff).to_bytes(2, "little")
    for a, b in CONNECTIONS:
        la = landmarks[a]
        lb = landmarks[b]
        if not la.get("ok") or not lb.get("ok"):
            continue
        x0 = ox + int(la["x"] * dw)
        y0 = oy + int(la["y"] * dh)
        x1 = ox + int(lb["x"] * dw)
        y1 = oy + int(lb["y"] * dh)
        draw_line(buf, fb_w, fb_h, stride, x0, y0, x1, y1, cyan)
    for landmark in landmarks:
        if not landmark.get("ok"):
            continue
        cx = ox + int(landmark["x"] * dw)
        cy = oy + int(landmark["y"] * dh)
        for yy in range(cy - 3, cy + 4):
            if 0 <= yy < fb_h:
                for xx in range(cx - 3, cx + 4):
                    if 0 <= xx < fb_w and (xx - cx) ** 2 + (yy - cy) ** 2 <= 10:
                        off = yy * stride + xx * 2
                        buf[off:off + 2] = yellow


def render_frame(yuv: bytes, args: argparse.Namespace, fb_w: int, fb_h: int, stride: int, scale: int, landmarks: list[dict[str, float]]) -> bytes:
    width = args.width
    height = args.height
    dw = width * scale
    dh = height * scale
    ox = (fb_w - dw) // 2
    oy = (fb_h - dh) // 2
    frame = width * height
    y_plane = yuv[:width * height]
    u_plane = yuv[frame:frame + frame // 4]
    v_plane = yuv[frame + frame // 4:frame + frame // 2]
    chroma_width = width // 2
    buf = bytearray(stride * fb_h)
    for row_index in range(height):
        y_row = row_index * width
        uv_row = (row_index // 2) * chroma_width
        row = bytearray(width * scale * 2)
        out = 0
        for col in range(width):
            uv_index = uv_row + (col // 2)
            r, g, b = yuv_to_rgb(y_plane[y_row + col], u_plane[uv_index], v_plane[uv_index])
            pixel = rgb565_bytes(r, g, b)
            for _ in range(scale):
                row[out:out + 2] = pixel
                out += 2
        dest_start = (oy + row_index * scale) * stride + ox * 2
        for repeat in range(scale):
            off = dest_start + repeat * stride
            buf[off:off + len(row)] = row
    draw_skeleton(buf, fb_w, fb_h, stride, ox, oy, dw, dh, landmarks)
    return bytes(buf)


def run_model_worker(yuv: bytes, frame_seq: int, args: argparse.Namespace, shared: SharedPose, log_handle: Any) -> None:
    start = time.monotonic()
    record: dict[str, Any] = {
        "event": "model",
        "frame": frame_seq,
        "engine": args.pose_engine,
        "threads": args.threads,
        "cores": args.cores,
    }
    try:
        if args.pose_engine == "iphone":
            pose = run_iphone_pose(yuv, args, record)
        else:
            with tempfile.TemporaryDirectory(prefix="gemma4-pose-") as tmp:
                tmpdir = Path(tmp)
                rgb_path = tmpdir / "frame.rgb"
                json_path = tmpdir / "pose.json"
                t0 = time.monotonic()
                rgb_path.write_bytes(yuv420_to_rgb24(yuv, args.width, args.height))
                record["rgb_ms"] = round((time.monotonic() - t0) * 1000, 2)
                command = [
                    str(args.pose_runtime),
                    "pipeline-rgb-track",
                    str(args.pose_data_dir),
                    str(rgb_path),
                    str(args.width),
                    str(args.height),
                    str(json_path),
                    str(args.threads),
                    "1",
                    str(args.refresh_interval),
                ]
                if shutil.which("taskset"):
                    command = ["taskset", "-c", args.cores] + command
                t1 = time.monotonic()
                result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=args.model_timeout, check=False)
                record["runtime_ms"] = round((time.monotonic() - t1) * 1000, 2)
                record["returncode"] = result.returncode
                if result.returncode != 0:
                    record["error"] = result.stdout[-500:]
                    pose = {"pose_count": 0, "pose_presence": 0.0, "pose_landmarks": []}
                else:
                    pose = json.loads(json_path.read_text())

        landmarks = extract_landmarks(pose, args.min_quality)
        record["pose_count"] = int(pose.get("pose_count", 0))
        record["pose_presence"] = round(float(pose.get("pose_presence", 0.0)), 4)
        with shared.lock:
            shared.landmarks = landmarks
            shared.pose_count = record["pose_count"]
            shared.pose_presence = record["pose_presence"]
            shared.last_model_ms = record.get("iphone_total_ms") or record.get("runtime_ms")
    except Exception as exc:
        record["error"] = str(exc)
    finally:
        record["total_ms"] = round((time.monotonic() - start) * 1000, 2)
        record["ts"] = time.time()
        print(json.dumps(record, separators=(",", ":")), file=log_handle, flush=True)
        with shared.lock:
            shared.model_running = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--framerate", type=float, default=8.0)
    parser.add_argument("--roi", default="", help="optional rpicam normalized crop x,y,w,h; example 0.25,0.25,0.5,0.5")
    parser.add_argument("--pose-runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--pose-data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--pose-engine", choices=["iphone", "local"], default="iphone")
    parser.add_argument("--iphone-pose-bridge", default=os.environ.get("GEMMA_IPHONE_POSE_BRIDGE", DEFAULT_IPHONE_POSE_BRIDGE))
    parser.add_argument("--iphone-pose-format", choices=["deflate_yuv420", "yuv420", "deflate_rgb24", "rgb24"], default="deflate_yuv420")
    parser.add_argument("--iphone-pose-backend", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--iphone-pose-model", choices=["lite", "full", "heavy"], default="full")
    parser.add_argument("--iphone-zlib-level", type=int, default=1)
    parser.add_argument("--iphone-pose-timeout", type=float, default=10.0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--cores", default="0,1")
    parser.add_argument("--model-interval", type=float, default=0.25)
    parser.add_argument("--model-timeout", type=float, default=20.0)
    parser.add_argument("--refresh-interval", type=int, default=8)
    parser.add_argument("--min-quality", type=float, default=0.35)
    parser.add_argument("--latency-log", type=Path, default=DEFAULT_LATENCY_LOG)
    parser.add_argument("--fb", default="/dev/fb0")
    parser.add_argument("--rpicam-vid", default="rpicam-vid")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.pose_runtime = args.pose_runtime.expanduser()
    args.pose_data_dir = args.pose_data_dir.expanduser()
    args.latency_log = args.latency_log.expanduser()
    args.latency_log.parent.mkdir(parents=True, exist_ok=True)
    stop = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    fb_w, fb_h, stride = parse_fbset()
    frame_size = args.width * args.height * 3 // 2
    scale = max(1, min(fb_w // args.width, fb_h // args.height))

    command = [
        args.rpicam_vid,
        "-t", "0",
        "--nopreview",
        "--codec", "yuv420",
        "--width", str(args.width),
        "--height", str(args.height),
        "--framerate", str(args.framerate),
        "--flush",
        "-o", "-",
    ]
    if args.roi:
        command.extend(["--roi", args.roi])
    shared = SharedPose()
    frame_seq = 0
    last_frame_at = 0.0

    with args.latency_log.open("a", buffering=1) as log_handle, open(args.fb, "r+b", buffering=0) as fb:
        print(json.dumps({
            "event": "start",
            "ts": time.time(),
            "frame": f"{args.width}x{args.height}",
            "display": f"{fb_w}x{fb_h}",
            "framerate": args.framerate,
            "threads": args.threads,
            "cores": args.cores,
        }, separators=(",", ":")), file=log_handle, flush=True)
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=sys.stderr)
        try:
            assert proc.stdout is not None
            while not stop:
                t_read = time.monotonic()
                frame = read_exact(proc.stdout, frame_size)
                if frame is None:
                    break
                frame_seq += 1
                with shared.lock:
                    landmarks = list(shared.landmarks)
                    can_start_model = not shared.model_running and (t_read - shared.last_model_start) >= args.model_interval
                    if can_start_model:
                        shared.model_running = True
                        shared.last_model_start = t_read
                if can_start_model:
                    threading.Thread(
                        target=run_model_worker,
                        args=(bytes(frame), frame_seq, args, shared, log_handle),
                        daemon=True,
                    ).start()
                t_render = time.monotonic()
                image = render_frame(frame, args, fb_w, fb_h, stride, scale, landmarks)
                fb.seek(0)
                fb.write(image)
                now = time.monotonic()
                display_fps = 1.0 / (now - last_frame_at) if last_frame_at else None
                last_frame_at = now
                if frame_seq % 30 == 0:
                    with shared.lock:
                        model_ms = shared.last_model_ms
                        pose_count = shared.pose_count
                        pose_presence = shared.pose_presence
                    print(json.dumps({
                        "event": "display",
                        "frame": frame_seq,
                        "ts": time.time(),
                        "read_ms": round((t_render - t_read) * 1000, 2),
                        "render_write_ms": round((now - t_render) * 1000, 2),
                        "display_fps": round(display_fps, 2) if display_fps else None,
                        "last_model_ms": model_ms,
                        "pose_count": pose_count,
                        "pose_presence": pose_presence,
                    }, separators=(",", ":")), file=log_handle, flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)
            print(json.dumps({"event": "stop", "ts": time.time(), "frame": frame_seq}, separators=(",", ":")), file=log_handle, flush=True)


if __name__ == "__main__":
    main()
