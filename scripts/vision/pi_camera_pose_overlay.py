#!/usr/bin/env python3
"""Pi camera preview with local framebuffer overlay and iPhone pose requests.

This is intentionally dependency-light for the Pi 3. It reads YUV420 frames
from the CSI Pi camera through rpicam-vid, draws the camera and overlay directly
to /dev/fb0, and sends a lower-resolution RGB frame to the iPhone bridge for
MediaPipe pose.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - fallback for older Python builds.
    ZoneInfo = None  # type: ignore[assignment]


ROOT = Path.home() / "gemma4-robot"
DEFAULT_STATUS = ROOT / "kiosk" / "status.json"
DEFAULT_SENSOR_STATE = ROOT / "kiosk" / "sensors.json"
DEFAULT_VISION_STATE = ROOT / "kiosk" / "vision_state.json"
DEFAULT_VISION_COMMAND = ROOT / "kiosk" / "vision_command.json"
LONDON_TZ = ZoneInfo("Europe/London") if ZoneInfo is not None else None

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]

FONT = {
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    ":": ["00000", "01100", "01100", "00000", "01100", "01100", "00000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "/": ["00001", "00010", "00100", "01000", "10000", "00000", "00000"],
    "!": ["01100", "01100", "01100", "01100", "01100", "00000", "01100"],
    "?": ["11110", "00001", "00001", "00110", "00100", "00000", "00100"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01110"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "J": ["00001", "00001", "00001", "00001", "10001", "10001", "01110"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
}


def clamp(value: int) -> int:
    return 0 if value < 0 else 255 if value > 255 else value


def rgb565(r: int, g: int, b: int) -> bytes:
    value = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
    return value.to_bytes(2, "little")


BLACK = rgb565(0, 0, 0)
WHITE = rgb565(245, 248, 255)
GREEN = rgb565(50, 230, 120)
CYAN = rgb565(60, 210, 255)
YELLOW = rgb565(255, 220, 70)
RED = rgb565(255, 70, 70)
PANEL = rgb565(12, 18, 26)


class SharedState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.pose: dict[str, Any] = {}
        self.last_pose_at = 0.0
        self.human_seen_at: float | None = None
        self.squat_count = 0
        self.squat_phase = "up"
        self.coach_active = False
        self.coach_done = False
        self.target_reps = 10
        self.milestones = [3, 6, 9, 10]
        self.session_id = ""
        self.pose_success_times: list[float] = []
        self.pose_error = ""


def parse_fbset() -> tuple[int, int, int, int]:
    output = subprocess.check_output(["fbset", "-i"], text=True, stderr=subprocess.DEVNULL)
    width = height = stride = depth = 0
    for line in output.splitlines():
        parts = line.strip().split()
        if parts[:1] == ["geometry"] and len(parts) >= 6:
            width = int(parts[1])
            height = int(parts[2])
            depth = int(parts[5])
        elif parts[:1] == ["LineLength"]:
            for part in reversed(parts[1:]):
                if part.isdigit():
                    stride = int(part)
                    break
    if width <= 0 or height <= 0 or stride <= 0:
        raise RuntimeError("could not parse framebuffer geometry")
    if depth != 16:
        raise RuntimeError(f"only RGB565 framebuffer is supported, got depth={depth}")
    return width, height, stride, depth


def clear_framebuffer(path: str, width: int, height: int, stride: int) -> None:
    row = BLACK * width
    padding = b"\x00" * max(0, stride - len(row))
    with open(path, "r+b", buffering=0) as fb:
        fb.seek(0)
        for _ in range(height):
            fb.write(row)
            if padding:
                fb.write(padding)


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


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


def yuv_to_rgb(y: int, u: int, v: int) -> tuple[int, int, int]:
    c = y - 16
    d = u - 128
    e = v - 128
    r = clamp((298 * c + 409 * e + 128) >> 8)
    g = clamp((298 * c - 100 * d - 208 * e + 128) >> 8)
    b = clamp((298 * c + 516 * d + 128) >> 8)
    return r, g, b


def sample_yuv420_rgb(frame: bytes, src_w: int, src_h: int, sx: int, sy: int) -> tuple[int, int, int]:
    y_offset = sy * src_w + sx
    uv_w = src_w // 2
    uv_h = src_h // 2
    uv_index = (sy // 2) * uv_w + (sx // 2)
    u_offset = src_w * src_h
    v_offset = u_offset + uv_w * uv_h
    return yuv_to_rgb(frame[y_offset], frame[u_offset + uv_index], frame[v_offset + uv_index])


def put_pixel(buf: bytearray, stride: int, width: int, height: int, x: int, y: int, color: bytes) -> None:
    if 0 <= x < width and 0 <= y < height:
        offset = y * stride + x * 2
        buf[offset:offset + 2] = color


def draw_rect(buf: bytearray, stride: int, width: int, height: int, x: int, y: int, w: int, h: int, color: bytes) -> None:
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(width, x + w)
    y1 = min(height, y + h)
    row = color * max(0, x1 - x0)
    for yy in range(y0, y1):
        offset = yy * stride + x0 * 2
        buf[offset:offset + len(row)] = row


def draw_line(buf: bytearray, stride: int, width: int, height: int, x0: int, y0: int, x1: int, y1: int, color: bytes) -> None:
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        for oy in (-1, 0, 1):
            for ox in (-1, 0, 1):
                put_pixel(buf, stride, width, height, x0 + ox, y0 + oy, color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def draw_circle(buf: bytearray, stride: int, width: int, height: int, cx: int, cy: int, radius: int, color: bytes) -> None:
    rr = radius * radius
    for y in range(cy - radius, cy + radius + 1):
        for x in range(cx - radius, cx + radius + 1):
            if (x - cx) * (x - cx) + (y - cy) * (y - cy) <= rr:
                put_pixel(buf, stride, width, height, x, y, color)


def draw_char(buf: bytearray, stride: int, width: int, height: int, x: int, y: int, ch: str, color: bytes, scale: int) -> None:
    glyph = FONT.get(ch.upper(), FONT.get("?", FONT[" "]))
    for row, pattern in enumerate(glyph):
        for col, bit in enumerate(pattern):
            if bit == "1":
                draw_rect(buf, stride, width, height, x + col * scale, y + row * scale, scale, scale, color)


def draw_text(buf: bytearray, stride: int, width: int, height: int, x: int, y: int, text: str, color: bytes = WHITE, scale: int = 2) -> None:
    cursor = x
    for ch in text:
        draw_char(buf, stride, width, height, cursor, y, ch, color, scale)
        cursor += 6 * scale


def wrap_text(text: str, max_chars: int, max_lines: int) -> list[str]:
    words = text.replace("\n", " ").split()
    lines: list[str] = []
    current = ""
    for word in words:
        word = word.upper()
        if len(word) > max_chars:
            word = word[:max_chars]
        candidate = word if not current else f"{current} {word}"
        if len(candidate) > max_chars:
            if current:
                lines.append(current)
            current = word
        else:
            current = candidate
        if len(lines) >= max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    return lines


def render_camera_frame(
    frame: bytes,
    src_w: int,
    src_h: int,
    fb_w: int,
    fb_h: int,
    stride: int,
    state: SharedState,
    status_path: Path,
    sensor_path: Path,
    upscale: bool,
    max_display_width: int,
    max_display_height: int,
) -> bytearray:
    buf = bytearray(BLACK * (stride * fb_h // 2))
    max_w = min(fb_w, max_display_width if max_display_width > 0 else fb_w)
    max_h = min(fb_h, max_display_height if max_display_height > 0 else fb_h)
    fit_scale = min(max_w / src_w, max_h / src_h)
    scale = fit_scale if upscale else min(1.0, fit_scale)
    display_w = min(fb_w, int(src_w * scale))
    display_h = min(fb_h, int(src_h * scale))
    ox = (fb_w - display_w) // 2
    oy = (fb_h - display_h) // 2

    scale_x = display_w // src_w if display_w % src_w == 0 else 0
    scale_y = display_h // src_h if display_h % src_h == 0 else 0
    if scale_x > 0 and scale_y > 0:
        render_yuv420_integer_scaled(frame, src_w, src_h, buf, stride, ox, oy, scale_x, scale_y)
    else:
        for dy in range(display_h):
            sy = min(src_h - 1, dy * src_h // display_h)
            out = (oy + dy) * stride + ox * 2
            for dx in range(display_w):
                sx = min(src_w - 1, dx * src_w // display_w)
                r, g, b = sample_yuv420_rgb(frame, src_w, src_h, sx, sy)
                value = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
                buf[out:out + 2] = value.to_bytes(2, "little")
                out += 2

    with state.lock:
        pose = dict(state.pose)
        squat_count = state.squat_count
        phase = state.squat_phase
        active = state.coach_active
        target = state.target_reps
        human_present = (time.time() - state.last_pose_at) < 1.5 and bool(pose.get("pose_count"))
        pose_fps = pose_fps_from_times(state.pose_success_times)
        pose_status = pose_status_text(state.pose_error, state.last_pose_at)

    draw_pose(buf, stride, fb_w, fb_h, ox, oy, display_w, display_h, pose)
    draw_status_overlay(
        buf,
        stride,
        fb_w,
        fb_h,
        status_path,
        human_present,
        active,
        squat_count,
        target,
        phase,
        pose_fps,
        pose_status,
        sensor_path,
        ox,
        oy,
        display_w,
    )
    return buf


def render_yuv420_integer_scaled(
    frame: bytes,
    src_w: int,
    src_h: int,
    buf: bytearray,
    stride: int,
    ox: int,
    oy: int,
    scale_x: int,
    scale_y: int,
) -> None:
    for sy in range(src_h):
        row = bytearray()
        for sx in range(src_w):
            r, g, b = sample_yuv420_rgb(frame, src_w, src_h, sx, sy)
            row.extend(rgb565(r, g, b) * scale_x)
        for repeat in range(scale_y):
            out = (oy + sy * scale_y + repeat) * stride + ox * 2
            buf[out:out + len(row)] = row


def draw_pose(buf: bytearray, stride: int, fb_w: int, fb_h: int, ox: int, oy: int, dw: int, dh: int, pose: dict[str, Any]) -> None:
    poses = pose.get("pose_landmarks")
    if not isinstance(poses, list) or not poses:
        return
    landmarks = poses[0]
    if not isinstance(landmarks, list):
        return

    points: list[tuple[int, int, float]] = []
    for item in landmarks:
        if not isinstance(item, dict):
            points.append((0, 0, 0.0))
            continue
        visibility = float(item.get("visibility") or item.get("presence") or 0.0)
        x = ox + int(float(item.get("x") or 0.0) * dw)
        y = oy + int(float(item.get("y") or 0.0) * dh)
        points.append((x, y, visibility))

    for a, b in POSE_CONNECTIONS:
        if a < len(points) and b < len(points) and points[a][2] >= 0.25 and points[b][2] >= 0.25:
            draw_line(buf, stride, fb_w, fb_h, points[a][0], points[a][1], points[b][0], points[b][1], CYAN)
    for x, y, visibility in points:
        if visibility >= 0.25:
            draw_circle(buf, stride, fb_w, fb_h, x, y, 3, YELLOW)


def draw_status_overlay(
    buf: bytearray,
    stride: int,
    fb_w: int,
    fb_h: int,
    status_path: Path,
    human_present: bool,
    coach_active: bool,
    squat_count: int,
    target: int,
    phase: str,
    pose_fps: float,
    pose_status: str,
    sensor_path: Path,
    overlay_x: int,
    overlay_y: int,
    overlay_w: int,
) -> None:
    status = read_json(status_path)
    output = str(status.get("output") or "")
    state_text = str(status.get("state") or "idle").upper()
    panel_x = overlay_x
    panel_y = overlay_y
    panel_w = min(fb_w - panel_x, max(overlay_w, 360))
    draw_rect(buf, stride, fb_w, fb_h, panel_x, panel_y, panel_w, 98, PANEL)
    sensor_suffix = format_sensor_suffix(sensor_path)
    first_line = "HUMAN YES" if human_present else "WAITING FOR HUMAN"
    if sensor_suffix:
        first_line = f"{first_line}  {sensor_suffix}"
    first_line = f"{first_line}  {format_london_time()}"
    draw_text(buf, stride, fb_w, fb_h, panel_x + 14, panel_y + 10, truncate_text(first_line, max(20, panel_w // 12)), GREEN if human_present else YELLOW, 2)
    if coach_active:
        draw_text(buf, stride, fb_w, fb_h, panel_x + 14, panel_y + 38, f"SQUATS {squat_count}/{target} {phase}", GREEN, 2)
    else:
        draw_text(buf, stride, fb_w, fb_h, panel_x + 14, panel_y + 38, f"STATE {state_text}", WHITE, 2)
    draw_text(buf, stride, fb_w, fb_h, panel_x + 14, panel_y + 66, f"POSE {pose_fps:.1f} FPS {pose_status}", GREEN if pose_fps > 0 else RED, 2)

    if output:
        lines = wrap_text(output, max_chars=max(20, fb_w // 14), max_lines=3)
        panel_h = 20 + len(lines) * 18
        draw_rect(buf, stride, fb_w, fb_h, 0, fb_h - panel_h, fb_w, panel_h, PANEL)
        for index, line in enumerate(lines):
            draw_text(buf, stride, fb_w, fb_h, 14, fb_h - panel_h + 10 + index * 18, line, WHITE, 2)


def truncate_text(text: str, max_chars: int) -> str:
    text = text.replace("\n", " ").strip().upper()
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[:max_chars - 3] + "..."


def render_native_status_bar(
    fb_w: int,
    status_h: int,
    stride: int,
    state: SharedState,
    status_path: Path,
    sensor_path: Path,
) -> bytearray:
    status_h = max(1, status_h)
    buf = bytearray(BLACK * (stride * status_h // 2))
    draw_rect(buf, stride, fb_w, status_h, 0, 0, fb_w, status_h, PANEL)

    with state.lock:
        pose = dict(state.pose)
        squat_count = state.squat_count
        phase = state.squat_phase
        active = state.coach_active
        target = state.target_reps
        human_present = (time.time() - state.last_pose_at) < 1.5 and bool(pose.get("pose_count"))
        pose_fps = pose_fps_from_times(state.pose_success_times)
        pose_status = pose_status_text(state.pose_error, state.last_pose_at)

    voice = read_json(status_path)
    voice_state = str(voice.get("state") or "idle").upper()
    turn = voice.get("turn")
    output = str(voice.get("output") or "")
    error = str(voice.get("error") or "")
    tokens_per_second = voice.get("tokens_per_second")
    token_text = ""
    if isinstance(tokens_per_second, (int, float)) and tokens_per_second > 0:
        token_text = f" {tokens_per_second:.1f} TPS"

    max_chars = max(24, (fb_w - 24) // 12)
    human = "YES" if human_present else "NO"
    sensor_suffix = format_sensor_suffix(sensor_path)
    first_line = f"POSE {pose_fps:.1f} FPS {pose_status} HUMAN {human}"
    if sensor_suffix:
        first_line = f"{first_line}  {sensor_suffix}"
    first_line = f"{first_line}  {format_london_time()}"
    draw_text(
        buf,
        stride,
        fb_w,
        status_h,
        12,
        8,
        truncate_text(first_line, max_chars),
        GREEN if pose_status == "OK" else RED,
        2,
    )
    turn_text = f" TURN {turn}" if isinstance(turn, int) and turn > 0 else ""
    draw_text(
        buf,
        stride,
        fb_w,
        status_h,
        12,
        30,
        truncate_text(f"GEMMA {voice_state}{turn_text}{token_text}", max_chars),
        RED if voice_state == "ERROR" else WHITE,
        2,
    )
    if active:
        line = f"SQUATS {squat_count}/{target} {phase}"
        color = GREEN
    else:
        line = "CAMERA FAST PREVIEW   POSE LOW FPS"
        color = CYAN
    draw_text(buf, stride, fb_w, status_h, 12, 52, truncate_text(line, max_chars), color, 2)

    detail = error or output
    if detail and status_h >= 90:
        draw_text(buf, stride, fb_w, status_h, 12, 74, truncate_text(detail, max_chars), YELLOW if error else WHITE, 2)
    return buf


def format_sensor_suffix(sensor_path: Path) -> str:
    sensor = read_json(sensor_path)
    co2_value = sensor.get("co2_value")
    co2_raw = sensor.get("co2_raw")
    if co2_value is None and co2_raw is None:
        return ""

    parts = ["CO2"]
    if isinstance(co2_value, (int, float)):
        parts.append(str(int(co2_value)))
    if isinstance(co2_raw, (int, float)):
        parts.extend(["RAW", str(int(co2_raw))])

    updated_at_unix = sensor.get("updated_at_unix")
    if isinstance(updated_at_unix, (int, float)) and time.time() - float(updated_at_unix) > 12:
        parts.append("STALE")
    return " ".join(parts)


def format_london_time() -> str:
    if LONDON_TZ is not None:
        return datetime.now(LONDON_TZ).strftime("%H:%M:%S %d-%m-%Y")
    return time.strftime("%H:%M:%S %d-%m-%Y", time.localtime())


def camera_to_pose_rgb(frame: bytes, src_w: int, src_h: int, dst_w: int, dst_h: int) -> bytes:
    out = bytearray(dst_w * dst_h * 3)
    target = 0
    for y in range(dst_h):
        sy = min(src_h - 1, y * src_h // dst_h)
        for x in range(dst_w):
            sx = min(src_w - 1, x * src_w // dst_w)
            r, g, b = sample_yuv420_rgb(frame, src_w, src_h, sx, sy)
            out[target] = r
            out[target + 1] = g
            out[target + 2] = b
            target += 3
    return bytes(out)


def post_pose(bridge_url: str, payload: bytes, width: int, height: int, backend: str, model: str, timeout: float) -> dict[str, Any]:
    query = urllib.parse.urlencode({
        "format": "rgb24",
        "width": width,
        "height": height,
        "pose_backend": backend,
        "pose_model": model,
        "timeout": timeout,
    })
    request = urllib.request.Request(
        f"{bridge_url.rstrip('/')}/pose-frame?{query}",
        data=payload,
        headers={"Content-Type": "application/octet-stream"},
        method="POST",
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout + 2) as response:
        result = json.loads(response.read().decode("utf-8"))
    result["http_wall_seconds"] = time.monotonic() - started
    return result


def pose_worker(args: argparse.Namespace, shared: SharedState, frame: bytes) -> None:
    try:
        payload = camera_to_pose_rgb(frame, args.width, args.height, args.pose_width, args.pose_height)
        result = post_pose(args.bridge_url, payload, args.pose_width, args.pose_height, args.pose_backend, args.pose_model, args.pose_timeout)
    except Exception as exc:  # noqa: BLE001 - visible in state/log for field debugging.
        result = {"error": str(exc), "pose_count": 0, "pose_presence": 0.0}

    with shared.lock:
        now = time.time()
        shared.pose = result
        shared.last_pose_at = now
        error = str(result.get("error") or "")
        shared.pose_error = error
        if not error:
            shared.pose_success_times.append(now)
            cutoff = now - 6.0
            shared.pose_success_times = [timestamp for timestamp in shared.pose_success_times if timestamp >= cutoff]
        update_squat_count(shared, result)


def pose_fps_from_times(timestamps: list[float]) -> float:
    if len(timestamps) < 2:
        return 0.0
    span = timestamps[-1] - timestamps[0]
    if span <= 0:
        return 0.0
    return (len(timestamps) - 1) / span


def pose_status_text(error: str, last_pose_at: float) -> str:
    if time.time() - last_pose_at > 5.0:
        return "NO DATA"
    if not error:
        return "OK"
    upper = error.upper()
    if "500" in upper or "NO IPHONE" in upper or "WORKER" in upper:
        return "PHONE OFF"
    if "TIME" in upper:
        return "TIMEOUT"
    return "ERROR"


def landmark_angle(landmarks: list[Any], a: int, b: int, c: int) -> float | None:
    try:
        pa = landmarks[a]
        pb = landmarks[b]
        pc = landmarks[c]
        visibility = min(
            float(pa.get("visibility") or pa.get("presence") or 0.0),
            float(pb.get("visibility") or pb.get("presence") or 0.0),
            float(pc.get("visibility") or pc.get("presence") or 0.0),
        )
        if visibility < 0.25:
            return None
        ax, ay = float(pa["x"]), float(pa["y"])
        bx, by = float(pb["x"]), float(pb["y"])
        cx, cy = float(pc["x"]), float(pc["y"])
    except (IndexError, TypeError, ValueError, KeyError, AttributeError):
        return None
    v1 = (ax - bx, ay - by)
    v2 = (cx - bx, cy - by)
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.hypot(*v1)
    mag2 = math.hypot(*v2)
    if mag1 == 0 or mag2 == 0:
        return None
    cosine = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cosine))


def update_squat_count(shared: SharedState, pose: dict[str, Any]) -> None:
    if not shared.coach_active or shared.coach_done:
        return
    poses = pose.get("pose_landmarks")
    if not isinstance(poses, list) or not poses:
        return
    landmarks = poses[0]
    if not isinstance(landmarks, list):
        return
    angles = [
        angle for angle in (
            landmark_angle(landmarks, 23, 25, 27),
            landmark_angle(landmarks, 24, 26, 28),
        )
        if angle is not None
    ]
    if not angles:
        return
    knee_angle = sum(angles) / len(angles)
    if shared.squat_phase == "up" and knee_angle < 115:
        shared.squat_phase = "down"
    elif shared.squat_phase == "down" and knee_angle > 155:
        shared.squat_phase = "up"
        shared.squat_count += 1
        if shared.squat_count >= shared.target_reps:
            shared.coach_done = True
            shared.coach_active = False


def apply_command(shared: SharedState, command_path: Path) -> None:
    command = read_json(command_path)
    if command.get("command") != "coach_squats":
        return
    session_id = str(command.get("session_id") or "")
    with shared.lock:
        if session_id and session_id != shared.session_id:
            shared.session_id = session_id
            shared.squat_count = 0
            shared.squat_phase = "up"
            shared.coach_done = False
        shared.coach_active = True
        shared.target_reps = max(1, int(command.get("target_reps") or 10))
        milestones = command.get("milestones")
        if isinstance(milestones, list):
            parsed: list[int] = []
            for item in milestones:
                try:
                    value = int(item)
                except (TypeError, ValueError):
                    continue
                if value > 0:
                    parsed.append(value)
            if parsed:
                shared.milestones = sorted(set(parsed))


def write_state(shared: SharedState, path: Path) -> None:
    with shared.lock:
        pose = dict(shared.pose)
        age = time.time() - shared.last_pose_at if shared.last_pose_at else None
        human_present = bool(pose.get("pose_count")) and age is not None and age <= 1.5
        pose_fps = pose_fps_from_times(shared.pose_success_times)
        pose_status = pose_status_text(shared.pose_error, shared.last_pose_at)
        payload = {
            "updated_at_unix": time.time(),
            "human_present": human_present,
            "pose_presence": pose.get("pose_presence"),
            "pose_age_seconds": age,
            "pose_fps": pose_fps,
            "pose_status": pose_status,
            "pose_error": shared.pose_error,
            "coach_active": shared.coach_active,
            "coach_done": shared.coach_done,
            "squat_count": shared.squat_count,
            "squat_phase": shared.squat_phase,
            "target_reps": shared.target_reps,
            "milestones": shared.milestones,
            "session_id": shared.session_id,
            "last_pose": pose,
        }
    write_json_atomic(path, payload)


def start_camera_stream(args: argparse.Namespace) -> subprocess.Popen[bytes]:
    command = [
        "rpicam-vid",
        "--camera", str(args.camera_index),
        "-t", "0",
        "--width", str(args.width),
        "--height", str(args.height),
        "--framerate", str(args.framerate),
        "--codec", "yuv420",
        "--flush",
        "-o", "-",
    ]
    if args.native_preview:
        preview_y = max(0, args.status_height)
        command[1:1] = [
            "--preview", f"0,{preview_y},{args.preview_width},{args.preview_height}",
            "--info-text", "",
            "--viewfinder-width", str(args.preview_width),
            "--viewfinder-height", str(args.preview_height),
        ]
    else:
        command.insert(1, "-n")
    return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=sys.stderr)


def frame_size_for_camera(args: argparse.Namespace) -> int:
    return args.width * args.height * 3 // 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--fb", default="/dev/fb0")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--framerate", type=int, default=10)
    parser.add_argument("--display-fps", type=float, default=4.0)
    parser.add_argument("--display-width", type=int, default=320)
    parser.add_argument("--display-height", type=int, default=240)
    parser.add_argument("--upscale", action="store_true", help="Scale camera to fill the display height instead of drawing native size.")
    parser.add_argument("--native-preview", action="store_true", help="Let rpicam render a high-FPS fullscreen preview while Python samples frames for pose.")
    parser.add_argument("--preview-width", type=int, default=1024)
    parser.add_argument("--preview-height", type=int, default=576)
    parser.add_argument("--status-height", type=int, default=96)
    parser.add_argument("--status-fps", type=float, default=2.0)
    parser.add_argument("--pose-width", type=int, default=192)
    parser.add_argument("--pose-height", type=int, default=144)
    parser.add_argument("--pose-fps", type=float, default=5.0)
    parser.add_argument("--pose-backend", default="gpu")
    parser.add_argument("--pose-model", default="lite")
    parser.add_argument("--pose-timeout", type=float, default=10.0)
    parser.add_argument("--bridge-url", default=os.environ.get("GEMMA_IOS_BRIDGE_URL", "http://127.0.0.1:8765"))
    parser.add_argument("--status-file", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--sensor-file", type=Path, default=DEFAULT_SENSOR_STATE)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_VISION_STATE)
    parser.add_argument("--command-file", type=Path, default=DEFAULT_VISION_COMMAND)
    args = parser.parse_args()

    frame_size = frame_size_for_camera(args)
    shared = SharedState()
    next_pose_at = 0.0
    next_display_at = 0.0
    next_status_at = 0.0
    next_state_at = 0.0
    pose_busy = False

    fb_w, fb_h, stride, _depth = parse_fbset()
    clear_framebuffer(args.fb, fb_w, fb_h, stride)

    proc = start_camera_stream(args)
    if proc.stdout is None:
        raise RuntimeError("camera stream has no stdout")

    try:
        if args.native_preview:
            status_h = max(0, min(args.status_height, fb_h))
            with open(args.fb, "r+b", buffering=0) as fb:
                while True:
                    frame = read_exact(proc.stdout, frame_size)
                    if frame is None:
                        break
                    apply_command(shared, args.command_file)
                    now = time.monotonic()
                    if now >= next_pose_at and not pose_busy:
                        pose_busy = True

                        def run_pose(payload: bytes = bytes(frame)) -> None:
                            nonlocal pose_busy
                            try:
                                pose_worker(args, shared, payload)
                            finally:
                                pose_busy = False

                        threading.Thread(target=run_pose, daemon=True).start()
                        next_pose_at = now + 1.0 / max(0.1, args.pose_fps)
                    if status_h > 0 and now >= next_status_at:
                        status_bar = render_native_status_bar(fb_w, status_h, stride, shared, args.status_file, args.sensor_file)
                        fb.seek(0)
                        fb.write(status_bar)
                        next_status_at = now + 1.0 / max(0.1, args.status_fps)
                    if now >= next_state_at:
                        write_state(shared, args.state_file)
                        next_state_at = now + 0.25
            return

        with open(args.fb, "r+b", buffering=0) as fb:
            while True:
                frame = read_exact(proc.stdout, frame_size)
                if frame is None:
                    break
                apply_command(shared, args.command_file)
                now = time.monotonic()
                if now >= next_pose_at and not pose_busy:
                    pose_busy = True

                    def run_pose(payload: bytes = bytes(frame)) -> None:
                        nonlocal pose_busy
                        try:
                            pose_worker(args, shared, payload)
                        finally:
                            pose_busy = False

                    threading.Thread(target=run_pose, daemon=True).start()
                    next_pose_at = now + 1.0 / max(0.1, args.pose_fps)

                if now >= next_display_at:
                    image = render_camera_frame(
                        frame,
                        args.width,
                        args.height,
                        fb_w,
                        fb_h,
                        stride,
                        shared,
                        args.status_file,
                        args.sensor_file,
                        args.upscale,
                        args.display_width,
                        args.display_height,
                    )
                    fb.seek(0)
                    fb.write(image)
                    next_display_at = now + 1.0 / max(0.1, args.display_fps)
                if now >= next_state_at:
                    write_state(shared, args.state_file)
                    next_state_at = now + 0.25
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
