#!/usr/bin/env python3
"""End-to-end image pipeline around the low-level pose model executor.

This script is a validation harness: model inference is done by the
`pose_estimation/` C runtime binary passed on the command line, while Python
keeps the image IO and MediaPipe-compatible postprocessing readable until it is
ported into the target runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -100.0, 100.0)))


def resize_letterbox(rgb: np.ndarray, size: int) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    h, w = rgb.shape[:2]
    scale = size / max(h, w)
    resized_w = int(round(w * scale))
    resized_h = int(round(h * scale))
    resized = Image.fromarray(rgb).resize((resized_w, resized_h), Image.Resampling.BILINEAR)
    out = np.zeros((size, size, 3), dtype=np.float32)
    left = (size - resized_w) // 2
    top = (size - resized_h) // 2
    out[top : top + resized_h, left : left + resized_w] = (
        np.asarray(resized, dtype=np.float32) / 127.5 - 1.0
    )
    padding = (
        left / size,
        top / size,
        (size - left - resized_w) / size,
        (size - top - resized_h) / size,
    )
    return out, padding


def generate_pose_anchors() -> np.ndarray:
    strides = [8, 16, 32, 32, 32]
    min_scale = 0.1484375
    max_scale = 0.75
    input_size = 224
    anchors: list[tuple[float, float, float, float]] = []
    layer = 0
    while layer < len(strides):
        same_stride = layer
        widths: list[float] = []
        heights: list[float] = []
        while same_stride < len(strides) and strides[same_stride] == strides[layer]:
            scale = min_scale + (max_scale - min_scale) * same_stride / (len(strides) - 1)
            next_scale = (
                1.0
                if same_stride == len(strides) - 1
                else min_scale + (max_scale - min_scale) * (same_stride + 1) / (len(strides) - 1)
            )
            widths.append(scale)
            heights.append(scale)
            interpolated = math.sqrt(scale * next_scale)
            widths.append(interpolated)
            heights.append(interpolated)
            same_stride += 1
        feature = math.ceil(input_size / strides[layer])
        for y in range(feature):
            for x in range(feature):
                x_center = (x + 0.5) / feature
                y_center = (y + 0.5) / feature
                for _w, _h in zip(widths, heights):
                    # MediaPipe pose detector uses fixed anchor size.
                    anchors.append((x_center, y_center, 1.0, 1.0))
        layer = same_stride
    out = np.asarray(anchors, dtype=np.float32)
    if out.shape != (2254, 4):
        raise RuntimeError(f"unexpected anchor shape {out.shape}")
    return out


def decode_detections(
    raw_boxes: np.ndarray,
    raw_scores: np.ndarray,
    letterbox: tuple[float, float, float, float],
) -> list[dict[str, Any]]:
    anchors = generate_pose_anchors()
    scores = sigmoid(raw_scores.reshape(-1))
    boxes = raw_boxes.reshape(-1, 12)
    left, top, right, bottom = letterbox
    x_scale = 1.0 - left - right
    y_scale = 1.0 - top - bottom
    detections = []
    for i in range(boxes.shape[0]):
        if scores[i] < 0.5:
            continue
        raw = boxes[i]
        ax, ay, aw, ah = anchors[i]
        x_center = raw[0] / 224.0 * aw + ax
        y_center = raw[1] / 224.0 * ah + ay
        w = raw[2] / 224.0 * aw
        h = raw[3] / 224.0 * ah
        keypoints = []
        for k in range(4):
            kx = raw[4 + 2 * k] / 224.0 * aw + ax
            ky = raw[5 + 2 * k] / 224.0 * ah + ay
            keypoints.append({"x": float((kx - left) / x_scale), "y": float((ky - top) / y_scale)})
        detections.append(
            {
                "score": float(scores[i]),
                "box": {
                    "xmin": float((x_center - w * 0.5 - left) / x_scale),
                    "ymin": float((y_center - h * 0.5 - top) / y_scale),
                    "width": float(w / x_scale),
                    "height": float(h / y_scale),
                },
                "keypoints": keypoints,
            }
        )
    detections.sort(key=lambda d: d["score"], reverse=True)
    return detections


def rect_from_detection(det: dict[str, Any], image_w: int, image_h: int) -> dict[str, float]:
    kp0 = det["keypoints"][0]
    kp1 = det["keypoints"][1]
    x0 = kp0["x"] * image_w
    y0 = kp0["y"] * image_h
    x1 = kp1["x"] * image_w
    y1 = kp1["y"] * image_h
    box_size = math.hypot(x1 - x0, y1 - y0) * 2.0
    rotation = math.pi * 0.5 - math.atan2(-(y1 - y0), x1 - x0)
    while rotation > math.pi:
        rotation -= 2.0 * math.pi
    while rotation <= -math.pi:
        rotation += 2.0 * math.pi
    width = box_size / image_w
    height = box_size / image_h
    long_side = max(width * image_w, height * image_h)
    return {
        "x_center": float(kp0["x"]),
        "y_center": float(kp0["y"]),
        "width": float((long_side / image_w) * 1.25),
        "height": float((long_side / image_h) * 1.25),
        "rotation": float(rotation),
    }


def sample_rotated_rect(rgb: np.ndarray, rect: dict[str, float], size: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    out = np.zeros((size, size, 3), dtype=np.float32)
    c = math.cos(rect["rotation"])
    s = math.sin(rect["rotation"])
    for oy in range(size):
        v = (oy + 0.5) / size - 0.5
        for ox in range(size):
            u = (ox + 0.5) / size - 0.5
            x_norm = rect["x_center"] + (c * u - s * v) * rect["width"]
            y_norm = rect["y_center"] + (s * u + c * v) * rect["height"]
            x = x_norm * w - 0.5
            y = y_norm * h - 0.5
            x0 = math.floor(x)
            y0 = math.floor(y)
            wx = x - x0
            wy = y - y0
            if x0 < 0 or y0 < 0 or x0 + 1 >= w or y0 + 1 >= h:
                continue
            p00 = rgb[y0, x0].astype(np.float32)
            p01 = rgb[y0, x0 + 1].astype(np.float32)
            p10 = rgb[y0 + 1, x0].astype(np.float32)
            p11 = rgb[y0 + 1, x0 + 1].astype(np.float32)
            top = p00 + (p01 - p00) * wx
            bot = p10 + (p11 - p10) * wx
            out[oy, ox] = (top + (bot - top) * wy) / 127.5 - 1.0
    return out


def run_model(runtime: Path, data_dir: Path, model: str, input_tensor: np.ndarray, work_dir: Path, threads: int) -> dict[int, np.ndarray]:
    input_path = work_dir / f"{model}_input.bin"
    input_tensor.astype("<f4").tofile(input_path)
    out_dir = work_dir / model
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(runtime),
            str(data_dir),
            model,
            str(input_path),
            str(out_dir),
            str(threads),
            "1",
        ],
        check=True,
    )
    if model == "detector":
        return {
            441: np.fromfile(out_dir / "pose_detector_tensor_441.bin", dtype="<f4").reshape(1, 2254, 12),
            429: np.fromfile(out_dir / "pose_detector_tensor_429.bin", dtype="<f4").reshape(1, 2254, 1),
        }
    outputs = {
        310: np.fromfile(out_dir / "pose_landmarker_tensor_310.bin", dtype="<f4").reshape(1, 195),
        315: np.fromfile(out_dir / "pose_landmarker_tensor_315.bin", dtype="<f4").reshape(1, 1),
        283: np.fromfile(out_dir / "pose_landmarker_tensor_283.bin", dtype="<f4").reshape(1, 64, 64, 39),
        312: np.fromfile(out_dir / "pose_landmarker_tensor_312.bin", dtype="<f4").reshape(1, 117),
    }
    seg = out_dir / "pose_landmarker_tensor_282.bin"
    if seg.exists():
        outputs[282] = np.fromfile(seg, dtype="<f4").reshape(1, 256, 256, 1)
    return outputs


def refine_landmarks(raw: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    landmarks = raw.reshape(39, 5).astype(np.float32).copy()
    landmarks[:, 0] /= 256.0
    landmarks[:, 1] /= 256.0
    landmarks[:, 2] /= 256.0
    landmarks[:, 3] = sigmoid(landmarks[:, 3])
    landmarks[:, 4] = sigmoid(landmarks[:, 4])
    hm = heatmap.reshape(64, 64, 39)
    out = landmarks.copy()
    for lm in range(39):
        center_col = int(out[lm, 0] * 64)
        center_row = int(out[lm, 1] * 64)
        if center_col < 0 or center_col >= 64 or center_row < 0 or center_row >= 64:
            continue
        begin_col = max(0, center_col - 3)
        end_col = min(64, center_col + 4)
        begin_row = max(0, center_row - 3)
        end_row = min(64, center_row + 4)
        weights = sigmoid(hm[begin_row:end_row, begin_col:end_col, lm])
        total = float(weights.sum())
        if total <= 0.0:
            continue
        rows = np.arange(begin_row, end_row, dtype=np.float32)[:, None]
        cols = np.arange(begin_col, end_col, dtype=np.float32)[None, :]
        out[lm, 0] = float((weights * cols).sum() / 64.0 / total)
        out[lm, 1] = float((weights * rows).sum() / 64.0 / total)
    return out


def project_landmarks(landmarks: np.ndarray, rect: dict[str, float]) -> list[dict[str, float]]:
    c = math.cos(rect["rotation"])
    s = math.sin(rect["rotation"])
    out = []
    for lm in landmarks[:33]:
        x = lm[0] - 0.5
        y = lm[1] - 0.5
        new_x = (c * x - s * y) * rect["width"] + rect["x_center"]
        new_y = (s * x + c * y) * rect["height"] + rect["y_center"]
        out.append(
            {
                "x": float(new_x),
                "y": float(new_y),
                "z": float(lm[2] * rect["width"]),
                "visibility": float(lm[3]),
                "presence": float(lm[4]),
            }
        )
    return out


def decode_world(world: np.ndarray, landmarks: np.ndarray, rect: dict[str, float]) -> list[dict[str, float]]:
    raw = world.reshape(39, 3)
    c = math.cos(rect["rotation"])
    s = math.sin(rect["rotation"])
    out = []
    for i in range(33):
        x = float(c * raw[i, 0] - s * raw[i, 1])
        y = float(s * raw[i, 0] + c * raw[i, 1])
        out.append(
            {
                "x": x,
                "y": y,
                "z": float(raw[i, 2]),
                "visibility": float(landmarks[i, 3]),
                "presence": float(landmarks[i, 4]),
            }
        )
    return out


def compare_landmarks(result: list[dict[str, float]], reference: list[dict[str, float]]) -> dict[str, float]:
    dx = []
    dy = []
    dz = []
    for a, b in zip(result, reference):
        dx.append(abs(a["x"] - b["x"]))
        dy.append(abs(a["y"] - b["y"]))
        dz.append(abs(a["z"] - b["z"]))
    return {
        "max_abs_x": float(max(dx)),
        "max_abs_y": float(max(dy)),
        "max_abs_z": float(max(dz)),
        "mean_abs_x": float(np.mean(dx)),
        "mean_abs_y": float(np.mean(dy)),
        "mean_abs_z": float(np.mean(dz)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runtime", type=Path)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--official-json", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--work-dir", type=Path, default=Path("out/pose_runtime_pipeline"))
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(Image.open(args.image).convert("RGB"))
    h, w = rgb.shape[:2]
    detector_input, letterbox = resize_letterbox(rgb, 224)
    detector_out = run_model(args.runtime, args.data_dir, "detector", detector_input[None], args.work_dir, args.threads)
    detections = decode_detections(detector_out[441][0], detector_out[429][0], letterbox)
    if not detections:
        payload = {"pose_count": 0, "detections": []}
    else:
        rect = rect_from_detection(detections[0], w, h)
        landmarker_input = sample_rotated_rect(rgb, rect, 256)
        landmarker_out = run_model(args.runtime, args.data_dir, "landmarker", landmarker_input[None], args.work_dir, args.threads)
        internal_landmarks = refine_landmarks(landmarker_out[310][0], landmarker_out[283][0])
        pose_landmarks = project_landmarks(internal_landmarks, rect)
        world_landmarks = decode_world(landmarker_out[312][0], internal_landmarks, rect)
        payload = {
            "pose_count": 1,
            "detections": detections[:5],
            "selected_rect": rect,
            "pose_presence": float(landmarker_out[315][0, 0]),
            "pose_landmarks": [pose_landmarks],
            "pose_world_landmarks": [world_landmarks],
        }
        if args.official_json:
            official = json.loads(args.official_json.read_text())
            if official.get("pose_landmarks"):
                payload["official_compare"] = compare_landmarks(
                    pose_landmarks, official["pose_landmarks"][0]
                )
                payload["official_world_compare"] = compare_landmarks(
                    world_landmarks, official["pose_world_landmarks"][0]
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.output} pose_count={payload['pose_count']}")
    if "official_compare" in payload:
        print(json.dumps(payload["official_compare"], indent=2))


if __name__ == "__main__":
    main()
