#!/usr/bin/env python3
"""Run official MediaPipe Pose Landmarker on one image and save JSON output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mediapipe as mp
import numpy as np
from PIL import Image


def landmark_to_dict(lm) -> dict[str, float | str]:
    out: dict[str, float | str] = {
        "x": float(lm.x),
        "y": float(lm.y),
        "z": float(lm.z),
        "visibility": float(lm.visibility),
        "presence": float(lm.presence),
    }
    name = getattr(lm, "name", None)
    if name:
        out["name"] = str(name)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    rgb = np.asarray(Image.open(args.image).convert("RGB"))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    base_options = mp.tasks.BaseOptions(model_asset_path=str(args.task))
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.0,
        min_pose_presence_confidence=0.0,
        min_tracking_confidence=0.0,
        output_segmentation_masks=False,
    )
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    payload = {
        "task": str(args.task),
        "image": str(args.image),
        "image_shape": list(rgb.shape),
        "pose_count": len(result.pose_landmarks),
        "pose_landmarks": [
            [landmark_to_dict(lm) for lm in pose] for pose in result.pose_landmarks
        ],
        "pose_world_landmarks": [
            [landmark_to_dict(lm) for lm in pose] for pose in result.pose_world_landmarks
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(
        f"wrote {args.output} poses={payload['pose_count']} "
        f"shape={payload['image_shape']}"
    )


if __name__ == "__main__":
    main()
