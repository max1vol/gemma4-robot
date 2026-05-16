#!/usr/bin/env python3
"""Dump official MediaPipe Pose Landmarker outputs useful for validation."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mediapipe as mp
import numpy as np
from PIL import Image
from mediapipe.framework.formats import detection_pb2, landmark_pb2, rect_pb2
from mediapipe.python import packet_creator, packet_getter
from mediapipe.tasks.cc.vision.pose_detector.proto import (
    pose_detector_graph_options_pb2,
)
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.core import task_info as task_info_module
from mediapipe.tasks.python.vision import pose_landmarker
from mediapipe.tasks.python.vision.core import base_vision_task_api
from mediapipe.tasks.python.vision.core import (
    vision_task_running_mode as running_mode_module,
)


def landmark_dict(lm) -> dict[str, float]:
    return {
        "x": float(lm.x),
        "y": float(lm.y),
        "z": float(lm.z),
        "visibility": float(lm.visibility),
        "presence": float(lm.presence),
    }


def world_landmark_dict(lm) -> dict[str, float]:
    return {
        "x": float(lm.x),
        "y": float(lm.y),
        "z": float(lm.z),
        "visibility": float(lm.visibility),
        "presence": float(lm.presence),
    }


def rect_dict(rect: rect_pb2.NormalizedRect) -> dict[str, float]:
    return {
        "x_center": float(rect.x_center),
        "y_center": float(rect.y_center),
        "width": float(rect.width),
        "height": float(rect.height),
        "rotation": float(rect.rotation),
    }


def detection_dict(det: detection_pb2.Detection) -> dict[str, Any]:
    loc = det.location_data
    box = loc.relative_bounding_box
    return {
        "label_id": [int(x) for x in det.label_id],
        "score": [float(x) for x in det.score],
        "relative_bounding_box": {
            "xmin": float(box.xmin),
            "ymin": float(box.ymin),
            "width": float(box.width),
            "height": float(box.height),
        },
        "relative_keypoints": [
            {"x": float(kp.x), "y": float(kp.y)} for kp in loc.relative_keypoints
        ],
    }


def proto_list(packet, proto_cls):
    out = []
    for item in packet_getter.get_proto_list(packet):
        proto = proto_cls()
        proto.MergeFrom(item)
        out.append(proto)
    return out


@dataclass
class DetectorOptions:
    detector_model: Path
    num_poses: int

    def to_pb2(self):
        options = pose_detector_graph_options_pb2.PoseDetectorGraphOptions()
        options.base_options.CopyFrom(
            base_options_module.BaseOptions(
                model_asset_path=str(self.detector_model)
            ).to_pb2()
        )
        options.min_detection_confidence = 0.0
        options.num_poses = self.num_poses
        return options


def run_detector_graph(detector_model: Path, image, norm_rect) -> dict[str, Any]:
    streams = [
        "DETECTIONS:detections",
        "POSE_RECTS:pose_rects",
        "EXPANDED_POSE_RECTS:expanded_pose_rects",
        "IMAGE:image_out",
    ]
    task_info = task_info_module.TaskInfo(
        task_graph="mediapipe.tasks.vision.pose_detector.PoseDetectorGraph",
        input_streams=["IMAGE:image_in", "NORM_RECT:norm_rect_in"],
        output_streams=streams,
        task_options=DetectorOptions(detector_model, num_poses=1),
    )
    api = base_vision_task_api.BaseVisionTaskApi(
        task_info.generate_graph_config(False),
        running_mode_module.VisionTaskRunningMode.IMAGE,
    )
    try:
        packets = api._process_image_data(
            {
                "image_in": packet_creator.create_image(image),
                "norm_rect_in": packet_creator.create_proto(norm_rect.to_pb2()),
            }
        )
        detections = proto_list(packets["detections"], detection_pb2.Detection)
        pose_rects = proto_list(packets["pose_rects"], rect_pb2.NormalizedRect)
        expanded_rects = proto_list(
            packets["expanded_pose_rects"], rect_pb2.NormalizedRect
        )
        return {
            "detections": [detection_dict(d) for d in detections],
            "pose_rects": [rect_dict(r) for r in pose_rects],
            "expanded_pose_rects": [rect_dict(r) for r in expanded_rects],
        }
    finally:
        api.close()


def run_landmarker_graph(task_model: Path, image, norm_rect) -> dict[str, Any]:
    options = pose_landmarker.PoseLandmarkerOptions(
        base_options=base_options_module.BaseOptions(
            model_asset_path=str(task_model)
        ),
        running_mode=running_mode_module.VisionTaskRunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.0,
        min_pose_presence_confidence=0.0,
        min_tracking_confidence=0.0,
        output_segmentation_masks=False,
    )
    streams = [
        "NORM_LANDMARKS:norm_landmarks",
        "WORLD_LANDMARKS:world_landmarks",
        "AUXILIARY_LANDMARKS:aux_landmarks",
        "POSE_RECTS_NEXT_FRAME:pose_rects_next_frame",
        "DETECTIONS:detections",
        "IMAGE:image_out",
    ]
    task_info = pose_landmarker._TaskInfo(
        task_graph=pose_landmarker._TASK_GRAPH_NAME,
        input_streams=["IMAGE:image_in", "NORM_RECT:norm_rect_in"],
        output_streams=streams,
        task_options=options,
    )
    api = base_vision_task_api.BaseVisionTaskApi(
        task_info.generate_graph_config(False),
        running_mode_module.VisionTaskRunningMode.IMAGE,
    )
    try:
        packets = api._process_image_data(
            {
                "image_in": packet_creator.create_image(image),
                "norm_rect_in": packet_creator.create_proto(norm_rect.to_pb2()),
            }
        )
        pose_landmarks = proto_list(
            packets["norm_landmarks"], landmark_pb2.NormalizedLandmarkList
        )
        world_landmarks = proto_list(
            packets["world_landmarks"], landmark_pb2.LandmarkList
        )
        aux_landmarks = proto_list(
            packets["aux_landmarks"], landmark_pb2.NormalizedLandmarkList
        )
        next_rects = proto_list(
            packets["pose_rects_next_frame"], rect_pb2.NormalizedRect
        )
        detections = proto_list(packets["detections"], detection_pb2.Detection)
        return {
            "pose_landmarks": [
                [landmark_dict(lm) for lm in pose.landmark]
                for pose in pose_landmarks
            ],
            "pose_world_landmarks": [
                [world_landmark_dict(lm) for lm in pose.landmark]
                for pose in world_landmarks
            ],
            "auxiliary_landmarks": [
                [landmark_dict(lm) for lm in pose.landmark]
                for pose in aux_landmarks
            ],
            "pose_rects_next_frame": [rect_dict(r) for r in next_rects],
            "detections": [detection_dict(d) for d in detections],
        }
    finally:
        api.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=Path)
    parser.add_argument("detector", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    rgb = np.asarray(Image.open(args.image).convert("RGB"))
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    dummy_api = object.__new__(base_vision_task_api.BaseVisionTaskApi)
    norm_rect = dummy_api.convert_to_normalized_rect(None, image, roi_allowed=False)

    payload = {
        "task": str(args.task),
        "detector": str(args.detector),
        "image": str(args.image),
        "image_shape": list(rgb.shape),
        "input_norm_rect": rect_dict(norm_rect.to_pb2()),
        "pose_detector_graph": run_detector_graph(args.detector, image, norm_rect),
        "pose_landmarker_graph": run_landmarker_graph(args.task, image, norm_rect),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(
        f"wrote {args.output} "
        f"detections={len(payload['pose_landmarker_graph']['detections'])} "
        f"landmarks={len(payload['pose_landmarker_graph']['pose_landmarks'])}"
    )


if __name__ == "__main__":
    main()
