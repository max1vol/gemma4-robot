#!/usr/bin/env python3
"""Extract the first Pose Landmarker Lite landmarker conv into flat test data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tflite import Model, TensorType


def tensor_shape(tensor) -> list[int]:
    return [tensor.Shape(i) for i in range(tensor.ShapeLength())]


def tensor_array(model, subgraph, tensor_idx: int) -> np.ndarray:
    tensor = subgraph.Tensors(tensor_idx)
    buffer = model.Buffers(tensor.Buffer())
    data = bytes(buffer.DataAsNumpy()) if buffer.DataLength() else b""
    shape = tensor_shape(tensor)
    tensor_type = tensor.Type()
    if tensor_type == TensorType.FLOAT16:
        return np.frombuffer(data, dtype=np.float16).reshape(shape).astype(np.float32)
    if tensor_type == TensorType.FLOAT32:
        return np.frombuffer(data, dtype=np.float32).reshape(shape)
    if tensor_type == TensorType.INT32:
        return np.frombuffer(data, dtype=np.int32).reshape(shape)
    raise ValueError(f"unsupported tensor {tensor_idx} type {tensor_type}")


def reference_first_conv(input_image: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    padded = np.pad(input_image, ((1, 1), (1, 1), (0, 0)), mode="constant")
    output = np.empty((128, 128, 24), dtype=np.float32)
    for oy in range(128):
        iy = oy * 2
        for ox in range(128):
            ix = ox * 2
            patch = padded[iy : iy + 3, ix : ix + 3, :]
            vals = np.tensordot(weights, patch, axes=([1, 2, 3], [0, 1, 2])) + bias
            output[oy, ox, :] = np.minimum(np.maximum(vals, 0), 6)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("landmarker_tflite", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    model = Model.GetRootAsModel(args.landmarker_tflite.read_bytes(), 0)
    subgraph = model.Subgraphs(0)

    # For pose_landmarks_detector.tflite:
    # op 3 consumes padded input tensor 187, dequantized weights tensor 355
    # from source tensor 67, and dequantized bias tensor 488 from source tensor 11.
    weights = tensor_array(model, subgraph, 67)
    bias = tensor_array(model, subgraph, 11)
    paddings = tensor_array(model, subgraph, 2)

    rng = np.random.default_rng(args.seed)
    input_image = rng.random((256, 256, 3), dtype=np.float32)
    reference = reference_first_conv(input_image, weights, bias)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    weights.astype("<f4").tofile(args.out_dir / "w_ohwi_f32.bin")
    bias.astype("<f4").tofile(args.out_dir / "b_f32.bin")
    input_image.astype("<f4").tofile(args.out_dir / "input_f32.bin")
    reference.astype("<f4").tofile(args.out_dir / "ref_out_f32.bin")
    (args.out_dir / "meta.json").write_text(
        json.dumps(
            {
                "input_shape": [1, 256, 256, 3],
                "pad_shape": [1, 258, 258, 3],
                "output_shape": [1, 128, 128, 24],
                "weights_shape": list(weights.shape),
                "bias_shape": list(bias.shape),
                "paddings": paddings.tolist(),
                "stride": [2, 2],
                "activation": "relu6",
                "seed": args.seed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
