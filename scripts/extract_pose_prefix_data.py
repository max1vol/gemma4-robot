#!/usr/bin/env python3
"""Extract reproducible Pose Landmarker Lite prefix tensors for low-level probes.

The generated files are intentionally flat little-endian blobs so C probes on
the Pi can run without TensorFlow Lite, MediaPipe, NumPy, or model parsers.
"""

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


def relu6(x: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, 0.0), 6.0).astype(np.float32)


def conv3x3_stride2_same(input_image: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    padded = np.pad(input_image, ((1, 1), (1, 1), (0, 0)), mode="constant")
    output = np.empty((128, 128, 24), dtype=np.float32)
    for oy in range(128):
        iy = oy * 2
        for ox in range(128):
            ix = ox * 2
            patch = padded[iy : iy + 3, ix : ix + 3, :]
            output[oy, ox, :] = np.tensordot(weights, patch, axes=([1, 2, 3], [0, 1, 2])) + bias
    return relu6(output)


def depthwise_same(input_image: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    # TFLite depthwise filter layout is [1, H, W, input_channels] for multiplier=1.
    padded = np.pad(input_image, ((1, 1), (1, 1), (0, 0)), mode="constant")
    output = np.broadcast_to(bias, input_image.shape).astype(np.float32).copy()
    for ky in range(3):
        for kx in range(3):
            output += padded[ky : ky + input_image.shape[0], kx : kx + input_image.shape[1], :] * weights[0, ky, kx, :]
    return relu6(output)


def depthwise_stride2_same(input_image: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    # TFLite graph pads by 1 then runs a VALID stride-2 depthwise convolution.
    padded = np.pad(input_image, ((1, 1), (1, 1), (0, 0)), mode="constant")
    out_h = input_image.shape[0] // 2
    out_w = input_image.shape[1] // 2
    output = np.broadcast_to(bias, (out_h, out_w, input_image.shape[2])).astype(np.float32).copy()
    for oy in range(out_h):
        iy = oy * 2
        for ox in range(out_w):
            ix = ox * 2
            patch = padded[iy : iy + 3, ix : ix + 3, :]
            output[oy, ox, :] += np.sum(patch * weights[0, :, :, :], axis=(0, 1))
    return relu6(output)


def pointwise(input_image: np.ndarray, weights: np.ndarray, bias: np.ndarray, activation: str) -> np.ndarray:
    # TFLite conv filter layout is [output_channels, 1, 1, input_channels].
    output = np.tensordot(input_image, weights[:, 0, 0, :], axes=([2], [1])) + bias
    output = output.astype(np.float32)
    if activation == "relu6":
        output = relu6(output)
    return output


def add_tensors(a: np.ndarray, b: np.ndarray, activation: str) -> np.ndarray:
    output = (a + b).astype(np.float32)
    if activation == "relu6":
        output = relu6(output)
    return output


def q_relu6(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = np.rint(np.clip(x, 0.0, 6.0) * (255.0 / 6.0)).astype(np.uint8)
    return q, q.astype(np.float32) * (6.0 / 255.0)


def affine_u8(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    xmin = float(x.min())
    xmax = float(x.max())
    scale = (xmax - xmin) / 255.0 if xmax > xmin else 1.0
    q = np.rint((x - xmin) / scale).clip(0, 255).astype(np.uint8)
    dq = q.astype(np.float32) * scale + xmin
    return q, dq, {"min": xmin, "max": xmax, "scale": scale, "zero": xmin}


def stats(x: np.ndarray) -> dict[str, float | list[int]]:
    return {
        "shape": list(x.shape),
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "std": float(x.std()),
    }


def write_f32(path: Path, x: np.ndarray) -> None:
    x.astype("<f4").tofile(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("landmarker_tflite", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    model = Model.GetRootAsModel(args.landmarker_tflite.read_bytes(), 0)
    subgraph = model.Subgraphs(0)
    rng = np.random.default_rng(args.seed)
    input_image = rng.random((256, 256, 3), dtype=np.float32)

    op3_w = tensor_array(model, subgraph, 67)
    op3_b = tensor_array(model, subgraph, 11)
    op6_w = tensor_array(model, subgraph, 121)
    op6_b = tensor_array(model, subgraph, 12)
    op9_w = tensor_array(model, subgraph, 68)
    op9_b = tensor_array(model, subgraph, 122)
    op12_w = tensor_array(model, subgraph, 69)
    op12_b = tensor_array(model, subgraph, 26)
    op15_w = tensor_array(model, subgraph, 123)
    op15_b = tensor_array(model, subgraph, 59)
    op18_w = tensor_array(model, subgraph, 70)
    op18_b = tensor_array(model, subgraph, 124)
    op23_w = tensor_array(model, subgraph, 125)
    op23_b = tensor_array(model, subgraph, 34)
    op26_w = tensor_array(model, subgraph, 71)
    op26_b = tensor_array(model, subgraph, 126)

    op3 = conv3x3_stride2_same(input_image, op3_w, op3_b)
    op3_u8, op3_dq = q_relu6(op3)
    op6 = depthwise_same(op3, op6_w, op6_b)
    op6_from_qop3 = depthwise_same(op3_dq, op6_w, op6_b)
    op6_u8, op6_dq = q_relu6(op6_from_qop3)
    op9 = pointwise(op6, op9_w, op9_b, activation="none")
    op9_from_qop6 = pointwise(op6_dq, op9_w, op9_b, activation="none")
    op9_u8, op9_dq, op9_quant = affine_u8(op9_from_qop6)
    op12 = pointwise(op9, op12_w, op12_b, activation="relu6")
    op12_from_qop9 = pointwise(op9_dq, op12_w, op12_b, activation="relu6")
    op15 = depthwise_same(op9, op15_w, op15_b)
    op18 = pointwise(op15, op18_w, op18_b, activation="none")
    op19 = add_tensors(op9, op18, activation="relu6")
    op23 = depthwise_stride2_same(op12, op23_w, op23_b)
    op26 = pointwise(op23, op26_w, op26_b, activation="none")
    op19_u8, op19_dq = q_relu6(op19)
    op26_u8, op26_dq, op26_quant = affine_u8(op26)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_f32(args.out_dir / "input_f32.bin", input_image)
    write_f32(args.out_dir / "op3_w_ohwi_f32.bin", op3_w)
    write_f32(args.out_dir / "op3_b_f32.bin", op3_b)
    write_f32(args.out_dir / "op3_ref_f32.bin", op3)
    op3_u8.tofile(args.out_dir / "op3_ref_relu6_u8.bin")
    write_f32(args.out_dir / "op6_w_1hwc_f32.bin", op6_w)
    write_f32(args.out_dir / "op6_b_f32.bin", op6_b)
    write_f32(args.out_dir / "op6_ref_f32.bin", op6)
    write_f32(args.out_dir / "op6_ref_from_qop3_f32.bin", op6_from_qop3)
    op6_u8.tofile(args.out_dir / "op6_ref_from_qop3_relu6_u8.bin")
    write_f32(args.out_dir / "op9_w_ohwi_f32.bin", op9_w)
    write_f32(args.out_dir / "op9_b_f32.bin", op9_b)
    write_f32(args.out_dir / "op9_ref_f32.bin", op9)
    write_f32(args.out_dir / "op9_ref_from_qop6_f32.bin", op9_from_qop6)
    op9_u8.tofile(args.out_dir / "op9_ref_from_qop6_affine_u8.bin")
    write_f32(args.out_dir / "op12_w_ohwi_f32.bin", op12_w)
    write_f32(args.out_dir / "op12_b_f32.bin", op12_b)
    write_f32(args.out_dir / "op12_ref_f32.bin", op12)
    write_f32(args.out_dir / "op12_ref_from_qop9_f32.bin", op12_from_qop9)
    write_f32(args.out_dir / "op15_w_1hwc_f32.bin", op15_w)
    write_f32(args.out_dir / "op15_b_f32.bin", op15_b)
    write_f32(args.out_dir / "op15_ref_f32.bin", op15)
    write_f32(args.out_dir / "op18_w_ohwi_f32.bin", op18_w)
    write_f32(args.out_dir / "op18_b_f32.bin", op18_b)
    write_f32(args.out_dir / "op18_ref_f32.bin", op18)
    write_f32(args.out_dir / "op19_ref_f32.bin", op19)
    op19_u8.tofile(args.out_dir / "op19_ref_relu6_u8.bin")
    write_f32(args.out_dir / "op23_w_1hwc_f32.bin", op23_w)
    write_f32(args.out_dir / "op23_b_f32.bin", op23_b)
    write_f32(args.out_dir / "op23_ref_f32.bin", op23)
    write_f32(args.out_dir / "op26_w_ohwi_f32.bin", op26_w)
    write_f32(args.out_dir / "op26_b_f32.bin", op26_b)
    write_f32(args.out_dir / "op26_ref_f32.bin", op26)
    op26_u8.tofile(args.out_dir / "op26_ref_affine_u8.bin")

    # Compatibility names for the first-conv benchmark.
    write_f32(args.out_dir / "w_ohwi_f32.bin", op3_w)
    write_f32(args.out_dir / "b_f32.bin", op3_b)
    write_f32(args.out_dir / "ref_out_f32.bin", op3)

    meta = {
        "seed": args.seed,
        "source": str(args.landmarker_tflite),
        "ops": {
            "op3_conv": {"weights": stats(op3_w), "bias": stats(op3_b), "output": stats(op3)},
            "op6_depthwise": {"weights": stats(op6_w), "bias": stats(op6_b), "output": stats(op6)},
            "op9_pointwise": {
                "weights": stats(op9_w),
                "bias": stats(op9_b),
                "output": stats(op9),
                "from_qop6": stats(op9_from_qop6),
                "affine_u8": op9_quant,
            },
            "op12_pointwise_relu6": {
                "weights": stats(op12_w),
                "bias": stats(op12_b),
                "output": stats(op12),
                "from_qop9": stats(op12_from_qop9),
            },
            "op15_depthwise_relu6": {"weights": stats(op15_w), "bias": stats(op15_b), "output": stats(op15)},
            "op18_pointwise": {"weights": stats(op18_w), "bias": stats(op18_b), "output": stats(op18)},
            "op19_add_relu6": {"output": stats(op19)},
            "op23_depthwise_stride2_relu6": {"weights": stats(op23_w), "bias": stats(op23_b), "output": stats(op23)},
            "op26_pointwise": {
                "weights": stats(op26_w),
                "bias": stats(op26_b),
                "output": stats(op26),
                "affine_u8": op26_quant,
            },
        },
    }
    (args.out_dir / "prefix_meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
