#!/usr/bin/env python3
"""Dump TFLite graph structure and rough MAC counts for pose models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import tflite
from tflite import BuiltinOperator, BuiltinOptions, Model, TensorType


BUILTIN_NAMES = {
    value: name
    for name, value in vars(BuiltinOperator).items()
    if name.isupper() and isinstance(value, int)
}
TENSOR_TYPE_NAMES = {
    value: name
    for name, value in vars(TensorType).items()
    if name.isupper() and isinstance(value, int)
}


def tensor_shape(tensor) -> list[int]:
    return [int(tensor.Shape(i)) for i in range(tensor.ShapeLength())]


def tensor_name(tensor) -> str:
    raw = tensor.Name()
    return raw.decode("utf-8", errors="replace") if raw else ""


def tensor_info(model, subgraph, tensor_idx: int) -> dict[str, Any]:
    tensor = subgraph.Tensors(tensor_idx)
    buffer_idx = int(tensor.Buffer())
    buffer = model.Buffers(buffer_idx)
    return {
        "index": int(tensor_idx),
        "name": tensor_name(tensor),
        "shape": tensor_shape(tensor),
        "type": TENSOR_TYPE_NAMES.get(int(tensor.Type()), str(int(tensor.Type()))),
        "buffer": buffer_idx,
        "buffer_bytes": int(buffer.DataLength() if buffer else 0),
    }


def parse_options(op_name: str, op) -> dict[str, Any]:
    builtin = op.BuiltinOptions()
    if builtin is None:
        return {}

    def init(cls):
        opts = cls()
        opts.Init(builtin.Bytes, builtin.Pos)
        return opts

    if op_name == "CONV_2D":
        opts = init(tflite.Conv2DOptions)
        return {
            "padding": int(opts.Padding()),
            "stride_w": int(opts.StrideW()),
            "stride_h": int(opts.StrideH()),
            "dilation_w": int(opts.DilationWFactor()),
            "dilation_h": int(opts.DilationHFactor()),
            "activation": int(opts.FusedActivationFunction()),
        }
    if op_name == "DEPTHWISE_CONV_2D":
        opts = init(tflite.DepthwiseConv2DOptions)
        return {
            "padding": int(opts.Padding()),
            "stride_w": int(opts.StrideW()),
            "stride_h": int(opts.StrideH()),
            "dilation_w": int(opts.DilationWFactor()),
            "dilation_h": int(opts.DilationHFactor()),
            "depth_multiplier": int(opts.DepthMultiplier()),
            "activation": int(opts.FusedActivationFunction()),
        }
    if op_name in {"AVERAGE_POOL_2D", "MAX_POOL_2D"}:
        opts = init(tflite.Pool2DOptions)
        return {
            "padding": int(opts.Padding()),
            "stride_w": int(opts.StrideW()),
            "stride_h": int(opts.StrideH()),
            "filter_w": int(opts.FilterWidth()),
            "filter_h": int(opts.FilterHeight()),
            "activation": int(opts.FusedActivationFunction()),
        }
    if op_name == "ADD":
        opts = init(tflite.AddOptions)
        return {"activation": int(opts.FusedActivationFunction())}
    if op_name == "MUL":
        opts = init(tflite.MulOptions)
        return {"activation": int(opts.FusedActivationFunction())}
    if op_name == "RESIZE_BILINEAR":
        opts = init(tflite.ResizeBilinearOptions)
        return {
            "align_corners": bool(opts.AlignCorners()),
            "half_pixel_centers": bool(opts.HalfPixelCenters()),
        }
    if op_name == "PAD":
        opts = init(tflite.PadOptions)
        return {}
    if op_name == "RESHAPE":
        opts = init(tflite.ReshapeOptions)
        return {
            "new_shape": [int(opts.NewShape(i)) for i in range(opts.NewShapeLength())]
        }
    if op_name == "CONCATENATION":
        opts = init(tflite.ConcatenationOptions)
        return {
            "axis": int(opts.Axis()),
            "activation": int(opts.FusedActivationFunction()),
        }
    if op_name == "DEPTH_TO_SPACE":
        opts = init(tflite.DepthToSpaceOptions)
        return {"block_size": int(opts.BlockSize())}
    return {}


def mac_count(op_name: str, inputs: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> int:
    if not inputs or not outputs:
        return 0
    out_shape = outputs[0]["shape"]
    if len(out_shape) != 4:
        return 0
    batch, out_h, out_w, out_c = out_shape
    if batch != 1:
        return 0
    if op_name == "CONV_2D" and len(inputs) >= 2:
        w_shape = inputs[1]["shape"]
        if len(w_shape) == 4:
            return int(out_h * out_w * out_c * w_shape[1] * w_shape[2] * w_shape[3])
    if op_name == "DEPTHWISE_CONV_2D" and len(inputs) >= 2:
        w_shape = inputs[1]["shape"]
        if len(w_shape) == 4:
            return int(out_h * out_w * out_c * w_shape[1] * w_shape[2])
    if op_name in {"AVERAGE_POOL_2D", "MAX_POOL_2D"} and len(inputs) >= 1:
        in_shape = inputs[0]["shape"]
        if len(in_shape) == 4:
            # The options parser records filter size separately, but for a rough
            # report this lower-cost pool estimate is sufficient.
            return int(out_h * out_w * out_c)
    return 0


def inventory(path: Path) -> dict[str, Any]:
    model = Model.GetRootAsModel(path.read_bytes(), 0)
    subgraph = model.Subgraphs(0)
    opcodes = []
    for i in range(model.OperatorCodesLength()):
        opcode = model.OperatorCodes(i)
        code = int(opcode.BuiltinCode())
        opcodes.append(BUILTIN_NAMES.get(code, str(code)))

    tensors = [
        tensor_info(model, subgraph, i) for i in range(subgraph.TensorsLength())
    ]
    tensor_by_idx = {t["index"]: t for t in tensors}
    operators = []
    macs_total = 0
    op_counts: dict[str, int] = {}

    for i in range(subgraph.OperatorsLength()):
        op = subgraph.Operators(i)
        op_name = opcodes[op.OpcodeIndex()]
        input_ids = [int(op.Inputs(j)) for j in range(op.InputsLength())]
        output_ids = [int(op.Outputs(j)) for j in range(op.OutputsLength())]
        inputs = [tensor_by_idx[idx] for idx in input_ids if idx >= 0]
        outputs = [tensor_by_idx[idx] for idx in output_ids if idx >= 0]
        macs = mac_count(op_name, inputs, outputs)
        macs_total += macs
        op_counts[op_name] = op_counts.get(op_name, 0) + 1
        operators.append(
            {
                "index": i,
                "op": op_name,
                "inputs": input_ids,
                "input_shapes": [t["shape"] for t in inputs],
                "outputs": output_ids,
                "output_shapes": [t["shape"] for t in outputs],
                "options": parse_options(op_name, op),
                "macs": macs,
            }
        )

    return {
        "path": str(path),
        "description": model.Description().decode("utf-8", errors="replace")
        if model.Description()
        else "",
        "inputs": [
            tensor_by_idx[int(subgraph.Inputs(i))]
            for i in range(subgraph.InputsLength())
        ],
        "outputs": [
            tensor_by_idx[int(subgraph.Outputs(i))]
            for i in range(subgraph.OutputsLength())
        ],
        "op_counts": dict(sorted(op_counts.items())),
        "macs_total": macs_total,
        "operators": operators,
        "tensors": tensors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", type=Path, nargs="+")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    payload = [inventory(path) for path in args.models]
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))

    for item in payload:
        print(item["path"])
        print(f"  description: {item['description']}")
        print(f"  inputs: {[(t['index'], t['name'], t['shape'], t['type']) for t in item['inputs']]}")
        print(f"  outputs: {[(t['index'], t['name'], t['shape'], t['type']) for t in item['outputs']]}")
        print(f"  op_counts: {item['op_counts']}")
        print(f"  conv/depthwise macs: {item['macs_total']:,}")
        print("  first operators:")
        for op in item["operators"][:16]:
            print(
                f"    #{op['index']:03d} {op['op']:<18} "
                f"in={op['input_shapes']} out={op['output_shapes']} macs={op['macs']:,}"
            )
        print()


if __name__ == "__main__":
    main()
