#!/usr/bin/env python3
"""Generate flat runtime data for the hand-written pose executor.

This is an offline build step.  The Pi runtime must not link TensorFlow Lite or
MediaPipe; it consumes the generated C include file and constant blobs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import tflite
from ai_edge_litert.interpreter import Interpreter, OpResolverType
from tflite import BuiltinOperator, Model, TensorType


SKIP_CONST_OPS = {"DEQUANTIZE", "DENSIFY"}
SUPPORTED_RUNTIME_OPS = {
    "ADD",
    "CONCATENATION",
    "CONV_2D",
    "DEPTHWISE_CONV_2D",
    "DEPTH_TO_SPACE",
    "LOGISTIC",
    "MAX_POOL_2D",
    "PAD",
    "RESHAPE",
    "RESIZE_BILINEAR",
}
OP_ENUM = {
    "ADD": "POSE_OP_ADD",
    "CONCATENATION": "POSE_OP_CONCAT",
    "CONV_2D": "POSE_OP_CONV2D",
    "DEPTHWISE_CONV_2D": "POSE_OP_DEPTHWISE",
    "DEPTH_TO_SPACE": "POSE_OP_DEPTH_TO_SPACE",
    "LOGISTIC": "POSE_OP_LOGISTIC",
    "MAX_POOL_2D": "POSE_OP_MAX_POOL2D",
    "PAD": "POSE_OP_PAD",
    "RESHAPE": "POSE_OP_RESHAPE",
    "RESIZE_BILINEAR": "POSE_OP_RESIZE_BILINEAR",
}
TYPE_ENUM = {
    TensorType.FLOAT32: "POSE_TENSOR_FLOAT32",
    TensorType.INT32: "POSE_TENSOR_INT32",
}
BUILTIN_NAMES = {
    value: name
    for name, value in vars(BuiltinOperator).items()
    if name.isupper() and isinstance(value, int)
}


def tensor_shape(tensor) -> list[int]:
    return [int(tensor.Shape(i)) for i in range(tensor.ShapeLength())]


def opcodes(model) -> list[str]:
    out = []
    for i in range(model.OperatorCodesLength()):
        opcode = model.OperatorCodes(i)
        out.append(BUILTIN_NAMES.get(int(opcode.BuiltinCode()), str(opcode.BuiltinCode())))
    return out


def init_options(op):
    builtin = op.BuiltinOptions()
    if builtin is None:
        return None
    return builtin.Bytes, builtin.Pos


def parse_options(op_name: str, op) -> dict[str, Any]:
    raw = init_options(op)
    if raw is None:
        return {}
    data, pos = raw

    def init(cls):
        opts = cls()
        opts.Init(data, pos)
        return opts

    if op_name == "CONV_2D":
        opts = init(tflite.Conv2DOptions)
        return {
            "padding": int(opts.Padding()),
            "stride_h": int(opts.StrideH()),
            "stride_w": int(opts.StrideW()),
            "dilation_h": int(opts.DilationHFactor()),
            "dilation_w": int(opts.DilationWFactor()),
            "activation": int(opts.FusedActivationFunction()),
        }
    if op_name == "DEPTHWISE_CONV_2D":
        opts = init(tflite.DepthwiseConv2DOptions)
        return {
            "padding": int(opts.Padding()),
            "stride_h": int(opts.StrideH()),
            "stride_w": int(opts.StrideW()),
            "dilation_h": int(opts.DilationHFactor()),
            "dilation_w": int(opts.DilationWFactor()),
            "depth_multiplier": int(opts.DepthMultiplier()),
            "activation": int(opts.FusedActivationFunction()),
        }
    if op_name == "ADD":
        opts = init(tflite.AddOptions)
        return {"activation": int(opts.FusedActivationFunction())}
    if op_name == "MAX_POOL_2D":
        opts = init(tflite.Pool2DOptions)
        return {
            "padding": int(opts.Padding()),
            "stride_h": int(opts.StrideH()),
            "stride_w": int(opts.StrideW()),
            "filter_h": int(opts.FilterHeight()),
            "filter_w": int(opts.FilterWidth()),
            "activation": int(opts.FusedActivationFunction()),
        }
    if op_name == "RESIZE_BILINEAR":
        opts = init(tflite.ResizeBilinearOptions)
        return {
            "align_corners": int(bool(opts.AlignCorners())),
            "half_pixel_centers": int(bool(opts.HalfPixelCenters())),
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


def const_buffer_tensors(model, subgraph) -> set[int]:
    out = set()
    for i in range(subgraph.TensorsLength()):
        tensor = subgraph.Tensors(i)
        buffer = model.Buffers(tensor.Buffer())
        if buffer and buffer.DataLength():
            out.add(i)
    return out


def folded_constants(model, subgraph, names: list[str]) -> set[int]:
    consts = const_buffer_tensors(model, subgraph)
    changed = True
    while changed:
        changed = False
        for i in range(subgraph.OperatorsLength()):
            op = subgraph.Operators(i)
            op_name = names[op.OpcodeIndex()]
            if op_name not in SKIP_CONST_OPS:
                continue
            inputs = [int(op.Inputs(j)) for j in range(op.InputsLength()) if int(op.Inputs(j)) >= 0]
            outputs = [int(op.Outputs(j)) for j in range(op.OutputsLength()) if int(op.Outputs(j)) >= 0]
            if inputs and all(idx in consts for idx in inputs):
                for out_idx in outputs:
                    if out_idx not in consts:
                        consts.add(out_idx)
                        changed = True
    return consts


def reachable_runtime_ops(subgraph, names: list[str], wanted_outputs: set[int] | None) -> list[int]:
    if wanted_outputs is None:
        return list(range(subgraph.OperatorsLength()))
    required = set(wanted_outputs)
    selected: list[int] = []
    for i in range(subgraph.OperatorsLength() - 1, -1, -1):
        op = subgraph.Operators(i)
        outputs = [int(op.Outputs(j)) for j in range(op.OutputsLength()) if int(op.Outputs(j)) >= 0]
        if not any(idx in required for idx in outputs):
            continue
        inputs = [int(op.Inputs(j)) for j in range(op.InputsLength()) if int(op.Inputs(j)) >= 0]
        required.update(inputs)
        if names[op.OpcodeIndex()] not in SKIP_CONST_OPS:
            selected.append(i)
    return sorted(selected)


def used_constants(subgraph, names: list[str], consts: set[int], selected_ops: set[int]) -> set[int]:
    used = set()
    for i in range(subgraph.OperatorsLength()):
        if i not in selected_ops:
            continue
        op = subgraph.Operators(i)
        op_name = names[op.OpcodeIndex()]
        if op_name in SKIP_CONST_OPS:
            continue
        if op_name not in SUPPORTED_RUNTIME_OPS:
            raise ValueError(f"unsupported runtime op {op_name} at op {i}")
        for j in range(op.InputsLength()):
            idx = int(op.Inputs(j))
            if idx in consts:
                used.add(idx)
    return used


def read_interpreter_tensor(interpreter: Interpreter, idx: int) -> np.ndarray:
    value = interpreter.get_tensor(idx)
    if value.dtype == np.float32:
        return np.ascontiguousarray(value.astype("<f4", copy=False))
    if value.dtype == np.int32:
        return np.ascontiguousarray(value.astype("<i4", copy=False))
    if value.dtype == np.float16:
        return np.ascontiguousarray(value.astype("<f4"))
    raise ValueError(f"unsupported constant dtype at tensor {idx}: {value.dtype}")


def tensor_c_def(
    model,
    subgraph,
    idx: int,
    is_const: bool,
    const_offsets: dict[int, int],
    const_bytes: dict[int, int],
) -> str:
    tensor = subgraph.Tensors(idx)
    shape = tensor_shape(tensor)
    dims = shape + [1] * (4 - len(shape))
    tensor_type = int(tensor.Type())
    type_name = TYPE_ENUM.get(tensor_type)
    if type_name is None:
        # Folded-away float16/sparse constants are not directly consumed by the
        # runtime. Keep a descriptor so tensor indices stay stable.
        type_name = "POSE_TENSOR_UNSUPPORTED"
    offset = const_offsets.get(idx, 0)
    byte_count = const_bytes.get(idx, 0)
    return (
        f"  {{{idx}, {type_name}, {len(shape)}, "
        f"{{{dims[0]}, {dims[1]}, {dims[2]}, {dims[3]}}}, "
        f"{1 if is_const else 0}, {offset}u, {byte_count}u}},"
    )


def op_c_def(op_index: int, op_name: str, inputs: list[int], outputs: list[int], options: dict[str, Any]) -> str:
    ins = inputs + [-1] * (8 - len(inputs))
    outs = outputs + [-1] * (4 - len(outputs))
    fields = {
        "activation": 0,
        "padding": 0,
        "stride_h": 1,
        "stride_w": 1,
        "dilation_h": 1,
        "dilation_w": 1,
        "filter_h": 1,
        "filter_w": 1,
        "axis": 0,
        "block_size": 0,
        "half_pixel_centers": 0,
        "align_corners": 0,
        "depth_multiplier": 1,
    }
    fields.update({k: int(v) for k, v in options.items()})
    return (
        f"  {{{op_index}, {OP_ENUM[op_name]}, {len(inputs)}, "
        f"{{{', '.join(str(x) for x in ins)}}}, {len(outputs)}, "
        f"{{{', '.join(str(x) for x in outs)}}}, "
        f"{fields['activation']}, {fields['padding']}, "
        f"{fields['stride_h']}, {fields['stride_w']}, "
        f"{fields['dilation_h']}, {fields['dilation_w']}, "
        f"{fields['filter_h']}, {fields['filter_w']}, {fields['axis']}, "
        f"{fields['block_size']}, {fields['half_pixel_centers']}, "
        f"{fields['align_corners']}, {fields['depth_multiplier']}}},"
    )


def write_model(
    model_name: str,
    path: Path,
    out_dir: Path,
    wanted_outputs: set[int] | None = None,
) -> dict[str, Any]:
    model = Model.GetRootAsModel(path.read_bytes(), 0)
    subgraph = model.Subgraphs(0)
    names = opcodes(model)
    consts = folded_constants(model, subgraph, names)
    selected_ops = set(reachable_runtime_ops(subgraph, names, wanted_outputs))
    used_consts = used_constants(subgraph, names, consts, selected_ops)

    interpreter = Interpreter(
        model_path=str(path),
        num_threads=1,
        experimental_op_resolver_type=OpResolverType.BUILTIN_REF,
        experimental_preserve_all_tensors=True,
    )
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    zero_input = np.zeros(input_detail["shape"], dtype=np.float32)
    interpreter.set_tensor(input_detail["index"], zero_input)
    interpreter.invoke()

    const_path = out_dir / f"{model_name}_constants.bin"
    const_offsets: dict[int, int] = {}
    const_bytes: dict[int, int] = {}
    with const_path.open("wb") as f:
        for idx in sorted(used_consts):
            value = read_interpreter_tensor(interpreter, idx)
            while f.tell() % 64:
                f.write(b"\0")
            const_offsets[idx] = f.tell()
            data = value.tobytes(order="C")
            const_bytes[idx] = len(data)
            f.write(data)

    runtime_ops = []
    skipped_ops = 0
    for i in range(subgraph.OperatorsLength()):
        if i not in selected_ops:
            continue
        op = subgraph.Operators(i)
        op_name = names[op.OpcodeIndex()]
        if op_name in SKIP_CONST_OPS:
            skipped_ops += 1
            continue
        inputs = [int(op.Inputs(j)) for j in range(op.InputsLength()) if int(op.Inputs(j)) >= 0]
        outputs = [int(op.Outputs(j)) for j in range(op.OutputsLength()) if int(op.Outputs(j)) >= 0]
        runtime_ops.append((i, op_name, inputs, outputs, parse_options(op_name, op)))

    header_lines = [
        f"static const PoseTensorDef {model_name}_tensors[] = {{",
    ]
    for idx in range(subgraph.TensorsLength()):
        header_lines.append(
            tensor_c_def(
                model,
                subgraph,
                idx,
                idx in used_consts,
                const_offsets,
                const_bytes,
            )
        )
    header_lines.append("};")
    header_lines.append("")
    header_lines.append(f"static const PoseOpDef {model_name}_ops[] = {{")
    for op in runtime_ops:
        header_lines.append(op_c_def(*op))
    header_lines.append("};")
    header_lines.append("")
    input_ids = [int(subgraph.Inputs(i)) for i in range(subgraph.InputsLength())]
    output_ids = [int(subgraph.Outputs(i)) for i in range(subgraph.OutputsLength())]
    if wanted_outputs is not None:
        output_ids = [idx for idx in output_ids if idx in wanted_outputs]
    padded_inputs = input_ids + [-1] * (4 - len(input_ids))
    padded_outputs = output_ids + [-1] * (8 - len(output_ids))
    header_lines.append(
        f"static const PoseModelDef {model_name}_model = "
        f"{{\"{model_name}\", \"{const_path.name}\", "
        f"{len(input_ids)}, {{{', '.join(str(x) for x in padded_inputs)}}}, "
        f"{len(output_ids)}, {{{', '.join(str(x) for x in padded_outputs)}}}, "
        f"{model_name}_tensors, {subgraph.TensorsLength()}, "
        f"{model_name}_ops, {len(runtime_ops)}}};"
    )

    (out_dir / f"{model_name}_plan.inc").write_text("\n".join(header_lines) + "\n")
    return {
        "name": model_name,
        "path": str(path),
        "const_path": str(const_path),
        "tensor_count": subgraph.TensorsLength(),
        "runtime_op_count": len(runtime_ops),
        "skipped_const_op_count": skipped_ops,
        "used_constant_count": len(used_consts),
        "constant_bytes": const_path.stat().st_size,
        "inputs": input_ids,
        "outputs": output_ids,
        "wanted_outputs": sorted(wanted_outputs) if wanted_outputs is not None else None,
        "op_counts": {
            name: sum(1 for _, op_name, _, _, _ in runtime_ops if op_name == name)
            for name in sorted(SUPPORTED_RUNTIME_OPS)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--landmarker", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("out/pose_runtime_data"))
    parser.add_argument(
        "--keep-segmentation",
        action="store_true",
        help="Keep landmarker segmentation output tensor 282. The default pose runtime prunes it.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    landmarker_outputs = None if args.keep_segmentation else {310, 315, 283, 312}
    summaries = [
        write_model("pose_detector", args.detector, args.out_dir),
        write_model(
            "pose_landmarker",
            args.landmarker,
            args.out_dir,
            wanted_outputs=landmarker_outputs,
        ),
    ]
    combined = []
    for name in ("pose_detector", "pose_landmarker"):
        combined.append(f'#include "{name}_plan.inc"')
    (args.out_dir / "pose_models_plan.inc").write_text("\n".join(combined) + "\n")
    (args.out_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
