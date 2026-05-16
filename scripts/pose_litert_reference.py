#!/usr/bin/env python3
"""Run one pose TFLite model with LiteRT and dump raw tensor outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", choices=["pose_detector", "pose_landmarker"])
    parser.add_argument("model", type=Path)
    parser.add_argument("input", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--random-input", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    interpreter = Interpreter(
        model_path=str(args.model),
        num_threads=1,
        experimental_op_resolver_type=OpResolverType.BUILTIN_REF,
    )
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    shape = input_detail["shape"]
    if args.random_input:
        rng = np.random.default_rng(args.seed)
        data = rng.random(shape, dtype=np.float32)
        args.input.parent.mkdir(parents=True, exist_ok=True)
        data.astype("<f4").tofile(args.input)
    else:
        data = np.fromfile(args.input, dtype="<f4").reshape(shape)
    interpreter.set_tensor(input_detail["index"], data.astype(np.float32, copy=False))
    interpreter.invoke()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for output in interpreter.get_output_details():
        value = interpreter.get_tensor(output["index"]).astype("<f4", copy=False)
        out_path = args.out_dir / f"{args.model_name}_tensor_{output['index']}.bin"
        value.tofile(out_path)
        print(
            f"{out_path} shape={list(value.shape)} "
            f"min={float(value.min()):.9g} max={float(value.max()):.9g}"
        )


if __name__ == "__main__":
    main()
