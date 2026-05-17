#!/usr/bin/env python3
"""Benchmark Pi-to-iPhone pose frame encodings over the binary bridge."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.parse
import urllib.request
import zlib
from pathlib import Path
from typing import Any, Callable


def parse_size(value: str) -> tuple[int, int]:
    if "x" not in value:
        raise argparse.ArgumentTypeError("size must be WIDTHxHEIGHT")
    width_text, height_text = value.lower().split("x", 1)
    width = int(width_text)
    height = int(height_text)
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise argparse.ArgumentTypeError("width and height must be positive even numbers")
    return width, height


def clamp_u8(value: int) -> int:
    if value < 0:
        return 0
    if value > 255:
        return 255
    return value


def rgb_to_yuv(r: int, g: int, b: int) -> tuple[int, int, int]:
    y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16
    u = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128
    v = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128
    return clamp_u8(y), clamp_u8(u), clamp_u8(v)


def resize_rgb_nearest(rgb: bytes, src_width: int, src_height: int, dst_width: int, dst_height: int) -> bytes:
    if dst_width == src_width and dst_height == src_height:
        return rgb
    output = bytearray(dst_width * dst_height * 3)
    out = 0
    for row in range(dst_height):
        source_row = (row * src_height // dst_height) * src_width * 3
        for col in range(dst_width):
            source = source_row + (col * src_width // dst_width) * 3
            output[out] = rgb[source]
            output[out + 1] = rgb[source + 1]
            output[out + 2] = rgb[source + 2]
            out += 3
    return bytes(output)


def rgb_to_yuv420(rgb: bytes, width: int, height: int) -> bytes:
    frame = width * height
    y_plane = bytearray(frame)
    u_plane = bytearray(frame // 4)
    v_plane = bytearray(frame // 4)
    chroma_width = width // 2

    for row in range(height):
        row_offset = row * width
        for col in range(width):
            src = (row_offset + col) * 3
            y, _u, _v = rgb_to_yuv(rgb[src], rgb[src + 1], rgb[src + 2])
            y_plane[row_offset + col] = y

    for row in range(0, height, 2):
        for col in range(0, width, 2):
            u_total = 0
            v_total = 0
            for dy in (0, 1):
                for dx in (0, 1):
                    src = ((row + dy) * width + col + dx) * 3
                    _y, u, v = rgb_to_yuv(rgb[src], rgb[src + 1], rgb[src + 2])
                    u_total += u
                    v_total += v
            chroma_index = (row // 2) * chroma_width + (col // 2)
            u_plane[chroma_index] = u_total // 4
            v_plane[chroma_index] = v_total // 4

    return bytes(y_plane + u_plane + v_plane)


def raw_deflate(data: bytes, level: int) -> bytes:
    compressor = zlib.compressobj(level, zlib.DEFLATED, -15)
    return compressor.compress(data) + compressor.flush()


def parse_jpeg_size(data: bytes) -> tuple[int, int]:
    if not data.startswith(b"\xff\xd8"):
        raise ValueError("not a jpeg file")
    index = 2
    while index < len(data):
        if data[index] != 0xFF:
            index += 1
            continue
        while index < len(data) and data[index] == 0xFF:
            index += 1
        if index >= len(data):
            break
        marker = data[index]
        index += 1
        if marker in (0xD8, 0xD9):
            continue
        if index + 2 > len(data):
            break
        length = int.from_bytes(data[index:index + 2], "big")
        if length < 2 or index + length > len(data):
            break
        if marker in {
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        }:
            height = int.from_bytes(data[index + 3:index + 5], "big")
            width = int.from_bytes(data[index + 5:index + 7], "big")
            return width, height
        index += length
    raise ValueError("could not find jpeg dimensions")


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def post_pose(
    bridge_url: str,
    payload: bytes,
    frame_format: str,
    width: int,
    height: int,
    backend: str,
    model: str,
    timeout: float,
) -> dict[str, Any]:
    query = urllib.parse.urlencode(
        {
            "format": frame_format,
            "width": width,
            "height": height,
            "pose_backend": backend,
            "pose_model": model,
            "timeout": str(timeout),
        }
    )
    url = f"{bridge_url.rstrip('/')}/pose-frame?{query}"
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/octet-stream"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout + 10) as response:
        body = response.read()
    result = json.loads(body.decode("utf-8"))
    result["http_wall_ms"] = (time.perf_counter() - started) * 1000
    return result


def build_variant_maker(
    source_rgb: bytes,
    source_width: int,
    source_height: int,
    frame_format: str,
    width: int,
    height: int,
    level: int,
) -> Callable[[], tuple[bytes, float]]:
    def make_rgb() -> tuple[bytes, float]:
        started = time.perf_counter()
        resized = resize_rgb_nearest(source_rgb, source_width, source_height, width, height)
        return resized, (time.perf_counter() - started) * 1000

    def make_yuv() -> tuple[bytes, float]:
        started = time.perf_counter()
        resized = resize_rgb_nearest(source_rgb, source_width, source_height, width, height)
        yuv = rgb_to_yuv420(resized, width, height)
        return yuv, (time.perf_counter() - started) * 1000

    def make_deflate_rgb() -> tuple[bytes, float]:
        started = time.perf_counter()
        resized = resize_rgb_nearest(source_rgb, source_width, source_height, width, height)
        compressed = raw_deflate(resized, level)
        return compressed, (time.perf_counter() - started) * 1000

    def make_deflate_yuv() -> tuple[bytes, float]:
        started = time.perf_counter()
        resized = resize_rgb_nearest(source_rgb, source_width, source_height, width, height)
        yuv = rgb_to_yuv420(resized, width, height)
        compressed = raw_deflate(yuv, level)
        return compressed, (time.perf_counter() - started) * 1000

    makers = {
        "rgb24": make_rgb,
        "yuv420": make_yuv,
        "deflate_rgb24": make_deflate_rgb,
        "deflate_yuv420": make_deflate_yuv,
        "zlib_rgb24": make_deflate_rgb,
        "zlib_yuv420": make_deflate_yuv,
    }
    return makers[frame_format]


def benchmark_variant(args: argparse.Namespace, name: str, maker: Callable[[], tuple[bytes, float]], frame_format: str, width: int, height: int) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for run_index in range(args.warmup + args.repeats):
        payload, prep_ms = maker()
        result = post_pose(
            args.bridge_url,
            payload,
            frame_format,
            width,
            height,
            args.backend,
            args.model,
            args.timeout,
        )
        record = {
            "run": run_index,
            "warmup": run_index < args.warmup,
            "prep_ms": prep_ms,
            "payload_bytes": len(payload),
            "http_wall_ms": result.get("http_wall_ms", 0.0),
            "decode_ms": float(result.get("decode_seconds", 0.0)) * 1000,
            "load_ms": float(result.get("load_seconds", 0.0)) * 1000,
            "inference_ms": float(result.get("inference_seconds", 0.0)) * 1000,
            "iphone_total_ms": float(result.get("total_seconds", 0.0)) * 1000,
            "pose_count": result.get("pose_count"),
            "pose_presence": result.get("pose_presence"),
        }
        runs.append(record)
        print(json.dumps({"variant": name, **record}, separators=(",", ":")), flush=True)

    measured = [run for run in runs if not run["warmup"]]
    return {
        "name": name,
        "format": frame_format,
        "width": width,
        "height": height,
        "runs": runs,
        "summary": {
            "payload_bytes": int(statistics.median([run["payload_bytes"] for run in measured])) if measured else 0,
            "prep_ms": summarize([run["prep_ms"] for run in measured]),
            "http_wall_ms": summarize([run["http_wall_ms"] for run in measured]),
            "decode_ms": summarize([run["decode_ms"] for run in measured]),
            "inference_ms": summarize([run["inference_ms"] for run in measured]),
            "iphone_total_ms": summarize([run["iphone_total_ms"] for run in measured]),
            "pose_presence": summarize([float(run["pose_presence"] or 0.0) for run in measured]),
        },
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# iPhone Pose Transport Benchmark",
        "",
        f"- Bridge: `{report['bridge_url']}`",
        f"- Backend/model: `{report['backend']}` / `{report['model']}`",
        f"- Repeats: {report['repeats']} measured, {report['warmup']} warmup",
        "",
        "| Variant | Bytes | Prep ms | HTTP wall ms | iPhone decode ms | Inference ms | iPhone total ms | Presence |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in report["variants"]:
        summary = variant["summary"]
        lines.append(
            "| {name} | {bytes} | {prep:.1f} | {wall:.1f} | {decode:.1f} | {infer:.1f} | {total:.1f} | {presence:.3f} |".format(
                name=variant["name"],
                bytes=summary["payload_bytes"],
                prep=summary["prep_ms"]["median"],
                wall=summary["http_wall_ms"]["median"],
                decode=summary["decode_ms"]["median"],
                infer=summary["inference_ms"]["median"],
                total=summary["iphone_total_ms"]["median"],
                presence=summary["pose_presence"]["median"],
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-url", default="http://127.0.0.1:8765")
    parser.add_argument("--rgb-file", type=Path, required=True)
    parser.add_argument("--rgb-size", type=parse_size, required=True)
    parser.add_argument("--jpeg-file", type=Path)
    parser.add_argument("--sizes", type=parse_size, nargs="+", default=[(320, 240), (256, 192), (160, 120)])
    parser.add_argument("--formats", nargs="+", default=["rgb24", "yuv420", "deflate_rgb24", "deflate_yuv420"])
    parser.add_argument("--zlib-level", type=int, default=1)
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--model", choices=["lite", "full", "heavy"], default="lite")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    source_width, source_height = args.rgb_size
    source_rgb = args.rgb_file.read_bytes()
    expected = source_width * source_height * 3
    if len(source_rgb) != expected:
        raise SystemExit(f"{args.rgb_file} has {len(source_rgb)} bytes, expected {expected}")

    variants: list[dict[str, Any]] = []
    for width, height in args.sizes:
        if width > source_width or height > source_height:
            continue
        for frame_format in args.formats:
            maker = build_variant_maker(
                source_rgb,
                source_width,
                source_height,
                frame_format,
                width,
                height,
                args.zlib_level,
            )
            name = f"{frame_format}-{width}x{height}"
            variants.append(benchmark_variant(args, name, maker, frame_format, width, height))

    if args.jpeg_file:
        jpeg_data = args.jpeg_file.read_bytes()
        jpeg_width, jpeg_height = parse_jpeg_size(jpeg_data)
        maker = lambda: (jpeg_data, 0.0)
        variants.append(benchmark_variant(args, f"jpeg-{jpeg_width}x{jpeg_height}", maker, "jpeg", jpeg_width, jpeg_height))

    report = {
        "bridge_url": args.bridge_url,
        "backend": args.backend,
        "model": args.model,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "source_rgb": str(args.rgb_file),
        "source_size": {"width": source_width, "height": source_height},
        "variants": variants,
    }
    if args.output_json:
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.output_md:
        write_markdown(args.output_md, report)


if __name__ == "__main__":
    main()
