#!/usr/bin/env python3
"""Stdlib-only Pi bridge for an iPhone-initiated Gemma worker connection."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import struct
import sys
import time
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator
from urllib.parse import parse_qs, urlsplit


GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
POSE_BINARY_MAGIC = b"G4POSE01"
GENERATE_BINARY_MAGIC = b"G4GEN01"
TTS_BINARY_MAGIC = b"G4TTS01"
POSE_WS_CHUNK_CHARS = 120_000


@dataclass
class PendingJob:
    queue: asyncio.Queue[dict[str, Any]]
    started_at: float = field(default_factory=time.monotonic)


class BridgeState:
    def __init__(self) -> None:
        self.worker: WebSocketPeer | None = None
        self.worker_status: dict[str, Any] = {}
        self.pending: dict[str, PendingJob] = {}
        self.lock = asyncio.Lock()
        self.total_requests = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_pose_requests = 0
        self.total_tts_requests = 0
        self.last_text = ""
        self.last_prompt = ""
        self.last_pose = {}
        self.last_tts = {}

    async def set_worker(self, worker: "WebSocketPeer | None") -> None:
        stale_worker: WebSocketPeer | None = None
        pending_to_fail: list[PendingJob] = []
        async with self.lock:
            if self.worker and self.worker is not worker:
                stale_worker = self.worker
            self.worker = worker
            if worker is None:
                self.worker_status = {}
            if stale_worker is not None or worker is None:
                pending_to_fail = list(self.pending.values())
        if stale_worker is not None:
            try:
                await stale_worker.close()
            except Exception:
                pass
        for job in pending_to_fail:
            await job.queue.put({
                "type": "error",
                "message": "iPhone worker disconnected",
            })

    async def send_to_worker_json(self, worker: "WebSocketPeer", payload: dict[str, Any]) -> None:
        try:
            await worker.send_json(payload)
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            if self.worker is worker:
                await self.set_worker(None)
            raise RuntimeError("iPhone worker disconnected while sending request") from exc

    async def send_to_worker_binary(self, worker: "WebSocketPeer", payload: bytes) -> None:
        try:
            await worker.send_binary(payload)
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            if self.worker is worker:
                await self.set_worker(None)
            raise RuntimeError("iPhone worker disconnected while sending request") from exc

    async def handle_worker_message(self, message: dict[str, Any]) -> None:
        msg_type = message.get("type")
        if msg_type == "ready":
            self.worker_status = message
            return

        job_id = message.get("id")
        if isinstance(job_id, str) and job_id in self.pending:
            await self.pending[job_id].queue.put(message)

    async def handle_worker_binary(self, payload: bytes) -> None:
        message = unpack_worker_binary(payload)
        job_id = message.get("id")
        if isinstance(job_id, str) and job_id in self.pending:
            await self.pending[job_id].queue.put(message)

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int,
        timeout: float,
        media: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        worker = self.worker
        if not worker:
            raise RuntimeError("No iPhone worker connected")

        job_id = uuid.uuid4().hex
        job = PendingJob(queue=asyncio.Queue())
        self.pending[job_id] = job
        self.total_requests += 1
        self.last_prompt = prompt
        chunks: list[str] = []

        if media:
            await self.send_to_worker_binary(
                worker,
                pack_generate_binary(
                    {
                        "type": "generate_media",
                        "id": job_id,
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                    },
                    media,
                ),
            )
        else:
            await self.send_to_worker_json(
                worker,
                {
                    "type": "generate",
                    "id": job_id,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                }
            )

        try:
            while True:
                event = await asyncio.wait_for(job.queue.get(), timeout=timeout)
                event_type = event.get("type")
                if event_type == "token":
                    token = str(event.get("text", ""))
                    chunks.append(token)
                    self.total_output_tokens += max(1, len(token.split()))
                    yield {"type": "token", "id": job_id, "text": token}
                elif event_type == "done":
                    text = str(event.get("text") or "".join(chunks))
                    elapsed = time.monotonic() - job.started_at
                    input_tokens = int(event.get("input_tokens_estimate") or max(1, len(prompt.split())))
                    output_tokens = int(event.get("output_tokens_estimate") or max(1, len(text.split())))
                    self.total_input_tokens += input_tokens
                    self.last_text = text
                    yield {
                        "type": "done",
                        "id": job_id,
                        "text": text,
                        "elapsed_seconds": elapsed,
                        "input_tokens_estimate": input_tokens,
                        "output_tokens_estimate": output_tokens,
                        "tokens_per_second": event.get("tokens_per_second"),
                    }
                    return
                elif event_type == "error":
                    raise RuntimeError(str(event.get("message", "iPhone worker returned an error")))
        finally:
            self.pending.pop(job_id, None)

    async def generate(
        self,
        prompt: str,
        max_tokens: int,
        timeout: float,
        media: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        async for event in self.generate_stream(
            prompt,
            max_tokens=max_tokens,
            timeout=timeout,
            media=media,
        ):
            if event.get("type") == "done":
                result = dict(event)
                result.pop("type", None)
                return result
        raise RuntimeError("iPhone worker ended stream without a final done event")

    async def pose(
        self,
        payload: dict[str, Any],
        timeout: float,
        frame_bytes: bytes | None = None,
    ) -> dict[str, Any]:
        worker = self.worker
        if not worker:
            raise RuntimeError("No iPhone worker connected")

        job_id = uuid.uuid4().hex
        job = PendingJob(queue=asyncio.Queue())
        self.pending[job_id] = job
        self.total_pose_requests += 1

        request = dict(payload)
        request["id"] = job_id
        if frame_bytes is not None:
            request["type"] = "pose_binary"
            request["input_bytes"] = len(frame_bytes)
            await self.send_to_worker_binary(worker, pack_pose_binary(request, frame_bytes))
        else:
            data = request.get("data")
            if isinstance(data, str) and len(data) > POSE_WS_CHUNK_CHARS:
                chunks = [
                    data[index : index + POSE_WS_CHUNK_CHARS]
                    for index in range(0, len(data), POSE_WS_CHUNK_CHARS)
                ]
                start = dict(request)
                start.pop("data", None)
                start["type"] = "pose_start"
                start["chunk_count"] = len(chunks)
                await self.send_to_worker_json(worker, start)
                for index, chunk in enumerate(chunks):
                    await self.send_to_worker_json(
                        worker,
                        {
                            "type": "pose_chunk",
                            "id": job_id,
                            "chunk_index": index,
                            "data": chunk,
                        }
                    )
            else:
                request["type"] = "pose"
                await self.send_to_worker_json(worker, request)

        try:
            while True:
                event = await asyncio.wait_for(job.queue.get(), timeout=timeout)
                event_type = event.get("type")
                if event_type == "pose_done":
                    result = dict(event)
                    result.pop("type", None)
                    result.pop("id", None)
                    self.last_pose = result
                    return result
                if event_type == "error":
                    raise RuntimeError(str(event.get("message", "iPhone worker returned an error")))
        finally:
            self.pending.pop(job_id, None)

    async def tts_stream(
        self,
        text: str,
        backend: str,
        voice: str,
        timeout: float,
    ) -> AsyncIterator[bytes]:
        worker = self.worker
        if not worker:
            raise RuntimeError("No iPhone worker connected")

        job_id = uuid.uuid4().hex
        job = PendingJob(queue=asyncio.Queue())
        self.pending[job_id] = job
        self.total_tts_requests += 1
        started = time.monotonic()
        audio_bytes = 0
        chunks = 0

        await self.send_to_worker_json(
            worker,
            {
                "type": "tts",
                "id": job_id,
                "text": text,
                "tts_backend": backend,
                "voice": voice,
                "audio_format": "s16le",
                "sample_rate": 24000,
            },
        )

        try:
            while True:
                event = await asyncio.wait_for(job.queue.get(), timeout=timeout)
                event_type = event.get("type")
                if event_type == "tts_audio":
                    chunk = event.get("audio_bytes")
                    if isinstance(chunk, bytes):
                        audio_bytes += len(chunk)
                        chunks += 1
                        yield chunk
                elif event_type == "tts_done":
                    self.last_tts = {
                        "backend": event.get("backend", backend),
                        "voice": event.get("voice", voice),
                        "sample_rate": event.get("sample_rate", 24000),
                        "audio_bytes": audio_bytes,
                        "chunks": chunks,
                        "elapsed_seconds": time.monotonic() - started,
                        "iphone_elapsed_seconds": event.get("elapsed_seconds"),
                        "iphone_first_audio_seconds": event.get("first_audio_seconds"),
                    }
                    return
                elif event_type == "error":
                    raise RuntimeError(str(event.get("message", "iPhone worker returned an error")))
        finally:
            self.pending.pop(job_id, None)

    async def tts_benchmark(
        self,
        text: str,
        timeout: float,
    ) -> dict[str, Any]:
        worker = self.worker
        if not worker:
            raise RuntimeError("No iPhone worker connected")

        job_id = uuid.uuid4().hex
        job = PendingJob(queue=asyncio.Queue())
        self.pending[job_id] = job
        self.total_tts_requests += 1

        await self.send_to_worker_json(
            worker,
            {
                "type": "tts_benchmark",
                "id": job_id,
                "text": text,
            },
        )

        try:
            while True:
                event = await asyncio.wait_for(job.queue.get(), timeout=timeout)
                event_type = event.get("type")
                if event_type == "tts_benchmark_done":
                    result = dict(event)
                    result.pop("type", None)
                    result.pop("id", None)
                    self.last_tts = result
                    return result
                if event_type == "error":
                    raise RuntimeError(str(event.get("message", "iPhone worker returned an error")))
        finally:
            self.pending.pop(job_id, None)

    def health(self) -> dict[str, Any]:
        return {
            "worker_connected": self.worker is not None,
            "worker_status": self.worker_status,
            "pending_jobs": list(self.pending.keys()),
            "total_requests": self.total_requests,
            "total_pose_requests": self.total_pose_requests,
            "total_tts_requests": self.total_tts_requests,
            "total_input_tokens_estimate": self.total_input_tokens,
            "total_output_tokens_estimate": self.total_output_tokens,
            "last_prompt": self.last_prompt[-240:],
            "last_text": self.last_text[-400:],
            "last_pose": self.last_pose,
            "last_tts": self.last_tts,
        }


class WebSocketPeer:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.reader = reader
        self.writer = writer
        self.write_lock = asyncio.Lock()

    async def send_json(self, payload: dict[str, Any]) -> None:
        await self.send_text(json.dumps(payload, separators=(",", ":")))

    async def send_text(self, text: str) -> None:
        data = text.encode("utf-8")
        async with self.write_lock:
            self.writer.write(encode_ws_frame(data, opcode=0x1))
            await self.writer.drain()

    async def send_binary(self, data: bytes) -> None:
        async with self.write_lock:
            self.writer.write(encode_ws_frame(data, opcode=0x2))
            await self.writer.drain()

    async def close(self) -> None:
        try:
            async with self.write_lock:
                self.writer.write(encode_ws_frame(b"", opcode=0x8))
                await self.writer.drain()
        finally:
            self.writer.close()
            await self.writer.wait_closed()


async def read_http_request(reader: asyncio.StreamReader) -> tuple[str, str, dict[str, str], bytes]:
    header_data = await reader.readuntil(b"\r\n\r\n")
    header_text = header_data.decode("iso-8859-1")
    lines = header_text.split("\r\n")
    method, path, _version = lines[0].split(" ", 2)
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        headers[key.strip().lower()] = value.strip()

    content_length = int(headers.get("content-length", "0") or "0")
    body = await reader.readexactly(content_length) if content_length else b""
    return method, path, headers, body


async def handle_connection(
    state: BridgeState,
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    try:
        method, path, headers, body = await read_http_request(reader)
        parsed_path = urlsplit(path)
        route = parsed_path.path
        query = {key: values[-1] for key, values in parse_qs(parsed_path.query).items() if values}
        if headers.get("upgrade", "").lower() == "websocket":
            try:
                await handle_websocket(state, route, headers, reader, writer)
            except (asyncio.IncompleteReadError, ConnectionResetError):
                pass
            return

        if method == "GET" and route == "/health":
            await send_http_json(writer, 200, state.health())
            return

        if method == "POST" and route == "/generate-stream":
            payload = json.loads(body.decode("utf-8") or "{}")
            prompt = str(payload.get("prompt", ""))
            max_tokens = int(payload.get("max_tokens", 128))
            timeout = float(payload.get("timeout", 300))
            await send_http_ndjson_stream(
                writer,
                state.generate_stream(prompt, max_tokens=max_tokens, timeout=timeout),
            )
            return

        if method == "POST" and route == "/generate-media-stream":
            prompt, max_tokens, timeout, media = unpack_generate_http_request(body)
            await send_http_ndjson_stream(
                writer,
                state.generate_stream(
                    prompt,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    media=media,
                ),
            )
            return

        if method == "POST" and route == "/generate":
            payload = json.loads(body.decode("utf-8") or "{}")
            prompt = str(payload.get("prompt", ""))
            max_tokens = int(payload.get("max_tokens", 128))
            timeout = float(payload.get("timeout", 300))
            result = await state.generate(prompt, max_tokens=max_tokens, timeout=timeout)
            await send_http_json(writer, 200, result)
            return

        if method == "POST" and route == "/tts-stream":
            payload = json.loads(body.decode("utf-8") or "{}")
            text = str(payload.get("text", ""))
            backend = str(payload.get("tts_backend") or payload.get("backend") or "piper-ryan-high")
            voice = str(payload.get("voice") or "")
            timeout = float(payload.get("timeout", 120))
            await send_http_binary_stream(
                writer,
                state.tts_stream(text=text, backend=backend, voice=voice, timeout=timeout),
                headers={
                    "Content-Type": "application/octet-stream",
                    "X-Audio-Format": "s16le",
                    "X-Audio-Sample-Rate": "24000",
                    "X-Audio-Channels": "1",
                },
            )
            return

        if method == "POST" and route == "/tts-benchmark":
            payload = json.loads(body.decode("utf-8") or "{}")
            text = str(payload.get("text", "Hello from the iPhone TTS benchmark."))
            timeout = float(payload.get("timeout", 240))
            result = await state.tts_benchmark(text=text, timeout=timeout)
            await send_http_json(writer, 200, result)
            return

        if method == "POST" and route in {"/pose", "/pose-frame"}:
            content_type = headers.get("content-type", "").split(";", 1)[0].strip().lower()
            if route == "/pose-frame" or content_type == "application/octet-stream":
                payload = {
                    "format": query.get("format", "yuv420"),
                    "width": int(query["width"]),
                    "height": int(query["height"]),
                    "pose_backend": query.get("pose_backend", query.get("backend", "gpu")),
                    "pose_model": query.get("pose_model", query.get("model", "lite")),
                }
                timeout = float(query.get("timeout", "30"))
                result = await state.pose(payload, timeout=timeout, frame_bytes=body)
            else:
                payload = json.loads(body.decode("utf-8") or "{}")
                timeout = float(payload.pop("timeout", 30))
                result = await state.pose(payload, timeout=timeout)
            await send_http_json(writer, 200, result)
            return

        await send_http_json(writer, 404, {"error": "not found"})
    except Exception as exc:  # noqa: BLE001 - keep bridge errors visible to caller.
        await send_http_json(writer, 500, {"error": str(exc)})


async def handle_websocket(
    state: BridgeState,
    path: str,
    headers: dict[str, str],
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    if not path.startswith("/worker"):
        await send_http_json(writer, 404, {"error": "websocket path must be /worker"})
        return

    key = headers.get("sec-websocket-key")
    if not key:
        await send_http_json(writer, 400, {"error": "missing Sec-WebSocket-Key"})
        return

    accept = base64.b64encode(hashlib.sha1((key + GUID).encode("ascii")).digest()).decode("ascii")
    writer.write(
        (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n"
            "\r\n"
        ).encode("ascii")
    )
    await writer.drain()

    peer = WebSocketPeer(reader, writer)
    await state.set_worker(peer)
    await peer.send_json({"type": "hello", "bridge": "gemma4-robot-pi", "time": time.time()})
    print(f"iPhone worker connected from {writer.get_extra_info('peername')}", flush=True)

    try:
        while True:
            opcode, payload = await read_ws_frame(reader)
            if opcode == 0x8:
                break
            if opcode == 0x9:
                writer.write(encode_ws_frame(payload, opcode=0xA))
                await writer.drain()
                continue
            if opcode == 0x2:
                try:
                    await state.handle_worker_binary(payload)
                except Exception as exc:  # noqa: BLE001
                    print(f"ignored bad worker binary frame: {exc}", flush=True)
                continue
            if opcode != 0x1:
                continue
            try:
                message = json.loads(payload.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            await state.handle_worker_message(message)
    finally:
        await state.set_worker(None)
        print(f"iPhone worker disconnected from {writer.get_extra_info('peername')}", flush=True)


async def read_ws_frame(reader: asyncio.StreamReader) -> tuple[int, bytes]:
    first = await reader.readexactly(2)
    b1, b2 = first[0], first[1]
    opcode = b1 & 0x0F
    masked = bool(b2 & 0x80)
    length = b2 & 0x7F

    if length == 126:
        length = struct.unpack("!H", await reader.readexactly(2))[0]
    elif length == 127:
        length = struct.unpack("!Q", await reader.readexactly(8))[0]

    mask = await reader.readexactly(4) if masked else b""
    payload = await reader.readexactly(length) if length else b""
    if masked:
        payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
    return opcode, payload


def encode_ws_frame(payload: bytes, opcode: int = 0x1) -> bytes:
    header = bytearray([0x80 | opcode])
    length = len(payload)
    if length < 126:
        header.append(length)
    elif length <= 0xFFFF:
        header.append(126)
        header.extend(struct.pack("!H", length))
    else:
        header.append(127)
        header.extend(struct.pack("!Q", length))
    return bytes(header) + payload


def pack_pose_binary(header: dict[str, Any], frame_bytes: bytes) -> bytes:
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return POSE_BINARY_MAGIC + struct.pack("!I", len(header_bytes)) + header_bytes + frame_bytes


def pack_generate_binary(header: dict[str, Any], media: list[dict[str, Any]]) -> bytes:
    offset = 0
    media_headers = []
    payload = bytearray()
    for item in media:
        data = bytes(item["data"])
        media_headers.append(
            {
                "mime_type": item["mime_type"],
                "display_name": item.get("display_name"),
                "offset": offset,
                "bytes": len(data),
            }
        )
        payload.extend(data)
        offset += len(data)

    request = dict(header)
    request["media"] = media_headers
    header_bytes = json.dumps(request, separators=(",", ":")).encode("utf-8")
    return GENERATE_BINARY_MAGIC + struct.pack("!I", len(header_bytes)) + header_bytes + bytes(payload)


def unpack_generate_http_request(payload: bytes) -> tuple[str, int, float, list[dict[str, Any]]]:
    header_start = len(GENERATE_BINARY_MAGIC) + 4
    if len(payload) < header_start or not payload.startswith(GENERATE_BINARY_MAGIC):
        raise ValueError("generate media request must use G4GEN01 binary framing")
    header_len = struct.unpack("!I", payload[len(GENERATE_BINARY_MAGIC) : header_start])[0]
    header_end = header_start + header_len
    if len(payload) < header_end:
        raise ValueError("generate media request header is truncated")
    header = json.loads(payload[header_start:header_end].decode("utf-8"))
    body_start = header_end
    media = []
    for item in header.get("media", []):
        offset = int(item["offset"])
        size = int(item["bytes"])
        start = body_start + offset
        end = start + size
        if start < body_start or end > len(payload) or end < start:
            raise ValueError("generate media request has an out-of-range media item")
        media.append(
            {
                "mime_type": str(item["mime_type"]),
                "display_name": item.get("display_name"),
                "data": payload[start:end],
            }
        )
    return (
        str(header.get("prompt", "")),
        int(header.get("max_tokens", 128)),
        float(header.get("timeout", 300)),
        media,
    )


def unpack_worker_binary(payload: bytes) -> dict[str, Any]:
    if payload.startswith(TTS_BINARY_MAGIC):
        header_start = len(TTS_BINARY_MAGIC) + 4
        if len(payload) < header_start:
            raise ValueError("TTS binary frame is too short")
        header_len = struct.unpack("!I", payload[len(TTS_BINARY_MAGIC) : header_start])[0]
        header_end = header_start + header_len
        if len(payload) < header_end:
            raise ValueError("TTS binary frame header is truncated")
        header = json.loads(payload[header_start:header_end].decode("utf-8"))
        header["audio_bytes"] = payload[header_end:]
        return header
    raise ValueError("unknown worker binary frame magic")


async def send_http_json(writer: asyncio.StreamWriter, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, indent=2).encode("utf-8")
    reason = {
        200: "OK",
        400: "Bad Request",
        404: "Not Found",
        500: "Internal Server Error",
        502: "Bad Gateway",
    }.get(status, "OK")
    writer.write(
        (
            f"HTTP/1.1 {status} {reason}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n"
            "\r\n"
        ).encode("ascii")
        + body
    )
    try:
        await writer.drain()
    except ConnectionResetError:
        return
    writer.close()
    try:
        await writer.wait_closed()
    except ConnectionResetError:
        pass


async def send_http_ndjson_stream(
    writer: asyncio.StreamWriter,
    events: AsyncIterator[dict[str, Any]],
) -> None:
    writer.write(
        (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: application/x-ndjson\r\n"
            "Transfer-Encoding: chunked\r\n"
            "Cache-Control: no-store\r\n"
            "Connection: close\r\n"
            "\r\n"
        ).encode("ascii")
    )
    try:
        await writer.drain()
        async for event in events:
            await send_http_chunk(
                writer,
                json.dumps(event, separators=(",", ":")).encode("utf-8") + b"\n",
            )
    except Exception as exc:  # noqa: BLE001 - stream errors must reach the HTTP caller.
        try:
            await send_http_chunk(
                writer,
                json.dumps(
                    {"type": "error", "message": str(exc)},
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n",
            )
        except ConnectionResetError:
            pass
    finally:
        await finish_http_chunks(writer)


async def send_http_chunk(writer: asyncio.StreamWriter, body: bytes) -> None:
    writer.write(f"{len(body):X}\r\n".encode("ascii") + body + b"\r\n")
    await writer.drain()


async def finish_http_chunks(writer: asyncio.StreamWriter) -> None:
    try:
        writer.write(b"0\r\n\r\n")
        await writer.drain()
    except ConnectionResetError:
        return
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except ConnectionResetError:
            pass


async def send_http_binary_stream(
    writer: asyncio.StreamWriter,
    chunks: AsyncIterator[bytes],
    headers: dict[str, str],
) -> None:
    header_sent = False
    response_closed = False

    async def send_headers() -> None:
        nonlocal header_sent
        if header_sent:
            return
        header_lines = [
            "HTTP/1.1 200 OK",
            "Transfer-Encoding: chunked",
            "Cache-Control: no-store",
            "Connection: close",
        ]
        header_lines.extend(f"{name}: {value}" for name, value in headers.items())
        writer.write(("\r\n".join(header_lines) + "\r\n\r\n").encode("ascii"))
        await writer.drain()
        header_sent = True

    try:
        async for chunk in chunks:
            if chunk:
                await send_headers()
                await send_http_chunk(writer, chunk)
    except Exception as exc:  # noqa: BLE001 - stream errors must reach the caller if possible.
        print(f"binary stream failed: {exc}", file=sys.stderr, flush=True)
        if not header_sent:
            await send_http_json(writer, 502, {"error": str(exc)})
            response_closed = True
            return
        # A chunked binary response cannot change HTTP status after audio bytes have
        # already been sent. Close the stream without appending text into raw PCM.
        return
    finally:
        if response_closed:
            return
        if header_sent:
            await finish_http_chunks(writer)
        else:
            await send_headers()
            await finish_http_chunks(writer)


async def serve(args: argparse.Namespace) -> None:
    state = BridgeState()
    server = await asyncio.start_server(
        lambda reader, writer: handle_connection(state, reader, writer),
        host=args.host,
        port=args.port,
    )
    sockets = ", ".join(str(sock.getsockname()) for sock in (server.sockets or []))
    print(f"listening on {sockets}", flush=True)
    async with server:
        await server.serve_forever()


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=payload.get("timeout", 300) + 5) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    subcommands = parser.add_subparsers(dest="command", required=True)

    serve_parser = subcommands.add_parser("serve")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8765)

    prompt_parser = subcommands.add_parser("prompt")
    prompt_parser.add_argument("prompt")
    prompt_parser.add_argument("--host", default="127.0.0.1")
    prompt_parser.add_argument("--port", type=int, default=8765)
    prompt_parser.add_argument("--max-tokens", type=int, default=128)
    prompt_parser.add_argument("--timeout", type=float, default=300)

    health_parser = subcommands.add_parser("health")
    health_parser.add_argument("--host", default="127.0.0.1")
    health_parser.add_argument("--port", type=int, default=8765)

    args = parser.parse_args()

    if args.command == "serve":
        try:
            asyncio.run(serve(args))
        except KeyboardInterrupt:
            return 130
        return 0

    if args.command == "prompt":
        result = post_json(
            f"http://{args.host}:{args.port}/generate",
            {
                "prompt": args.prompt,
                "max_tokens": args.max_tokens,
                "timeout": args.timeout,
            },
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "health":
        print(json.dumps(get_json(f"http://{args.host}:{args.port}/health"), indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
