#!/usr/bin/env python3
"""Drive a LEGO hub motor from the hub's left/right arrow buttons.

This talks to the Robot Inventor/SPIKE uJSON-RPC runtime over Bluetooth
Classic RFCOMM channel 1. Right arrow rotates the motor forward by one step;
left arrow rotates it backward by one step.
"""

from __future__ import annotations

import argparse
import base64
import json
import select
import socket
import sys
import time
from typing import Any


DEFAULT_HUB = "A8:E2:C1:9A:5D:04"


class HubController:
    def __init__(self, hub: str, port: str, degrees: int, speed: int) -> None:
        self.hub = hub
        self.port = port.upper()
        self.degrees = degrees
        self.speed = speed
        self.sock: socket.socket | None = None
        self.buffer = b""
        self.sequence = 0
        self.button_down = {"left": False, "right": False}
        self.last_action = {"left": 0.0, "right": 0.0}

    def connect(self) -> None:
        sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
        sock.settimeout(10)
        sock.connect((self.hub, 1))
        sock.setblocking(False)
        self.sock = sock
        print(f"connected to {self.hub} on RFCOMM channel 1", flush=True)

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()
        self.sock = None

    def next_id(self) -> str:
        self.sequence += 1
        return f"ctrl{self.sequence:04d}"

    def send_rpc(self, method: str, params: dict[str, Any] | None = None) -> str:
        if self.sock is None:
            raise RuntimeError("not connected")
        message: dict[str, Any] = {"i": self.next_id(), "m": method}
        if params is not None:
            message["p"] = params
        payload = json.dumps(message, separators=(",", ":")).encode("utf-8") + b"\r"
        self.sock.sendall(payload)
        return message["i"]

    def rotate(self, direction: int) -> None:
        degrees = self.degrees * direction
        request_id = self.send_rpc(
            "scratch.motor_run_for_degrees",
            {
                "port": self.port,
                "speed": self.speed,
                "degrees": degrees,
                "stall": True,
                "stop": 1,
            },
        )
        print(f"sent {request_id}: motor {self.port} {degrees:+d} degrees", flush=True)

    def read_messages(self, timeout: float = 0.2) -> list[dict[str, Any]]:
        if self.sock is None:
            raise RuntimeError("not connected")
        readable, _, _ = select.select([self.sock], [], [], timeout)
        if not readable:
            return []

        data = self.sock.recv(4096)
        if not data:
            raise ConnectionError("hub closed the socket")
        self.buffer += data

        messages = []
        while b"\r" in self.buffer:
            raw, self.buffer = self.buffer.split(b"\r", 1)
            if not raw or not raw.lstrip().startswith(b"{"):
                continue
            try:
                messages.append(json.loads(raw.decode("utf-8", "replace")))
            except json.JSONDecodeError:
                continue
        return messages

    def handle_message(self, message: dict[str, Any]) -> None:
        if "i" in message:
            if "e" in message:
                try:
                    error = base64.b64decode(message["e"]).decode("utf-8", "replace")
                except Exception:
                    error = str(message["e"])
                print(f"response {message['i']}: error {error}", flush=True)
            else:
                print(f"response {message['i']}: {message.get('r')}", flush=True)
            return

        if message.get("m") != 3 or not isinstance(message.get("p"), list):
            return

        payload = message["p"]
        if len(payload) < 2:
            return

        button, duration = payload[0], payload[1]
        if button not in ("left", "right"):
            return

        now = time.monotonic()
        is_release = isinstance(duration, (int, float)) and duration > 0

        should_rotate = False
        if not is_release:
            if not self.button_down[button] and now - self.last_action[button] > 0.25:
                should_rotate = True
                self.button_down[button] = True
        else:
            if not self.button_down[button] and now - self.last_action[button] > 0.25:
                should_rotate = True
            self.button_down[button] = False

        if not should_rotate:
            return

        self.last_action[button] = now
        direction = -1 if button == "left" else 1
        print(f"{button} button -> {'backward' if direction < 0 else 'forward'}", flush=True)
        self.rotate(direction)

    def run(self) -> None:
        self.connect()
        try:
            print(
                f"ready: right arrow = +{self.degrees} deg, left arrow = -{self.degrees} deg on motor {self.port}",
                flush=True,
            )
            while True:
                for message in self.read_messages():
                    self.handle_message(message)
        finally:
            self.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub", default=DEFAULT_HUB, help="Bluetooth address of the LEGO hub")
    parser.add_argument("--port", default="B", help="Motor port to drive")
    parser.add_argument("--degrees", type=int, default=90, help="Degrees per button press")
    parser.add_argument("--speed", type=int, default=40, help="Motor speed")
    parser.add_argument("--retry-delay", type=float, default=2.0, help="Seconds to wait before reconnecting")
    parser.add_argument("--once", action="store_true", help="Exit instead of reconnecting after an error")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    while True:
        controller = HubController(args.hub, args.port, args.degrees, args.speed)
        try:
            controller.run()
        except KeyboardInterrupt:
            print("stopped", flush=True)
            return 0
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr, flush=True)
            controller.close()
            if args.once:
                return 1
            print(f"reconnecting in {args.retry_delay:g}s", flush=True)
            time.sleep(args.retry_delay)


if __name__ == "__main__":
    raise SystemExit(main())
