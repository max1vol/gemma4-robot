#!/usr/bin/env python3
"""Upload a hub-side slow arrow-button motor controller.

Run this on the Raspberry Pi that has the LEGO Technic Large Hub connected over
USB serial. It writes a Python program to a hub slot and starts it.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import select
import string
import sys
import termios
import time
from typing import Any


DEFAULT_DEVICE = "/dev/serial/by-id/usb-LEGO_System_A_S_LEGO_Technic_Large_Hub_in_FS_Mode_334C396A3338-if00"


PROGRAM_BODY = r'''
import hub

motor = hub.port.A.motor
speed = 20
state = 0

while True:
    left = hub.button.left.is_pressed()
    right = hub.button.right.is_pressed()

    target = 0
    if left and not right:
        target = -speed
    elif right and not left:
        target = speed

    if target != state:
        if target == 0:
            motor.brake()
        else:
            motor.run_at_speed(target)
        state = target

    yield 20
'''


def build_program(body: str) -> str:
    body = body.strip("\n").replace("\t", "    ")
    indented = "".join(f"    {line}\n" for line in body.splitlines())
    return (
        "from runtime import VirtualMachine\n\n"
        "# Stack for execution:\n"
        "async def stack_1(vm, stack):\n"
        f"{indented}"
        "# Setup for execution:\n"
        "def setup(rpc, system, stop):\n\n"
        "    # Initialize VM:\n"
        "    vm = VirtualMachine(rpc, system, stop, \"Target__1\")\n\n"
        "    # Register stack on VM:\n"
        "    vm.register_on_start(\"stack_1\", stack_1)\n\n"
        "    return vm"
    )


class HubSerial:
    def __init__(self, path: str) -> None:
        self.path = path
        self.fd: int | None = None
        self.buffer = b""

    def open(self) -> None:
        fd = os.open(self.path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        attrs = termios.tcgetattr(fd)
        attrs[0] = 0
        attrs[1] = 0
        attrs[2] = attrs[2] | termios.CLOCAL | termios.CREAD
        attrs[3] = 0
        attrs[4] = termios.B115200
        attrs[5] = termios.B115200
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
        self.fd = fd

    def close(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def send(self, message: dict[str, Any]) -> str | None:
        if self.fd is None:
            raise RuntimeError("serial device is not open")
        payload = json.dumps(message, separators=(",", ":")).encode("utf-8") + b"\r"
        os.write(self.fd, payload)
        return message.get("i")

    def read_messages(self, timeout: float) -> list[dict[str, Any]]:
        if self.fd is None:
            raise RuntimeError("serial device is not open")
        readable, _, _ = select.select([self.fd], [], [], timeout)
        if not readable:
            return []

        self.buffer += os.read(self.fd, 4096)
        messages: list[dict[str, Any]] = []
        while b"\r" in self.buffer:
            raw, self.buffer = self.buffer.split(b"\r", 1)
            if not raw.strip().startswith(b"{"):
                continue
            try:
                messages.append(json.loads(raw.decode("utf-8", "replace")))
            except json.JSONDecodeError:
                continue
        return messages

    def wait_for_id(self, message_id: str, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for message in self.read_messages(0.2):
                if message.get("m") == "runtime_error":
                    raise RuntimeError(f"hub runtime error: {message}")
                if message.get("i") == message_id:
                    return message
        raise TimeoutError(f"timed out waiting for response {message_id}")


def make_id() -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(random.choice(alphabet) for _ in range(4))


def rpc(hub: HubSerial, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"i": make_id(), "m": method}
    if params is not None:
        message["p"] = params
    message_id = hub.send(message)
    assert message_id is not None
    response = hub.wait_for_id(message_id, 8.0)
    if "e" in response:
        raise RuntimeError(f"{method} failed: {response}")
    return response


def upload_program(hub: HubSerial, name: str, slot: int, program: str) -> None:
    if program.encode("ascii", "strict").decode("ascii") != program:
        raise ValueError("hub Python upload must be ASCII")

    now_ms = int(time.time() * 1000)
    start = rpc(
        hub,
        "start_write_program",
        {
            "meta": {
                "created": now_ms,
                "modified": now_ms,
                "name": base64.b64encode(name.encode("ascii")).decode("ascii"),
                "type": 0,
                "project_id": random.randint(1000, 999999),
            },
            "fname": name,
            "size": len(program.encode("utf-8")),
            "slotid": slot,
        },
    )
    details = start.get("r") or {}
    blocksize = int(details["blocksize"])
    transferid = str(details["transferid"])

    remaining = program
    while remaining:
        chunk = remaining[:blocksize]
        remaining = remaining[blocksize:]
        rpc(
            hub,
            "write_package",
            {
                "data": base64.b64encode(chunk.encode("ascii")).decode("ascii"),
                "transferid": transferid,
            },
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--slot", type=int, default=2)
    parser.add_argument("--name", default="A Slow Arrows")
    parser.add_argument("--speed", type=int, default=20)
    args = parser.parse_args()

    body = PROGRAM_BODY.replace("speed = 20", f"speed = {args.speed}")
    program = build_program(body)

    hub = HubSerial(args.device)
    try:
        hub.open()
        print(f"connected: {args.device}", flush=True)
        try:
            rpc(hub, "program_terminate")
        except Exception as exc:
            print(f"warning: program_terminate failed: {exc}", file=sys.stderr, flush=True)
        upload_program(hub, args.name, args.slot, program)
        print(f"uploaded {args.name!r} to slot {args.slot}", flush=True)
        rpc(hub, "program_execute", {"slotid": args.slot})
        print(f"started slot {args.slot}", flush=True)

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            for message in hub.read_messages(0.2):
                print(json.dumps(message, separators=(",", ":")), flush=True)
    finally:
        hub.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
