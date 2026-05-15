#!/usr/bin/env python3
"""Build a micro:bit MicroPython .hex locally and flash it via the Pi USB port."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


PI = "max@pi3"
SSH = ["tailscale", "ssh", PI]
PY2HEX_CANDIDATES = [
    Path.home() / "Library/Python/3.9/bin/py2hex",
    Path("/usr/local/bin/py2hex"),
    Path("/usr/bin/py2hex"),
]


def run(command: list[str], *, input_text: str | None = None) -> None:
    subprocess.run(command, input=input_text, text=True, check=True)


def upload(local: Path, remote: str) -> None:
    with local.open("rb") as handle:
        subprocess.run(
            [*SSH, f"cat > {shlex.quote(remote)}"],
            stdin=handle,
            check=True,
        )


def get_pi_password() -> str:
    secret = Path.home() / ".maxpi3"
    for line in secret.read_text(encoding="utf-8").splitlines():
        if line.startswith("PASSWORD="):
            return line.split("=", 1)[1]
    raise RuntimeError(f"PASSWORD= not found in {secret}")


def find_py2hex() -> Path:
    for path in PY2HEX_CANDIDATES:
        if path.exists():
            return path
    raise RuntimeError("py2hex not found; install with: /usr/bin/python3 -m pip install --user uflash")


def build_hex(source: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    py2hex = find_py2hex()
    run([str(py2hex), "-o", str(out_dir), str(source)])
    return out_dir / f"{source.stem}.hex"


def flash_hex(hex_path: Path) -> None:
    remote_hex = f"/home/max/{hex_path.name}"
    upload(hex_path, remote_hex)
    password = get_pi_password()
    remote_command = f"""set -eu
sudo -S -p "" sh -c {shlex.quote(f'''
mkdir -p /mnt/microbit
umount /mnt/microbit 2>/dev/null || true
mount -t vfat /dev/sda /mnt/microbit
cp {remote_hex} /mnt/microbit/MICROBIT.HEX
sync
umount /mnt/microbit
''')}
"""
    run([*SSH, remote_command], input_text=password + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="MicroPython source file for micro:bit")
    parser.add_argument("--out-dir", type=Path, default=Path("out"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    hex_path = build_hex(source, args.out_dir.resolve())
    flash_hex(hex_path)
    print(f"Flashed {hex_path} via {PI}")


if __name__ == "__main__":
    main()
