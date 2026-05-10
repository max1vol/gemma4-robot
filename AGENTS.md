# AGENTS.md

## Robot Hub Access

The LEGO hub used for this project is a LEGO MINDSTORMS Robot Inventor / SPIKE Prime compatible Technic Large Hub. It is reachable from the Raspberry Pi 3 over Bluetooth Classic RFCOMM, not over USB in the current setup.

Connect to the Pi with:

```sh
tailscale ssh pi@100.95.196.115
```

The hub Bluetooth address discovered in this session was:

```text
A8:E2:C1:9A:5D:04
```

The Pi has a working Bluetooth adapter and can pair with the hub when the hub is on and its blue Bluetooth ring is flashing. If the hub is not reachable, power it on, wait for boot, then press the Bluetooth button once so the blue ring flashes.

## Confirming Connectivity

On the Pi, scan for the hub:

```sh
sudo btmgmt power on
sudo btmgmt bredr on
sudo btmgmt le on
sudo btmgmt connectable on
sudo btmgmt bondable on
sudo btmgmt pairable on
sudo btmgmt ssp on
sudo btmgmt sc off
sudo btmgmt io-cap 4
sudo timeout 12s btmgmt find
```

The hub should appear as:

```text
LEGO Hub A8:E2:C1:9A:5D:04
```

Pair if needed:

```sh
sudo timeout 25s btmgmt pair -c 4 -t 0 A8:E2:C1:9A:5D:04
bluetoothctl info A8:E2:C1:9A:5D:04
```

The useful service is Bluetooth Classic Serial Port Profile on RFCOMM channel 1. The hub speaks line-oriented JSON messages terminated by carriage return (`\r`). This is the Robot Inventor/SPIKE uJSON-RPC runtime protocol.

Do not use EV3/NXT protocols. Do not use raw MicroPython REPL unless explicitly debugging the hub firmware. The reliable control path found in this session is JSON-RPC over RFCOMM channel 1.

## Python RFCOMM Helper

Use Python's Bluetooth socket support from the Pi:

```python
import json
import socket

HUB = "A8:E2:C1:9A:5D:04"

def send_rpc(message):
    sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
    sock.settimeout(8)
    sock.connect((HUB, 1))
    sock.sendall(json.dumps(message, separators=(",", ":")).encode("utf-8") + b"\r")
    sock.close()
```

For robust scripts, read telemetry for about a second after connecting before sending commands, then read until the response with the same `i` field appears.

## Display And Button Light

The 5x5 hub display is brightness-only, not RGB. To show a heart and make the center button red, send:

```json
{"i":"h101","m":"program_terminate"}
{"i":"h102","m":"scratch.display_clear"}
{"i":"h103","m":"scratch.display_image","p":{"image":"09090:99999:99999:09990:00900"}}
{"i":"h104","m":"scratch.center_button_lights","p":{"color":9}}
```

Color codes observed from the Robot Inventor hub API:

```text
0 off
3 blue
6 green
9 red
```

Example to set the center button blue:

```json
{"i":"b001","m":"scratch.center_button_lights","p":{"color":3}}
```

Example to set it green:

```json
{"i":"g001","m":"scratch.center_button_lights","p":{"color":6}}
```

## Motors

Motor commands that worked use:

```json
{"i":"m001","m":"scratch.motor_run_for_degrees","p":{"port":"A","speed":40,"degrees":90,"stall":true,"stop":1}}
```

Use `degrees: 90` for a quarter turn and `degrees: 360` for one full revolution.

In this session, the motor initially believed to be on `A` was reported by telemetry on `F`:

```text
ports F:dev=75,data=[0, 0, -5, 0]
```

After a 90 degree command to `F`, telemetry showed the position had changed to about 89:

```text
last non-empty port F [75, [0, 89, 83, 0]]
```

If a motor command returns success but the physical motor does not move, inspect the telemetry first and drive the port where the motor is actually reported.

## Working Command Patterns

Rotate port `B` a small amount:

```json
{"i":"mb01","m":"scratch.motor_run_for_degrees","p":{"port":"B","speed":40,"degrees":30,"stall":true,"stop":1}}
```

Rotate port `B` one revolution:

```json
{"i":"mb02","m":"scratch.motor_run_for_degrees","p":{"port":"B","speed":40,"degrees":360,"stall":true,"stop":1}}
```

Rotate port `A` by 90 degrees:

```json
{"i":"ma01","m":"scratch.motor_run_for_degrees","p":{"port":"A","speed":40,"degrees":90,"stall":true,"stop":1}}
```

## Notes From Discovery

USB did not enumerate in this session. `lsusb` did not show a LEGO device and no `/dev/ttyACM*` appeared. Treat Bluetooth RFCOMM as the current working transport.

Useful source references for this protocol:

- LEGO Robot Inventor hub API: `https://lego.github.io/MINDSTORMS-Robot-Inventor-hub-API/`
- `lego-hub-tk`: `https://github.com/smr99/lego-hub-tk`
- Tufts SPIKE Web Interface source: `https://tuftsceeo.github.io/SPIKE-Web-Interface/ServiceDock_SPIKE.js.html`

## AIY Voice Kit HAT

The same Pi at `100.95.196.115` also has a Google AIY Voice Kit HAT/Bonnet-style audio board with dual microphones, a button, and a speaker. Use it only as local hardware. Do not use Google Assistant, Google Cloud Speech, or any cloud speech APIs unless explicitly requested.

The official AIY Python library exposes audio as ALSA command wrappers. The important module is `aiy.voice.audio`; its `aplay()` and `arecord()` helpers build normal `aplay` and `arecord` commands, and its `play_wav()` / `record_file()` helpers call those commands. That means the reliable low-level interface is plain ALSA:

```sh
aplay -l
arecord -l
aplay -D default /path/to/effect.wav
arecord -D default -f S16_LE -r 44100 -c 2 -d 3 /tmp/test.wav
```

If the HAT is installed correctly, ALSA should expose a Google/AIY/Voice sound card. Historical device names for this board include strings like `snd_rpi_googlevoicehat_soundcard`, but always inspect the live Pi with `aplay -l`, `arecord -l`, and `/proc/asound/cards`.

The official AIY button API is `aiy.board.Board`. It uses GPIO, with defaults from the source:

```text
BUTTON_PIN = 23
LED_PIN = 25
```

Use `Board().button.wait_for_press()` or `button.when_pressed = callback` for the hardware button. For the LED, `Board().led` is the compatibility path that works with Voice HAT v1; `aiy.leds` is for newer Voice/Vision Bonnet RGB LEDs and is not compatible with the Voice HAT v1.

To play a local test effect without Google services, prefer ALSA directly. A safe quick test is a short generated WAV played through the default PCM:

```sh
python3 - <<'PY'
import math
import struct
import wave

path = "/tmp/voice-kit-test.wav"
rate = 44100
with wave.open(path, "wb") as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(rate)
    frames = []
    for i in range(int(rate * 0.45)):
        t = i / rate
        freq = 660 if t < 0.22 else 880
        amp = int(13000 * math.sin(2 * math.pi * freq * t))
        frames.append(struct.pack("<h", amp))
    wav.writeframes(b"".join(frames))
print(path)
PY
aplay -D default /tmp/voice-kit-test.wav
```

If `default` does not route to the HAT speaker, repeat with the concrete device from `aplay -l`, for example `plughw:0,0` or `plughw:1,0`.

Official references checked:

- Voice Kit guide: `https://aiyprojects.withgoogle.com/voice/`
- AIY Python API docs: `https://aiyprojects.readthedocs.io/`
- AIY source repository: `https://github.com/google/aiyprojects-raspbian`
- AIY audio source: `https://raw.githubusercontent.com/google/aiyprojects-raspbian/aiyprojects/src/aiy/voice/audio.py`
- AIY board/button source: `https://raw.githubusercontent.com/google/aiyprojects-raspbian/aiyprojects/src/aiy/board.py`
