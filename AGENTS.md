# AGENTS.md

## Robot Hub Access

The LEGO hub used for this project is a LEGO MINDSTORMS Robot Inventor / SPIKE Prime compatible Technic Large Hub. It is reachable from the Raspberry Pi 3 over Bluetooth Classic RFCOMM, not over USB in the current setup.

The current Raspberry Pi OS setup is reachable over Tailscale as:

```sh
tailscale ssh max@pi3
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

USB did not enumerate in the original session. `lsusb` did not show a LEGO device and no `/dev/ttyACM*` appeared. Treat Bluetooth RFCOMM as the current working transport.

After replacing the failed SD card and installing fresh Raspberry Pi OS, Bluetooth had to be powered explicitly:

```sh
sudo systemctl restart bluetooth
sudo hciconfig hci0 up
sudo btmgmt power on
```

The same hub then paired over Bluetooth Classic with:

```sh
sudo btmgmt connectable on
sudo btmgmt fast-conn on
sudo btmgmt discov on
sudo btmgmt bondable on
sudo btmgmt pairable on
sudo btmgmt ssp on
sudo btmgmt io-cap 4
sudo timeout 25s btmgmt pair -c 4 -t 0 A8:E2:C1:9A:5D:04
```

On the fresh OS, rotating motor `B` by 90 degrees succeeded with:

```json
{"i":"mb90","m":"scratch.motor_run_for_degrees","p":{"port":"B","speed":40,"degrees":90,"stall":true,"stop":1}}
```

Telemetry confirmed the motor on `B`:

```text
ports {'B': [75, [0, 90, 85, 0]]}
```

For interactive button control, use `scripts/hub_button_motor_controller.py`. It listens to the hub's left/right arrow button events over the same RFCOMM JSON stream. The default mapping is:

```text
right arrow -> motor B +90 degrees
left arrow  -> motor B -90 degrees
```

Copy it to the Pi and run it detached:

```sh
tailscale ssh max@pi3 'mkdir -p ~/gemma4-robot/scripts'
tailscale ssh max@pi3 'cat > ~/gemma4-robot/scripts/hub_button_motor_controller.py' < scripts/hub_button_motor_controller.py
tailscale ssh max@pi3 'nohup python3 ~/gemma4-robot/scripts/hub_button_motor_controller.py --port B --degrees 90 --speed 40 > /tmp/hub_button_motor_controller.log 2>&1 & echo $! > /tmp/hub_button_motor_controller.pid'
```

Check it with:

```sh
tailscale ssh max@pi3 'tail -80 /tmp/hub_button_motor_controller.log; ps -fp $(cat /tmp/hub_button_motor_controller.pid)'
```

Observed working log:

```text
connected to A8:E2:C1:9A:5D:04 on RFCOMM channel 1
ready: right arrow = +90 deg, left arrow = -90 deg on motor B
left button -> backward
sent ctrl0001: motor B -90 degrees
response ctrl0001: 0
right button -> forward
sent ctrl0002: motor B +90 degrees
response ctrl0002: 0
```

Useful source references for this protocol:

- LEGO Robot Inventor hub API: `https://lego.github.io/MINDSTORMS-Robot-Inventor-hub-API/`
- `lego-hub-tk`: `https://github.com/smr99/lego-hub-tk`
- Tufts SPIKE Web Interface source: `https://tuftsceeo.github.io/SPIKE-Web-Interface/ServiceDock_SPIKE.js.html`

## micro:bit And ELECFREAKS Nezha Pro

The ELECFREAKS Nezha Pro carrier uses an inserted BBC micro:bit V2. Program the micro:bit with MicroPython by flashing a `.hex` through the Pi's USB connection. The verified Pi access path is:

```sh
tailscale ssh max@pi3
```

When the micro:bit is connected correctly, the Pi sees:

```text
/dev/ttyACM0
/dev/serial/by-id/usb-Arm_BBC_micro:bit_CMSIS-DAP_...-if01
USB mass-storage label: MICROBIT
```

Verify from the Mac with:

```sh
tailscale ssh max@pi3 'lsusb | grep -Ei "mbed|micro|0d28"; ls -l /dev/ttyACM0 /dev/serial/by-id/*micro* 2>/dev/null || true; lsblk -o NAME,TRAN,SIZE,FSTYPE,LABEL,MOUNTPOINTS'
```

Use `scripts/microbit/flash_microbit_via_pi.py` to build and flash MicroPython programs. It builds a `.hex` with `py2hex`, streams it to the Pi through `tailscale ssh max@pi3`, mounts the Pi's `MICROBIT` drive, and copies the file as `MICROBIT.HEX` for DAPLink flashing.

For ELECFREAKS PlanetX sensors that need MakeCode extensions, use the local MakeCode/PXT project under:

```text
scripts/microbit/makecode_planetx_readings/
```

Build it in a temporary PXT workspace and flash the micro:bit V2/CODAL hex, not the combined `binary.hex`. The combined MakeCode `binary.hex` is larger and DAPLink on `pi3` has reported:

```text
FAIL.TXT: error: The transfer timed out.
```

The smaller V2-only output that worked is:

```text
/tmp/gemma4-pxt/sensor-readings/built/mbcodal-binary.hex
```

You can flash an already-built MakeCode hex with the existing helper function:

```sh
python3 - <<'PY'
import importlib.util
from pathlib import Path

helper_path = Path("scripts/microbit/flash_microbit_via_pi.py").resolve()
spec = importlib.util.spec_from_file_location("flash_microbit_via_pi", helper_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.flash_hex(Path("/tmp/gemma4-pxt/sensor-readings/built/mbcodal-binary.hex"))
PY
```

Install the local build tool once if needed:

```sh
/usr/bin/python3 -m pip install --user uflash
```

Safe heart display test:

```sh
python3 scripts/microbit/flash_microbit_via_pi.py scripts/microbit/microbit_heart.py
```

Smoke test:

```sh
python3 scripts/microbit/flash_microbit_via_pi.py scripts/microbit/microbit_nezha_test.py
```

After flashing `microbit_nezha_test.py`, the micro:bit scrolls `PI`; micro:bit button A scrolls `A`, and button B scrolls `B`.

Nezha V2/Pro motor example:

```sh
python3 scripts/microbit/flash_microbit_via_pi.py scripts/microbit/microbit_nezha_v2_motor_buttons.py
```

That example assumes a safe smart motor/mechanism on Nezha motor port A/M1:

```text
micro:bit button A -> motor A clockwise 90 degrees at 30% speed
micro:bit button B -> motor A counterclockwise 90 degrees at 30% speed
micro:bit logo touch -> stop motor A
```

Do not flash the motor example unless the mechanism is safe to move.

The verified MicroPython REPL check over `/dev/ttyACM0` returned `42` for `print(6*7)`, confirming the flashed MicroPython runtime was active on the micro:bit.

### PlanetX Sensor Findings

The PlanetX MakeCode package maps Nezha RJ ports as:

```text
J1 analog -> micro:bit P1
J1 digital -> micro:bit P8
J2 analog -> micro:bit P2
J2 digital -> micro:bit P12
```

For the current sensor setup:

```text
J2: EF05030 CO2 sensor
J1: temperature sensor under investigation
```

Verified J2 CO2 readings over USB serial were stable. Observed examples:

```text
co2_raw=14..19
co2_value=1002..1013
```

`co2_value` is the ELECFREAKS package value (`1024 - analogRead(P2)`), not calibrated ppm.

The J1 temperature sensor did not behave like the documented EF05041 DS18B20 before the cable was reseated. Native `PlanetX_Basic.Ds18b20Temp(J1)` returned `-Infinity`, and a direct DS18B20 pin scan on P1/P2/P8/P12/P14/P16 also returned `-Infinity`. A DHT11-style PlanetX probe on J1 returned `0` temperature and `0` humidity. A clean analog-only J1 probe showed a real analog signal around:

```text
j1_p1_raw=631..643
millivolts=2035..2074
```

After reseating the J1 cable, `serial_alive_probe.ts` flashed as the V2/CODAL hex and printed `alive sample=...`, confirming the micro:bit USB serial path was healthy. The guarded V2/CODAL scan in `main.ts` then ran and printed progress through all protocol calls. Current post-reseat observations were:

```text
co2_raw=10..15
co2_value=1006..1012
j1_analog_p1=209..214
j1_digital_p8=0
dht11_temp_j1=0
dht11_humidity_j1=0
temp_j1=-Infinity
temp_p1/temp_p2/temp_p8/temp_p12/temp_p14/temp_p16=-Infinity
```

So J1 still does not decode as EF05041 DS18B20 or DHT11 after reseating. It does present a nonzero analog P1 signal, but no temperature conversion formula has been verified for that signal.

Reference files:

- `MICROBIT_NEZHA_PRO.md`
- `scripts/microbit/flash_microbit_via_pi.py`
- `scripts/microbit/microbit_heart.py`
- `scripts/microbit/microbit_nezha_test.py`
- `scripts/microbit/microbit_nezha_v2_motor_buttons.py`
- `scripts/microbit/makecode_planetx_readings/main.ts`
- `scripts/microbit/makecode_planetx_readings/analog_j1_temperature_probe.ts`
- `scripts/microbit/makecode_planetx_readings/serial_alive_probe.ts`

## AIY Voice Kit HAT

The same Pi reachable as `max@pi3` over Tailscale also has a Google AIY Voice Kit HAT/Bonnet-style audio board with dual microphones, a button, and a speaker. Use it only as local hardware. Do not use Google Assistant, Google Cloud Speech, or any cloud speech APIs unless explicitly requested.

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

## Pi 3B+ No-Boot Recovery Checklist

If `max@pi3` drops off Tailscale after adding the Voice Kit HAT and the Pi 3B+ shows only the red power LED with no HDMI output, do not assume the Pi is burned immediately. Diagnose in this order:

1. Power off and unplug everything.
2. Remove the AIY Voice Kit HAT/Bonnet completely from the 40-pin header.
3. Disconnect the HAT speaker, button, microphone board, LEGO hub USB cable, and any other USB devices.
4. Inspect the 40-pin header carefully. A one-pin offset HAT install can short 5V, 3V3, ground, or GPIO pins.
5. Boot the bare Pi only: Pi + known-good microSD + known-good 5V/2.5A power supply + HDMI monitor already powered on.
6. Watch both LEDs. On Pi 3B+, red is power. Green ACT should flash for SD-card activity. A solid red LED with no green ACT usually means the board is powered but is not reading/booting from the SD card.
7. Try a freshly imaged Raspberry Pi OS Lite card, preferably a different known-good microSD. Pi 3B+ needs boot files new enough for 3B+ hardware.
8. If green ACT flashes irregularly but HDMI is blank, then the Pi may be booting and the issue may be HDMI mode. Try another cable/monitor or add conservative HDMI settings to `/boot/config.txt`.
9. If the bare Pi will not show green ACT with multiple known-good SD cards and power supplies, measure rails with a multimeter: pin 2 or 4 to pin 6 should be about 5V; pin 1 to pin 6 should be about 3.3V.
10. If 3.3V is missing or the SoC gets hot within seconds, assume hardware damage and stop powering it.

Once the bare Pi boots again, shut down cleanly and reconnect the Voice Kit HAT only after confirming it is aligned on all 40 pins. Then re-test Tailscale and ALSA before attaching the LEGO hub or other peripherals.
