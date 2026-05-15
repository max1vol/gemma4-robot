# Programming ELECFREAKS Nezha Pro from the Pi

The micro:bit V2 appears on the Pi as:

```text
/dev/ttyACM0
/dev/serial/by-id/usb-Arm_BBC_micro:bit_CMSIS-DAP_...-if01
USB mass storage label: MICROBIT
```

Use MicroPython for Python programs. The local workflow is:

```sh
python3 scripts/microbit/flash_microbit_via_pi.py scripts/microbit/microbit_nezha_test.py
```

That command:

1. runs `py2hex` locally to build a MicroPython `.hex`;
2. streams the `.hex` to the Pi via `tailscale ssh max@pi3`;
3. mounts the Pi's `MICROBIT` mass-storage device;
4. copies the file as `MICROBIT.HEX`, which makes DAPLink flash it.

The tested smoke program is:

```sh
python3 scripts/microbit/flash_microbit_via_pi.py scripts/microbit/microbit_nezha_test.py
```

After flashing, the micro:bit scrolls `PI`. Pressing micro:bit button A scrolls
`A`; pressing button B scrolls `B`.

The Nezha V2/Pro motor example is:

```sh
python3 scripts/microbit/flash_microbit_via_pi.py scripts/microbit/microbit_nezha_v2_motor_buttons.py
```

It assumes a smart motor is connected to Nezha motor port A/M1:

- micro:bit button A: move motor A clockwise 90 degrees at 30% speed
- micro:bit button B: move motor A counterclockwise 90 degrees at 30% speed
- micro:bit logo touch: stop motor A

Do not flash the motor example unless the mechanism is safe to move.

## MakeCode / PlanetX Sensor Programs

PlanetX sensors that rely on ELECFREAKS MakeCode APIs live in:

```text
scripts/microbit/makecode_planetx_readings/
```

Use MakeCode/PXT for these, not MicroPython bit-banging, when checking the EF05030 CO2 sensor or the EF05041 DS18B20 sensor.

The reliable flash artifact for the micro:bit V2 is the V2/CODAL hex:

```text
/tmp/gemma4-pxt/sensor-readings/built/mbcodal-binary.hex
```

Avoid flashing the larger combined MakeCode `binary.hex` through the Pi. DAPLink has reported this transient failure for it:

```text
error: The transfer timed out.
```

To build in the temporary PXT workspace:

```sh
rm -rf /tmp/gemma4-pxt/sensor-readings
mkdir -p /tmp/gemma4-pxt/sensor-readings
cp scripts/microbit/makecode_planetx_readings/main.ts scripts/microbit/makecode_planetx_readings/pxt.json /tmp/gemma4-pxt/sensor-readings/
cd /tmp/gemma4-pxt/sensor-readings
npx pxt install
npx pxt build
```

Then flash `built/mbcodal-binary.hex` with `flash_hex()` from `scripts/microbit/flash_microbit_via_pi.py`.

Current live findings:

- J2 CO2 sensor: reads on analog P2. Observed `co2_raw=14..19`, `co2_value=1002..1013`. `co2_value` is ELECFREAKS' inverse analog value, not ppm.
- J1 temperature sensor before cable reseat: did not respond as EF05041 DS18B20 (`-Infinity`), did not respond as DHT11 (`0` temperature / `0` humidity), but did show an analog P1 signal around raw `631..643`.
- J1 after cable reseat: V2/CODAL `serial_alive_probe.ts` printed successfully, proving USB serial still works. The guarded V2/CODAL full sensor scan in `main.ts` then printed through every protocol step. Current post-reseat readings were `j1_analog_p1=209..214`, `j1_digital_p8=0`, DHT11 temperature/humidity `0 / 0`, and DS18B20 `-Infinity` on J1 plus direct scan pins P1/P2/P8/P12/P14/P16. J1 still does not decode as EF05041 DS18B20 or DHT11.

Useful MakeCode source files:

- `scripts/microbit/makecode_planetx_readings/main.ts`: guarded CO2/J1 sensor scan
- `scripts/microbit/makecode_planetx_readings/analog_j1_temperature_probe.ts`: raw analog J1 probe
- `scripts/microbit/makecode_planetx_readings/serial_alive_probe.ts`: no-sensor serial sanity check
