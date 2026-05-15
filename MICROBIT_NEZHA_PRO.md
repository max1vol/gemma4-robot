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
