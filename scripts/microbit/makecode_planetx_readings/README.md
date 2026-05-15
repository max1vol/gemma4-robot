# PlanetX sensor readings

MakeCode/PXT project for reading ELECFREAKS PlanetX sensors on the Nezha Pro with a micro:bit V2.

## Files

- `main.ts`: guarded scan for J2 CO2 plus J1 temperature protocol checks.
- `analog_j1_temperature_probe.ts`: raw analog-only J1 probe.
- `serial_alive_probe.ts`: no-sensor serial heartbeat used to prove flashing and USB serial.
- `pxt.json`: MakeCode project metadata and ELECFREAKS `pxt-PlanetX` dependency.

## Build and flash

Build in a temporary PXT workspace:

```sh
rm -rf /tmp/gemma4-pxt/sensor-readings
mkdir -p /tmp/gemma4-pxt/sensor-readings
cp scripts/microbit/makecode_planetx_readings/main.ts scripts/microbit/makecode_planetx_readings/pxt.json /tmp/gemma4-pxt/sensor-readings/
cd /tmp/gemma4-pxt/sensor-readings
npx pxt install
npx pxt build
```

Flash the V2/CODAL output, not the combined `binary.hex`:

```text
/tmp/gemma4-pxt/sensor-readings/built/mbcodal-binary.hex
```

The combined `binary.hex` has timed out during DAPLink copy on `pi3`.

## Port mapping

From ELECFREAKS `pxt-PlanetX`:

```text
J1 analog -> P1
J1 digital -> P8
J2 analog -> P2
J2 digital -> P12
```

## Observed readings

J2 EF05030 CO2 produced stable readings:

```text
co2_raw=14..19
co2_value=1002..1013
```

`co2_value` is the ELECFREAKS package value (`1024 - analogRead(P2)`), not calibrated ppm.

Before the J1 cable was reseated, the temperature sensor did not respond as the expected EF05041 DS18B20:

```text
PlanetX_Basic.Ds18b20Temp(J1) -> -Infinity
direct DS18B20 scan P1/P2/P8/P12/P14/P16 -> -Infinity
DHT11 J1 temperature/humidity -> 0 / 0
analog J1 P1 -> raw 631..643, about 2.04..2.07 V
```

After reseating the J1 cable:

- `serial_alive_probe.ts` flashed as `mbcodal-binary.hex` and printed `alive sample=...`.
- The guarded full sensor scan in `main.ts` flashed as `mbcodal-binary.hex` and printed through all protocol checks.
- Current J2 readings were `co2_raw=10..15`, `co2_value=1006..1012`.
- Current J1 readings were `j1_analog_p1=209..214`, `j1_digital_p8=0`, DHT11 temperature/humidity `0 / 0`, DS18B20 J1 `-Infinity`, and direct DS18B20 scan on P1/P2/P8/P12/P14/P16 all `-Infinity`.
- J1 still does not decode as EF05041 DS18B20 or DHT11 after reseating. It has a nonzero analog signal, but no temperature conversion has been verified.
