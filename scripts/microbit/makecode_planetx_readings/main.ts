serial.redirectToUSB()
basic.showString("R")

let sample = 0

//% shim=dstemp::celsius
function ds18b20C(pin: DigitalPin): number {
    return -9999
}

basic.forever(function () {
    sample += 1
    serial.writeLine("sample=" + sample + " begin")

    const co2Raw = pins.analogReadPin(AnalogPin.P2)
    const co2Value = PlanetX_Basic.gasValue(PlanetX_Basic.GasList.Co2, PlanetX_Basic.AnalogRJPin.J2)
    serial.writeLine("sample=" + sample + " co2_raw=" + co2Raw + " co2_value=" + co2Value)

    const j1AnalogP1 = pins.analogReadPin(AnalogPin.P1)
    const j1DigitalP8 = pins.digitalReadPin(DigitalPin.P8)
    serial.writeLine("sample=" + sample + " j1_analog_p1=" + j1AnalogP1 + " j1_digital_p8=" + j1DigitalP8)

    serial.writeLine("sample=" + sample + " dht11_j1_before")
    const dht11TempJ1 = PlanetX_Basic.dht11Sensor(PlanetX_Basic.DigitalRJPin.J1, PlanetX_Basic.DHT11_state.DHT11_temperature_C)
    const dht11HumidityJ1 = PlanetX_Basic.dht11Sensor(PlanetX_Basic.DigitalRJPin.J1, PlanetX_Basic.DHT11_state.DHT11_humidity)
    serial.writeLine("sample=" + sample + " dht11_temp_j1=" + dht11TempJ1 + " dht11_humidity_j1=" + dht11HumidityJ1)

    serial.writeLine("sample=" + sample + " ds18b20_j1_before")
    const tempJ1 = PlanetX_Basic.Ds18b20Temp(PlanetX_Basic.DigitalRJPin.J1, PlanetX_Basic.ValType.DS18B20_temperature_C)
    serial.writeLine("sample=" + sample + " temp_j1=" + tempJ1)

    serial.writeLine("sample=" + sample + " ds18b20_pin_scan_before")
    const tempP1 = ds18b20C(DigitalPin.P1)
    const tempP2 = ds18b20C(DigitalPin.P2)
    const tempP8 = ds18b20C(DigitalPin.P8)
    const tempP12 = ds18b20C(DigitalPin.P12)
    const tempP14 = ds18b20C(DigitalPin.P14)
    const tempP16 = ds18b20C(DigitalPin.P16)

    serial.writeLine(
        "sample=" + sample +
        " temp_p1=" + tempP1 +
        " temp_p2=" + tempP2 +
        " temp_p8=" + tempP8 +
        " temp_p12=" + tempP12 +
        " temp_p14=" + tempP14 +
        " temp_p16=" + tempP16 +
        " end"
    )

    basic.pause(1500)
})
