serial.redirectToUSB()
basic.showString("T")

let sample = 0

function presence(pin: DigitalPin): number {
    pins.setPull(pin, PinPullMode.PullUp)
    pins.digitalWritePin(pin, 0)
    control.waitMicros(600)
    pins.digitalWritePin(pin, 1)
    control.waitMicros(30)
    const ack = pins.digitalReadPin(pin)
    control.waitMicros(600)
    return ack
}

function temp(pin: DigitalPin): number {
    return DS18B20.Ds18b20Temp(DS18B20.ValType.DS18B20_temperature_C, pin)
}

basic.forever(function () {
    sample += 1
    serial.writeLine("sample=" + sample + " begin")

    const co2Raw = pins.analogReadPin(AnalogPin.P2)
    const co2Value = PlanetX_Basic.gasValue(PlanetX_Basic.GasList.Co2, PlanetX_Basic.AnalogRJPin.J2)
    serial.writeLine("sample=" + sample + " co2_raw=" + co2Raw + " co2_value=" + co2Value)

    serial.writeLine(
        "sample=" + sample +
        " p1_ack=" + presence(DigitalPin.P1) + " p1_temp=" + temp(DigitalPin.P1) +
        " p8_ack=" + presence(DigitalPin.P8) + " p8_temp=" + temp(DigitalPin.P8) +
        " p12_ack=" + presence(DigitalPin.P12) + " p12_temp=" + temp(DigitalPin.P12) +
        " p14_ack=" + presence(DigitalPin.P14) + " p14_temp=" + temp(DigitalPin.P14) +
        " p16_ack=" + presence(DigitalPin.P16) + " p16_temp=" + temp(DigitalPin.P16) +
        " end"
    )

    basic.pause(1500)
})
