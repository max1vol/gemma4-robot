serial.redirectToUSB()
basic.showString("A")

let sample = 0

basic.forever(function () {
    sample += 1
    const raw = pins.analogReadPin(AnalogPin.P1)
    const millivolts = Math.round(raw * 3300 / 1023)
    const tmp36C = Math.round(((millivolts - 500) / 10) * 10) / 10
    const lm35C = Math.round((millivolts / 10) * 10) / 10

    serial.writeLine(
        "sample=" + sample +
        " j1_p1_raw=" + raw +
        " millivolts=" + millivolts +
        " tmp36_c_est=" + tmp36C +
        " lm35_c_est=" + lm35C
    )

    basic.pause(1000)
})
