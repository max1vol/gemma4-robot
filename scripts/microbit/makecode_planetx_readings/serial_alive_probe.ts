serial.redirectToUSB()
basic.showString("S")

let sample = 0

basic.forever(function () {
    sample += 1
    serial.writeLine("alive sample=" + sample)
    basic.pause(1000)
})
