from microbit import *

uart.init(baudrate=115200)

last = None
display.show("A")

while True:
    pressed = button_a.is_pressed()
    if pressed != last:
        uart.write("A:down\n" if pressed else "A:up\n")
        display.show(Image.YES if pressed else "A")
        last = pressed
    sleep(20)
