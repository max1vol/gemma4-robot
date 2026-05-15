from microbit import *


display.scroll("PI")

while True:
    if button_a.was_pressed():
        display.scroll("A")
    if button_b.was_pressed():
        display.scroll("B")
    sleep(50)
