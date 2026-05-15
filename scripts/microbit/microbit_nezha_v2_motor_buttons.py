from microbit import *


NEZHA_V2_ADDR = 0x10


class Motor:
    A = 1
    B = 2
    C = 3
    D = 4


class Direction:
    CLOCKWISE = 1
    COUNTERCLOCKWISE = 2


class NezhaV2:
    def __init__(self):
        i2c.init()

    def start(self, motor, speed):
        if speed < -100:
            speed = -100
        if speed > 100:
            speed = 100
        direction = Direction.CLOCKWISE if speed >= 0 else Direction.COUNTERCLOCKWISE
        i2c.write(
            NEZHA_V2_ADDR,
            bytearray([0xFF, 0xF9, motor, direction, 0x60, abs(speed), 0xF5, 0x00]),
        )

    def stop(self, motor):
        i2c.write(
            NEZHA_V2_ADDR,
            bytearray([0xFF, 0xF9, motor, 0x00, 0x5F, 0x00, 0xF5, 0x00]),
        )

    def move_degrees(self, motor, speed, direction, degrees):
        if speed < 0:
            speed = 0
        if speed > 100:
            speed = 100
        if degrees < 0:
            degrees = -degrees
        self.set_servo_speed(speed)
        high = (degrees >> 8) & 0xFF
        low = degrees & 0xFF
        i2c.write(
            NEZHA_V2_ADDR,
            bytearray([0xFF, 0xF9, motor, direction, 0x70, high, 2, low]),
        )

    def set_servo_speed(self, speed):
        value = max(0, min(100, speed)) * 9
        high = (value >> 8) & 0xFF
        low = value & 0xFF
        i2c.write(
            NEZHA_V2_ADDR,
            bytearray([0xFF, 0xF9, 0x00, 0x00, 0x77, high, 0x00, low]),
        )


nezha = NezhaV2()
display.scroll("NEZHA")

while True:
    if button_a.was_pressed():
        display.show("A")
        nezha.move_degrees(Motor.A, 30, Direction.CLOCKWISE, 90)
    if button_b.was_pressed():
        display.show("B")
        nezha.move_degrees(Motor.A, 30, Direction.COUNTERCLOCKWISE, 90)
    if pin_logo.is_touched():
        display.show("S")
        nezha.stop(Motor.A)
    sleep(50)
