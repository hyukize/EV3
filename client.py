#!/usr/bin/env pybricks-micropython


from pybricks.hubs import EV3Brick
from pybricks.ev3devices import Motor, ColorSensor
from pybricks.parameters import Port, Direction
from pybricks.tools import wait
from pybricks.robotics import DriveBase
from pybricks.media.ev3dev import SoundFile
import usocket as socket

addr = ('192.168.1.2', 8000)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(socket.getaddrinfo(*addr)[0][-1])

ev3 = EV3Brick()
left_motor = Motor(Port.B, positive_direction=Direction.COUNTERCLOCKWISE)
right_motor = Motor(Port.C, positive_direction=Direction.COUNTERCLOCKWISE)

line_sensor = ColorSensor(Port.S2)
left_sensor = ColorSensor(Port.S3)
right_sensor = ColorSensor(Port.S1)

robot = DriveBase(left_motor, right_motor, wheel_diameter=56, axle_track=115)

BLACK = 4
WHITE = (38, 51, 29)

THRESHOLD = 50

DRIVE_SPEED = 80
PROPORTIONAL_GAIN = 0.3
direction = "down"


def turn(angle):
    global robot

    if angle == 90:
        robot.turn(50)
        while True:
            robot.drive(0, 50)
            line = int((line_sensor.reflection() - BLACK) / (WHITE[0] - BLACK) * 100)
            if line < 50:
                robot.stop()
                break

    elif angle == -90:
        robot.turn(-130)
        while True:
            robot.drive(0, 50)
            line = int((line_sensor.reflection() - BLACK) / (WHITE[0] - BLACK) * 100)
            if line < 50:
                robot.stop()
                break
    else:
        robot.turn(140)
        while True:
            robot.drive(0, 50)
            line = int((line_sensor.reflection() - BLACK) / (WHITE[0] - BLACK) * 100)
            if line < 50:
                robot.stop()
                break


def change_direction(target, current_dir):
    global robot
    dir_info = {"left": 0, "up": 1, "right": 2, "down": 3}
    tar = dir_info[target]
    c_dir = dir_info[current_dir]
    if c_dir != tar:
        diff = tar - c_dir
        if diff < 0:
            diff += 4
        if diff == 2:
            turn(180)
        elif diff == 1:
            turn(-90)
        else:
            turn(90)


def move():
    global THRESHOLD, BLACK, WHITE, PROPORTIONAL_GAIN, robot

    left = int((left_sensor.reflection() - BLACK) / (WHITE[1] - BLACK) * 100)
    right = int((right_sensor.reflection() - BLACK) / (WHITE[2] - BLACK) * 100)

    if left < 20 and right < 20:
        robot.straight(30)
    will_arrive = False
    arrived = False

    while True:
        line = int((line_sensor.reflection() - BLACK) / (WHITE[0] - BLACK) * 100)
        left = int((left_sensor.reflection() - BLACK) / (WHITE[1] - BLACK) * 100)
        right = int((right_sensor.reflection() - BLACK) / (WHITE[2] - BLACK) * 100)

        if left < 20 and right < 20:
            if not will_arrive:
                will_arrive = True
        elif will_arrive and left > 80 and right > 80:
            robot.stop()
            will_arrive = False
            arrived = True
        else:
            deviation = line - THRESHOLD
            turn_rate = int(PROPORTIONAL_GAIN * deviation)
            robot.drive(DRIVE_SPEED, turn_rate)

        if arrived:
            break


def collision():
    global robot
    robot.straight(50)
    robot.straight(-50)


while True:
    obs = False

    data = client.recv(1024).decode()
    print(data)

    if data == "end":
        ev3.speaker.beep()
        client.send(b"received")
        continue

    elif data[:3] == "obs":
        target_direction = data[4:]
        obs = True
    else:
        target_direction = data

    change_direction(target_direction, direction)
    direction = target_direction

    if obs:
        collision()
    else:
        move()

    client.send(b"done")
