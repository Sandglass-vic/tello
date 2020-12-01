#!/usr/bin/python3
from sys import argv
from time import sleep
from pymouse import PyMouse
from pykeyboard import PyKeyboard

if __name__ == "__main__":
    if(not (len(argv) == 1 or (len(argv) == 2 and argv[1] == "-m"))):
        print("Usage %s [-m]" % argv[0])
        exit(-1)

    # Constants
    command_interval = 1.5

    m = PyMouse()
    k = PyKeyboard()

    # Open a new window
    k.press_keys([k.control_key, 't'])
    sleep(command_interval)

    # Launch roscore
    k.type_string("roscore")
    k.tap_key(k.enter_key)

    # Open a new tab
    k.press_keys([k.shift_key, 't'])
    sleep(command_interval)

    # Launch state.py
    k.type_string("rosrun tello_control tello_state.py")
    k.tap_key(k.enter_key)

    # Open a new tab
    k.press_keys([k.shift_key, 't'])
    sleep(command_interval)

    # Launch control.py
    k.type_string("rosrun tello_control tello_control.py")
    k.tap_key(k.enter_key)

    # Open a new tab
    k.press_keys([k.shift_key, 't'])
    sleep(command_interval)

    k.type_string("rosrun tello_control manual.py")
    k.tap_key(k.enter_key)