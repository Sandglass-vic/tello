#!/usr/bin/python3
from subprocess import run

while True:
    cmd = "".join(map(str,input("[Command] = ")))
    run(f"rostopic pub /tello/cmd_string/ std_msgs/String \"{cmd}\" -1", shell=True)
