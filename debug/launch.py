#!/usr/bin/python3
from time import sleep
from pymouse import PyMouse
from pykeyboard import PyKeyboard

# Constants
command_interval = 0.8
gazebo_delay = 10
judge_delay = 3

m = PyMouse()
k = PyKeyboard()

# Open a new window
k.press_keys([k.control_key,'t'])
sleep(command_interval)
k.press_keys([k.windows_l_key,k.left_key])

# Launch package
k.type_string("roscd uav_sim && python ./scripts/modify.py && roslaunch uav_sim arena.launch")
k.tap_key(k.enter_key)
sleep(gazebo_delay)

# Open a new tab
m.click(int(m.screen_size()[0]/4), int(m.screen_size()[1]/2))
k.press_keys([k.shift_key,'t'])
sleep(command_interval)

# Launch judge
k.type_string("roscd uav_sim && rosrun uav_sim judge.py")
k.tap_key(k.enter_key)
sleep(judge_delay)
k.tap_key(k.enter_key)

# Open a new tab
m.click(int(m.screen_size()[0]/4), int(m.screen_size()[1]/2))
k.press_keys([k.shift_key,'t'])
sleep(command_interval)

# Launch cmd
k.type_string("roscd uav_sim && ./command.py")
k.tap_key(k.enter_key)
sleep(command_interval)

# Open a new tab
m.click(int(m.screen_size()[0]/4), int(m.screen_size()[1]/2))
k.press_keys([k.shift_key,'t'])
sleep(command_interval)

# Launch rviz
k.type_string("rosrun rviz rviz")
k.tap_key(k.enter_key)
sleep(command_interval)

# Return
k.press_keys([k.shift_key,k.left_key])
k.press_keys([k.shift_key,k.left_key])
k.press_keys([k.shift_key,k.left_key])

# Combine windows
m.click(int(2.2*m.screen_size()[0]/4), 40)
k.press_keys([k.windows_l_key,k.right_key])
