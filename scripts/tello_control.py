#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tello_base as tello
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
import rospy
import numpy as np
import random
import threading
import time

import sys
sys.path.append("/home/thudrone/catkin_ws/src/tello_control/lib/yolov3_detect")
from json import load
from collections import deque
from tello_detect import Detector
from tools import getPackagePath

# if you can not find cv2 in your python, you can try this. usually happen when you use conda.
# sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')

y_max_th = 200
y_min_th = 170

img = None
tello_state = 'mid:-1;x:100;y:100;z:-170;mpry:1,180,1;pitch:0;roll:0;yaw:-19;'
tello_state_lock = threading.Lock()
img_lock = threading.Lock()

# subscribe tello_state and tello_image


class info_updater():
    def __init__(self):
        rospy.Subscriber("tello_state", String, self.update_state)
        rospy.Subscriber("tello_image", Image, self.update_img)
        self.con_thread = threading.Thread(target=rospy.spin)
        self.con_thread.start()

    def update_state(self, data):
        global tello_state, tello_state_lock
        tello_state_lock.acquire()  # thread locker
        tello_state = data.data
        tello_state_lock.release()
        # print(tello_state)

    def update_img(self, data):
        global img, img_lock
        img_lock.acquire()  # thread locker
        img = CvBridge().imgmsg_to_cv2(data, desired_encoding="passthrough")
        img_lock.release()
        # print(img)


# put string into dict, easy to find
def parse_state():
    global tello_state, tello_state_lock
    tello_state_lock.acquire()
    statestr = tello_state.split(';')
    print (statestr)
    dict = {}
    for item in statestr:
        if 'mid:' in item:
            mid = int(item.split(':')[-1])
            dict['mid'] = mid
        elif 'x:' in item:
            x = int(item.split(':')[-1])
            dict['x'] = x
        elif 'z:' in item:
            z = int(item.split(':')[-1])
            dict['z'] = z
        elif 'mpry:' in item:
            mpry = item.split(':')[-1]
            mpry = mpry.split(',')
            dict['mpry'] = [int(mpry[0]), int(mpry[1]), int(mpry[2])]
        # y can be recognized as mpry, so put y first
        elif 'y:' in item:
            y = int(item.split(':')[-1])
            dict['y'] = y
        elif 'pitch:' in item:
            pitch = int(item.split(':')[-1])
            dict['pitch'] = pitch
        elif 'roll:' in item:
            roll = int(item.split(':')[-1])
            dict['roll'] = roll
        elif 'yaw:' in item:
            yaw = int(item.split(':')[-1])
            dict['yaw'] = yaw
    tello_state_lock.release()
    return dict


def showimg():
    global img, img_lock
    img_lock.acquire()
    cv2.imshow("tello_image", img)
    cv2.waitKey(2)
    img_lock.release()

class task_handle():
    class taskstages():
        fly_through_window = 1
        finished = 6  # task done signal

    def __init__(self, ctrl):
        self.delay = 4
        self.States_Dict = None
        self.ctrl = ctrl
        self.dectector = Detector()
        self.now_stage = self.taskstages.fly_through_window
        self.nav_route_ = load(
            open(getPackagePath('tello_control') + "/scripts/route.json", "r"))
        for key in self.nav_route_:
            self.nav_route_[key] = map(deque, self.nav_route_[key])
        self.rospy.Publisher('command', String, queue_size=1)
        ctrl.mon()
        time.sleep(2)
        ctrl.takeoff()
        time.sleep(2)
        ctrl.up(100)
        time.sleep(4)

        # Main
        while not (self.now_stage == self.taskstages.finished):
            if(self.now_stage == self.taskstages.fly_through_window):
                self.fly_through_window()
        self.ctrl.land()
        print("Task Done!")

    # Navigation
    def adjustPose(self):
        adjust_order = deque([2, 3, 0, 1])  # [z, yaw, x, y]
        if self.next_nav_dimension_:
            adjust_order.remove(self.next_nav_dimension_)
            adjust_order.appendleft(self.next_nav_dimension_)
        while True:
            self.updateCurrentPose()
            adjust_times = 0
            for dimension in adjust_order:
                error = self.updatePoseError(dimension)
                if abs(error) > self.allowed_pose_error_[dimension]:
                    cmd = self.commands_[dimension][0 if error > 0 else 1] + \
                        str(min(self.max_pose_adjustment_[
                            dimension], abs(error)))
                    adjust_times += 1
                    self.cmd_queue_.append(cmd)
                    self.publishCommand()
                    time.sleep(1.0)
            if adjust_times == 0:
                return

    def switchNavigatingState(self):
        if self.nav_nodes_ == None or len(self.nav_nodes_) == 0:
            self.flight_state_ = self.next_state_
            # Log info
            info_vector = ["WAITING", "NAVIGATING",
                           "DETECTING_TARGET", "DETECTING_OBJECT", "LANDING", "FINISHED"]
            rospy.logfatal(
                "State change->{}".format(info_vector[self.flight_state_.value]))
            # End log info
        else:
            next_nav_node_ = self.nav_nodes_.popleft()
            self.flight_state_ = self.FlightState.NAVIGATING
            if len(next_nav_node_) == 1:
                # Directly publish a str command
                self.cmd_queue_.append(next_nav_node_[0])
                self.publishCommand()
                time.sleep(2.0)
                self.switchNavigatingState()
            else:
                # Modify target pose
                self.next_nav_dimension_ = self.dimension_map_[
                    next_nav_node_[0]]
                self.next_nav_pose_[
                    self.next_nav_dimension_] = next_nav_node_[1]
                if self.next_nav_pose_[3] == 90:
                    self.commands_ = [["right ", "left "], [
                        "forward ", "back "], ["up ", "down "], ["cw ", "ccw "]]
                elif self.next_nav_pose_[3] == 0:
                    self.commands_ = [["forward ", "back "], [
                        "left ", "right "], ["up ", "down "], ["cw ", "ccw "]]
                elif self.next_nav_pose_[3] == -90:
                    self.commands_ = [["left ", "right "], [
                        "back ", "forward "], ["up ", "down "], ["cw ", "ccw "]]
                elif self.next_nav_pose_[3] == 180:
                    self.commands_ = [["back ", "forward "], [
                        "right ", "left "], ["up ", "down "], ["cw ", "ccw "]]

    def navigate(self, route):
        print("Enter navigation.")
        while len(route) > 0:
            self.States_Dict = parse_state()
            node = route.pop()
            if node[0] == "x":
                diff = node[1] - self.States_Dict["x"] 
                if diff > 0:
                    self.ctrl.forward(diff)
                else:
                    self.ctrl.back(abs(diff))
            elif node[0] == "y":
                diff = node[1] - self.States_Dict["y"]
                if diff > 0:
                    self.ctrl.left(diff)
                else:
                    self.ctrl.right(abs(diff))
            elif node[0] == "z":
                diff = node[1] - self.States_Dict["z"]
                if diff > 0:
                    self.ctrl.up(diff)
                else:
                    self.ctrl.down(abs(diff))
            print("Command published")
            time.sleep(self.delay)

    def fly_through_window(self):
        assert (self.now_stage == self.taskstages.fly_through_window)
        window_index = None
        for dpos_index in range(2):
            print("Flying to detect position" + str(dpos_index))
            self.navigate(self.nav_route_["window_dectect_pos"][dpos_index])
            raw_index = self.dectector.detectWindow(img)
            print("==============================\n tello_state = " + tello_state)
            if raw_index == 0:
                print("Window not found.")
                continue
            else:
                window_index = dpos_index*4 + raw_index - 1 
                print("Window index = " + str(window_index) + "\n")
                break
        # Fail to recognize target
        if window_index == None:
            self.now_stage = self.taskstages.finished
            return
        # Fly to the right window
        self.navigate(self.nav_route_["window_pos"][window_index])

if __name__ == '__main__':
    rospy.init_node('tello_control', anonymous=True)

    info_updater()
    task_handle()
