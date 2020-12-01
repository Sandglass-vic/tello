#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("/home/thudrone/catkin_ws/src/tello_control/lib/yolov3_detect")

from tools import getPackagePath
from tello_detect import Detector
from collections import deque
from json import load
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



y_max_th = 200
y_min_th = 170

tello_img = None
tello_state = 'mid:-1;x:100;y:100;z:-170;mpry:1,180,1;pitch:0;roll:0;yaw:-19;'
feedback = ["None", "None"]
tello_state_lock = threading.Lock()
tello_img_lock = threading.Lock()

# subscribe tello_state and tello_image


class info_updater():
    def __init__(self):
        rospy.Subscriber("tello_state", String, self.update_state)
        rospy.Subscriber("tello_image", Image, self.update_tello_img)
        rospy.Subscriber('command_feedback', String, self.update_feedback)
        self.con_thread = threading.Thread(target=rospy.spin)
        self.con_thread.start()

    def update_state(self, data):
        global tello_state, tello_state_lock
        tello_state_lock.acquire()  # thread locker
        tello_state = data.data
        tello_state_lock.release()
        # print(tello_state)

    def update_tello_img(self, data):
        global tello_img, tello_img_lock
        tello_img_lock.acquire()  # thread locker
        tello_img = CvBridge().imgmsg_to_cv2(data, desired_encoding="passthrough")
        tello_img_lock.release()
        # print(tello_img)

    def update_feedback(self, data):
        global feedback
        feedback = data.data.split("!")
        # print(feedback)


class image_show():
    def __init__(self):
        self.thread = threading.Thread(target=self.show_tello_img)
        self.thread.start()

    def show_tello_img(self):
        global tello_img, tello_img_lock
        while True:
            tello_img_lock.acquire()
            if type(tello_img) != type(None):
                cv2.imshow("tello_image", tello_img)
                cv2.waitKey(2)
            tello_img_lock.release()
            time.sleep(0.05)


class Controller():
    class FlightState():
        WAITING = 0
        NAVIGATING = 1
        # Fire pos
        DETECTING_WINDOW = 2
        # Balls
        DETECTING_BALL = 3
        LANDING = 4
        FINISHED = 5

    def __init__(self):
        # Command Publisher
        self.ctrl_ = rospy.Publisher('command', String, queue_size=10)

        # The pose of drone in the world's coordinate system

        # For navigation
        self.dimension_map_ = {"x": 0, "y": 1, "z": 2, "yaw": 3}
        self.dimensions_ = ["x", "y", "z", "yaw"]
        self.allowed_pose_error_ = [10, 10, 8, 5]  # [x, y, z, yaw]
        self.max_pose_adjustment_ = [100, 100, 100, 30]  # [x, y, z, yaw]
        self.min_pose_adjustment_ = [20, 20, 20, 0]  # [x, y, z, yaw]
        self.nav_nodes_ = None  # a deque of target poses
        self.current_pose_ = dict()
        self.next_nav_pose_ = {"x": None, "y": None, "z": 100, "yaw": 0}
        self.next_nav_dimension_ = None
        self.commands_ = [["forward ", "back "], [
            "left ", "right "], ["up ", "down "], ["cw ", "ccw "]]
        self.cmd_queue_ = deque()
        self.flight_state_ = self.FlightState.WAITING
        self.next_state_ = None
        # Load routes
        self.nav_route_ = load(
            open(getPackagePath('tello_control') + "/scripts/route.json", "r"))
        for key in self.nav_route_:
            self.nav_route_[key] = map(deque, self.nav_route_[key])

        # For detection
        self.detector = Detector()  # Time-consuming
        self.bridge_ = CvBridge()
        self.dpos_index_ = -1
        self.win_index_ = -1
        self.ball_index_ = -1
        # Answers
        self.detected_ball_num__ = 0
        self.ball_type_ = ['e', 'e', 'e', 'e', 'e']

        # Main
        rospy.loginfo("Ready~")
        while not rospy.is_shutdown():
            self.decide()
        rospy.logwarn('Controller node shut down.')
        # self.ctrl_.publish("mon")
        # while True:
        #     self.updateCurrentPose()
        #     time.sleep(1)

    # Publications
    def publishCommand(self):
        while self.cmd_queue_:
            command_str = self.cmd_queue_.popleft()
            msg = String()
            msg.data = command_str
            rospy.logfatal("Command: " + command_str +
                           ", waiting for feedback")
            time_start = time.time()
            while True:
                self.ctrl_.publish(msg)
                while feedback == None or feedback[0] != command_str:
                    time.sleep(0.1)
                    if time.time() - time_start > 10:
                        self.ctrl_.publish("land")
                        time.sleep(2)
                        exit(-1)
                info = "Feedback received.(" + " ".join(feedback) + ")"
                rospy.loginfo(info)
                if feedback[1] == "True":
                    break
            rospy.logfatal("Command: " + command_str +
                           " finished.")

    def publishAnswer(self):
        pass

    # Navigation
    def updateCurrentPose(self):
        global tello_state, tello_state_lock
        tello_state_lock.acquire()
        statestr = tello_state.split(';')
        for item in statestr:
            if 'mid:' in item:
                mid = int(item.split(':')[-1])
                self.current_pose_['mid'] = mid
            elif 'x:' in item:
                x = int(item.split(':')[-1])
                self.current_pose_['x'] = x
            elif 'z:' in item:
                z = int(item.split(':')[-1])
                self.current_pose_['z'] = z
            elif 'mpry:' in item:
                mpry = item.split(':')[-1]
                mpry = mpry.split(',')
                self.current_pose_['mpry'] = [
                    int(mpry[0]), int(mpry[1]), int(mpry[2])]
            # y can be recognized as mpry, so put y first
            elif 'y:' in item:
                y = int(item.split(':')[-1])
                self.current_pose_['y'] = y
            elif 'pitch:' in item:
                pitch = int(item.split(':')[-1])
                self.current_pose_['pitch'] = pitch
            elif 'roll:' in item:
                roll = int(item.split(':')[-1])
                self.current_pose_['roll'] = roll
            elif 'yaw:' in item:
                yaw = int(item.split(':')[-1])
                self.current_pose_['yaw'] = yaw
        tello_state_lock.release()
        rospy.loginfo("Current pose: " + str(self.current_pose_))
        rospy.loginfo("Next nav pose: " + str(self.next_nav_pose_))

    def updatePoseError(self, dimension):
        assert(dimension >= 0 and dimension <= 3)
        dimension_str = self.dimensions_[dimension]
        if dimension == 3:  # Yaw
            yaw_error = self.current_pose_[
                dimension_str] - self.next_nav_pose_[dimension_str]
            if abs(yaw_error) > 180:
                sig = -1 if yaw_error > 0 else 1
                yaw_error += sig * 360
            return int(yaw_error)
        else:  # x, y, z
            return int(self.next_nav_pose_[dimension_str] - self.current_pose_[dimension_str])

    def adjustPose(self):
        adjust_order = deque([2, 0, 1])  # [z, yaw, x, y]
        if self.next_nav_dimension_:
            adjust_order.remove(self.next_nav_dimension_)
            adjust_order.appendleft(self.next_nav_dimension_)
        while True:
            adjust_times = 0
            for dimension in adjust_order:
                self.updateCurrentPose()
                error = self.updatePoseError(dimension)
                if abs(error) > self.allowed_pose_error_[dimension]:
                    cmd = self.commands_[dimension][0 if error > 0 else 1] + \
                        str(max(self.min_pose_adjustment_[dimension], min(self.max_pose_adjustment_[
                            dimension], abs(error))))
                    adjust_times += 1
                    self.cmd_queue_.append(cmd)
                    self.publishCommand()
            if adjust_times == 0:
                return

    def switchNavigatingState(self):
        if self.nav_nodes_ == None or len(self.nav_nodes_) == 0:
            self.flight_state_ = self.next_state_
            # Log info
            info_vector = ["WAITING", "NAVIGATING",
                           "DETECTING_WINDOW", "DETECTING_BALL", "LANDING", "FINISHED"]
            rospy.logfatal(
                "State change->{}".format(info_vector[self.flight_state_]))
            # End log info
        else:
            next_nav_node_ = self.nav_nodes_.popleft()
            self.flight_state_ = self.FlightState.NAVIGATING
            if len(next_nav_node_) == 1:
                # Directly publish a str command
                self.cmd_queue_.append(next_nav_node_[0])
                self.publishCommand()
                self.switchNavigatingState()
            else:
                # Modify target pose
                self.next_nav_dimension_ = self.dimension_map_[
                    next_nav_node_[0]]
                self.next_nav_pose_[next_nav_node_[0]] = next_nav_node_[1]
                if self.next_nav_pose_["yaw"] == 0:
                    self.commands_ = [["forward ", "back "], [
                        "left ", "right "], ["up ", "down "], ["cw ", "ccw "]]
                elif self.next_nav_pose_["yaw"] == 90:
                    self.commands_ = [["right ", "left "], [
                        "forward ", "back "], ["up ", "down "], ["cw ", "ccw "]]
                elif self.next_nav_pose_["yaw"] == -90:
                    self.commands_ = [["left ", "right "], [
                        "back ", "forward "], ["up ", "down "], ["cw ", "ccw "]]
                elif self.next_nav_pose_["yaw"] == 180:
                    self.commands_ = [["back ", "forward "], [
                        "right ", "left "], ["up ", "down "], ["cw ", "ccw "]]

    def decide(self):
        if self.flight_state_ == self.FlightState.WAITING:
            # Wait until our drone successfully takes off
            self.cmd_queue_.append('mon')
            self.cmd_queue_.append('takeoff')
            self.publishCommand()

            self.updateCurrentPose()
            while(self.current_pose_["mid"] < 0):
                self.updateCurrentPose()
                rospy.loginfo("Blanket not found!")
                time.sleep(0.2)
            rospy.logfatal("Find blanket!")

            self.next_nav_pose_['x'] = self.current_pose_['x']
            self.next_nav_pose_['y'] = self.current_pose_['y']

            # Change route
            self.dpos_index_ = 0
            self.nav_nodes_ = self.nav_route_[
                "window_detect_pos"][self.dpos_index_]
            rospy.loginfo("Flying to detect position" + str(self.dpos_index_))
            self.next_state_ = self.FlightState.DETECTING_WINDOW

        elif self.flight_state_ == self.FlightState.LANDING:
            self.publishAnswer()
            self.cmd_queue_.append('land')
            self.publishCommand()
            self.nav_nodes_ = None
            self.next_state_ = self.FlightState.FINISHED

        elif self.flight_state_ == self.FlightState.FINISHED:
            return

        else:
            # Being always navigating
            self.adjustPose()

            if self.flight_state_ == self.FlightState.DETECTING_WINDOW:
                cv2.imwrite("/home/thudrone/catkin_ws/src/tello_control/detect_position_" +
                            str(self.dpos_index_) + ".jpg", tello_img)
                raw_index = self.detector.detectWindow(tello_img)
                if raw_index == 0:
                    rospy.logfatal("Window not found.")
                    if self.dpos_index_ == 0:
                        # Continue detecting
                        self.dpos_index_ += 1
                        self.nav_nodes_ = self.nav_route_[
                            "window_detect_pos"][self.dpos_index_]
                        rospy.loginfo(
                            "Flying to detect position" + str(self.dpos_index_))
                        self.next_state_ = self.FlightState.DETECTING_WINDOW
                    else:
                        rospy.logfatal(
                            "Window detection failed. Landing......")
                        self.nav_nodes_ = None
                        self.next_state_ = self.FlightState.LANDING
                else:
                    # Target detected
                    self.win_index_ = self.dpos_index_ * 4 + raw_index - 1
                    rospy.logfatal("Window index = " +
                                   str(self.win_index_) + "\n")
                    self.nav_nodes_ = self.nav_route_[
                        "through_window"][self.win_index_]
                    self.next_state_ = self.FlightState.DETECTING_BALL

            elif self.flight_state_ == self.FlightState.DETECTING_BALL:
                self.nav_nodes_ = None
                self.next_state_ = self.FlightState.LANDING
                """  self.detectObject()
                if self.ball_index_ == 4:
                    self.nav_nodes_ = self.nav_route_["normal_land"][0]
                    self.next_state_ = self.FlightState.LANDING

                elif self.detected_ball_num__ == 3:
                    self.nav_nodes_ = self.nav_route_["pre_land"][0]
                    self.next_state_ = self.FlightState.LANDING

                else:
                    self.updateBallIndex()
                    self.next_state_ = self.FlightState.DETECTING_BALL
                    self.nav_nodes_ = self.nav_route_[
                        "detect_next_ball"][self.ball_index_] """
        self.switchNavigatingState()


if __name__ == '__main__':
    rospy.init_node('tello_control', anonymous=True)
    rospy.logwarn('Controller node set up.')
    info_updater()
    # Show img
    image_show()
    Controller()
