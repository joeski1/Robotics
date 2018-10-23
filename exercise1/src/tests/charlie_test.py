#!/usr/bin/env python

import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import math
import rospy
import roslib

# Messages
from sensor_msgs.msg import LaserScan

# local modules
import movement


mov = None
laser_data = []

def log(x):
    rospy.loginfo(x)

def laser_callback(msg):
    global laser_data
    #laser_data = msg.scan_data


if __name__ == '__main__':
    rospy.init_node('charlies_node', anonymous=True)

    mov = movement.Movement()

    rospy.Subscriber('base_scan', LaserScan, laser_callback)

    #while not rospy.is_shutdown():
        # do stuff with laser_data
    mov.turn_angle(math.radians(-90), 1)
    while True:
        mov.turn_vel(1)
    mov.stop()
    mov.forward_distance(1.5, 1)
    while True:
        mov.forward_vel(1)
    mov.stop()

    rospy.spin()
