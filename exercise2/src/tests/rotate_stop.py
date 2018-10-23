#!/usr/bin/env python
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "../common"))
sys.path.append(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "../pf_localisation"))
import math
import movement
import rospy
from nav_msgs.msg import Odometry
from util import getHeading

def odometry_callback(odometry):
    print(math.degrees(getHeading(odometry.pose.pose.orientation)))

if __name__ == '__main__':
    rospy.init_node('movement', anonymous=True)
    rospy.Subscriber("/odom", Odometry, odometry_callback, queue_size=1)

    mov = movement.Movement()
    mov.debug = True

    while not rospy.is_shutdown():
        print('running again')
        mov.rotate_angle(math.radians(180))
        raw_input()
