#!/usr/bin/env python
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "../common"))
import movement
import rospy
from nav_msgs.msg import Odometry

def odometry_callback(odometry):
    print(odometry.pose)

if __name__ == '__main__':
    rospy.init_node('movement', anonymous=True)
    rospy.Subscriber("/odom", Odometry, odometry_callback, queue_size=1)

    mov = movement.Movement()
    mov.debug = True

    while not rospy.is_shutdown():
        print('running again')
        finished = mov.forward_distance(4)
        if not finished:
            quit()
        raw_input()
