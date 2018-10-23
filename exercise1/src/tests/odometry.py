#!/usr/bin/env python

import rospy
import roslib

# Messages
from nav_msgs.msg import Odometry
#from geometry_msgs.msg import Point, Quaternion



def odometry_callback(msg):
    print msg.pose.pose
    print msg.twist.twist


if __name__ == '__main__':
    rospy.init_node('odometry', anonymous=True)
    rospy.Subscriber('odom', Odometry, odometry_callback)
    rospy.spin()

