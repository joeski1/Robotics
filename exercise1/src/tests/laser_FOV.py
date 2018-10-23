#!/usr/bin/env python
#
# documentation:
# http://wiki.ros.org/APIs
# http://docs.ros.org/api/rospy/html/
# http://wiki.ros.org/common_msgs


import rospy
import roslib

# Messages
#from nav_msgs.msg import
from sensor_msgs.msg import LaserScan

def laser_callback(msg):
    rospy.loginfo('far left: {}   far right: {}'.format(msg.ranges[511], msg.ranges[0]))


if __name__ == '__main__':
    rospy.init_node('laser_FOV')
    rospy.Subscriber('base_scan', LaserScan, laser_callback)
    rospy.spin()

