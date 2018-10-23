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
#from sensor_msgs.msg import


if __name__ == '__main__':
    rospy.init_node('NAME')
    #rospy.Subscriber('CHANNEL', ABC, SOME_callback)
    rospy.spin()

