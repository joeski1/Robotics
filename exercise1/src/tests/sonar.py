#!/usr/bin/env python
#
# documentation:
# http://wiki.ros.org/APIs
# http://docs.ros.org/api/rospy/html/
# http://wiki.ros.org/common_msgs
# http://wiki.ros.org/p2os_driver?distro=groovy
#   messages in p2os_msgs.msg


import rospy
import roslib

# messages
from std_msgs.msg import Bool, Float32
# found from grepping /opt/ros/indigo that the message was sensor_msgs.msg/BatteryState
# but the p2os driver said differently in the error output
from p2os_msgs.msg import SonarArray


def sonar_callback(msg):
    # index 0-7 going left to right
    rospy.loginfo(msg.ranges)


def main():
    rospy.init_node('sonar')

    rospy.Subscriber('sonar', SonarArray, sonar_callback)

    rospy.spin()


if __name__ == '__main__':
    main()

