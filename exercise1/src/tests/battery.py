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
from p2os_msgs.msg import BatteryState

battery_threshold = 10.0 # volts
pub = None

def battery_callback(msg):
    rospy.loginfo(msg)
    if msg.voltage < battery_threshold:
        pub.publish(True)


def main():
    global pub

    rospy.init_node('battery_monitor')

    rospy.Subscriber('battery_state', BatteryState, battery_callback)

    pub = rospy.Publisher('battery_low', Bool, queue_size='None')

    rospy.spin()


if __name__ == '__main__':
    main()

