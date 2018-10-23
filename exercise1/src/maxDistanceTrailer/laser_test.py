#!/usr/bin/env python
#file to test the laser simplifier

import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "../common"))


import rospy
from laser_simplifier import LaserFiveWays
from sensor_msgs.msg import LaserScan

if __name__ == "__main__":
    rospy.init_node("laser_testing_node",anonymous=True)
    laser = LaserFiveWays()
    rate = rospy.Rate(2)
    while not rospy.is_shutdown():
        results = laser.getFiveDirection()
        if results is not None:
            #rospy.loginfo("North: " +  str(results[0]))
            dirs = ['East','North East', 'North', 'North West', 'West']
            for i in range(0,5):
                rospy.loginfo(dirs[i] + ": " + str(results[i]))
        rate.sleep()
