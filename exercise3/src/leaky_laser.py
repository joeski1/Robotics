#!/usr/bin/env python
# node to add some noise to the laser readings

import rospy
import random
from sensor_msgs.msg import LaserScan

MAX_LASER = 5.75  # maximum range the laser can detect
SHOULD_EXTEND = 0.95  # the probability that a particular laser value should be extended
ONE_MINUS_SHOULD_EXTEND = 1.0 - SHOULD_EXTEND  # only calculate once
MAX_EXTEND = 1.5  # the maximum we should ever extend a single laser reading

def laser_callback(scan):
    '''callback function loops through each item in the range
    values provided by the laser scan. 1- SHOULD_EXTEND*100 % of the time we mess up our laser reading
    to provide a mimicked version of real life'''
    my_ranges = scan.ranges
    for i in range(len(my_ranges)):
        new_num = random.random()  # random number between 0 and 1
        if new_num >= SHOULD_EXTEND:
            # decide how much to add using random number generated
            to_add = ((new_num-SHOULD_EXTEND)/ONE_MINUS_SHOULD_EXTEND) * MAX_EXTEND
            my_ranges[i] = min(my_ranges[i] + to_add, MAX_LASER)  # cap the value at the range of our laser

    scan.ranges = my_ranges  # change the message to hold my updated ranges
    leaky_pub.publish(scan)

if __name__ == '__main__':
    rospy.init_node('leaky_laser')
    leaky_pub = rospy.Publisher('leaky_laser', LaserScan, queue_size=100)
    rospy.Subscriber('base_scan', LaserScan, laser_callback)
    rospy.spin()
