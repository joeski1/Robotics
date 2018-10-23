#!/usr/bin/env python

import rospy
from sensor_msgs.msg import LaserScan

# Calculate min reading between i - clean_spread and i + clean_spread
def laser_callback(raw_laser):
    clean_spread = 10

    raw_ranges = raw_laser.ranges
    clean_ranges = []
    for i in range(0, len(raw_ranges)):
        for adjust in range(-clean_spread, clean_spread + 1):
            clean_ranges[i] = raw_ranges[i]
            if (0 <= i + adjust and i + adjust < len(raw_ranges)):
                clean_ranges[i] = min(clean_ranges[i], raw_ranges[i + clean_spread])

    pub.publish(LaserScan(raw_laser.header,
              raw_laser.angle_min,
              raw_laser.angle_max,
              raw_laser.angle_increment,
              raw_laser.time_increment,
              raw_laser.scan_time,
              raw_laser.range_min,
              raw_laser.range_max))

if __name__ == '__main__':
    global pub
    pub = rospy.Publisher('base_scan_clean', LaserScan, queue_size=10)
    rospy.init_node('base_scan_clean')
    rospy.Subscriber('base_scan', LaserScan, laser_callback)
