#!/usr/bin/env python
import rospy
import math
import json
import os
import sys
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point, Quaternion, Vector3

outdir = os.path.dirname(os.path.realpath(__file__))

scans = []
odoms = []

if len(sys.argv) < 2:
    print('you must enter a prefix for the output files')
    sys.exit(0)
else:
    prefix = sys.argv[1]


def timestamp(t):
    return t.secs * int(10e9) + t.nsecs

def dictify(obj):
    try:
        obj.__slots__
    except AttributeError:
        # object was primitive
        return obj

    d = {}
    for attrib in obj.__slots__:
        d[attrib] = dictify(obj.__getattribute__(attrib))
    return d

def laser_callback(scan):
    global start
    time = dictify(rospy.get_rostime() - start)
    scan = dictify(scan)
    scan['time'] = time
    scans.append(scan)

def odometry_callback(odom):
    global start
    time = dictify(rospy.get_rostime() - start)
    odom = dictify(odom)
    odom['time'] = time
    odoms.append(odom)


if __name__ == '__main__':
    rospy.init_node('message_gather')


    raw_input('press enter to start recording')
    global start
    start = rospy.get_rostime()

    rospy.Subscriber('odom', Odometry, odometry_callback, queue_size=100)
    rospy.Subscriber('base_scan', LaserScan, laser_callback, queue_size=100)


    try:
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            print('{} lasers, {} odoms'.format(len(scans), len(odoms)))
            r.sleep()
    except KeyboardInterrupt:
        print('Ctrl+C caught')

    with open(os.path.join(outdir, prefix + 'scans.json'), 'w') as f:
        json.dump(scans, f)

    with open(os.path.join(outdir, prefix + 'odoms.json'), 'w') as f:
        json.dump(odoms, f)

    print('messages gathered in {} seconds'.format(timestamp(rospy.get_rostime()-start)/int(10e9)))
    print('files written')


