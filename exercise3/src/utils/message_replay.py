#!/usr/bin/env python
import rospy
from collections import deque
import math
import json
import os
import sys
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovariance, Pose, Point, Quaternion, TwistWithCovariance, Twist, Vector3
from std_msgs.msg import Header

indir = os.path.dirname(os.path.realpath(__file__))
prefix = '' if len(sys.argv) < 2 else sys.argv[1]

scans = []
odoms = []
next_laser = None
next_odom = None
laser_start = 0
odom_start = 0

def get_msg_class(name, parent_name):
    if name == 'LaserScan':
        return LaserScan
    elif name == 'Odometry':
        return Odometry
    elif name == 'header':
        return Header
    elif name == 'stamp':
        return rospy.Time
    elif name == 'pose':
        if parent_name == 'pose':
            return Pose
        else:
            return PoseWithCovariance
    elif name == 'twist':
        if parent_name == 'twist':
            return Twist
        else:
            return TwistWithCovariance
    elif name == 'position':
        return Point
    elif name == 'orientation':
        return Quaternion
    elif name == 'linear' or name == 'angular':
        return Vector3
    else:
        assert False




def undictify(dic, name=None, parent_name=None):
    if not isinstance(dic, dict): # is a primitive
        return dic

    obj = get_msg_class(name, parent_name)()

    for attr, val in dic.iteritems():
        setattr(obj, attr, undictify(val, attr, name))

    return obj

def timestamp(t):
    return t.secs * int(10e9) + t.nsecs

def msg_time(msg):
    return timestamp(msg[1])

def try_publish(next_val, q, publisher, message_start, now):
    if next_val is not None and now >= msg_time(next_val)-message_start:
        publisher.publish(next_val[0])
        if len(q) > 0:
            return q.popleft(), True
        else:
            return None, True
    return next_val, False


def json_load(filename, msg_name):
    with open(os.path.join(indir, prefix + filename), 'r') as f:
        data = json.load(f)

    timestamps = [undictify(x['time'], 'stamp') for x in data]

    # remove custom timestamps
    for msg in data:
        del msg['time']

    objects = [undictify(x, msg_name) for x in data]
    #scans.sort(key=msg_time)
    return deque(zip(objects, timestamps))


if __name__ == '__main__':
    rospy.init_node('message_replay')

    odom_pub = rospy.Publisher('odom', Odometry, queue_size=10)
    laser_pub = rospy.Publisher('base_scan', LaserScan, queue_size=10)

    scans = json_load('scans.json', 'LaserScan')
    odoms = json_load('odoms.json', 'Odometry')

    print('{} lasers and {} odoms ready to replay'.format(len(scans), len(odoms)))
    raw_input('press enter to start')

    next_laser = scans.popleft()
    next_odom = odoms.popleft()
    laser_start = msg_time(next_laser)
    odom_start = msg_time(next_odom)

    start = rospy.get_rostime()
    r = rospy.Rate(500)
    while not rospy.is_shutdown():
        now = rospy.get_rostime() - start
        now = now.secs * int(10e9) + now.nsecs

        next_laser, sent_laser = try_publish(next_laser, scans, laser_pub, laser_start, now)
        next_odom,  sent_odom  = try_publish(next_odom, odoms, odom_pub, odom_start, now)

        if sent_laser or sent_odom:
            print('{} lasers and {} odoms left to replay'.format(len(scans), len(odoms)))

        if next_laser is None and next_odom is None:
            rospy.signal_shutdown('done')

        r.sleep()

    print('replay complete in {} seconds'.format(now / int(10e9)))


