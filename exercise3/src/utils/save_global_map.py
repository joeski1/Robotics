#!/usr/bin/env python
# node to save the global map when CTRL + C is pressed

import rospy
import sys
import os
from exercise3.msg import global_map

outdir = os.path.dirname(os.path.realpath(__file__))

def global_map_callback(msg):
    global last_msg
    print('Got global map {}'.format(msg.counter))
    last_msg = msg

if __name__ == '__main__':
    global last_msg
    last_msg = None
    rospy.init_node('save_global_map')
    rospy.Subscriber('global_map', global_map, global_map_callback, queue_size=1)
    try:
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            r.sleep()
    except KeyboardInterrupt:
        print('Saving map')
        msg = last_msg
        arr = last_msg.global_map
        xs = last_msg.cols
        ys = len(arr) / xs
        with open(outdir + '/global_map.pbm', 'w') as f:
            f.write("P2\n{} {}\n".format(xs, ys))
            for y in range(0, ys):
                for x in range(0, xs):
                    value = int(255 * arr[x + y * xs])
                    string = ("{} " if x + 1 < xs else "{}").format(value)
                    f.write(string)
                f.write("\n")
