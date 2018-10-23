#!/usr/bin/env python
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "../pf_localisation"))

import time
import pf
import rospy
from geometry_msgs.msg import ( PoseStamped, PoseWithCovarianceStamped,
                                PoseArray, Quaternion )

a = pf.PFLocaliser()

def cloud_callback(cloud):
    #print(str(cloud.poses))
    a.particlecloud = cloud
    start = time.clock()
    print('estimating pose')
    p = a.estimate_pose()
    print('done. Took {} seconds'.format(time.clock()-start))
    # p is a ROS pose with a quaternion but estimate_pose() prints a human
    # readable pose prediction as well
    print(p)
    #rospy.signal_shutdown("Done")

if __name__ == "__main__":
    rospy.init_node("record_poses")
    rospy.Subscriber("/particlecloud", PoseArray, cloud_callback, queue_size=1)
    rospy.spin()
