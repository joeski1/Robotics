#!/usr/bin/env python
# this ros node is used to publish the face detection data to the face channel
import rospy
from exercise3.msg import face_msg
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import face
import time
import atexit

if __name__ == '__main__':

    rospy.init_node('face_publisher')
    face.open_cam()
    atexit.register(face.close_cam) # make sure to close nicely

    face_pub = rospy.Publisher('faces', face_msg, queue_size=100)
    r = rospy.Rate(5)
    while not rospy.is_shutdown():
        distances, angles = face.get_faces()
        #distances, angles = [1.0], [0.0]
        rospy.loginfo([(distances[i], angles[i]) for i in range(len(distances))])
        new_msg = face_msg()
        new_msg.distances = distances
        new_msg.angles = angles
        face_pub.publish(new_msg)
        r.sleep()

    face.close_cam()
