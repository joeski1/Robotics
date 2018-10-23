#!/usr/bin/env python
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import rospy
import threading
import time
import numpy as np
import math

from exercise3.msg import global_map, pose_update, global_map_request, navigation

class World:

    def __init__(self):
        self.map = None
        self.pose_msg = None
        rospy.Subscriber('global_map', global_map, self._map_callback)
        rospy.Subscriber('pose_update', pose_update, self._pose_callback)

        self._update_lock = threading.Lock()
        self._need_update = False
        self.request_pub = rospy.Publisher('global_map_request', global_map_request, queue_size='None')
        self.navigation_pub = rospy.Publisher('navigation', navigation, queue_size='None')

    def publish_path(self, path):
        xs = map(lambda (x, y): x, path)
        ys = map(lambda (x, y): y, path)
        self.navigation_pub.publish(navigation(xs, ys))

    def _pose_callback(self, pose_msg):
        self.pose_msg = pose_msg


    def _map_callback(self, map_msg):
        if self._need_update:
            self.map = map_msg
            self._need_update = False


    def update(self):
        with self._update_lock:
            self._need_update = True
            while self._need_update and not rospy.is_shutdown():
                start_time = time.time()
                self.request_pub.publish(global_map_request())
                while time.time() - start_time < 2 and self._need_update and not rospy.is_shutdown():
                    time.sleep(0)
            if rospy.is_shutdown():
                sys.exit(0)


    def get_prob(self, coord):
        if self.map is None:
            raise Exception("Cannot view a map without a map")
        (x, y) = coord
        maxX = self.map.cols
        maxY = len(self.map.global_map) / maxX

        # TODO: Allow no limits
        if x < 0 or y < 0 or x >= maxX or y >= maxY:
            return 1

        return self.map.global_map[x + y * maxX]


    def pose_grid(self, pose_msg=None):
        if pose_msg is None:
            if self.map is None:
                raise Exception("Cannot get a pose without a map")
            pose_msg = self.map.pose
        return self.real_to_grid(self.pose_real(pose_msg=pose_msg))


    def pose_real(self, pose_msg=None):
        if pose_msg is None:
            pose_msg = self.pose_msg
        return (pose_msg.x, pose_msg.y, pose_msg.rotation)


    def grid_to_real(self, grid_coord):
        return (real_x, real_y, None)


    def real_to_grid(self, real_coord):
        real_x, real_y, _ = real_coord
        res = self.map.resolution
        return (int(round(real_x / res)), int(round(real_y / res)))

    def pose_rotation(self, pose_msg = None):
        if pose_msg is None:
            pose_msg = self.pose_msg
        return (3 * math.pi - pose_msg.rotation) % (2 * math.pi) # Undoing Matt's fix
