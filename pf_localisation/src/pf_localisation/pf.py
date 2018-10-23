#!/usr/bin/env python
from geometry_msgs.msg import Pose, PoseArray, Quaternion, Point, PoseWithCovarianceStamped
from pf_base import PFLocaliserBase
import math
import rospy
from numpy.random import vonmises
from util import rotateQuaternion, getHeading
import random

import time


class PFLocaliser(PFLocaliserBase):
    def __init__(self):
        # Call the superclass constructor
        super(PFLocaliser, self).__init__()

        # Set motion model parameters
        self.ODOM_ROTATION_NOISE = 2  # Odometry model rotation noise
        self.ODOM_TRANSLATION_NOISE = 2  # Odometry x axis (forward) noise
        self.ODOM_DRIFT_NOISE = 2  # Odometry y axis (side-side) noise
        # Sensor model parameters
        self.NUMBER_PREDICTED_READINGS = 20  # Number of readings to predict

        # Generating particles
        self.NUMBER_PARTICLES = 100
        self.POSITION_STANDARD_DEVIATION = 2
        self.ORIENTATION_STANDARD_DEVIATION = 2

    def point_to_cell(self, point):
        """
        Converts a point into a cell location
        :Args:
            |  point (geometry_msgs.msg.Point) point using meter units
        :Return:
            | (int, int) cell location on map
        """
        res = self.occupancy_map.info.resolution
        return (int(math.floor(point.x / res)), int(math.floor(point.y / res)))

    def is_valid_position(self, point):
        """
        Checks whether a point is a valid location for the robot on the map
        :Args:
            |  point (geometry_msgs.msg.Point) point using meter units
        :Return:
            | (boolean) whether or not the robot can be in that position
        """
        if point is None:
            return False
        (x, y) = self.point_to_cell(point)
        width = self.occupancy_map.info.width
        value = self.occupancy_map.data[x * width + y]  # data is row major TODO: Check operation is correct
        return value != 0  # TODO: Check cells around the point to see if it there is an obstacle nearby

    def initialise_particle_cloud(self, initial_pose):
        """
        Set particle cloud to initial pose plus noise
        """
        # Easy access constants
        psd = self.POSITION_STANDARD_DEVIATION
        osd = self.ORIENTATION_STANDARD_DEVIATION

        pos_array = PoseArray()

        # For each particle
        for x in range(0, self.NUMBER_PARTICLES):
            # Create a new pose
            pos = Pose(position=None)

            # Find a random valid position from the initial position
            init_pos = initial_pose.pose.pose.position
            while self.is_valid_position(pos.position):
                pos.position = Point(random.gauss(init_pos.x, psd),
                                     random.gauss(init_pos.y, psd), init_pos.z)

            # Rotate heading a random amount
            init_ori = initial_pose.pose.pose.orientation
            rotate_amount = vonmises(getHeading(initial_pose.orientation.y), osd)
            pos.orientation = rotateQuaternion(init_ori, rotate_amount)

            # Add to the pose array
            pos_array.poses.append(pos)
        return pos_array

    def update_particle_cloud(self, scan):
        # Update particle cloud, given map and laser scan
        likelihoods = []
        for p in self.particlecloud.poses:
            likelihoods.append((p, self.sensor_model.get_weight(scan, p)))
            # TODO: Finish this

    def estimate_pose(self):
        """
        Create new estimated pose, given particle cloud
        E.g. just average the location and orientation values of each of
        the particles and return this.

        Better approximations could be made by doing some simple clustering,
        e.g. taking the average location of half the particles after
        throwing away any which are outliers
        """
        return Pose()  # TODO: This function
