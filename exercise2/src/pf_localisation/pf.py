#!/usr/bin/env python
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    ".."))

from geometry_msgs.msg import Pose, PoseArray, Quaternion, Point, PoseWithCovarianceStamped
from numpy.random.mtrand import vonmises

from pf_base import PFLocaliserBase
import math
import rospy
import tf  # for converting to and from quaternions
import DBSCAN

from util import rotateQuaternion, getHeading
from random import random, randrange, gauss, shuffle
import time

import resampler
import kld_sampler
import noisifier

class PFLocaliser(PFLocaliserBase):
    def __init__(self):
        # Call the superclass constructor
        super(PFLocaliser, self).__init__()

        # Set motion model parameters
        self.ODOM_ROTATION_NOISE = 0.006081294 		# Odometry model rotation noise
        self.ODOM_TRANSLATION_NOISE = 0.007881588 	# Odometry x axis (forward) noise
        self.ODOM_DRIFT_NOISE = 0.03284 #32.84 	# Odometry y axis (side-side) noise

        # Sensor model parameters
        self.NUMBER_PREDICTED_READINGS = 50  # Number of readings to predict

        # Generating particles
        self.USE_INITIAL_POSE = False
        self.NUMBER_PARTICLES = 500
        self.POSITION_STANDARD_DEVIATION = 2 * 2
        self.ORIENTATION_STANDARD_DEVIATION = 5

        # Resampler
        self.USE_KLD_SAMPLER = True

        # Percentage of our worst particles (inside the map) are placed randomly (set to 0.0 or negative to disable)
        self._MOVE_BAD_PARTICLES = 0.01

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
        if point is None: return False
        (x, y) = self.point_to_cell(point)
        width = self.occupancy_map.info.width
        height = self.occupancy_map.info.height
        index = x + y * width

        # data is row major TODO: Check operation is correct
        value = -1 if index > len(self.occupancy_map.data) or x >= width or x < 0 or y >= height or y < 0 else self.occupancy_map.data[index]
        return value != -1  # TODO: Check cells around the point to see if it there is an obstacle nearby

    def generate_random_pose(self):
        """
        Generates a random pose on the map
        :Return:
            | (geometry_msgs.msg.Pose) the random pose on the map
        """
        res = self.occupancy_map.info.resolution
        width = int(self.occupancy_map.info.width * res) + 1
        height = int(self.occupancy_map.info.height * res) + 1

        # Choose a random position
        pos = Pose()
        position = None
        while not self.is_valid_position(position):
            position = Point(random() * width, random() * height, 0)

        pos.position = position

        # Choose a random heading
        pos.orientation = Quaternion(w=1.0) #initial_pose.pose.pose.orientation
        rotate_amount = randrange(0, 360)
        pos.orientation = rotateQuaternion(pos.orientation, math.radians(rotate_amount))
        return pos

    def initialise_particle_cloud(self, initial_pose):
        """
        Set particle cloud to initial pose plus noise
        """
        # Easy access constants
        psd = self.POSITION_STANDARD_DEVIATION
        osd = self.ORIENTATION_STANDARD_DEVIATION

        pos_array = PoseArray()

        if self.USE_INITIAL_POSE:

            # For each particle
            init_pos = initial_pose.pose.pose.position
            init_ori = initial_pose.pose.pose.orientation
            for x in range(0, self.NUMBER_PARTICLES):
                # Create a new pose
                pos = Pose()

                # Find a random valid position from the inital position
                position = None
                while not self.is_valid_position(position):
                    position = Point(gauss(init_pos.x, psd),
                                     gauss(init_pos.y, psd), 0)
                pos.position = position

                # Rotate heading a random amount
                heading = getHeading(init_ori)
                rotate_amount = vonmises(heading, osd) - heading
                pos.orientation = rotateQuaternion(init_ori, rotate_amount)

                # Add to the pose array
                pos_array.poses.append(pos)
        else:
            for x in range(0, self.NUMBER_PARTICLES):
                pos = self.generate_random_pose()
                pos_array.poses.append(pos)

        return pos_array

    def update_particle_cloud(self, scan):
        # Update particle cloud, given map and laser scan
        #add noise to action model

        i = 0
        likelihoods = []

        for p in self.particlecloud.poses:
            newp = noisifier.noisifier(p)  # add noise due to action model

            # Check whether we are a valid pose
            generateParticle = random() <= 0.025
            if not self.is_valid_position(newp.position) and generateParticle:
                newp = self.generate_random_pose()

            likelihood = self.sensor_model.get_weight(scan, newp) if self.is_valid_position(newp.position) else -1
            likelihoods.append((newp, likelihood))

        if self._MOVE_BAD_PARTICLES > 0.0:
            likelihoods = sorted(likelihoods, lambda (x, a), (y, b): cmp(a, b))
            without_negatives = [item for item in likelihoods if item[1] >= 0.0]
            #print("a: %2.6f, b: %2.6f" % (without_negatives[0][1], without_negatives[len(without_negatives) - 1][1]))
            moveTotal = int(0.01 * math.pow(len(without_negatives), 2) * self._MOVE_BAD_PARTICLES)
            print("Moving %d particles" % moveTotal)
            for i in range(0, moveTotal):
                likelihoods.remove(without_negatives[i])
                likelihoods.append((self.generate_random_pose(), self.sensor_model.get_weight(scan, newp)))

        if self.USE_KLD_SAMPLER:
            self.particlecloud.poses = kld_sampler.kld_sample(likelihoods)
            print("Number of particles %d" % (len(self.particlecloud.poses)))
        else:
            self.particlecloud.poses = resampler.resample(likelihoods)

    def estimate_pose(self, draw_stuff=True, print_stuff=True):
        """
        Create new estimated pose, given particle cloud
        E.g. just average the location and orientation values of each of
        the particles and return this.

        Better approximations could be made by doing some simple clustering,
        e.g. taking the average location of half the particles after
        throwing away any which are outliers
        """

        # just in case the tuning made it worse
        use_old_values = True

        # pose sampling: take a random sample of N poses and throw away the
        # rest. Useful when working with the navstack as it uses a maximum of
        # 5000 particles by default.
        do_sampling = True
        max_poses = 500
        ros_poses = self.particlecloud.poses

        if do_sampling:
            # shallow copy since shuffle is in-place
            ros_poses_copy = list(ros_poses)
            shuffle(ros_poses_copy)
            ros_poses = ros_poses_copy[0:max_poses]


        poses = []

        # convert poses
        for p in ros_poses:
            o = p.orientation
            q = (o.x, o.y, o.z, o.w)  # extract object into tuple
            e = tf.transformations.euler_from_quaternion(q)
            yaw = e[2] if use_old_values else math.degrees(e[2])
            poses.append((p.position.x, p.position.y, yaw))

        if use_old_values:
            d = DBSCAN.DBSCAN_weighted(20)
            clusters = DBSCAN.dbscan(poses, eps=0.5, minPts=4, distance_metric=d)
        else:
            d = DBSCAN.DBSCAN_weighted(0.1)
            clusters = DBSCAN.dbscan(poses, eps=0.3, minPts=5, distance_metric=d)

        #d = DBSCAN.DBSCAN_translated(poses, 5)
        #clusters = DBSCAN.dbscan(poses, eps=0.5, minPts=4, distance_metric=d)

        if len(clusters) == 0:
            return Pose()

        largest_c = None
        largest_len = 0
        for c in clusters:
            if len(c) > largest_len:
                largest_c = c
                largest_len = len(c)

        c = largest_c
        cl = len(c)

        # rviz drawing
        if draw_stuff:
            DBSCAN.v.clear_markers()
            in_cluster = set()
            for i in clusters:
                in_cluster.update(i)
            #not_c = [p for p in poses if p not in c]
            not_c = [p for p in poses if p not in c and p in in_cluster]
            draw_c = c
            if not use_old_values:
                not_c = map(lambda (px,py,pt): (px,py,math.radians(pt)), not_c)
                draw_c = map(lambda (px,py,pt): (px,py,math.radians(pt)), c)
            DBSCAN.draw_cluster(draw_c, color=(0,0,1,0.5))

            DBSCAN.draw_cluster(not_c, color=(0,1,0,0.8))
            DBSCAN.draw_cluster([p for p in poses if p not in in_cluster], color=(1,0,0,0.5))

        x_bar = sum([x for (x,_,_) in c])/cl
        y_bar = sum([y for (_,y,_) in c])/cl
        t_bar = sum([t for (_,_,t) in c])/cl
        if not use_old_values:
            t_bar = math.radians(t_bar)

        if draw_stuff:
            DBSCAN.draw_pose((x_bar, y_bar, t_bar))


        # printing for tests
        if print_stuff:
            import numpy as np
            print('-'*30)
            print('pose estimate: {}'.format((x_bar, y_bar, t_bar)))

            N = len(clusters)
            print('number of clusters = ' + str(N))
            print('certainty: P(real_pose in largest_c) = {}/{} = {}'.format(cl,
                len(poses), (cl/float(len(poses)))))

            lens = [len(i) for i in clusters]

            print('number of noise poses = ' + str(len(poses)-sum(lens)) +
                    ' (not in any cluster)')

            print('min {}, max {}, avg {}, var {} particles per cluster'.format(
                min(lens), max(lens), sum(lens)/N, np.var(lens)))

            # distances within the largest cluster
            ds = []
            px, py, ptheta = (x_bar, y_bar, t_bar)
            for qi,q in enumerate(largest_c):
                (qx,qy,qtheta) = q
                ds.append((qx-px)**2 + (qy-py)**2 + d.weight*(qtheta-ptheta)**2)
            print(('min {}, max {}, avg {}, var {} ' +
                'distance metric (squared) within largest cluster').format(
                min(ds), max(ds), sum(ds)/len(largest_c), np.var(ds)))


        p = DBSCAN.pose_to_ros_pose(x_bar, y_bar, t_bar)

        return p
