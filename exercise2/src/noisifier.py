#!/usr/bin/env python
# file is used to add noise to a Pose msg
# @author: Charlie Street


import random

from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion

from pf_localisation.util import getHeading
from pf_localisation.util import rotateQuaternion

SIGMA_NORMAL = 0.1 # values for distributions, mean will be point in question
KAPPA_VON_MISES = 40 # joe says so

# function will take a particle and add a small amount of noise to it
# gaussian noise will be used for translations
# von mises noise will be used for rotations
def noisifier(particle):

    # adding noise to position
    point = particle.position
    newx = random.normalvariate(point.x,SIGMA_NORMAL)  # new x value
    newy = random.normalvariate(point.y,SIGMA_NORMAL)  # new y value
    new_point = Point()
    new_point.x = newx
    new_point.y = newy
    new_point.z = point.z  # carry z over

    quat = particle.orientation  # get me the quaternion
    quat_rads = getHeading(quat)
    quat_rad_noise = random.vonmisesvariate(quat_rads,KAPPA_VON_MISES)  # get new angle in radians
    quat_diff = quat_rad_noise - quat_rads  # get the difference so we know how much to rotate the particle
    new_quat = rotateQuaternion(quat,quat_diff)  # get the new quaternion

    new_particle = Pose()
    new_particle.position = new_point
    new_particle.orientation = new_quat

    return new_particle
