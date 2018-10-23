#!/usr/bin/env python

# based on the paper:
# "A density-based algorithm for discovering clusters in large spatial databases
# with noise"

import viz
import math
from geometry_msgs.msg import Pose
import tf  # for converting to and from quaternions


v = viz.Viz()


def dbscan(poses, eps, minPts, distance_metric):
    '''
        Density-based clustering of poses with a given distance metric and
        parameters for the DBSCAN algorithm

        Noise points are discarded rather than being placed in their own cluster

        see: http://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf
    '''
    # cluster index
    visitedi = set() # set of visited indices
    anyClusteri = set() # set of indices to test whether pose is in any cluster
    # list of list of poses
    clusters = []

    # pi => index of pose p
    # p  => pose p

    for pi,p in enumerate(poses):
        if pi in visitedi:
            continue
        visitedi.add(pi)

        neighbourhoodi = distance_metric.regionQuery(poses, pi, eps)
        if len(neighbourhoodi) >= minPts: # p is not noise

            # new cluster
            cluster = [p]

            # expand cluster by searching through neighbours
            # continue search with neighbours of q if q is not on the edge
            while len(neighbourhoodi) > 0:
                qi = neighbourhoodi.pop(0)
                if qi not in visitedi:
                    visitedi.add(qi)

                    q_neighbourhoodi = distance_metric.regionQuery(poses, qi, eps)
                    if len(q_neighbourhoodi) >= minPts:
                        # add all q's neighbours to neighbours
                        # duplicates are handled by the visited set
                        neighbourhoodi.extend(q_neighbourhoodi)

                if qi not in anyClusteri:
                    q = poses[qi]
                    cluster.append(q)
                    anyClusteri.add(qi)

            clusters.append(cluster)

    return clusters

class DBSCAN_weighted:
    '''
        A weight metric using euclidean distance with the orientation dimension
        weighted differently to the others
    '''
    def __init__(self, weight):
        self.weight = weight

    def regionQuery(self, poses, index, eps):
        neighboursi = []
        (px, py, ptheta) = poses[index]
        for qi,q in enumerate(poses):
            (qx,qy,qtheta) = q
            if (qx-px)**2 + (qy-py)**2 + self.weight*(qtheta-ptheta)**2 < eps:
                neighboursi.append(qi)
        return neighboursi

class DBSCAN_translated:
    '''
        A weight metric using translated poses in the direction of their
        orientation and clustering on the resulting points
    '''
    def __init__(self, poses, oradius):
        self.translated = []
        for x,y,t in poses:
            self.translated.append((x+(oradius*math.sin(t)), y+oradius*math.cos(t)))

    def regionQuery(self, _, pi, eps):
        '''
        returns a list of indices of poses from self.translated that are within
        distance eps to the point in the list at the given index
        '''
        neighboursi = []
        px,py = self.translated[pi]
        for qi,(qx,qy) in enumerate(self.translated):
            if (qx-px)**2 + (qy-py)**2 < eps:
                neighboursi.append(qi)
        return neighboursi


def draw_cluster(cluster, color=(0,1,0,1)):
    # r = radius = length of lines
    r = 0.2
    ps = []
    for (x,y,t) in cluster:
        ps.append((x,y))
        ps.append((x+r*math.cos(t), y+r*math.sin(t)))

    v.draw_lines(ps, size=0.01, color=color)

def draw_pose(p, color=(0, 0, 1, 1)):
    r = 0.4
    ps = []
    (x,y,t) = p
    ps.append((x,y))
    ps.append((x+r*math.cos(t), y+r*math.sin(t)))

    v.draw_lines(ps, size=0.04, color=color)

def pose_to_ros_pose(x, y, t):
    '''
    given (x,y,theta) return a ros Pose object
    '''
    p = Pose()
    p.position.x = x
    p.position.y = y
    # quat_from_euler returns a numpy array
    q = tf.transformations.quaternion_from_euler(0, 0, t)
    p.orientation.x = q[0]
    p.orientation.y = q[1]
    p.orientation.z = q[2]
    p.orientation.w = q[3]
    return p

