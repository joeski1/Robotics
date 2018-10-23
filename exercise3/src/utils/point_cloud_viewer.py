#!/usr/bin/env python
import rospy
from Queue import Queue, Empty
import time
import threading
import numpy as np
import copy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection

# messages
from exercise3.msg import stitching_points


class PointsViewer:
    def __init__(self):
        self.points_queue = Queue()
        self.fig = plt.figure()
        self.ax  = self.fig.add_subplot(111, aspect='equal')

        self.global_pts, = plt.plot([], [], 'bo')
        self.local_pts, = plt.plot([], [], 'r.', markersize=8)
        self.local_trans_pts, = plt.plot([], [], 'k.')
        self.local_lines = LineCollection([], linewidths=0.5, zorder=0, colors='black')
        self.ax.add_collection(self.local_lines)
        self.robot_pos, = plt.plot([], [], 'ro')

        d = 5.75 # laser range
        plt.xlim(-d, d)
        plt.ylim(-d, d)
        self.ani = animation.FuncAnimation(self.fig, self._updatefig,
                interval=100)

    def show(self, blocking=False):
        plt.ion() # interactive
        plt.show(block=blocking)

    def _updatefig(self, x):
        try:
            msg = self.points_queue.get_nowait()
        except Empty:
            if __name__ == '__main__' and rospy.is_shutdown():
                plt.close() # this will crash tk but oh well

            return (self.global_pts, self.local_pts, self.local_trans_pts,
                    self.robot_pos, self.local_lines)

        #with open('local_map.py', 'w') as f:
            #f.write('cells = {}\n'.format(new_data))
            #print('local map written to file')

        if msg.counter is not None:
            self.ax.set_title('local map {}'.format(msg.counter))
        #plt.draw()


        self.global_pts.set_xdata(msg.global_xs)
        self.global_pts.set_ydata(msg.global_ys)

        self.local_pts.set_xdata(msg.local_xs)
        self.local_pts.set_ydata(msg.local_ys)

        self.local_trans_pts.set_xdata(msg.local_trans_xs)
        self.local_trans_pts.set_ydata(msg.local_trans_ys)

        local_pts = zip(msg.local_xs, msg.local_ys)
        local_trans_pts = zip(msg.local_trans_xs, msg.local_trans_ys)
        self.local_lines.set_segments(list(zip(local_pts, local_trans_pts)))

        self.robot_pos.set_xdata([msg.rx, msg.trans_rx])
        self.robot_pos.set_ydata([msg.ry, msg.trans_ry])

        return (self.global_pts, self.local_pts, self.local_trans_pts,
                self.robot_pos, self.local_lines)

    def update_points(self, msg):
        self.points_queue.put(msg)


def points_callback(msg):
    global pv
    pv.update_points(msg)

def ros_spin():
    rospy.spin()

def fake_points_msg():

    ps = stitching_points()
    ps.global_xs = [1, 1]
    ps.global_ys = [0, 1]
    ps.local_xs = [2, 2]
    ps.local_ys = [0, 1]
    ps.local_trans_xs = [1.5, 1.5]
    ps.local_trans_ys = [0, 1]
    ps.rx = 0
    ps.ry = 0
    ps.trans_rx = 1
    ps.trans_ry = -1
    points_callback(ps)

if __name__ == '__main__':
    pv = PointsViewer()

    #fake_points_msg()

    rospy.init_node('points_viewer')

    ros_thread = threading.Thread(target=ros_spin)
    ros_thread.daemon = True
    rospy.Subscriber('local_map_points', stitching_points, points_callback)
    ros_thread.start()


    # non-blocking isn't working great
    block = True
    if block:
        pv.show(blocking=True)
    else:
        pv.show(blocking=False)

        while not rospy.is_shutdown():
            pv.fig.canvas.draw()
            #plt.draw()
            time.sleep(0.1)

        plt.close()
