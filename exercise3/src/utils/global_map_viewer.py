#!/usr/bin/env python
import math
import rospy
import os
from Queue import Queue, Empty
import time
import threading
import numpy as np
import copy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# messages
from exercise3.msg import global_map, global_map_request, navigation, face_msg


outdir = os.path.dirname(os.path.realpath(__file__))

SHOW_DIR = False
SHOW_LOCAL_MAP = True
SHOW_PATH = True
SHOW_FACES = True


MAX_FACES = 1000
face_separation = 10 # min distance of cells (Euclidean distance) between faces
# Cartesian coordinates of detected faces relative to the global map
faces = []
# polar coordinates of faces relative to the robot now
new_faces = []

def pol_to_cart(pol):
    r = pol[0]
    theta = pol[1]
    return (r*math.cos(theta), r*math.sin(theta))

def distance(a, b):
    ax, ay = a
    bx, by = b
    return math.sqrt((ax-bx)**2 + (ay-by)**2)

class CellSetter:
    def __init__(self, robot_x, robot_y, resolution, cells):
        self.rx = robot_x
        self.ry = robot_y
        self.resolution = resolution
        self.cells = cells

    def set_cell(self, r, theta, value):
        x, y = pol_to_cart((r, theta))
        x_cell = self.rx + (x // self.resolution)
        y_cell = self.ry + (y // self.resolution)
        self.cells[y_cell, x_cell] = value # row major indexing

BELOW = -1.0
ABOVE = 2.0


class MapViewer:
    def __init__(self):
        self.raw_array = []
        self.cols = 0
        self.cmap = copy.copy(matplotlib.cm.gray) # modify color map
        self.cmap._init()
        self.cmap.set_under('#ff0000') # color for <0 (BELOW)
        self.cmap.set_over('#00ff00') # color for >1 (ABOVE)
        self.cmap.set_bad('#5555ff')  # color for NaN
        self.image_queue = Queue()
        self.fig = plt.figure()
        self.ax  = self.fig.add_subplot(111, aspect='equal')
        # aspect='equal' => match image
        self.im = plt.imshow(np.ndarray(shape=(2, 2)), animated=True,
                cmap=self.cmap, interpolation='none', vmin=0, vmax=1)
        # blit=True is a bad idea since it only re-draws the image and not the grid
        self.ani = animation.FuncAnimation(self.fig, self._updatefig,
                interval=500)
    def show(self, blocking=False):
        plt.ion() # interactive
        plt.show(block=blocking)

    def _updatefig(self, x):
        try:
            new_data, num_cols, resolution, pose, counter = self.image_queue.get_nowait()
        except Empty:
            if __name__ == '__main__' and rospy.is_shutdown():
                plt.close() # this will crash tk but oh well

            return self.im,

        #print('new global map {}'.format(counter))
        if counter is not None:
            self.ax.set_title('global map {}'.format(counter))
        #plt.draw()

        # keep for saving at the end
        self.cols = num_cols

        width  = num_cols
        height = len(new_data)/num_cols
        rows, cols = height, width
        self.raw_arr = np.array(new_data, dtype=float).reshape((rows, cols), order='C')
        arr = np.array(self.raw_arr, copy=True)

        # Show path on the screen
        if SHOW_PATH and path is not None:
            render_path = path
            for cell in render_path:
                x_cell, y_cell = cell
                arr[y_cell, x_cell] = ABOVE

        # highlight where the robot is
        # x, y relative to 0, 0 = middle of global map
        rx, ry, rt = pose
        rx_cell = rx // resolution
        ry_cell = ry // resolution
        arr[ry_cell, rx_cell] = BELOW # row major indexing

        setter = CellSetter(rx_cell, ry_cell, resolution, arr)

        # I think this is because +y is down on the graph? x is probably flipped as well?
        rt += math.pi # 180 degrees out

        # highlight where the robot is facing
        if SHOW_DIR:
            setter.set_cell(3*resolution, rt, ABOVE)
            setter.set_cell(2*resolution, rt, ABOVE)

        # highlight where the local map should be
        if SHOW_LOCAL_MAP:
            laser_range = 5.75 # metres
            local_width = laser_range*2 / resolution
            local_height = laser_range / resolution
            w = local_width*resolution
            h = local_height*resolution

            hypot = math.sqrt((w/2)**2 + h**2)
            angle = math.acos(h/hypot)
            setter.set_cell(hypot, rt + angle, BELOW)
            setter.set_cell(hypot, rt - angle, BELOW)
            setter.set_cell(w/2, rt + math.pi/2, BELOW)
            setter.set_cell(w/2, rt - math.pi/2, BELOW)

        # make newly discovered faces relative to the global map using the
        # current robot position
        for f in new_faces:
            face_x, face_y = pol_to_cart(f)
            face_x_cell = int(rx_cell + (face_x // resolution))
            face_y_cell = int(ry_cell + (face_y // resolution))
            face = (face_x_cell, face_y_cell)

            # check that this face is not too close to other faces on the map
            too_close = False
            for of in faces:
                if distance(face, of) < face_separation:
                    too_close = True
                    break

            if not too_close and len(faces) < MAX_FACES:
                faces.append(face)
        new_faces[:] = [] # empty existing list without making a new one

        if SHOW_FACES:
            # draw faces
            for f in faces:
                face_x_cell, face_y_cell = f
                arr[face_y_cell, face_x_cell] = float('nan')

        #print(arr)
        self.im.set_array(arr)
        return self.im,

    def update_image(self, data_tuple):
        if self.image_queue.empty(): # don't buffer
            self.image_queue.put(data_tuple)


def global_map_callback(msg):
    global mv
    pose = (msg.pose.x, msg.pose.y, msg.pose.rotation)
    mv.update_image((msg.global_map, msg.cols, msg.resolution, pose, msg.counter))

def path_callback(msg):
    global path
    p = []
    for i in range(0, len(msg.xs)):
        p.append((msg.xs[i], msg.ys[i]))
    path = p


def face_callback(msg):
    global faces
    distances = msg.distances
    angles = msg.angles
    for i in range(len(distances)):
        new_faces.append((distances[i], angles[i]))
    #rospy.loginfo('faces = {}'.format(faces))

def ros_spin():
    request_pub = rospy.Publisher('global_map_request', global_map_request, queue_size='None')

    try:
        r = rospy.Rate(1) # Hz
        while not rospy.is_shutdown():
            request_pub.publish(global_map_request())
            r.sleep()
    except rospy.ROSInterruptException:
        pass

    #rospy.spin()

if __name__ == '__main__':
    global path
    path = []

    mv = MapViewer()

    rospy.init_node('global_map_viewer')

    ros_thread = threading.Thread(target=ros_spin)
    ros_thread.daemon = True
    rospy.Subscriber('global_map', global_map, global_map_callback)
    rospy.Subscriber('navigation', navigation, path_callback)
    rospy.Subscriber('faces', face_msg, face_callback)
    ros_thread.start()


    # non-blocking isn't working great
    block = True
    if block:
        mv.show(blocking=True)
    else:
        mv.show(blocking=False)

        while not rospy.is_shutdown():
            mv.fig.canvas.draw()
            #plt.draw()
            time.sleep(0.1)

        plt.close()


    print('Saving map')
    arr = mv.im.get_array()
    if not os.path.exists('global_map'):
        os.mkdir('global_map')

    np.savetxt(outdir + '/global_map/global_map.csv', mv.raw_arr, delimiter=",")
    plt.imsave(outdir + '/global_map/global_map.png', mv.cmap(arr))
    plt.imsave(outdir + '/global_map/global_map_raw.png',
            np.dstack((mv.raw_arr, mv.raw_arr, mv.raw_arr)))

