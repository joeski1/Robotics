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

# messages
from exercise3.msg import local_map


class MapViewer:
    def __init__(self, grid=True):
        self.cmap = copy.copy(matplotlib.cm.gray) # modify color map
        self.cmap._init()
        self.cmap.set_under('#000099') # color for -1
        self.image_queue = Queue()
        self.fig = plt.figure()
        self.ax  = self.fig.add_subplot(111, aspect='equal')
        self.grid = grid
        if self.grid:
            self.ax.grid(True, axis='both', color='r', linestyle='dotted')
        self.size = (2, 2)
        # aspect='equal' => match image
        self.im = plt.imshow(np.ndarray(shape=self.size), animated=True,
                cmap=self.cmap, interpolation='none', vmin=0, vmax=1)
        # blit=True is a bad idea since it only re-draws the image and not the grid
        self.ani = animation.FuncAnimation(self.fig, self._updatefig,
                interval=200)
    def show(self, blocking=False):
        plt.ion() # interactive
        plt.show(block=blocking)

    def _updatefig(self, x):
        try:
            new_data, num_cols, counter = self.image_queue.get_nowait()
        except Empty:
            if __name__ == '__main__' and rospy.is_shutdown():
                plt.close() # this will crash tk but oh well

            return self.im,

        print('new local map {}'.format(counter))

        #with open('local_map.py', 'w') as f:
            #f.write('cells = {}\n'.format(new_data))
            #print('local map written to file')

        width  = num_cols
        height = len(new_data)/num_cols
        assert len(new_data) % num_cols == 0
        assert width % 2 == 0

        if self.grid and self.size != (width, height):
            self.size = (width, height)

            # the *2 is because of the extent
            self.ax.set_aspect(height*2/float(width))

            left, right = -width//2, width//2
            top, bottom = 0, height
            self.ax.set_xticks(list(range(left, right)))
            self.ax.set_yticks(list(range(top, bottom)))

            self.im.set_extent((left, right, bottom, top))
        elif not self.grid:
            self.size = (width, height)

            # the *2 is because of the extent
            self.ax.set_aspect(height/float(width))

        if counter is not None:
            self.ax.set_title('local map {}'.format(counter))
        #plt.draw()

        rows, cols = height, width
        arr = np.array(new_data, dtype=float).reshape((rows, cols), order='C')
        #print(arr)
        self.im.set_array(arr)
        return self.im,

    def update_image(self, new_data, num_cols, counter=None):
        self.image_queue.put((new_data, num_cols, counter))


def local_map_callback(msg):
    global mv
    mv.update_image(msg.local_map, msg.cols, msg.counter)

def ros_spin():
    rospy.spin()

if __name__ == '__main__':
    mv = MapViewer(grid=False)

    rospy.init_node('local_map_viewer')

    ros_thread = threading.Thread(target=ros_spin)
    ros_thread.daemon = True
    rospy.Subscriber('local_map', local_map, local_map_callback)
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
