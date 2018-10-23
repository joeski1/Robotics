#!/usr/bin/env python

#TODO: try matplotlib instead of tkinter

# renamed in python3
try:
    import tkinter as tk
except:
    import Tkinter as tk
from PIL import Image

import sys
import time
from Queue import Queue, Empty
import threading
import rospy
import numpy as np

# messages
from exercise3.msg import local_map



class MapViewer:
    def __init__(self):
        self.image_queue = Queue()

    def start(self):
        print('starting map viewer')
        self.root = tk.Tk()
        self.root.title('Map Viewer')
        self.root.geometry('{}x{}'.format(200, 200))
        self.canvas = tk.Canvas(self.root, width=1000, height=1000)
        self.canvas.pack()
        self.canvas_image = self.canvas.create_image(0, 0, image=tk.PhotoImage())

        self.root.after(0, self._tk_process_new_images)
        self.root.mainloop() # runs forever

    def _tk_process_new_images(self):
        try:
            new_data, num_cols = self.image_queue.get_nowait()
        except Empty:
            self.root.after(10, self._tk_process_new_images)
            return
        print('new image')

        width  = num_cols
        height = len(new_data)/num_cols
        assert len(new_data) % num_cols == 0

        start = time.clock()

        # so, tkinter is complete shit:
        # - the documentation is garbage (or non-existent)
        # - the mainloop() and root.after() nonsense is horrible
        # - the only way to set the pixels of an image is to first make a string!
        # - setting the window size requires constructing a string like '1x1'
        # - there is a 'zoom' method but it doesn't work!!!
        # - there is a module called ImageTk but it isn't standard (requires a package)
        # - if the image isn't stored in self then it gets garbage collected
        #   because tk doesn't keep a copy!
        # - tkinter does threading _so_ badly. I have needed to completely
        #   rewrite this module and have ros execute in another thread and tkinter
        #   in the main thread otherwise it throws a tantrum. But this makes it
        #   hard to use this module as a library :(

        # create a numpy array, then a PIL image, do the resize and get the pixel data
        im = Image.fromarray(
                np.array(new_data, dtype=float)
                .reshape((width, height), order='C'))
        im = im.resize((200, 200), resample=Image.NEAREST)
        pixel_data = im.getdata()
        width,height = im.size

        # generate a string of the pixels
        pixels = ''
        # write image data in the form:
        # {#aabbcc#aabbcc}{#aabbcc#aabbcc}
        for r in range(height):
            pixels += '{'
            for c in range(width):
                # greyscale
                val = int(pixel_data[r*width + c] * 255)
                pixels += '#{:x}{:x}{:x} '.format(val, val, val)
            pixels += '} '

        # load the data into tkinter
        self.image = tk.PhotoImage(width=width, height=height)
        self.image.put(pixels)

        self.canvas.itemconfig(self.canvas_image, image=self.image)

        print('resize done in {}ms'.format((time.clock()-start)*1000))

        self.root.after(10, self._tk_process_new_images)

    def update_image(self, new_data, num_cols):
        self.image_queue.put((new_data, num_cols))



def local_map_callback(msg):
    global mv
    mv.update_image(msg.local_map, msg.cols)

def ros_spin():
    rospy.spin()

if __name__ == '__main__':
    rospy.init_node('local_map_viewer')

    ros_thread = threading.Thread(target=ros_spin)
    ros_thread.daemon = True
    rospy.Subscriber('local_map', local_map, local_map_callback)
    ros_thread.start()

    mv = MapViewer()
    mv.start()


