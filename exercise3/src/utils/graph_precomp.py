#!/usr/bin/env python

import precomp

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.lines as lines
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

grid_y = 23
grid_x = 46
grid_resolution = 0.25 # metre/cell

# from: http://stackoverflow.com/a/26757297
def cart2pol(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return(r, theta)
def pol2cart(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return(x, y)

xs = np.linspace(-grid_x*grid_resolution, grid_x*grid_resolution)

def rect(x, y, w, h, color='white', edge='#000000'):
    return mpatches.Rectangle((x, y), w, h, facecolor=color, edgecolor=edge)

def plot_line(a, b, color='r'):
    x1, y1 = a
    x2, y2 = b
    ls = lines.Line2D([x1, x2], [y1, y2], color=color)
    ax.add_line(ls)

def plot_polar_line(theta, color='r'):
    xp, yp = pol2cart(100, theta)
    plot_line((0, 0), (xp, yp), color)

def update_grid():
    width  = grid_x*grid_resolution
    height = grid_y*grid_resolution

    ax.set_aspect(height*2/float(width))

    plt.xlim((-width/2, width/2))
    plt.ylim((0, height))

    ax.grid(True, axis='both', color='r', linestyle='dotted')

class Cells:
    patches = []
    half_width = 0.5*grid_resolution
    for i in range(grid_y*grid_x):
        color = 'white'
        x,y = pol2cart(*precomp.cells_centers[i])
        patches.append(rect(x-half_width, y-half_width,
            grid_resolution, grid_resolution, color=color))
    # match_original to preserve styles
    cells = PatchCollection(patches, match_original=True)
    cell_colors = ['w']*len(patches)

def update_fig(t):
    ax.clear()
    update_grid()

    t += (grid_y-1)*grid_x # starting offset

    last_t = (t-1) % (grid_y*grid_x)
    t = t % (grid_y*grid_x)

    # check angles are correct, 0 on the right, pi on the left
    #plot_polar_line(0.01, color='r')
    #plot_polar_line(math.pi-0.01, color='g')

    lasers = precomp.cell_laser_angles[t]
    plot_polar_line(lasers[0], color='r')
    plot_polar_line(lasers[1], color='g')
    plot_polar_line(lasers[2], color='b')
    plot_polar_line(lasers[3], color='orange')

    Cells.cell_colors[last_t] = 'white'
    Cells.cell_colors[t] = '#ffaaaa'

    Cells.cells.set_facecolor(Cells.cell_colors)

    ax.add_collection(Cells.cells)


if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ani = animation.FuncAnimation(fig, update_fig, interval=10)
    plt.show()
