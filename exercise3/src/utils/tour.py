#!/usr/bin/env python
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "..", "exploration", "common"))

import rospy
import time
import math
from movement import Movement


start = 0

def msg():
    total = 173.0 # seconds
    dur = time.time()-start
    prop = dur/total
    rem = total-dur
    mins, secs = divmod(rem, 60)
    return '\t{}%, {}:{} remaining'.format(int(prop*100), int(mins), int(secs))

def f(x):
    print('forward {}{}'.format(x, msg()))
    mov.forward_distance(x, vel=1)#0.4)
def r(angle):
    print('rotate {}{}'.format(angle, msg()))
    mov.rotate_angle(math.radians(angle), 1)#vel=0.4)


if __name__ == '__main__':
    print('starting tour')
    rospy.init_node('tour')
    mov = Movement()

    start = time.time()

    # starting from (-14.34, -13.82, 0, 90) in the simulator
    # (which is facing the front of the building at the end of the corridor by
    # the lift)
    f(6)
    r(-20)
    f(5)
    r(-30)
    f(5)
    r(-30)
    print('by the vending machines')

    f(18)
    r(-55)
    print('facing down bottom right corridor')
    f(8.5)
    r(-30)
    f(4)
    r(180)
    print('end of the corridor, turn around')
    f(26)
    print('end of the other corridor')
    r(175)
    f(15)
    print('heading back')
    r(-50)
    f(5)
    r(-35)
    f(20)
    print('last corridor')
    r(-45)
    f(5)
    r(-35)
    f(9.5)
    print('back to the start')
    r(175)
    f(26)


    print('tour finished in {:.2f} seconds'.format(time.time()-start))

