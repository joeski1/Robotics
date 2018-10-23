#!/usr/bin/env python
#class for direction enum
#@author charlie street

class Direction:
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

def toDegrees(d):
    if d == Direction.NORTH:
        return 0
    elif d == Direction.WEST:
        return 90
    elif d == Direction.SOUTH:
        return 180
    elif d == Direction.EAST:
        return -90
    else:
        raise Exception("Not a valid direction")
