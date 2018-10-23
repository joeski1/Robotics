#!/usr/bin/env python
# this file is used to collect the training samples for the neural network in order to build the inverse sensor model
# the algorithm used is based upon an algorithm named 'Sense and Drive' originally invented by Dam et al
# in the paper 'Neural Network Applications in Sensor Fusion For An Autonomous Mobile Robot'
# Author: Charlie Street

import sys
import rospy
import csv
import random
import math
import tf
import time
import os
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

INTENDED_SAMPLES = 10000  # number of samples we would like to obtain, time permitted
TRAINING_FILE = 'training_data.csv'  # file where our training data will be written to


def is_simulated():
    import subprocess
    p = subprocess.Popen(['ps', '-ax'], stdout=subprocess.PIPE)
    stdout, _ = p.communicate()
    if 'stage_ros/stageros' in stdout:
        return True

if is_simulated():
    INITIAL_FORWARD_VEL = 0.5
    TURN_THRESHOLD = 0.001
    DISTANCE_THRESHOLD = 0.3  # distance threshold needs to be a lot less than grid size to get accurate samples
else:
    INITIAL_FORWARD_VEL = 0.2
    TURN_THRESHOLD = 0.05
    DISTANCE_THRESHOLD = 0.2  # distance threshold needs to be a lot less than grid size to get accurate samples


LASER_READINGS = 512
GRID_RESOLUTION = 0.25  # metres per cell (i.e grid cell size)
MAX_RANGE = None  # max range of laser sensor
MIN_RANGE = None
GRID_Y = 23  # maximum y value of grid cells
GRID_X = 46  # maximum x value of grid cells
GRID_CELLS = []

current_laser_data = None # list of ranges
current_position = None
current_orientation = None
num_samples = 0  # number of samples received
num_its     = 0  # number of sense_and_drive iterations
start_time  = 0
last_time   = 0 # time spent during previous runs


# create null twist as useful for stopping
zero_twist = Twist()
zero_twist.linear.x = 0.0
zero_twist.angular.z = 0.0


def log(x):
    rospy.loginfo(x)

tau = 2*math.pi
# from http://stackoverflow.com/a/1878936
def smallest_angle_between(x, y):
    a = (x-y) % tau
    b = (y-x) % tau
    return -a if a < b else b


def pol_to_cart(pol):
    r = pol[0]
    theta = pol[1]
    return (r*math.cos(theta), r*math.sin(theta))

def cart_to_pol(p):
    x, y = p
    r = math.sqrt(x**2 + y**2)
    theta = math.atan2(y, x)
    return (r, theta)

def laser_angle(number):
    # exercise1 movement.py get_laser_xy does something different
    # reading 0 is the far right, reading 511 is the far left
    # the angle goes from 0 to pi
    return (float(number)/float(LASER_READINGS)) * math.pi


def Gauss_num_sols(m):
    '''
    Using Gaussian elimination: return the number of solutions that a system of
    linear equations has
    Many thanks to Achim's handouts!
    '''
    assert(m.dtype==float)

    def swap_rows(m, frm, to):
        m[[frm, to],:] = m[[to, frm],:]

    n, cols = m.shape # n = number of equations/rows

    if n < 1 or cols < 1:
        raise Exception()
    elif n == 1:
        # doesn't handle (x y | z) etc but we cannot reach this case when only
        # accepting input of shape (2, 3)
        assert(cols <= 2)

        if cols == 2:
            # base case ax=b
            a, b = m.item(0), m.item(1)
            a_zero = np.isclose(a, 0)
            b_zero = np.isclose(b, 0)
            if a_zero and b_zero:
                return 'inf'
            elif a_zero:
                return 0 # contradictory
            else:
                return 1
        elif cols == 1:
            # 1x1 matrix
            if np.isclose(m.item(0), 0):
                return 'inf'
            else:
                return 0
    elif cols == 1:
        if np.allclose(m, 0):
            return 'inf'
        else:
            return 0
    else:
        # recursive case n>1
        first_col = m[:,0]
        # index of row with largest abs element
        pivot_row = np.argmax(np.abs(first_col)).item(0)
        if pivot_row != 0:
            swap_rows(m, 0, pivot_row)
        pivot = first_col.item(0)

        if np.isclose(pivot, 0):
            # advance column
            return Gauss_num_sols(m[:,1:])
        else:
            for r in range(1, n):
                if np.isclose(first_col.item(r), 0): # item already 0
                    continue
                else: # make it equal zero
                    factor = pivot / first_col.item(r)
                    m[r,:] *= factor
                    m[r,:] -= m[0,:]
            return Gauss_num_sols(m[1:,1:])


def lines_intersect(line1, line2):
    '''
    test whether the given line segments intersect
    lines are specified by a pair of tuples for the end points
    '''
    # convert a pair of pair of something into a pair of pair of floats
    def to_flt(tup): (x1,y1),(x2,y2) = tup; return ((float(x1), float(y1)), (float(x2), float(y2)))

    # convert from 2-point representation to parametric representation
    a, b = to_flt(line1)
    PQ1 = (b[0]-a[0], b[1]-a[1])
    P1 = a
    # now line1 parametric form: X = P1 + s*PQ1  (for all s)
    c, d = to_flt(line2)
    PQ2 = (d[0]-c[0], d[1]-c[1])
    P2 = c
    # now line2 parametric form: X = P2 + t*PQ2  (for all t)

    # construct the linear equations to solve for s and t where the line
    # equations are equal
    coeffs = np.matrix([
        [PQ1[0], -PQ2[0]],
        [PQ1[1], -PQ2[1]]], dtype=float)
    depends = np.matrix([
        [P2[0]-P1[0]],
        [P2[1]-P1[1]]], dtype=float)

    mat = np.hstack((coeffs, depends)) # glue together side by side

    # return the number of solutions to the system of equations but don't solve it
    num_sols = Gauss_num_sols(mat)

    if num_sols == 'inf':
        # lines are parallel and co-linear but may not touch
        # find the start and end points in terms of the parametric equation of
        # the first line (project the points onto the line)
        # we know that these points lie on the line so don't have to check the y-axis
        def dist(x, y):
            return math.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)

        def choose_min_in_dim(points, dimension):
            ''' return a list of the points with their values in the given dimension are lowest '''
            min_val = float('inf')
            mins = []
            for p in points:
                if p[dimension] == min_val:
                    mins.append(p)
                elif p[dimension] < min_val:
                    mins = [p]
                    min_val = p[dimension]
            return mins

        # choose one point to measure the distance to all the other points. This
        # is to avoid the issue with distinguishing between positive and
        # negative movement along the line
        chosen = None
        mins_x = choose_min_in_dim([a,b,c,d], 0)
        if len(mins_x) > 1:
            mins_y = choose_min_in_dim(mins_x, 1)
            chosen = mins_y[0] # doesn't matter if its a tie, choose first
        else:
            chosen = mins_x[0]

        # overlaps assumes that a comes before b and c comes before d, they are interchangeable anyway
        da, db = sorted([dist(chosen, a), dist(chosen, b)])
        dc, dd = sorted([dist(chosen, c), dist(chosen, d)])
        overlaps = min(db, dd)-max(da, dc) >= 0 # from http://stackoverflow.com/a/2953979
        return overlaps
    elif num_sols == 1:
        # if there is a single solution then numpy can find it for us. Make sure
        # that the solution is inside the line *segment* for both segments
        s, t = np.linalg.solve(coeffs, depends)
        return 0<= s <=1 and 0<= t <=1
    else:
        # no solutions => no intersection
        return False




def draw_cells(cells, grid_x=GRID_X, grid_y=GRID_Y, file_out=None):
    '''
    draw row-major 1D array of cells with elements:
    (traversed, blocked)
    '''
    free = ' '
    traversed = '+'
    blocked = '#'
    print('blocked:')
    print([c for c in range(len(cells)) if cells[c][1]])
    print('O' + ('-'*grid_x) + 'O')
    for r in range(grid_y):
        sys.stdout.write('|')
        for c in range(grid_x):
            cell = cells[r*grid_x + c]
            if cell[1]:  # blocked
                char = blocked
            elif cell[0]:  # traversed
                char = traversed
            else:
                char = free
            sys.stdout.write(char)
        print '|'
    print('O' + ('-'*grid_x) + 'O')

def draw_cells_ppm(cells, file_out, grid_x=GRID_X, grid_y=GRID_Y):
    '''
    draw row-major 1D array of cells with elements:
    (traversed, blocked)
    '''
    with open(file_out, 'w') as f:
        f.write('P2\n')
        f.write(str(grid_x) + ' ' + str(grid_y) + '\n')
        for r in range(grid_y):
            for c in range(grid_x):
                cell = cells[r*grid_x + c]
                if cell[1]:  # blocked
                    f.write('000')
                elif cell[0]:  # traversed
                    f.write('100')
                else:
                    f.write('255')

                if c != grid_x-1:
                    f.write(' ')
            f.write('\n')


class AABB:
    '''
    class representing a box/cell in our local map
    '''
    def __init__(self, minx, miny, maxx, maxy):
        self.minx = minx
        self.miny = miny
        self.maxx = maxx
        self.maxy = maxy

        cx, cy = minx+((maxx - minx)/2.0), miny+((maxy - miny)/2.0)
        (r, theta) = cart_to_pol((cx, cy))
        self.distance_to_cell = r
        self.angle_to_cell = theta

        readings_per_cell = 4
        width = readings_per_cell/2
        left_edge = LASER_READINGS - width
        for i in range(LASER_READINGS):
            if laser_angle(i) > theta:
                if i < width:
                    offset = 0
                elif i >= left_edge:
                    offset = LASER_READINGS-readings_per_cell
                else:
                    offset = i - width
                self.laser_indices = [offset, offset+1, offset+2, offset+3]
                break

    def __str__(self):
        return 'AABB(x={}, y={}, w={}, h={})'.format(self.minx, self.miny,
                self.maxx-self.minx, self.maxy-self.miny)

    def __repr__(self):
        return self.__str__()

    def get_angle_distance(self):
        return self.angle_to_cell, self.distance_to_cell

    def intersects_line(self, line):
        '''
        determine whether a line intersects any of the sides of the AABB
        eg line = ((0,0), (x,y))
        :param x: relative to 0,0 at the robot
        :param y: relative to 0,0 at the robot
        :return: does the line pass through this box
        '''

        tl = (self.minx, self.maxy)
        tr = (self.maxx, self.maxy)
        bl = (self.minx, self.miny)
        br = (self.maxx, self.miny)

        return (lines_intersect(line, (tl, tr)) or
                lines_intersect(line, (tr, br)) or
                lines_intersect(line, (br, bl)) or
                lines_intersect(line, (bl, tl)))

    def intersects_point(self, end_x, end_y):
        '''
        returns the occupied status of the grid cell (1 = occupied, 0 otherwise)
        :param end_x: the x point of stopping
        :param end_y: the y point of stopping
        :return: 1 for occupied, 0 otherwise
        '''
        in_x = self.minx <= end_x <= self.maxx
        in_y = self.miny <= end_y <= self.maxy
        return in_x and in_y


def generate_grid_cells():
    '''
    function generates our grid cells for the local map
    :return: nothing, though the global variable will be filled with multi use cells
    '''
    global GRID_CELLS
    GRID_CELLS = []
    x_max_values = [x*GRID_RESOLUTION for x in range(-(GRID_X/2 - 1), (GRID_X/2 + 1))]
    y_max_values = [y*GRID_RESOLUTION for y in range(1, GRID_Y+1)]
    y_max_values.reverse()
    for y in y_max_values:
        for x in x_max_values:
            GRID_CELLS.append(AABB(x-GRID_RESOLUTION, y-GRID_RESOLUTION, x, y))


def get_avg_front_laser_reading():
    '''
    function gets average of front 5 readings from laser
    :return: the average reading at the front of the robot
    '''
    midpoint = LASER_READINGS/2
    l = current_laser_data
    total_sum = sum([l[254], l[255], l[256], l[257]])
    return total_sum/4.0


def get_distance(point_1, point_2):
    '''
    gets euclidean distance between point 1 and point 2
    :param point_1: the first point to consider
    :param point_2: the second point to consider
    :return: the euclidean distance between the two points
    '''
    return math.sqrt((point_1[0] - point_2[0])**2 + (point_1[1] - point_2[1])**2)


def get_min_reading():
    '''
    gets the minimum laser reading for the current laser scan
    :return: the minimum laser scan
    '''
    min_val = float('inf')
    for i in range(len(current_laser_data)):
        if MIN_RANGE <= current_laser_data[i] <= MAX_RANGE and not math.isnan(current_laser_data[i]):
            if current_laser_data[i] < min_val:
                min_val = current_laser_data[i]

    return min_val


def extrapolate_front_laser(end_x, end_y, current_lasers):
    '''
    adds the front laser reading to our line
    :param end_x: the end x value of the line (line starts at origin) (metres)
    :param end_y: the end y value in the line (metres)
    :param current_lasers: the current set of laser readings
    :return: the adjusted end_x and end_y with
    '''
    middle_laser_reading = current_lasers[(LASER_READINGS/2) - 1]  # the - 1 is there due to zero indexing
    if middle_laser_reading > 0.3: # magic number
        return end_x, end_y
    else:
        original_hyp = math.sqrt(end_x**2 + end_y**2)
        new_hyp = original_hyp + middle_laser_reading  # imagine as a triangle
        angle_a_h = math.atan2(end_y, end_x)  # angle between adjacent and hypotenuse
        new_end_x = math.sin(angle_a_h) * new_hyp  # basic trig
        new_end_y = math.cos(angle_a_h)*new_hyp  # using trig to get the new x and y values
        return new_end_x, new_end_y


def get_end_point(start_pos, end_pos, start_angle, end_angle):
    '''
    function takes starting position and orientation and the same for the end
    and calculates new relative position
    :param start_pos: the (x,y) starting point
    :param end_pos: the (x,y) end point
    :param start_angle: the initial robot angle in radians
    :param end_angle:  the end robot angle in radians
    :return:
    '''

    delta_x = end_pos[0] - start_pos[0]  # get change in x and y
    delta_y = end_pos[1] - start_pos[1]
    distance_travelled = math.sqrt(delta_x**2 + delta_y**2)
    angle_turned = smallest_angle_between(start_angle,end_angle)  # get relative angle turned

    end_of_x = -math.sin(angle_turned) * distance_travelled  # use trig to calculate relatively where the robot ended up
    end_of_y = math.cos(angle_turned) * distance_travelled

    return end_of_x, end_of_y


def turn(start_ori, to_turn):
    ''' function turns the robot a specified number of degrees
        prevents infinite turning by increasing buffer on one side of the turn
        :param start_ori the starting_orientation of the robot pre-turning
        :param to_turn the number of radians to turn through
        :return nothing, the robot will be turned as a result of this method
    '''

    turn_dir = math.copysign(1, to_turn)
    target = start_ori + to_turn - turn_dir*TURN_THRESHOLD # cut motors at the target


    turn_twist = Twist()

    angle_left = smallest_angle_between(current_orientation, target)
    while math.copysign(1, angle_left) == turn_dir:
        if math.fabs(angle_left) < math.radians(20):
            turn_twist.angular.z = turn_dir*0.1
        else:
            turn_twist.angular.z = turn_dir*0.5
        pub_vel.publish(turn_twist)
        angle_left = smallest_angle_between(current_orientation, target)


    pub_vel.publish(zero_twist)  # STOP movement


def laser_callback(base_scan):
    '''
    whenever a new laser scan is available update our global laser scan
    :param base_scan: scan from base_scan channel, the latest laser reading
    :return: nothing, function sets global variable
    '''
    global current_laser_data, MAX_RANGE, MIN_RANGE
    current_laser_data = base_scan.ranges
    if MAX_RANGE is None:
        MAX_RANGE = base_scan.range_max  # set to max range of laser
    if MIN_RANGE is None:
        MIN_RANGE = base_scan.range_min


def odometry_callback(odom_msg):
    '''
    Whenever a new odometry reading is made available
    :param odom_msg: the message published by the odometry
    :return: nothing, we just update the global variables
    '''
    global current_position, current_orientation
    pose = odom_msg.pose.pose
    current_position = (pose.position.x, pose.position.y)
    quart = pose.orientation
    quaternion = (quart.x, quart.y, quart.z, quart.w)

    euler = tf.transformations.euler_from_quaternion(quaternion)
    current_orientation = euler[2]  # the angle (in radians) is the yaw of the euler

def laser_cell_intersections(starting_reading, end_of_x, end_of_y, do_print=True):
    training_cells = []
    cells = []
    for i,c in enumerate(GRID_CELLS):  # loop through and get all cells we have traversed
        traversed = c.intersects_line(((0, 0), (end_of_x, end_of_y)))

        distance_to_cell = c.get_angle_distance()[1]
        if traversed and distance_to_cell <= MAX_RANGE:

            # step 5
            occupied = 1 if c.intersects_point(end_of_x, end_of_y) else 0
            angle, distance = c.get_angle_distance()
            four_laser = [starting_reading[x] for x in c.laser_indices]
            for j in range(len(four_laser)):  # get rid of stupid values
                if math.isnan(four_laser[j]):
                    four_laser[j] = MIN_RANGE
                elif four_laser[j] < MIN_RANGE:
                    four_laser[j] = MIN_RANGE
                elif four_laser[j] > MAX_RANGE:
                    four_laser[j] = MAX_RANGE

            sample = (four_laser[0], four_laser[1], four_laser[2],
                      four_laser[3], angle, distance, occupied)
            training_cells.append(sample)
            if do_print:
                log('step 5: {}'.format(sample))
        else:
            occupied = 0

        cells.append((traversed, occupied==1))
    return training_cells, cells

def sense_and_drive():
    '''
    function carries out one iteration of the sense and drive algorithm
    the steps are:
    1. take current laser reading (and initial starting point)
    2. choose a random direction to turn (be very accurate so do at a low speed
    3. move forward until we reach a wall (or at least very close to one)
    4. using odometry traverse path we followed
    5. the cell we hit should have ground truth 1; 0 otherwise
    6. generate 7-element tuples to write to csv file (and increment total number of samples written)
    7. go back 50cm such that the robot has free movement for the next round of the algorithm (this need not be accurate)
    :return: nothing, all results of this function will be written to our training file
    '''
    global num_samples

    log('step 1')
    # step 1
    starting_reading = current_laser_data
    starting_pos = current_position
    starting_ori = current_orientation

    log('step 2')
    # step 2
    to_turn = math.pi * random.uniform(-1.0, 1.0) / 2.0  # we can only turn within the robots laser range here
    turn(starting_ori, to_turn)
    time.sleep(0.2)
    target = starting_ori + to_turn # what we want to aim for (reality might be less accurate)
    log('I want to turn: {} degrees'.format(math.degrees(to_turn)))


    angle_turned = math.degrees(smallest_angle_between(starting_ori, current_orientation))
    log('I think I have turned: {} degrees\n'.format(angle_turned))
    #exit(1)

    log('step 3')
    # step 3
    min_read = get_min_reading()
    if min_read > DISTANCE_THRESHOLD:  # don't start moving if right by a wall
        go_forward = Twist()
        go_forward.linear.x = INITIAL_FORWARD_VEL

        total_to_travel = min_read - DISTANCE_THRESHOLD  # how much we expect to travel
        min_read_loop = get_min_reading()
        while min_read_loop > DISTANCE_THRESHOLD:  # keep going until we reach an obstacle
            go_forward.linear.x = INITIAL_FORWARD_VEL*(min_read_loop/total_to_travel)  # proportional velocity
            pub_vel.publish(go_forward)
            min_read_loop = get_min_reading()

        pub_vel.publish(zero_twist)  # STOP

    log('step 4')
    # step 4

    end_of_x, end_of_y = get_end_point(starting_pos, current_position, starting_ori, target)

    log('I got to this pos from my origin: (' + str(end_of_x) + ", " + str(end_of_y) + ')')
    #e_end_of_x, e_end_of_y = extrapolate_front_laser(end_of_x, end_of_y, current_laser_data)

    training_cells, cells = laser_cell_intersections(starting_reading, end_of_x, end_of_y, do_print=True)
    #e_training_cells, e_cells = laser_cell_intersections(starting_reading, e_end_of_x, e_end_of_y, do_print=False)


    log('step 6')
    # step 6
    num_samples += len(training_cells)
    training_cells.insert(0, ('#', num_its, get_elapsed_time()))

    rospy.loginfo('writing to file. Do not kill')
    with open(TRAINING_FILE, 'a') as csv_file:
        train_writer = csv.writer(csv_file)
        train_writer.writerows(training_cells)  # hopefully this writes all at once
    rospy.loginfo('writing done')

    log('step 7')
    # step 7
    back_twist = Twist()
    back_twist.linear.x = -0.5
    pos_before_reverse = current_position
    # this is just to move the robot out of harms way, no accuracy required here!
    # by choosing a slightly smaller distance, there is less of a chance of a robot being taken off course by an
    # obstacle. This wouldn't really matter though, our aim is to move the robot out of harms way and this should
    # do just that.
    r = rospy.Rate(5) # Hz
    while get_distance(pos_before_reverse, current_position) < 0.2:
        pub_vel.publish(back_twist)
        r.sleep()
    pub_vel.publish(zero_twist)

    draw_cells(cells)
    #draw_cells_ppm(cells, 'src/ppm_map.pgm')

    #draw_cells(e_cells)
    #print "extrapolation " + ("does not" if cells == e_cells else "DOES") + " make a difference"

    time.sleep(1.5) # wait for the robot to come to rest before turning again

    # sometimes rotate through a large angle
    if random.choice([True, False]):
        turn_angle = math.radians(random.choice([120, -120]))
        turn(current_orientation, turn_angle)
        time.sleep(0.2)


def get_elapsed_time():
    return time.time() - start_time + last_time

if __name__ == '__main__':
    random.seed(None) # use system time to seed PRNG

    if os.path.isfile(TRAINING_FILE):
        print 'training file exists'
        with open(TRAINING_FILE, 'r') as f:
            lines = f.read().split('\n')
            headers = [l for l in lines if l.startswith('#')]
            num_its = len(headers)
            num_samples = len([l for l in lines if not l.startswith('#') and l != ''])

            last_time = float(headers[-1].split(',')[2])
            print '{} samples over {} iterations in a total of {:.1f} mins'.format(
                    num_samples, num_its, last_time/60)
            raw_input()


    start_time = time.time()
    rospy.init_node('sense_and_drive', anonymous=True)
    rospy.Subscriber('base_scan', LaserScan, laser_callback)  # lets me listen to laser readings
    rospy.Subscriber('odom', Odometry, odometry_callback)  # lets me listen to the odometry
    pub_vel = rospy.Publisher('cmd_vel', Twist, queue_size=100)  # lets me adjust the velocity of the robot

    generate_grid_cells()  # generate our list of grid boxes

    rospy.loginfo('Waiting for first laser reading and odometry before starting\n')
    r = rospy.Rate(4)  # Hz
    while current_laser_data is None or current_position is None:
        r.sleep()

    rospy.loginfo('starting sense and drive\n')

    while num_samples < INTENDED_SAMPLES:
        num_its += 1 # 1-based indexing
        starting_num_samples = num_samples

        sense_and_drive()  # carry out an iteration of the sense and drive algorithm

        time_elapsed = get_elapsed_time()
        proportion_done = num_samples / float(INTENDED_SAMPLES)
        proportion_left = 1.0-proportion_done
        est_remaining = (proportion_left/proportion_done)*time_elapsed

        rospy.loginfo(('\n\tFinished iteration {} of Sense And Drive.\n' +
            '\t{} total training samples. {} this iteration\n' +
            '\t{:.1f} mins Est Time Remaining. Total Time: {:.1f} mins\n' +
            '\tAvg {:.3f} samples per second').format(
                num_its, num_samples, num_samples - starting_num_samples,
                est_remaining/60, time_elapsed/60, num_samples/time_elapsed))

