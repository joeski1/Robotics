#!/usr/bin/env python
# @author mbway

import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "../common"))

import math
import rospy
import roslib
import tf # for converting from Quaternion to Euler
import threading
import subprocess
import sys

# Messages
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
try:
    from p2os_msgs.msg import SonarArray
except ImportError:
    pass # not available in simulator

# local modules
import viz

# documentation:
# http://wiki.ros.org/APIs
# http://docs.ros.org/api/rospy/html/
# http://wiki.ros.org/common_msgs


tau = 2*math.pi
max_vel = 10

def log(x):
    rospy.loginfo(x)

def log_var(x, x_name):
    log(x_name + ' {}'.format(x))

def sign(x):
    '''
        return 1 if x >= 0 and -1 if x < 0
    '''
    return math.copysign(1, x)

def pol_to_cart(pol):
    r = pol[0]
    theta = pol[1]
    return (r*math.cos(theta), r*math.sin(theta))

class AABB:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def __str__(self):
        return 'AABB({}, {}, {}, {})'.format(self.x, self.y, self.w, self.h)

    def test(self, p):
        px, py = p
        inside_x = px >= self.x and px <= self.x + self.w
        inside_y = py >= self.y and py <= self.y + self.h
        return inside_x and inside_y

    def testpol(self, p):
        return self.test(pol_to_cart(p))

def point_in_circle(point, circle):
    '''
    point : (x, y)
    circle : (x, y, r)
    '''
    return (point[0] - circle[0])**2 + (point[1] - circle[1])**2 < circle[2]**2

# from http://stackoverflow.com/a/1878936
def smallest_angle(theta):
    a = (-theta) % tau
    b = theta    % tau
    return -a if a < b else b
def smallest_angle_between(x, y):
    a = (x-y) % tau
    b = (y-x) % tau
    return -a if a < b else b


def is_simulated():
    p = subprocess.Popen(['ps', '-ax'], stdout=subprocess.PIPE)
    stdout, _ = p.communicate()
    if 'stage_ros/stageros' in stdout:
        return True


def freeFront(ranges, left, right, minRange):
    '''
    Check if there seems to be an object in front of the robot, on the data from
    the left value to the right value, if any of them are less than minRange
    then return's false
    '''
    for i in range(left, right):
        ranger = ranges[i]
        if ranger < minRange:
            return False

    return True


def canMoveForward(ranges, distance):
    '''
    Returns true if the robot can move forward the specified distance
    (Takes into account width of robot)
    '''
    if is_simulated():
        robotWidth = 0.8
    else:
        robotWidth = 0.5
    # How many reading it will check (distributed evenly across 180 degrees)
    values = range(1, 30)
    for i in values:
        maxReading = abs((robotWidth/2)/math.cos(math.radians((180/values[-1])*i)))
        if maxReading > distance:
            maxReading = distance
        reading = ranges[(len(ranges) / values[-1]) * i]
        if reading < maxReading:
            return False
    return True

def getOffendingReading(ranges, distance):
    '''
    Returns true if the robot can move forward the specified distance
    (Takes into account width of robot)
    '''
    min_reading_index = ranges.index(min(ranges))

    if is_simulated():
        robotWidth = 0.8
    else:
        robotWidth = 0.5

    if ranges[min_reading_index] <= distance:
        return min_reading_index
    else:
        return None

def default_laser_check_for_danger(movement, ranges):
    return not canMoveForward(ranges, 1)

def default_sonar_check_for_danger(movement, ranges):
    return min(ranges) < 1


def get_laser_xy(r, number):
    x=0
    y=0
    if number == 255:
        y = r
    elif number < 255:
        angle = (float(number)/512.0) * 180.0
        x = math.cos(math.radians(angle)) * r
        y = math.sin(math.radians(angle)) * r
    elif number > 255:
        angle = 180.0-((float(number)/512.0) * 180.0)
        x = -1 * math.cos(math.radians(angle)) * r
        y = math.sin(math.radians(angle)) * r
    return x, y

def grid_laser_check_for_danger(movement, ranges):
    job = movement.job
    if isinstance(job, Movement.ForwardJob):
        distance_left = math.sqrt((movement.pos[0]-job.target_pos[0])**2+(movement.pos[1]-job.target_pos[1])**2)
        bb = AABB(-0.4, 0, 0.8, distance_left + 0.2)
        #movement.viz.clear_markers()
        #movement.viz.draw_AABB(bb)
        movement._dlog('checking for danger with : ' + str(bb))
        for i, r in enumerate(ranges):
            if i % 2 == 0:
                x, y = get_laser_xy(r, i)
                #movement.viz.place_robot_marker(x, y, color=(0.5, 0, 0.5, 0.7), size=0.05)

                if bb.test((x, y)):
                    print("Interrupted")
                    movement._dlog('{} is in the AABB! In danger!'.format((x, y)))
                    movement.viz.draw_AABB(bb, color=(1, 0, 0, 1))
                    movement.viz.place_robot_marker(x, y)
                    return True
    return False



# odometry information
class Movement:

    class JobState:
        QUEUED   = 0
        STARTING = 1 # job set as current, not yet built up speed
        STARTED  = 2 # job set as current, robot moving
        STOPPING = 3 # job set as current, rolling to a stop
        STOPPED  = 4 # job done, robot came to a stop

        @staticmethod
        def is_moving(state):
            return state == STARTING or state == STARTED or state == STOPPING

        @staticmethod
        def to_str(state):
            return {
                Movement.JobState.QUEUED   : 'QUEUED',
                Movement.JobState.STARTING : 'STARTING',
                Movement.JobState.STARTED  : 'STARTED',
                Movement.JobState.STOPPING : 'STOPPING',
                Movement.JobState.STOPPED  : 'STOPPED',
            }.get(state, 'UNKNOWN ({})'.format(state))


    class Job:
        def __init__(self, movement):
            self.movement = movement
            self.state = Movement.JobState.QUEUED
            self.aborted = False
            self.laser_check_for_danger = None
            self.sonar_check_for_danger = None

        def start(self):
            self.state = Movement.JobState.STARTING

        def __repr__(self):
            return str(self)

    class TurnJob(Job):
        def __init__(self, movement, rel_angle, threshold):
            Movement.Job.__init__(self, movement)

            self.rel_angle = rel_angle
            self.sign = sign(rel_angle)
            self.threshold = threshold
            self.target_angle = None # set in start()


        def __str__(self):
            return 'TurnJob(rel_angle={} degrees, state={})'.format(
                    math.degrees(self.rel_angle), Movement.JobState.to_str(self.state))

        def start(self):
            self.start_angle = self.movement.orientation
            self.target_angle = self.start_angle + self.rel_angle
            Movement.Job.start(self)


        def should_stop(self):
            if self.target_angle is None:
                raise Exception('Job not started! ' + str(self))

            remaining_angle = smallest_angle_between(self.movement.orientation, self.target_angle)

            self.movement._dlog('remaining {} < threshold {} : should stop = {}'.format(
                math.degrees(remaining_angle), math.degrees(self.threshold),
                remaining_angle < self.threshold))

            return math.fabs(remaining_angle) < self.threshold

    class ForwardJob(Job):
        def __init__(self, movement, rel_distance, threshold,
                laser_check_for_danger, sonar_check_for_danger):
            Movement.Job.__init__(self, movement)

            self.rel_distance = rel_distance
            self.threshold = threshold

            # check_for_danger(ranges) : bool (in danger)
            self.laser_check_for_danger = laser_check_for_danger
            self.sonar_check_for_danger = sonar_check_for_danger

        def __str__(self):
            return 'ForwardJob(rel_distance={}, state={})'.format(
                    self.rel_distance, Movement.JobState.to_str(self.state))

        def start(self):
            self.start_pos = self.movement.pos

            rel_move = pol_to_cart((self.rel_distance, self.movement.orientation))

            self.target_pos = (self.start_pos[0] + rel_move[0],
                               self.start_pos[1] + rel_move[1])

            # x, y, r
            self.target_circle = (self.target_pos[0], self.target_pos[1], self.threshold)
            Movement.Job.start(self)

        def should_stop(self):
            if self.target_circle is None:
                raise Exception('Job not started! ' + str(self))

            return point_in_circle(self.movement.pos, self.target_circle)






    def __init__(self):
        # these are set once an odometry callback comes in
        self.pos = None         # (x, y)
        self.orientation = None # radians, anti-clockwise is positive
        self.last_laser_data = None
        self.last_sonar_data = None
        self.robot_moving = False

        # TODO: add logging levels
        self.debug = False # whether to print diagnostic messages

        self.viz = viz.Viz(self)

        self.jobs = []
        self.job = None
        self.job_lock = threading.Lock()

        rospy.Subscriber('odom', Odometry, self._odometry_callback)
        rospy.Subscriber('base_scan', LaserScan, self._laser_callback)
        if not is_simulated(): # not available in simulator
            rospy.Subscriber('sonar', SonarArray, self._sonar_callback)

        self.vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size='None')


    def _dlog(self, x):
        if self.debug:
            log('Movement: ' + x)

    def _new_job(self, job):
        with self.job_lock:
            located = False
            r = rospy.Rate(20)
            while not located:
                # self.pos and self.orientation are set to None initially and are
                # only set once an odometry message is received
                located = self.pos is not None and self.orientation is not None
                r.sleep()

            if self.job is not None:
                # wait for other jobs to finish
                self.jobs.append(job)
                self._dlog('pushing new job. jobs = ' + str(self.jobs))
            else:
                job.start()
                self.job = job
                self._dlog('started new job: ' + str(self.job))

    def _check_using_odometry(self):
        '''
        check whether the current job needs to stop using the odometry data
        '''
        job = self.job
        if job is not None:
            old_state = job.state

            if job.state == Movement.JobState.STARTING and self.robot_moving:
                job.state = Movement.JobState.STARTED
            elif job.state == Movement.JobState.STARTED and not self.robot_moving:
                # crashed
                job.state = Movement.JobState.STOPPED
                self.stop()
                rospy.logerr('CRASHED')
                rospy.signal_shutdown('crashed')
            elif job.state == Movement.JobState.STARTED and job.should_stop():
                job.state = Movement.JobState.STOPPING
            elif job.state == Movement.JobState.STOPPING and not self.robot_moving:
                job.state = Movement.JobState.STOPPED
                self.stop()

            if job.state != old_state:
                self._dlog('job state changed from {} to {}'.format(
                    Movement.JobState.to_str(old_state), Movement.JobState.to_str(job.state)))

        with self.job_lock:
            # still the same job as above
            # current job finished
            if job == self.job and job is not None and job.state == Movement.JobState.STOPPED:
                self._dlog('removing current job because it stopped: ' + str(job))
                self.job = None

            # might have just been set to None above, or might be None from
            # being set somewhere else
            if self.job is None and len(self.jobs) > 0:
                job = self.jobs.pop(0)
                job.start()
                self.job = job
                self._dlog('started new buffered job: ' + str(self.job) + '. jobs = ' + str(self.jobs))


    def _odometry_callback(self, msg):
        '''
            Receives messages about the pose (position+orientation) and changes in
            linear and angular velocity.

            The data is extracted and stored in the globals pos and orientation
        '''
        # http://docs.ros.org/kinetic/api/nav_msgs/html/msg/Odometry.html
        # msg.twist = odometry velocity { linear : Vec3, angular: Vec3 }
        # msg.pose  = odometry position { position : Vec3, orientation : Q }

        # the outer pose contains a covariance matrix along with the inner pose
        pose = msg.pose.pose

        # the z component is not useful for our robot
        self.pos = (pose.position.x, pose.position.y)

        # transform from a quaternion to an Euler angle (roll, pitch, yaw)
        o = pose.orientation
        q = (o.x, o.y, o.z, o.w) # extract object into tuple
        e = tf.transformations.euler_from_quaternion(q)

        # orientation = yaw
        self.orientation = e[2]

        t = msg.twist.twist
        linear = t.linear.x
        angular = t.angular.z
        self.robot_moving = math.fabs(linear) > 0.002 or math.fabs(angular) > 0.003


        #self._dlog('odometry update: pos={}, orientation={}'.format(self.pos, self.orientation))
        #self._dlog('job queue: {}  current job: {}'.format(self.jobs, self.job))

        self._check_using_odometry()


    def _check_using_laser(self):
        '''
            check whether the current job needs to be aborted using the laser data
        '''
        job = self.job
        if (job is not None and (
                job.state == Movement.JobState.STARTING or
                job.state == Movement.JobState.STARTED) and
            self.last_laser_data is not None and
            job.laser_check_for_danger is not None):

            if job.laser_check_for_danger(self, self.last_laser_data):
                self.stop()
                job.aborted = True
                job.state = Movement.JobState.STOPPING
                self._dlog('aborting job because of laser: ' + str(job))

    def _laser_callback(self, msg):
        '''
            using the laser scan data, determine whether the robot should not
            continue on its current heading
        '''
        rmin = msg.range_min
        rmax = msg.range_max
        self.last_laser_data = [max(min(x, rmax), rmin) for x in msg.ranges]

        self._check_using_laser()

    def _check_using_sonar(self):
        '''
            check whether the current job needs to be aborted using the sonar data
        '''
        job = self.job
        if (job is not None and (
                job.state == Movement.JobState.STARTING or
                job.state == Movement.JobState.STARTED) and
            self.last_sonar_data is not None and
            job.sonar_check_for_danger is not None):

            if job.sonar_check_for_danger(self, self.last_sonar_data):
                self.stop()
                job.aborted = True
                job.state = Movement.JobState.STOPPING
                self._dlog('aborting job because of sonar: ' + str(self.job))


    def _sonar_callback(self, msg):
        # ranges has readings 0-7 going left to right
        forward_readings = msg.ranges[1:7] # exclude readings 0 and 7
        self.last_sonar_readings = forward_readings

        self._check_using_sonar()





    def rotate_angle(self, rel_angle, vel=max_vel, threshold=None):
        '''
            rotate the robot through a given relative angle at a given velocity
        '''

        if threshold is None:
            if is_simulated():
                threshold = math.radians(5)
            else:
                if rel_angle >= math.radians(45):
                    threshold = math.radians(35)
                else:
                    threshold = max(math.fabs(rel_angle) - 0.1, 0.1)


        rel_angle = smallest_angle(rel_angle)
        vel *= sign(rel_angle)
        self._dlog('job: turn angle {} degrees'.format(math.degrees(rel_angle)))

        job = Movement.TurnJob(self, rel_angle, threshold)

        self._new_job(job)

        r = rospy.Rate(20)

        while job.state == Movement.JobState.QUEUED:
            r.sleep()

        if job.should_stop():
            self._dlog('aborted job before starting because angle was too small: ' + str(self.job))
            job.state = Movement.JobState.STOPPED
            return # angle too small to bother moving at all

        while not rospy.is_shutdown() and (
                job.state == Movement.JobState.STARTING or
                job.state == Movement.JobState.STARTED):
            # must continuously publish otherwise the robot stops
            self.rotate_vel(vel)
            r.sleep()

        self.stop()

        while not rospy.is_shutdown() and job.state != Movement.JobState.STOPPED:
            r.sleep()

        start = job.start_angle
        stop = self.orientation
        actual_angle = smallest_angle_between(start, stop)
        self._dlog('actual angle turned: {} degrees'.format(math.degrees(actual_angle)))
        self._dlog('squared error: {}'.format((math.degrees(job.rel_angle) - math.degrees(actual_angle))**2))


    def rotate_vel(self, vel):
        '''
            rotate the robot at a given velocity
        '''
        t = Twist()
        t.angular.z = vel
        self.vel_pub.publish(t)

    def stop(self):
        self.vel_pub.publish(Twist())

    def forward_distance(self, distance, vel=max_vel, threshold=None,
            laser_check_for_danger=default_laser_check_for_danger,
            sonar_check_for_danger=default_sonar_check_for_danger):
        '''
            move the robot forward a given distance at a given velocity
            returns True if the job was aborted because of immanent collision
            returns False if the job ended of its own accord
        '''
        self._dlog('job: forward distance {} metres'.format(distance))

        if threshold is None:
            if is_simulated():
                threshold = 0.1
            else:
                threshold = 0.55

        job = Movement.ForwardJob(self, distance, threshold, laser_check_for_danger, sonar_check_for_danger)
        self._new_job(job)

        r = rospy.Rate(20)

        while job.state == Movement.JobState.QUEUED:
            r.sleep()

        # not moving yet

        if job.should_stop(): # already met target
            self._dlog('aborted job before starting because movement was too small: ' + str(self.job))
            job.state = Movement.JobState.STOPPED
            return True # distance too small to bother moving at all

        if (laser_check_for_danger is not None and self.last_laser_data is not None and
            laser_check_for_danger(self, self.last_laser_data)):
            job.state = Movement.JobState.STOPPED
            self._dlog('aborted job before starting because of laser: ' + str(job))
            return False # too close to wall, can't move

        if (sonar_check_for_danger is not None and self.last_sonar_data is not None and
            sonar_check_for_danger(self, self.last_sonar_data)):
            job.state = Movement.JobState.STOPPED
            self._dlog('aborted job before starting because of sonar: ' + str(job))
            return False # too close to wall, can't move

        while not rospy.is_shutdown() and (
                job.state == Movement.JobState.STARTING or
                job.state == Movement.JobState.STARTED):
            # must continuously publish otherwise the robot stops
            self.forward_vel(vel)
            r.sleep()

        self.stop()

        while not rospy.is_shutdown() and job.state != Movement.JobState.STOPPED:
            r.sleep()


        start = job.start_pos
        stop = self.pos
        actual_distance = math.sqrt((stop[0]-start[0])**2 + (stop[1]-start[1])**2)
        self._dlog('actual distance travelled: {}'.format(actual_distance))
        self._dlog('squared error: {}'.format((job.rel_distance - actual_distance)**2))

        return not job.aborted

    def forward_1m(self):
        '''
            move the robot forward 1 metre
        '''
        if is_simulated():
            return self.forward_distance(1, 1, 0.1)
        else:
            return self.forward_distance(1, max_vel, 0.55)


    def forward_vel(self, vel):
        '''
            move the robot forward at a given velocity
        '''
        t = Twist()
        t.linear.x = vel
        self.vel_pub.publish(t)


    def move_twist(twist):
        '''
            move the robot with a given twist
        '''
        self.vel_pub.publish(twist)


print('turn on motors and press enter to continue')
raw_input()

if __name__ == '__main__':
    rospy.init_node('movement', anonymous=True)

    if is_simulated():
        print('running in a simulator')
    else:
        print('running on the robot')

    mov = Movement()
    mov.debug = True


    #finished = mov.forward_1m()

    while not rospy.is_shutdown():
        mov.rotate_vel(999)
    #finished = mov.forward_distance(999)

    '''
    while not rospy.is_shutdown():
        print('running again')
        finished = mov.forward_distance(1, 1)
        mov.rotate_angle(math.radians(90), 1)
        if not finished:
            print('forward job was aborted')
        raw_input()
    '''

    rospy.spin()
