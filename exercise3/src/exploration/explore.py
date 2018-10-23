#!/usr/bin/env python
import os
import sys
import subprocess
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "common"))

from movement import Movement, default_laser_check_for_danger, smallest_angle_between, point_in_circle, is_simulated, canMoveForward
from viz import Viz

from world import World
import rospy
import roslib
import math
import threading
import copy
import time
import random
import heapq
# PLAN:
#   - check_current_plan:
#       * if no plan or "bad plan" then generate a new plan
#       * plan is bad if route ends up blocked or a new unknown cell is more interesting (and close)
#
#   - new_plan:
#     * Threshold map into wall(1), free(0) and unknown(0.5) (and possibly downscale for efficiency)
#     * Find closest unknown (choose 100 random unknown loctations on edge of current map)
#       and see if there is a path there through free cells (using A*?)
#     * Try to execute the best plan MINIMISE(A*distance of route + B*abs(cell-0.5) + C*number of unknown particles nearby)
#

# Because the normal python one is broken...
# Adapted From: https://www.safaribooksonline.com/library/view/python-cookbook-3rd/9781449357337/ch01s05.html
class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, item, priority):
        heapq.heappush(self._queue, (priority, self._index, item))
        self._index += 1

    def pop(self):
        return heapq.heappop(self._queue)[-1]

    def empty(self):
        return len(self._queue) == 0

class PlanStatus:
    WAITING  = 0
    PLANNING = 1
    FINISHED = 2


# TODO: I assume this whole thing works
class Planner(threading.Thread):
    def __init__(self, threads=22):
        super(Planner, self).__init__(name="Planner")
        self.enabled = True
        self.debug = 0
        self._lock = threading.Lock()
        self._world = World()

        # Plan generators
        self._generators = {}
        self._planning = False
        self._timeout = False
        self.timeout_with_plan    = 4  # Wait x seconds after getting the first plan for other plans to finish
        self.timeout_without_plan = 15  # Cancel planning session if no plans are found after x seconds
        for i in range(0, threads - 2): # Two other threads are being used (executor and planner)
            angle = 360.0 * (float(i) / float(threads - 2))
            generator = PlanGenerator(self, angle)
            self._generators[angle] = generator
            generator.start()

        # Plan executor
        self._executor = PlanExecutor(self)
        self._executor.start()


    def debug_message(self, level, message):
        if self.debug >= level:
            rospy.loginfo(message)


    def run(self):
        '''
        Finds the best plan from possible plans
        '''
        r = rospy.Rate(10)
        while not rospy.is_shutdown():

            # Wait for map to update
            self.debug_message(1, "Requesting a map update")
            self._world.update()
            self.debug_message(1, "Got a map update")

            # Start planning
            self._planning = True
            planning_time = time.time()

            # Wait until all generators have started planned
            self.debug_message(1, "Waiting for all generators to start planning")
            waiting = 1
            while waiting and not rospy.is_shutdown():
                r.sleep()
                waiting = len(filter(lambda g : self._generators[g].status == PlanStatus.WAITING, self._generators)) # Generators have all started
                self.debug_message(3, "waiting for %d" % waiting)

            self.debug_message(1, "All generators have started planning")

            # Wait until all generators have a plan or the timeout has passed
            start_time = time.time()
            found_plan = False
            plan_time = sys.maxsize
            waiting = True
            self.debug_message(1, "Waiting for generators to finish planning")
            while waiting and not rospy.is_shutdown():
                r.sleep()
                waiting = len(filter(lambda g : self._generators[g].status != PlanStatus.FINISHED, self._generators))
                self.debug_message(3, "waiting for {}".format(waiting))

                # Check for if any plans have finished and start the second timer
                if not found_plan:
                    finished = len(filter(lambda g : (self._generators[g].status == PlanStatus.FINISHED and
                                                      self._generators[g].plan != None), self._generators))
                    if finished > 0:
                        plan_time = time.time()
                        found_plan = True

                # Check to see if we are outside the alloted time
                if ((time.time() - start_time > self.timeout_without_plan and not found_plan) or
                    (time.time() - plan_time  > self.timeout_with_plan and found_plan)):
                    waiting = 0
                    self.debug_message(1, "Planning expired allotted time")

            self.debug_message(1, "Planning has ended, ensure all PlanGenerators have ended")
            self._planning = False

            # Wait until all generators have noticed the planning has ended
            waiting = 1
            while waiting and not rospy.is_shutdown():
                r.sleep()
                waiting = len(filter(lambda g : self._generators[g].status != PlanStatus.WAITING, self._generators))
                self.debug_message(3, "waiting for %d" % waiting)

            self.debug_message(1, "All generators have finished, finding the best plan")
            best_plan = None
            best_score = sys.maxsize
            total_plans = 0
            for g in self._generators:
                gen = self._generators[g]
                if gen.plan is not None:
                    total_plans += 1
                    score = gen.plan.score()
                    if score < best_score:
                        best_plan = gen.plan
                        best_score = score
            self._executor.update_route(best_plan)

            # Output time it took
            total_time = time.time() - start_time
            if total_plans == 0:
                self.debug_message(0, "Failed to formulate a plan after %.3fs" % total_time)
            else:
                self.debug_message(0, "Took %.3fs to formulate a plan (had %i plans)" % (total_time, total_plans))
                self.debug_message(0, "Best Plan: {}".format(best_plan.path))
                self.debug_message(0, "Best Score: {}".format(best_plan.score()))


    def is_free(self, grid_coord):
        return self._world.get_prob(grid_coord) < 0.35

    def is_occupied(self, grid_coord):
        return self._world.get_prob(grid_coord) > 0.65

    def is_unknown(self, grid_coord):
        return not (self.is_free(grid_coord) or self.is_occupied(grid_coord))


    def find_route(self, start, end, interrupt=lambda: False, is_goal=lambda c: False, dlevel=3):
        # Adapted from: http://www.redblobgames.com/pathfinding/a-star/implementation.html
        def heuristic(a, b):
            ax, ay = a
            bx, by = b
            return abs(ax - bx) + abs(ay - by)

        def can_robot_fit(coord):
            # Create a massive square where around the coord by combining cells
            neighbours = create_square(coord, True)
            for i in range(0, 2):
                neighbours = map(lambda c: create_square(c, True), neighbours)
                neighbours = set([item for sublist in neighbours for item in sublist])
            total_neighbours = len(neighbours)
            neighbours = filter(lambda (x, y, _): not self.is_occupied((x,y)), neighbours)
            self.debug_message(dlevel, "{}: {} robot can fit?".format(coord, len(neighbours) == total_neighbours))
            return len(neighbours) == total_neighbours

        max_steps = 1000
        frontier = PriorityQueue()
        frontier.push(start, 0)
        came_from = {}
        cost_so_far = {}
        came_from[start] = None
        cost_so_far[start] = 0

        r = rospy.Rate(10)
        while not frontier.empty() and not interrupt():
            r.sleep()
            current = frontier.pop()
            self.debug_message(dlevel, "Looking at current: {}".format(current))
            if current == end or is_goal(current):
                self.debug_message(dlevel, "It is goal")
                # Trace back the path
                path = []
                node = end
                while not node == start:
                    path.append(node)
                    node = came_from[node]
                path.append(start)
                path.reverse()
                return Plan(self, path)

            # Check that the current coordinate has a path shorter than the maximum
            if cost_so_far[current] < max_steps:
                (x, y) = current

                # Neighbours are all cells around the current point (minus the current point).
                neighbours = create_square((x, y, 0), False) - {(x, y, 0)}

                # For each neighbour, check that the robot will fit inside (is free or unknown) a 3x3 around the neighbour
                # Only keep the neighbours where the robot will fit.
                neighbours = filter(lambda (x, y, c): (x, y) == end or can_robot_fit((x,y,c)), neighbours)

                # For each of the neighbours that the robot will fit in
                for (x, y, c) in neighbours:
                    next = (x,y)

                    # If next cell is unknown, double the cost to prefer free cells
                    #if self.is_unknown(next):
                    #    c *= 2

                    new_cost = cost_so_far[current] + c
                    # Check that the neighbour has not been visited or it has a shorter path
                    if next not in cost_so_far or new_cost < cost_so_far[next]:
                        cost_so_far[next] = new_cost
                        priority = straight_distance(end, next)#new_cost + heuristic(end, next)
                        frontier.push(next, priority)
                        came_from[next] = current
                        self.debug_message(dlevel, "{}: Added to queue with {} priority".format(next, priority))
            else:
                self.debug_message(dlevel, "Outside max steps: {}".format(cost_so_far[current]))


class PlanGenerator(threading.Thread):

    def __init__(self, planner, angle, prefix_name = "PlanGenerator", max_view=1000):
        super(PlanGenerator, self).__init__(name=prefix_name + "(angle=%.2f, max_view=%i)" % (angle, max_view))
        self._planner = planner
        self.status = PlanStatus.WAITING
        self.plan = None
        self.max_view = max_view # max distance to check for unknowns

        # Sin and cos of angle (for ray tracing)
        self.angle = angle
        self.s = math.sin(math.radians(angle))
        self.c = math.cos(math.radians(angle))


    def run(self):
        r = rospy.Rate(10)
        dlevel = 2
        #if self.angle == 0.0:
        #    dlevel = -1
        while not rospy.is_shutdown():
            # Wait until we should start planning
            self.status = PlanStatus.WAITING
            while not self._planner._planning and not rospy.is_shutdown():
                r.sleep()
            if rospy.is_shutdown():
                return

            self.plan = None
            self.status = PlanStatus.PLANNING

            robot_grid_x, robot_grid_y = self._planner._world.pose_grid()

            # Ray tracing from the robot in given angle looking for unknowns
            distance = 0
            found_unknowns = False
            grid_coord = None
            while not found_unknowns and distance < self.max_view and self._planner._planning:
                r.sleep()
                distance += 1
                grid_coord = (int(round(robot_grid_x + distance * self.s)), int(round(robot_grid_y + distance * self.c)))
                if self._planner.is_unknown(grid_coord) and not grid_coord == (robot_grid_x, robot_grid_y):
                    found_unknowns = True

            # Find route if we have a target
            if found_unknowns:
                self._planner.debug_message(dlevel, "{}: Went {} cells to find an unknown".format(self.angle, distance))
                self.plan = self._planner.find_route(grid_coord, (robot_grid_x, robot_grid_y),
                                                     lambda: not self._planner._planning, # Interrupt route finding when planning is over
                                                     lambda c: straight_distance(c, (robot_grid_x, robot_grid_y)) < 1, # Route start is x meters away from robot
                                                     dlevel=dlevel)
                if self.plan is None:
                    self._planner.debug_message(dlevel, "{}: Could not find route from ({}, {}) to {} (interrupted?: {})".format(self.angle, robot_grid_x, robot_grid_y, grid_coord, not self._planner._planning))
                else:
                    self.plan.path.reverse() # Finding route from goal to robot so have to reverse path
                    self._planner.debug_message(dlevel, "{}: Found route from ({}, {}) to {} (interrupted?: {})".format(self.angle, robot_grid_x, robot_grid_y, grid_coord, not self._planner._planning))
            else:
                self._planner.debug_message(dlevel, "{}: Went {} cells but NO UNKNOWNS".format(self.angle, distance))

            # Wait until planning is over
            self.status = PlanStatus.FINISHED
            while self._planner._planning and not rospy.is_shutdown():
                r.sleep()
            if rospy.is_shutdown():
                return

class PlanExecutor(threading.Thread):
    def __init__(self, planner, name = "PlanExecutor"):
        super(PlanExecutor, self).__init__(name=name)
        self.plan = None
        self._planner = planner
        self._world = planner._world
        self._movement = Movement()
        self._lock = threading.Lock()
        self._latest_plan = None
        self._latest_changed = False
        if is_simulated():
            self.fallback = "hoover.py"
        else:
            self.fallback = "wall_hugger.py"


    def update_route(self, route):
        if route is not None:
            with self._lock:
                if route != self._latest_plan:
                    self._latest_plan = route
                    # Is new route significantly better than current route
                    if self.plan is None or route.score() + 3 < self.plan.score():
                        self._latest_better = True


    def _check_for_danger(self, movement, ranges):
        if self._latest_better or not self._planner.enabled:
            return True
        else:
            distance = 0.6
            if is_simulated():
                distance = 0.8
            return default_laser_check_for_danger(self._movement, ranges)#not canMoveForward(ranges, distance)


    def _rotate(self, angle):
        if not self._latest_better or not self._planner.enabled:
            self._movement.rotate_angle(angle, vel=0.4,threshold=math.radians(5))
        return True # Rotation never fails


    def _forward(self, distance):
        if not self._latest_better or not self._planner.enabled:
            return self._movement.forward_distance(distance,
                            vel=0.4,
                            threshold=0.1,
                            laser_check_for_danger=self._check_for_danger)
        return False

    def run(self):
        r = rospy.Rate(10)
        fallback = None
        FNULL = open(os.devnull, "w")
        while not rospy.is_shutdown():
            while not self._planner.enabled and not rospy.is_shutdown():
                r.sleep()

            if rospy.is_shutdown():
                return

            # Copy latest route so that it doesn't change while processing
            with self._lock:
                self.plan = self._latest_plan
                self._latest_plan = None
                self._latest_better = False

            if self.plan is None:
                # Don't know what to do so start wall hugger
                if fallback is None:
                    fallback = subprocess.Popen(["rosrun", "exercise3", self.fallback], stdout=FNULL)
                    self._planner.debug_message(0, "Using {}".format(self.fallback))
                    self._world.publish_path([])

                #self._forward(999)
                #self._rotate(random.random() * 2 * math.pi)
            else:
                # Kill wall hugger
                if fallback is not None:
                    fallback.kill()
                    fallback = None

                self._planner.debug_message(0, "Trying to execute plan: {}".format(self.plan.path))
                self._world.publish_path(self.plan.path)
                # Try to follow the route
                steps = 0
                while self.plan is not None and steps < 8 + 2 * len(self.plan.turns) and not self._latest_better: # and self.plan.num_walls_on_path() < 100
                    steps += 1
                    # Find robots grid location
                    robot_grid_x, robot_grid_y = self._world.pose_grid(pose_msg=self._world.pose_msg)

                    on_line = False

                    # Find closest node in path
                    closest_node = self.plan.nearest_node((robot_grid_x, robot_grid_y))
                    closest_turn = self.plan.nearest_turn((robot_grid_x, robot_grid_y))

                    # While the robot it at the closest cell in the path, use the next node
                    while closest_node is not None and robot_at_coord((robot_grid_x, robot_grid_y), closest_node):
                        on_line = True
                        closest_node = self.plan.next_node(closest_node)
                        self._planner.debug_message(1, "I'm in that cell, moving to the next cell: {}".format(closest_node))

                    # While the robot it at the closest turn in the path, use the next node
                    while closest_turn is not None and robot_at_coord((robot_grid_x, robot_grid_y), closest_turn):
                        closest_turn = self.plan.next_turn(closest_turn)
                        self._planner.debug_message(1, "I'm in that turn, moving to the next turn: {}".format(closest_turn))

                    self._planner.debug_message(0, "I'm at: {}, closest is {}, aiming for {}".format((robot_grid_x, robot_grid_y), closest_node, closest_turn))

                    # If the robot is in the last cell, or failed to execute the plan too many times then set the route to None
                    if closest_node is None or closest_turn is None:
                        self._planner.debug_message(0, "At destination")
                        self._planner.debug_message(-1, "Made {}/{} steps".format(steps, 2 * len(self.plan.turns)))
                        self.plan = None
                        closest_node = None

                    # If we have a closest node
                    if closest_node is not None:
                        target_x, target_y = closest_node
                        # Point robot towards closest_node TODO: Check this is correct
                        robot_rotation = self._world.pose_rotation()
                        node_rotation = (math.atan2(target_x - robot_grid_x, target_y - robot_grid_y) - math.pi / 2) % (2 * math.pi)

                        self._planner.debug_message(2, "Before: Robot_rotation: {} Node rotation: {}".format(math.degrees(robot_rotation), math.degrees(node_rotation)))
                        self._rotate(smallest_angle_between(robot_rotation, node_rotation))
                        self._planner.debug_message(2, "After: Robot_rotation: {}".format(math.degrees(self._world.pose_rotation())))

                        # Calculate how long robot can travel in a straight line on the path
                        target = closest_turn if on_line else closest_node
                        distance = straight_distance(target, (robot_grid_x, robot_grid_y)) * self._world.map.resolution
                        self._planner.debug_message(2, "Travelling: {}m".format(distance))
                        self._forward(distance)

                if self.plan is not None:
                    self._planner.debug_message(0, "Aborted Plan")
                    self.plan = None # Give up plan

class Plan:
    def __init__(self, planner, path):
        self._planner = planner
        self.path = path
        self.turns = []
        self.target = path[-1]

        # Find the turning points of the route
        prev_point = None
        prev_angle = None
        for c in path:
            if prev_point is None:
                prev_point = c
                self.turns.append(c)
            else:
                x, y = c
                px, py = prev_point
                angle = math.atan2(x - px, y - py)
                if prev_angle is None:
                    prev_angle = angle
                elif prev_angle != angle:
                    prev_angle = angle
                    prev_point = c
                    self.turns.append(c)


    def global_weight(self):
        return self._planner._world.get_prob(self.target)

    def _nearest(self, grid_coord, list):
        closest_node = None
        closest_score = sys.maxsize
        for index, node in enumerate(list):
            distance = straight_distance(grid_coord, node)
            if distance < closest_score:
                closest_node = node
                closest_score = distance
        return closest_node

    def _next(self, node, list):
        updated_node = False
        for n in self.path:
            if updated_node:
                return n
            elif n == node:
                updated_node = True

    def nearest_node(self, grid_coord):
        return self._nearest(grid_coord, self.path)

    def nearest_turn(self, grid_coord):
        return self._nearest(grid_coord, self.turns)

    def next_node(self, node):
        return self._next(node, self.path)

    def next_turn(self, node):
        return self._next(node, self.turns)


    def robot_distance(self):
        """
        Straight line distance the robot is from the goal
        """
        robot_grid_coord = self._planner._world.pose_grid(pose_msg=self._planner._world.pose_msg)
        return straight_distance(self.target, robot_grid_coord)


    def num_walls_on_path(self):
        return len(map(self._planner.is_occupied, self.path))


    def score(self):
        '''
        Calculates how good a plan is (lowest score is best)
        '''
        robot_coord = self._planner._world.pose_grid(pose_msg=self._planner._world.pose_msg)

        distance_weight     = 0.1 * len(self.path)
        uncertainty_weight  = 200 * math.fabs(self.global_weight() - 0.5)
        turning_weight      = 2   * len(self.turns)
        walls_weight        = 4   * self.num_walls_on_path() # Count how many cells on the path are walls
        nearest_node_weight = 8   * straight_distance(robot_coord, self.nearest_node(robot_coord)) # How far is the closest node
        nearby_weight       = 1   # TODO: Count number of unknowns in a circle around coord (1m radius?)
        return distance_weight + uncertainty_weight + turning_weight + walls_weight + nearest_node_weight + nearby_weight


def create_square(grid_coord, diags=False):
    """
    Creates a square of coordinates around the passed coordinates
    """
    (x, y, c) = grid_coord
    straight = 1
    diag = 1.42
    if diags:
        return {(x-1, y-1, c + diag),   (x-1, y, c + straight), (x, y+1, c + diag),
                (x, y-1, c + straight), (x, y, c),              (x, y+1, c + straight),
                (x+1, y-1, c + diag),   (x+1, y, c + straight), (x+1, y+1, c + diag)}
    else:
        return {(x-1, y, c + straight), (x, y-1, c + straight), (x, y, c), (x, y+1, c + straight), (x+1, y, c + straight)}

def straight_distance(a, b):
    x1, y1 = a
    x2, y2 = b
    return math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))

def robot_at_coord(robot_grid, grid_coord):
    """
    Is robot at coord
    """
    x, y = grid_coord
    return point_in_circle(robot_grid, (x, y, 5))

def init():
    planner = Planner(threads=40) # 360 generators + 2 control threads
    planner.debug = 0
    planner.start()

def dummy_explore():
    mov = Movement()
    mov.forward_distance(999, vel=0.4)
    print('dummy explore done ')
    sys.exit(0)


if __name__ == '__main__':
    rospy.init_node('explore')

    print('turn on motors and press enter to continue')
    raw_input()

    #dummy_explore()
    init()
    rospy.spin()
