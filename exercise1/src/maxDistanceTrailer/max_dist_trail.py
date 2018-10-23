#!/usr/bin/env python
#class used to do the max distance trailing algorithm
#@author Charlie Street

import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "../common"))


import rospy
from sensor_msgs.msg import LaserScan
import math
import random

from laser_simplifier import LaserFiveWays
from directions import Direction
from directions import toDegrees
from Distances import SENSOR_TO_CENTER
from movement import Movement, AABB, grid_laser_check_for_danger

class MaxDist:

    def __init__(self):
        '''
        constructor sets up node among other things
        '''

        #CONSTANTS
        self.MULTIPLIER = 3 #1/(Multiplier^visited)
        self.DECAY = 1.2 #times by score
        self.DIAGONAL_MOVE =  math.sqrt(2)#amount to move when travelling diagonally
        self.VH_MOVE = 1#amount to move when travelling vertically or horizontally
        self.THRESHOLD_VERTICAL = 2 *  1.5 - SENSOR_TO_CENTER #threshold for moving vertically
        self.THRESHOLD_HORIZONTAL = 2 * 1.5
        self.THRESHOLD_DIAGONAL = 2 * math.sqrt(math.pow(1.5-SENSOR_TO_CENTER,2)+math.pow(1.5,2))#threshold for moving diagonally
        self.GRIDWH = 1
        self.AMOUNT_TO_ADD = 0.9



        self.mover = Movement()
        self.laserStuff = LaserFiveWays(self.mover)
        #self.mover.debug = True
        self.currentDir = Direction.NORTH
        self.currentPos = (0,0)
        self.grid_coords = {}
        self.grid_coords[(0,0)] = 1
        self.hasMovedHorizontally = False#used to stop too much horizontal turning


    def roundOfMovement(self):
        '''method makes one action based on laser readings'''
        laser_readings = self.laserStuff.getFiveDirection()
        if laser_readings is None:
            return
        newPoints = self._getNewPoints()
        coordsWithScores = {}

        for i in range(len(newPoints)):#combining two arrays together
            if newPoints[i] not in self.grid_coords.keys():#check if in grid
                currentThreshold = self.THRESHOLD_VERTICAL
                if i==0 or i==4:
                    currentThreshold = self.THRESHOLD_HORIZONTAL
                elif i == 1 or i == 3:
                    currentThreshold = self.THRESHOLD_DIAGONAL

                #ADD VERTICAL HERE
                if laser_readings[i] >= currentThreshold:#if should be added to grid
                    self.grid_coords[newPoints[i]] = 0#add to grid


            #create new dictionary with coordinates to scores
            coordsWithScores[newPoints[i]] = self._generateScore(newPoints[i],laser_readings[i],i)

        if self.hasMovedHorizontally:
            self.hasMovedHorizontally = False
            #coordsWithScores[newPoints[0]] = 0
            #coordsWithScores[newPoints[4]] = 0

        rospy.loginfo('COORDINATES MOVABLE: ' + str(len(coordsWithScores.keys())))
        #generate statistics to output
        '''
        rospy.loginfo('MOVE STATISTICS')
        for i in coordsWithScores:
            if i in self.grid_coords.keys():
                rospy.loginfo('Position (' + str(i[0]) + ", " + str(i[1]) + ") has score: " + str(coordsWithScores[i]))
                if coordsWithScores[i]==0:
                    rospy.loginfo('ZERO: dictionary value is: ' + str(self.grid_coords[i]))
        '''

        try:
            bestCoord = self._maxScoreCoord(coordsWithScores)#this is where we want to move too

            #if east or west are best turn and have a look, don't move because we have a lack of certainty
            if bestCoord == newPoints[0]:
                self.mover.rotate_angle(math.radians(-90))#move clockwise 90 degrees
                self.currentDir = self._getNewPose(-90)
                self._decay()
                self.hasMovedHorizontally = True
                rospy.loginfo("EASTSIDE")
                return
            elif bestCoord == newPoints[4]:
                self.mover.rotate_angle(math.radians(90))#move anti-clockwise -90 degrees
                self.currentDir = self._getNewPose(90)
                self._decay()
                self.hasMovedHorizontally = True
                rospy.loginfo("WESTSIDE")
                return

            directionAsIndex = None
            finished = False
            newAngle = 0
            if bestCoord is not None:
                for i in range(len(newPoints)):
                    if(newPoints[i] == bestCoord):
                        directionAsIndex = i
                        break

                rospy.loginfo("Best Coordinate: ("+str(bestCoord[0])+", "+str(bestCoord[1])+") with " + str(coordsWithScores[bestCoord]))

                newAngle = self._angleToTurn(directionAsIndex) #get the angle to turn
                self.mover.rotate_angle(math.radians(newAngle),1.0)

                distanceToTravel = self.DIAGONAL_MOVE if (newAngle % 90 != 0) else self.VH_MOVE #selects diagonal distance if 45 or -45 degrees
                finished = self.mover.forward_distance(distanceToTravel, 1, None, grid_laser_check_for_danger, None)#true means finished, false means aborted
            else:
                # TODO: Should be possible to get this, but don't because of inaccuracies in distance detection
                rospy.loginfo("No good coordinate")

            saveOurAsses = random.choice([45, -45])
            while not finished: # TODO: Make a better choice
                if bestCoord is not None:
                    rospy.logerr("MISTAKES WERE MADE")
                self.mover.rotate_angle(math.radians(saveOurAsses),1.0)
                newAngle += saveOurAsses

                distanceToTravel = self.DIAGONAL_MOVE if (newAngle % 90 != 0) else self.VH_MOVE #selects diagonal distance if 45 or -45 degrees
                finished = self.mover.forward_distance(distanceToTravel, 1, None, grid_laser_check_for_danger, None)#true means finished, false means aborted

            #setting new coordinate and position
            angleMoved = (toDegrees(self.currentDir) + newAngle) % 360
            rospy.loginfo("was: {} now: {}".format(toDegrees(self.currentDir), angleMoved))
            if angleMoved == 0:
                self.currentPos = (self.currentPos[0]    , self.currentPos[1] + 1)
            elif angleMoved == 45:
                self.currentPos = (self.currentPos[0] - 1, self.currentPos[1] + 1)
            elif angleMoved == 90:
                self.currentPos = (self.currentPos[0] - 1, self.currentPos[1])
            elif angleMoved == 135:
                self.currentPos = (self.currentPos[0] - 1, self.currentPos[1] - 1)
            elif angleMoved == 180:
                self.currentPos = (self.currentPos[0]    , self.currentPos[1] - 1)
            elif angleMoved == 225:
                self.currentPos = (self.currentPos[0] + 1, self.currentPos[1] - 1)
            elif angleMoved == 270:
                self.currentPos = (self.currentPos[0] + 1, self.currentPos[1])
            else:
                self.currentPos = (self.currentPos[0] + 1, self.currentPos[1] + 1)

            rospy.loginfo("newCoord: {}".format(self.currentPos))
            #if 45 degree angle move a further 45 degrees to line up to vertical or horizontal
            if newAngle % 90 != 0:
                newOptions = [-45,45]
                secondAngle = random.choice(newOptions)
                self.mover.rotate_angle(math.radians(secondAngle),1.0)
                newAngle += secondAngle#used for calculating new pose


            self.currentDir = self._getNewPose(newAngle)

            self._decay()#now decay at end

            self._increaseCoordValues(self.currentPos)


        except Exception, e: #if None on the maxScoreCoord
            exc_type, exc_obj, exc_tb = sys.exc_info()
            rospy.loginfo("Line number: " + str(exc_tb.tb_lineno) + " " + str(e))

            rospy.loginfo("couldn't find a best coordinate to go to!!!")

    def _increaseCoordValues(self,coordinate):
        '''increases values for coordinate and surrounding coordinates'''
        x = coordinate[0]
        y = coordinate[1]
        self.grid_coords[coordinate] += (1-self.AMOUNT_TO_ADD)
        for i in range(x-1,x+1+1):
            for j in range(y-1,y+1+1):
                if (i,j) in self.grid_coords.keys():
                    self.grid_coords[(i,j)] += self.AMOUNT_TO_ADD
                else:
                    self.grid_coords[(i, j)] = self.AMOUNT_TO_ADD

    def _getNewPose(self,newAngle):
        '''gets new pose from angle moved'''
        newDir = Direction.NORTH
        if newAngle == 0:  # i.e stay as is
            newDir = self.currentDir
        else:
            newDir = (self.currentDir + (-1 * (newAngle%360)/90)) % 4#(self.currentDir + (-1 * (newAngle / 90))) % 4
        return newDir

    def _angleToTurn(self, index):
        '''given the index from my convention:
            E:0, NE: 1, N: 2, NW: 3, W: 4, I will give the amount to turn'''
        if index == 0:#E
            return -90
        elif index == 1:#NE
            return -45
        elif index == 2:#N
            return 0
        elif index == 3:#NW
            return 45
        elif index == 4:#W
            return 90

    def _maxScoreCoord(self,coordsWithScores):
        '''goes through a dictionary of coordinates with scores as values and returns best coordinate'''
        bestCoord = None
        bestScore = 0
        for i in coordsWithScores.keys(): #generic maximum finder
            if coordsWithScores[i] > bestScore:
                bestScore = coordsWithScores[i]
                bestCoord = i

        return bestCoord

    def _decay(self):
        '''decays each value in grid coordinates to account for inaccuracies in movement and measurement'''
        for i in self.grid_coords.keys():
            self.grid_coords[i] /= self.DECAY


    def _generateScore(self,coordinate,sensor_reading, direction):
        '''generates score based on visited and sensor_reading and whether it is diagonal or not'''

        shouldReturnZero =    (((direction==1 or direction==3) and (sensor_reading < self.THRESHOLD_DIAGONAL))
                            or ((direction==0 or direction==4) and (sensor_reading < self.THRESHOLD_HORIZONTAL))
                            or ((direction==2)                 and (sensor_reading < self.THRESHOLD_VERTICAL)))

        if shouldReturnZero:
            return 0
        else:
            return sensor_reading/(pow(self.MULTIPLIER, self.grid_coords[coordinate]))  # accounts for visited over distance

    def _getNewPoints(self):
        '''method uses current direction and position to get new grid cells it can see
           Please don't hate me for this, they have to be in a certain orientation to combine well with laser readings :)'''
        newPoints = []
        x = self.currentPos[0]
        y = self.currentPos[1]
        if self.currentDir == Direction.NORTH:
            newPoints.append((x+1,y))
            newPoints.append((x+1,y+1))
            newPoints.append((x,y+1))
            newPoints.append((x-1,y+1))
            newPoints.append((x-1,y))
        elif self.currentDir == Direction.EAST:
            newPoints.append((x,y-1))
            newPoints.append((x+1,y-1))
            newPoints.append((x+1,y))
            newPoints.append((x+1,y+1))
            newPoints.append((x,y+1))
        elif self.currentDir == Direction.WEST:
            newPoints.append((x,y+1))
            newPoints.append((x-1,y+1))
            newPoints.append((x-1,y))
            newPoints.append((x-1,y-1))
            newPoints.append((x,y-1))
        elif self.currentDir == Direction.SOUTH:
            newPoints.append((x-1,y))
            newPoints.append((x-1,y-1))
            newPoints.append((x,y-1))
            newPoints.append((x+1,y-1))
            newPoints.append((x+1,y))
        return newPoints

if __name__ == '__main__':
    rospy.init_node('max_distance_node', anonymous=True)
    max = MaxDist()
    while not rospy.is_shutdown():
        max.roundOfMovement()
