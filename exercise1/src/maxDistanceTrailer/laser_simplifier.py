#!/usr/bin/env python
'''file will deal with laser readings
#by simplifying it down to give readings for 5 directions
#N, E, W, NW, NE
@author Charlie Street'''

import rospy
from sensor_msgs.msg import LaserScan
from Distances import SENSOR_TO_CENTER
from numpy import *
import math
from movement import AABB, pol_to_cart, get_laser_xy


class LaserFiveWays :


    def __init__(self, mover):
        '''constructor sets up subscribers'''
        self.mover = mover
        self.last_laser_data = None
        rospy.Subscriber('base_scan', LaserScan, self._laser_callback)

    def _laser_callback(self, last_laser_data):
        self.last_laser_data = last_laser_data.ranges

    def getSmallMinRange(self, laser_data, index):
        #gets minimum over small range of laser data
        mymin = float('inf')
        for i in range(index-3,index+3+1):
            if (not isnan(laser_data[i])) and laser_data[i] < mymin:
                mymin = laser_data[i]
        return mymin

    def getlasernum(self, num):
        return int(round( (float(num)/float(512)) * len(self.last_laser_data)))

    def getFiveDirection(self):
        '''gets simplified array from full sensor reading
           returns in order E,NE,N,NW,W'''
        laser_data = self.last_laser_data

        if laser_data is None:
            return None

        '''
        size = len(laser_data)
        proportions = [0,0.23501,0.5,0.76499,1]#mathematically calculated considering distance from robots centre to robots sensor
        returnArr = []#array for returning all five directions
        for i in proportions:#get angles to check
            currentVal = int(round((i * size) - (0 if i == 0 else 1) ))#make as values within array
            currentVals = []#list for storing values
            for j in range(currentVal-10,currentVal+10+1):
                if j >=0 and j < size:
                    if not isnan(laser_data[j]):
                        currentVals.append(laser_data[j])

            returnArr.append(min(currentVals))#sum(currentVals)/len(currentVals))
            if len(returnArr) == 3 :
                returnArr[2] += SENSOR_TO_CENTER


        #returnArr[0] = 0#dont want east or west now
        #returnArr[4] = 0
        '''

        returnArr = [0,0,0,0,0]#array for storing return values from laser readings

        #Dealing with North. index 2
        minimumNorth = float('inf')#want largest value possible
        for i in range(self.getlasernum(245),self.getlasernum(265)+1):
            if (not isnan(laser_data[i])) and laser_data[i] < minimumNorth:
                minimumNorth = laser_data[i]
        returnArr[2] = minimumNorth

        #Dealing with East, index 0
        minimumEast = float('inf')
        for i in range(self.getlasernum(0),self.getlasernum(10)+1):
            if (not isnan(laser_data[i])) and laser_data[i] < minimumEast:
                minimumEast = laser_data[i]
        returnArr[0] = minimumEast
        returnArr[0] = 0


        #Dealing with West, index 4
        minimumWest = float('inf')
        rospy.loginfo(str(len(laser_data)))
        for i in range(self.getlasernum(502),self.getlasernum(511)+1):
            rospy.loginfo(str(i))
            if (not isnan(laser_data[i])) and laser_data[i] < minimumWest:
                minimumWest = laser_data[i]
        returnArr[4] = minimumWest
        returnArr[4] = 0

        #Dealing with North West, index 3
        #All numbers are calculated by doing trigonometry on a grid of 1x1m
        #and then trying to get the best spread throughout a box
        reading1 = self.getSmallMinRange(laser_data,self.getlasernum(391))
        reading2 = self.getSmallMinRange(laser_data,self.getlasernum(358))
        reading3 = self.getSmallMinRange(laser_data,self.getlasernum(312))
        reading4 = self.getSmallMinRange(laser_data,self.getlasernum(426))
        reading5 = self.getSmallMinRange(laser_data,self.getlasernum(472))
        boxisfine = (reading1 > 2.03 and reading2 > 1.69 and reading3 > 1.45
                 and reading4 > 1.73 and reading5 > 1.54)
        if boxisfine:#if we can go here get max reading
            returnArr[3] = max([reading1,reading2,reading3,reading4,reading5])


        #Dealing with North East, index 1
        reading6 = self.getSmallMinRange(laser_data,self.getlasernum(119))
        reading7 = self.getSmallMinRange(laser_data,self.getlasernum(152))
        reading8 = self.getSmallMinRange(laser_data,self.getlasernum(198))
        reading9 = self.getSmallMinRange(laser_data,self.getlasernum(84))
        reading10 = self.getSmallMinRange(laser_data,self.getlasernum(38))

        boxisfineNE = (reading6 > 2.03 and reading7 > 1.69 and reading8 > 1.45
                   and reading9 > 1.73 and reading10 > 1.54)
        if boxisfineNE:
            returnArr[1] = max([reading6,reading7,reading8,reading9,reading10])


        '''
        Somehow not detecting things in range of the box (it is very rare to get a detection)
        '''


        ranges = laser_data
        squares = [(0.5, -SENSOR_TO_CENTER - 0.5),(0.5, 0.5 - SENSOR_TO_CENTER),(-0.5, 0.5 - SENSOR_TO_CENTER),(-1.5, 0.5 - SENSOR_TO_CENTER),(-1.5, -SENSOR_TO_CENTER - 0.5)]#mathematically calculated considering distance from robots centre to robots sensor
        for s, square in enumerate(squares): #get angles to check
            bb = AABB(square[0], square[1], 1, 1)
            if s != 0 and s != 4:
                inBounds = False
                for i, r in enumerate(ranges):
                    #theta = math.radians(90 - ((i/512)*180))
                    x, y = get_laser_xy(r, i)

                    if bb.test((x, y)):
                        inBounds = True
                        self.mover.viz.place_robot_marker(x, y)

                #rospy.loginfo("square: " + str(square))
                if inBounds:
                    returnArr[s] = 0

            if returnArr[s] is 0:
                self.mover.viz.draw_AABB(bb, color=(1,0,0,1))
            else:
                self.mover.viz.draw_AABB(bb, color=(0,1,0,1))

        return returnArr
