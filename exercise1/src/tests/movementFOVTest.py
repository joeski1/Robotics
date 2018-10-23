#!/usr/bin/env python
#Nishanth test

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

#Check if there seems to be an object in front of the robot, on the data from the left value to the right value, if any of them are less than minRange then return's false
def freeFront(data, left, right, minRange):
	for i in [left, right]:
		ranger = data[i]
		if ranger < minRange:
			return False

	return True

#find the greater average from fLow->fHigh in data and sLow->sHigh in data.
def findGreater(data, fLow, fHigh, sLow, sHigh):
	firstRunningTotal = 0;
	for i in [fLow, fHigh]:
		firstRunningTotal = firstRunningTotal + data[i]
	firstAverage = firstRunningTotal / 128

	secondRunningTotal = 0
	for i in [sLow, sHigh]:
		secondRunningTotal = secondRunningTotal + data[i]
	secondAverage = secondRunningTotal / 128

	if firstAverage > secondAverage:
		return 1
	if  firstAverage < secondAverage:
		return 2

	return 1


def laser_callback(laser_data):
	new_speed = Twist()

	#Values of laser indicators that indicate front
	frontLeft = 150
	frontRight = 350
	#Range to initiate rotation
	minRange = 0.75

	#Range to indicate left
	firstLower = 64
	firstHigher = 128
	#Range to indicate right
	secondLower = 128
	secondHigher = 192

	if freeFront(laser_data.ranges, frontLeft, frontRight, minRange):
		#if ok then go forward, and make adjustments to keep within range of wall
		#pick left or right
		choice = findGreater(laser_data.ranges, firstLower, firstHigher, secondLower, secondHigher)

		new_speed.linear.x = 1
	else:
		#find new wall by rotating
		rospy.loginfo(str(choice))
		new_speed.linear.x = 0.0
		new_speed.angular.z = 5



	pub.publish(new_speed)

if __name__ == '__main__':
	rospy.init_node('obstacle_stopper')
	rospy.Subscriber('base_scan', LaserScan, laser_callback)
	pub = rospy.Publisher('cmd_vel', Twist, queue_size=100)
	rospy.spin()
